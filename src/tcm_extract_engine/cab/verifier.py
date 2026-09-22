"""Model B：忠实性核验（Verification）。

B 的职责（严格隔离）：
  - 拿 A 的记录与其章节原文比对，判定事实是否被原文支撑
  - 不判断医学正确性、不使用外部知识、不重新抽取
  - 最小修改原则：修正建议只进入 correction_log（待人工审核后生效），
    B 不直接改写记录内容；唯一允许写回记录的是质量标记 needs_review

判定 verdict（每条记录一个）：
  SUPPORTED            记录被原文完全支撑
  PARTIAL              部分支撑
  UNSUPPORTED          关键字段无原文依据（疑似幻觉）
  MISSING              原文有明确规则但记录遗漏
  WRONG_DOSE           剂量与原文不符
  WRONG_RELATION       配伍关系成员/方向与原文不符
  WRONG_NORMALIZATION  归一化错误（炮制品/别名映射错）
  OCR_UNCERTAIN        原文 OCR 质量不足以判定

产出（06_verify/）：
  verification_results.json —— 每条记录 verdict + 依据
  correction_log.json      —— B 建议的字段级修正（不直接生效）
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from tcm_extract_engine.cab.planner import PromptTemplateRegistry
from tcm_extract_engine.cab.profiler import load_blocks
from tcm_extract_engine.core.llm import UnifiedLLMClient
from tcm_extract_engine.core.records import SourceRecord

logger = logging.getLogger(__name__)

# 单批核验的记录条数（LLM 调用粒度）
VERIFIER_BATCH_SIZE = 8

# 单条记录可附带的原文上限（字符；超长截断并标记）
MAX_SOURCE_CHARS_PER_RECORD = 24_000

# B 阶段系统提示
_SYSTEM_PROMPT_B = "你是抽取结果核验器（Verifier）。严格按指令输出 JSON。"

# 需要人工复核的 verdict
_REVIEW_VERDICTS = frozenset({
    "UNSUPPORTED",
    "MISSING",
    "WRONG_DOSE",
    "WRONG_RELATION",
    "WRONG_NORMALIZATION",
    "OCR_UNCERTAIN",
})

# 记录中不参与 B 修正比对的字段（流水线管理字段，非事实内容）
_NON_FACTUAL_FIELDS = frozenset({
    "record_id", "book_id", "source", "confidence", "needs_review", "ambiguities",
})


class Verdict(str, enum.Enum):
    """B 核验判定。"""

    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    MISSING = "MISSING"
    WRONG_DOSE = "WRONG_DOSE"
    WRONG_RELATION = "WRONG_RELATION"
    WRONG_NORMALIZATION = "WRONG_NORMALIZATION"
    OCR_UNCERTAIN = "OCR_UNCERTAIN"


_VERDICT_MAP: dict[str, Verdict] = {v.value: v for v in Verdict}


class VerificationResult(BaseModel):
    """单条记录的核验结果。"""

    record_id: str
    verdict: Verdict
    evidence_quote: str | None = None  # B 引用的原文依据
    explanation: str | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)  # B 报告的字段问题


class CorrectionEntry(BaseModel):
    """correction_log 中的一条修正建议（不直接生效）。"""

    record_id: str
    field: str  # 建议修正的字段名
    original: Any  # 原值
    suggested: Any  # B 建议值
    reason: str = ""
    verdict: Verdict = Verdict.PARTIAL
    applied: bool = False  # 人工在前端「应用纠错」后置真（覆盖写回核验记录）


class VerificationReport(BaseModel):
    """verify_records 输出（06_verify 主体）。"""

    total: int = 0
    results: list[VerificationResult] = Field(default_factory=list)
    corrections: list[CorrectionEntry] = Field(default_factory=list)

    def verdict_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.verdict.value] = counts.get(r.verdict.value, 0) + 1
        return counts

    @property
    def needs_review_count(self) -> int:
        return sum(1 for r in self.results if r.verdict.value in _REVIEW_VERDICTS)

    def summary(self) -> str:
        counts = self.verdict_counts()
        return (
            f"{self.total} 条记录核验: {counts}，"
            f"{self.needs_review_count} 条需人工复核，"
            f"{len(self.corrections)} 条修正建议"
        )


class VerificationError(Exception):
    """核验阶段错误。"""


# ── 主入口 ──────────────────────────────────────────────────────────────────


async def verify_records(
    *,
    records: list[SourceRecord],
    blocks_path: Path,
    client: UnifiedLLMClient,
    registry: PromptTemplateRegistry | None = None,
    semaphore: asyncio.Semaphore | None = None,
    batch_size: int = VERIFIER_BATCH_SIZE,
    cache_dir: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> VerificationReport:
    """对全部记录执行 Model B 核验。

    Args:
        records: 待核验记录（其中 needs_review 标记会被就地更新）。
        blocks_path: 01_parse/blocks.jsonl（原文来源）。
        client: 核验模型客户端。
        cache_dir: 批级缓存目录（可选，断点续传）。
        on_progress: 进度回调 ``(done, total)``。

    Returns:
        VerificationReport（含 results 与 correction_log 数据）。

    Raises:
        VerificationError: 有批次在重试后仍失败（成功批次已缓存）。
    """
    registry = registry or PromptTemplateRegistry()
    template = registry.load("b", "factual_verifier.md")

    blocks = load_blocks(blocks_path)
    blocks_by_id = {str(b.get("blockid")): b for b in blocks}
    sections_text = _sections_text(blocks)

    # 1) 原文定位：优先 section 标题，回退 block_ids
    verifiable: list[tuple[SourceRecord, str]] = []
    results: list[VerificationResult] = []
    for record in records:
        text = _source_for(record, sections_text, blocks_by_id)
        if text is None:
            results.append(VerificationResult(
                record_id=record.record_id,
                verdict=Verdict.UNSUPPORTED,
                explanation="无法定位原文（无 section/block 定位），未送 LLM 核验",
            ))
            record.needs_review = True
        else:
            verifiable.append((record, text[:MAX_SOURCE_CHARS_PER_RECORD]))

    # 2) 分批
    batches = [verifiable[i:i + batch_size] for i in range(0, len(verifiable), batch_size)]
    total_batches = len(batches)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    done_count = 0

    async def _run_batch(batch: list[tuple[SourceRecord, str]]) -> dict[str, Any]:
        """返回 B 的原始响应 dict（由调用方解析）。"""
        cache_key = _batch_cache_key(batch, template.version)
        if cache_dir is not None:
            cached = _load_batch_cache(cache_dir, cache_key, template.version)
            if cached is not None:
                return cached
        payload = _build_payload(batch)
        response = await client.complete_json(
            system_prompt=_SYSTEM_PROMPT_B,
            user_prompt=template.render(json.dumps(payload, ensure_ascii=False)),
            semaphore=semaphore,
        )
        if cache_dir is not None:
            _save_batch_cache(cache_dir, cache_key, template.version, response)
        return response

    async def _tracked(batch: list[tuple[SourceRecord, str]]) -> dict[str, Any]:
        nonlocal done_count
        result = await _run_batch(batch)
        done_count += 1
        if on_progress:
            on_progress(done_count, total_batches)
        return result

    gathered = await asyncio.gather(
        *(_tracked(b) for b in batches), return_exceptions=True
    )

    failures: list[str] = []
    batch_results: list[dict[str, Any]] = []
    for batch, result in zip(batches, gathered):
        if isinstance(result, BaseException):
            failures.append("; ".join(r.record_id for r, _ in batch[:5]))
        else:
            batch_results.append(result)

    if failures:
        raise VerificationError(
            f"{len(failures)}/{total_batches} 批次核验失败（成功批次已缓存）: "
            + " | ".join(failures[:5])
        )

    # 3) 解析 B 输出（按输入顺序稳定输出）
    records_by_id = {r.record_id: r for r in records}
    verified_ids: set[str] = set()
    all_corrections: list[CorrectionEntry] = []
    for response in batch_results:
        for item in response.get("records") or []:
            if not isinstance(item, dict):
                continue
            record_id = str(item.get("record_id") or "")
            record = records_by_id.get(record_id)
            if record is None:
                logger.warning("[Verifier] B 返回了未知 record_id: %s", record_id)
                continue
            verified_ids.add(record_id)
            result, corrections = _parse_result(record, item)
            results.append(result)
            all_corrections.extend(corrections)

    # 4) B 未覆盖的记录（漏答）→ PARTIAL 兑底并标记复核
    for record in records:
        if record.record_id not in verified_ids:
            if not any(r.record_id == record.record_id for r in results):
                results.append(VerificationResult(
                    record_id=record.record_id,
                    verdict=Verdict.PARTIAL,
                    explanation="B 未返回该记录的核验结果（漏答），转人工复核",
                ))
                record.needs_review = True

    report = VerificationReport(
        total=len(records), results=results, corrections=all_corrections
    )
    logger.info("[Verifier] %s", report.summary())
    return report

# ── 原文定位 ─────────────────────────────────────────────────────────────────


def _sections_text(blocks: list[dict[str, Any]]) -> dict[str, str]:
    """heading → 该章节全部 block 文本（B 核验的原文来源）。"""
    sections: dict[str, list[str]] = {}
    for block in blocks:
        heading = str(block.get("heading") or "").strip()
        if not heading:
            continue
        content = str(block.get("content") or "")
        if content:
            sections.setdefault(heading, []).append(content)
    return {k: "\n".join(v) for k, v in sections.items()}


def _source_for(
    record: SourceRecord,
    sections_text: dict[str, str],
    blocks_by_id: dict[str, dict[str, Any]],
) -> str | None:
    """定位记录的核验原文：section 标题优先，回退 block_ids。"""
    if record.source.section_path:
        title = record.source.section_path[-1].strip()
        text = sections_text.get(title)
        if text:
            return text
    if record.source.block_ids:
        parts = [
            str(blocks_by_id[b].get("content") or "")
            for b in record.source.block_ids
            if b in blocks_by_id
        ]
        if parts:
            return "\n".join(parts)
    return None


def _build_payload(batch: list[tuple[SourceRecord, str]]) -> dict[str, Any]:
    """构造 B 请求载荷：sections 去重 + 记录紧凑视图。"""
    sections: dict[str, str] = {}
    records_payload: list[dict[str, Any]] = []
    for record, text in batch:
        if record.source.section_path:
            key = record.source.section_path[-1]
        elif record.source.block_ids:
            key = f"blocks:{record.source.block_ids[0]}"
        else:
            key = record.record_id
        sections.setdefault(key, text)

        dump = record.model_dump(mode="json")
        for k in _NON_FACTUAL_FIELDS:
            dump.pop(k, None)
        records_payload.append({"record_id": record.record_id, "section": key, **dump})
    return {"sections": sections, "records": records_payload}


# ── B 输出解析 ───────────────────────────────────────────────────────────────


def _parse_result(
    record: SourceRecord, item: dict[str, Any]
) -> tuple[VerificationResult, list[CorrectionEntry]]:
    """解析单条 B 核验输出；就地更新 record.needs_review。"""
    verdict_raw = str(item.get("verdict") or "").upper()
    verdict = _VERDICT_MAP.get(verdict_raw)
    unknown_verdict = verdict is None
    if verdict is None:
        verdict = Verdict.PARTIAL

    issues = [
        {"field": str(i.get("field") or ""), "problem": str(i.get("problem") or ""),
         "source_evidence": i.get("source_evidence")}
        for i in (item.get("issues") or [])
        if isinstance(i, dict)
    ]
    explanation = "; ".join(
        f"{i['field']}: {i['problem']}" for i in issues if i["field"] or i["problem"]
    )
    if unknown_verdict:
        explanation = (explanation + "; " if explanation else "") + \
            f"verdict 无法解析: {verdict_raw!r}"

    needs_review = bool(item.get("needs_human_review")) or \
        verdict.value in _REVIEW_VERDICTS or unknown_verdict
    if needs_review:
        record.needs_review = True

    result = VerificationResult(
        record_id=record.record_id,
        verdict=verdict,
        evidence_quote=item.get("source_evidence"),
        explanation=explanation or None,
        issues=issues,
    )
    corrections = _diff_corrections(record, item.get("corrected_record"), verdict)
    return result, corrections


def _diff_corrections(
    record: SourceRecord, corrected: Any, verdict: Verdict
) -> list[CorrectionEntry]:
    """对比 B 的 corrected_record 与原记录，产出字段级修正建议。

    修正不直接生效：只进 correction_log，待人工审核（最小修改原则）。
    """
    if not isinstance(corrected, dict) or not corrected:
        return []
    full = record.model_dump(mode="json")
    base = {k: v for k, v in full.items() if k not in _NON_FACTUAL_FIELDS}
    overrides = {
        k: v for k, v in corrected.items()
        if k in full and k not in _NON_FACTUAL_FIELDS
    }
    if not overrides:
        return []
    try:
        merged = SourceRecord.model_validate({**full, **overrides})
    except ValidationError as exc:
        return [CorrectionEntry(
            record_id=record.record_id, field="(record)",
            original=None, suggested=corrected,
            reason=f"B 修正未通过 SourceRecord 校验，转人工: {exc.error_count()} 处",
            verdict=verdict,
        )]
    merged_dump = merged.model_dump(mode="json", exclude=_NON_FACTUAL_FIELDS)
    entries = [
        CorrectionEntry(
            record_id=record.record_id, field=key,
            original=base[key], suggested=merged_dump[key],
            reason="B 依据原文建议修正（待人工确认后生效）",
            verdict=verdict,
        )
        for key in base if base[key] != merged_dump[key]
    ]
    return entries


# ── 批级缓存 ─────────────────────────────────────────────────────────────────


def _batch_cache_key(batch: list[tuple[SourceRecord, str]], template_version: str) -> str:
    """缓存键 = 记录内容 hash + 模板版本（记录内容变化即失效）。"""
    basis = "\n".join(
        json.dumps(r.model_dump(mode="json", exclude=_NON_FACTUAL_FIELDS),
                   ensure_ascii=False, sort_keys=True)
        for r, _ in batch
    )
    digest = hashlib.sha1(basis.encode()).hexdigest()[:12]
    return f"{digest}-{template_version[:8]}"


def _load_batch_cache(
    cache_dir: Path, key: str, template_version: str
) -> dict[str, Any] | None:
    path = cache_dir / f"verify_{key}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("prompt_version") != template_version:
            return None
        return data.get("response")
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def _save_batch_cache(
    cache_dir: Path, key: str, template_version: str, response: dict[str, Any]
) -> None:
    path = cache_dir / f"verify_{key}.json"
    payload = {"prompt_version": template_version, "response": response}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
