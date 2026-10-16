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
import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from tcm_extract_engine.cab.planner import PromptTemplateRegistry
from tcm_extract_engine.cab.profiler import load_blocks
from tcm_extract_engine.common.concurrency import get_verify_semaphore
from tcm_extract_engine.core.llm import (
    ImageContent,
    LLMError,
    TextContent,
    UnifiedLLMClient,
    is_truncation_error,
)
from tcm_extract_engine.core.records import SourceRecord
from tcm_extract_engine.core.verdict import Verdict
from tcm_extract_engine.core.vision import ImageLoader
from tcm_extract_engine.domains import get_domain_pack

logger = logging.getLogger(__name__)

# 单批核验的记录条数（LLM 调用粒度）
VERIFIER_BATCH_SIZE = 8

# 单条记录可附带的原文上限（字符；超长截断并标记）
MAX_SOURCE_CHARS_PER_RECORD = 24_000

# B 阶段系统提示
_SYSTEM_PROMPT_B = "你是抽取结果核验器（Verifier）。严格按指令输出 JSON。"

# 需要人工复核的 verdict 子集由领域包声明（manifest.review_verdicts）

# 记录中不参与 B 修正比对的字段（流水线管理字段，非事实内容）
_NON_FACTUAL_FIELDS = frozenset(
    {
        "record_id",
        "book_id",
        "domain_id",
        "schema_version",
        "generation",
        "source",
        "confidence",
        "needs_review",
        "ambiguities",
        "passthrough",
    }
)


_VERDICT_MAP: dict[str, Verdict] = {v.value: v for v in Verdict}


class VerificationResult(BaseModel):
    """单条记录的核验结果。"""

    record_id: str
    verdict: Verdict
    evidence_quote: str | None = None  # B 引用的原文依据
    explanation: str | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)  # B 报告的字段问题
    needs_review: bool = False


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
    domain_id: str = "tcm"
    results: list[VerificationResult] = Field(default_factory=list)
    corrections: list[CorrectionEntry] = Field(default_factory=list)

    def verdict_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.verdict.value] = counts.get(r.verdict.value, 0) + 1
        return counts

    @property
    def needs_review_count(self) -> int:
        review = get_domain_pack(self.domain_id).review_verdicts
        return sum(1 for r in self.results if r.needs_review or r.verdict.value in review)

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
    vision_client: UnifiedLLMClient | None = None,
    pack: Any = None,
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
    pack = pack or get_domain_pack(records[0].domain_id if records else "tcm")
    registry = registry or PromptTemplateRegistry()
    template = registry.load("b", "factual_verifier.md")
    # 默认走 verify 专属闸门（VERIFY_CONCURRENCY）：不传时会退到全局 LLM 信号量
    # （LLM_CONCURRENCY=200），全部批次一次性超发给只有数个槽位的本地校验服务，
    # 服务端排队时间计入客户端超时，引发「超时→重排队尾→再超时」空转。
    semaphore = semaphore or get_verify_semaphore()

    blocks = load_blocks(blocks_path)
    blocks_by_id = {str(b.get("blockid")): b for b in blocks}
    sections_text = _sections_text(blocks)

    # 1) 原文定位：优先 section 标题，回退 block_ids
    verifiable: list[tuple[SourceRecord, str]] = []
    results: list[VerificationResult] = []
    visual_records: list[SourceRecord] = []
    for record in records:
        if record.domain_id != pack.id:
            raise VerificationError("核验记录与领域包不匹配")
        if record.passthrough:
            # 表格直通：原表逐字透传，天然保真，不送 B 核验
            results.append(
                VerificationResult(
                    record_id=record.record_id,
                    verdict=Verdict.PASSTHROUGH,
                    explanation="表格直通记录（原表逐行透传，免 LLM 核验）",
                )
            )
            continue
        if record.record_type == "FIGURE":
            if vision_client is not None:
                visual_records.append(record)
            else:
                results.append(
                    VerificationResult(
                        record_id=record.record_id,
                        verdict=Verdict.FIGURE_REVIEW,
                        explanation="未配置视觉核验客户端，转人工复核",
                    )
                )
                record.needs_review = True
            continue
        text = _source_for(record, sections_text, blocks_by_id)
        if text is None:
            results.append(
                VerificationResult(
                    record_id=record.record_id,
                    verdict=Verdict.UNSUPPORTED,
                    explanation="无法定位原文（无 section/block 定位），未送 LLM 核验",
                )
            )
            record.needs_review = True
        else:
            verifiable.append((record, text[:MAX_SOURCE_CHARS_PER_RECORD]))

    # 2) 分批
    batches = [verifiable[i : i + batch_size] for i in range(0, len(verifiable), batch_size)]
    total_batches = len(batches)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    done_count = 0

    async def _run_batch(batch: list[tuple[SourceRecord, str]]) -> dict[str, Any]:
        """返回 B 的原始响应 dict（由调用方解析）。"""
        cache_key = _batch_cache_key(
            batch, template.version + ":" + str(getattr(client, "cache_identity", getattr(client, "model", "")))
        )
        if cache_dir is not None:
            cached = _load_batch_cache(cache_dir, cache_key, template.version)
            if cached is not None:
                return cached
        payload = _build_payload(batch)
        user_prompt = template.render(json.dumps(payload, ensure_ascii=False))
        try:
            response = await client.complete_json(
                system_prompt=_SYSTEM_PROMPT_B,
                user_prompt=user_prompt,
                semaphore=semaphore,
                max_retries=2,
            )
        except LLMError as exc:
            # 思考型模型可能把输出预算全耗在推理链上（服务端 n_ctx 余量不足或
            # 链长超 max_tokens），content 为空/JSON 截断；/no_think 指令在本部署
            # 实测无效（推理仍跑满上限），重采样的链长分布也常落不进预算。
            # 照 extractors 截断拆批的同款模式：递归拆半缩小单请求 prompt，
            # 为思考链换余量；子批独立缓存，后续重跑直接命中。
            if not (exc.kind == "empty" or is_truncation_error(exc)):
                raise
            if len(batch) <= 1:
                raise
            logger.warning(
                "[Verifier] 批次撞输出预算 (%s)，拆半重试",
                exc.kind,
            )
            mid = len(batch) // 2
            left, right = await asyncio.gather(_run_batch(batch[:mid]), _run_batch(batch[mid:]))
            response = {
                **right,
                "records": [*left.get("records", []), *right.get("records", [])],
            }
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

    gathered = await asyncio.gather(*(_tracked(b) for b in batches), return_exceptions=True)

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

    async def verify_image(record: SourceRecord) -> dict[str, Any]:
        assert vision_client is not None
        image_template = registry.load("b", "figure_verifier.md")
        paths = list(
            dict.fromkeys(
                str(blocks_by_id[b].get("asset_path") or "")
                for b in record.source.block_ids
                if b in blocks_by_id
            )
        )
        paths = [p for p in paths if p]
        if not paths:
            return {
                "records": [
                    {
                        "record_id": record.record_id,
                        "verdict": "FIGURE_REVIEW",
                        "needs_human_review": True,
                        "issues": [{"field": "image_ref", "problem": "原图来源无法定位"}],
                    }
                ]
            }
        contents: list[Any] = []
        hashes: list[str] = []
        for path in paths:
            try:
                contents.append(
                    ImageContent(image_url={"url": ImageLoader.to_data_url(path, max_size=2048)})
                )
                hashes.append(ImageLoader.image_hash(path))
            except (OSError, ValueError, RuntimeError) as exc:
                raise VerificationError(f"视觉核验无法加载原图: {path}") from exc
        text = _source_for(record, sections_text, blocks_by_id) or ""
        basis = _batch_cache_key(
            [(record, text)],
            image_template.version + str(hashes) + str(getattr(vision_client, "cache_identity", getattr(vision_client, "model", ""))),
        )
        if cache_dir is not None:
            cached = _load_batch_cache(cache_dir, basis, image_template.version)
            if cached is not None:
                return cached
        payload = _build_payload([(record, text)])
        contents.insert(
            0, TextContent(text=image_template.render(json.dumps(payload, ensure_ascii=False)))
        )
        response = await vision_client.complete_multimodal_json(
            system_prompt=_SYSTEM_PROMPT_B,
            user_contents=contents,
            semaphore=semaphore,
            max_retries=2,
        )
        if cache_dir is not None:
            _save_batch_cache(cache_dir, basis, image_template.version, response)
        return response

    if visual_records:
        batch_results.extend(await asyncio.gather(*(verify_image(r) for r in visual_records)))

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
            if record is None and record_id.startswith("SR-") and len(record_id) >= 16:
                matches = [r for rid, r in records_by_id.items() if rid.startswith(record_id)]
                if len(matches) == 1:
                    record = matches[0]
                    record_id = record.record_id
                    item = {**item, "record_id": record_id}
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
                results.append(
                    VerificationResult(
                        record_id=record.record_id,
                        verdict=Verdict.PARTIAL,
                        explanation="B 未返回该记录的核验结果（漏答），转人工复核",
                        needs_review=True,
                    )
                )
                record.needs_review = True

    report = VerificationReport(
        total=len(records), domain_id=pack.id, results=results, corrections=all_corrections
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
    """定位记录的核验原文：block_ids 优先，回退 section 标题。

    block_ids 是抽取时读到的确切块，对标题污染/大纲修复后的规范化标题都免疫；
    section 标题查不到时才回退（老记录可能缺 block_ids）。
    """
    if record.source.block_ids:
        parts = [
            str(blocks_by_id[b].get("content") or "")
            for b in record.source.block_ids
            if b in blocks_by_id
        ]
        if parts:
            return "\n".join(parts)
    if record.source.section_path:
        title = record.source.section_path[-1].strip()
        text = sections_text.get(title)
        if text:
            return text
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
        {
            "field": str(i.get("field") or ""),
            "problem": str(i.get("problem") or ""),
            "source_evidence": i.get("source_evidence"),
        }
        for i in (item.get("issues") or [])
        if isinstance(i, dict)
    ]
    explanation = "; ".join(
        f"{i['field']}: {i['problem']}" for i in issues if i["field"] or i["problem"]
    )
    if unknown_verdict:
        explanation = (
            explanation + "; " if explanation else ""
        ) + f"verdict 无法解析: {verdict_raw!r}"

    needs_review = (
        bool(item.get("needs_human_review"))
        or verdict.value in get_domain_pack(record.domain_id).review_verdicts
        or unknown_verdict
    )
    if needs_review:
        record.needs_review = True

    result = VerificationResult(
        record_id=record.record_id,
        verdict=verdict,
        evidence_quote=item.get("source_evidence"),
        explanation=explanation or None,
        issues=issues,
        needs_review=needs_review or record.needs_review,
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
    corrected = dict(corrected)
    # TCM templates historically emit typed fields at top level; normalize only
    # at the model-output boundary, never while loading serialized artifacts.
    corrected = get_domain_pack(record.domain_id).normalize_correction(corrected)
    overrides = {k: v for k, v in corrected.items() if k in {"payload", "subject_text", "raw_text"}}
    if "payload" in overrides and isinstance(overrides["payload"], dict):
        overrides["payload"] = _merge_dict(full["payload"], overrides["payload"])
    try:
        merged = SourceRecord.model_validate({**full, **overrides}).model_dump(mode="json")
    except (ValidationError, ValueError) as exc:
        return [
            CorrectionEntry(
                record_id=record.record_id,
                field="(record)",
                original=None,
                suggested=corrected,
                reason=f"B 修正未通过载荷校验，转人工: {exc}",
                verdict=verdict,
            )
        ]
    entries: list[CorrectionEntry] = []

    def walk(old: Any, new: Any, path: str) -> None:
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                escaped = key.replace("~", "~0").replace("/", "~1")
                walk(old.get(key), new.get(key), path + "/" + escaped)
        elif old != new:
            entries.append(
                CorrectionEntry(
                    record_id=record.record_id,
                    field=path,
                    original=old,
                    suggested=new,
                    reason="B 依据原文建议修正（待人工确认后生效）",
                    verdict=verdict,
                )
            )

    for key in overrides:
        walk(full[key], merged[key], "/" + key)
    return entries


def _merge_dict(base: dict, update: dict) -> dict:
    return {
        **base,
        **{
            k: _merge_dict(base[k], v)
            if isinstance(v, dict) and isinstance(base.get(k), dict)
            else v
            for k, v in update.items()
        },
    }


def apply_correction(record: SourceRecord, correction: CorrectionEntry) -> SourceRecord:
    import copy

    if correction.record_id != record.record_id or correction.field == "(record)":
        raise ValueError("整条无效修正不能自动应用")
    parts = [p.replace("~1", "/").replace("~0", "~") for p in correction.field.split("/")[1:]]
    if not parts or parts[0] not in {"payload", "subject_text", "raw_text"}:
        raise ValueError("仅允许应用事实字段路径修正")
    data = copy.deepcopy(record.model_dump(mode="json"))
    target = data
    for part in parts[:-1]:
        if not isinstance(target.get(part), dict):
            raise ValueError("无效修正路径")
        target = target[part]
    if target.get(parts[-1]) != correction.original:
        raise ValueError("修正原值已变化，请重新核对")
    target[parts[-1]] = correction.suggested
    result = SourceRecord.model_validate(data)
    result.needs_review = True
    return result


# ── 批级缓存 ─────────────────────────────────────────────────────────────────


def _batch_cache_key(batch: list[tuple[SourceRecord, str]], template_version: str) -> str:
    """缓存键 = 记录内容 hash + 模板版本（记录内容变化即失效）。"""
    basis = "\n".join(
        json.dumps(
            r.model_dump(mode="json", exclude=set(_NON_FACTUAL_FIELDS)),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\nSOURCE:"
        + text
        for r, text in batch
    )
    digest = hashlib.sha256((basis + "\nCONTEXT:" + template_version).encode()).hexdigest()[:24]
    return f"{digest}-{template_version[:8]}"


def _load_batch_cache(cache_dir: Path, key: str, template_version: str) -> dict[str, Any] | None:
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
