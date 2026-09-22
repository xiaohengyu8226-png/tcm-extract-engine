"""Model A：分章节、分类型忠实抽取。

职责（严格隔离）：
  A 只负责忠实抽取——不判断医学正确性、不修改原文、不补充外部知识。

执行流程：
  1. 读取 extraction_plan（每个 section → extractor + 模板 + prompt_version）
  2. 按 section 组块（结构驱动，非纯 token 长度；表格按完整表/行批次；关系组不切断）
  3. 渲染模板 → LLM（JSON mode）→ 解析为 SourceRecord
  4. 批级缓存（hash = section_id + block_ids + prompt_version）支持断点续传
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tcm_extract_engine.cab.planner import PromptTemplateRegistry, SectionPlan
from tcm_extract_engine.cab.profiler import load_blocks
from tcm_extract_engine.core.artifacts import load_artifact
from tcm_extract_engine.core.llm import LLMError, UnifiedLLMClient, is_truncation_error
from tcm_extract_engine.core.records import (
    ActionLevel,
    HerbDose,
    InteractionGroup,
    InteractionMember,
    PopulationRule,
    PregnancyRule,
    RecordType,
    SourceRecord,
)

logger = logging.getLogger(__name__)

# 单次 LLM 调用的 section 文本上限（字符）
MAX_SECTION_CHARS_PER_CALL = 24_000

# A 阶段系统提示（与模板内的角色定位呼应）
_SYSTEM_PROMPT = "你是忠实的信息抽取引擎。严格按指令输出 JSON。"


class ExtractionError(Exception):
    """抽取阶段错误。"""


@dataclass(slots=True)
class ExtractionProgress:
    """抽取进度回调。"""

    current: int
    total: int
    records: int


# ── 记录解析：LLM 输出 → SourceRecord ───────────────────────────────────────

_RECORD_TYPE_MAP: dict[str, RecordType] = {t.value: t for t in RecordType}

_ACTION_MAP: dict[str, ActionLevel] = {
    "PROHIBIT": ActionLevel.PROHIBIT,
    "AVOID": ActionLevel.AVOID,
    "CAUTION": ActionLevel.CAUTION,
    "MONITOR": ActionLevel.MONITOR,
    "INFO": ActionLevel.INFO,
}


def _parse_record(
    raw: dict[str, Any], *, book_id: str, seq: int, section: SectionPlan
) -> SourceRecord:
    """将 LLM 输出的单条记录解析为 SourceRecord（宽松容错：字段级降级）。"""
    record_type = _RECORD_TYPE_MAP.get(str(raw.get("record_type") or "").upper())
    if record_type is None:
        # 按 extractor 推断默认类型
        record_type = _infer_type_from_extractor(section.extractor)

    page_hint = raw.get("page_hint") or section.page_start
    page: int | None = None
    if page_hint is not None:
        try:
            page = int(str(page_hint))
        except (ValueError, TypeError):
            page = section.page_start

    dose = None
    dose_raw = raw.get("dose")
    if isinstance(dose_raw, dict) and any(
        dose_raw.get(k) is not None for k in ("min_dose", "max_dose", "frequency")
    ):
        dose = HerbDose(
            min_dose=_to_float(dose_raw.get("min_dose")),
            max_dose=_to_float(dose_raw.get("max_dose")),
            unit=str(dose_raw.get("unit") or "g"),
            per_use=bool(dose_raw.get("per_use")),
            frequency=dose_raw.get("frequency"),
            special_note=dose_raw.get("special_note"),
        )

    interaction = _parse_interaction(raw.get("interaction"))
    pregnancy = _parse_pregnancy(raw.get("pregnancy"))
    population = _parse_population(raw.get("population"))

    # 兼容旧模板输出：LLM 将类型化载荷放进 attributes（顶层字段缺失时回退）
    attributes = _clean_attributes(raw.get("attributes"))
    if pregnancy is None and record_type == RecordType.PREGNANCY:
        pregnancy = _parse_pregnancy(attributes)
    if population is None and record_type == RecordType.POPULATION:
        population = _parse_population(attributes)

    action = _ACTION_MAP.get(str(raw.get("action") or "").upper())

    return SourceRecord(
        record_id=f"SR-{book_id[:8]}-{seq:06d}",
        book_id=book_id,
        record_type=record_type,
        source={
            "page_start": page,
            "page_end": page,
            "section_path": section.heading_path or [section.title],
            "block_ids": list(section.block_ids[:5]),
        },
        raw_text=str(raw.get("raw_text") or "")[:500],
        herb_name=raw.get("herb_name") or None,
        subject_text=raw.get("subject_text") or None,
        toxicity_level=raw.get("toxicity_level") or None,
        dose=dose,
        interaction=interaction,
        pregnancy=pregnancy,
        population=population,
        action=action,
        attributes=attributes,
        ambiguities=[str(a) for a in (raw.get("ambiguities") or []) if a],
    )


def _infer_type_from_extractor(extractor: str) -> RecordType:
    mapping = {
        "GeneralRuleExtractor": RecordType.GENERAL_RULE,
        "HerbTableExtractor": RecordType.HERB_TABLE,
        "InteractionExtractor": RecordType.INTERACTION,
        "PregnancyRuleExtractor": RecordType.PREGNANCY,
    }
    return mapping.get(extractor, RecordType.GENERAL_RULE)


def _parse_interaction(value: Any) -> InteractionGroup | None:
    if not isinstance(value, dict):
        return None
    anchor_raw = value.get("anchor") or {}
    if not anchor_raw.get("herb_name"):
        return None
    members = [
        InteractionMember(
            herb_name=str(m.get("herb_name") or ""),
            processed_form=m.get("processed_form"),
            role="member",
        )
        for m in (value.get("members") or [])
        if isinstance(m, dict) and m.get("herb_name")
    ]
    return InteractionGroup(
        group_name=value.get("group_name"),
        interaction_type=str(value.get("interaction_type") or "配伍禁忌"),
        anchor=InteractionMember(
            herb_name=str(anchor_raw.get("herb_name")),
            processed_form=anchor_raw.get("processed_form"),
            role="anchor",
        ),
        members=members,
        direction=value.get("direction"),
    )


def _parse_pregnancy(value: Any) -> PregnancyRule | None:
    if not isinstance(value, dict):
        return None
    if not value.get("restriction_level"):
        return None
    return PregnancyRule(
        restriction_level=str(value["restriction_level"]),
        target_herbs=[str(h) for h in (value.get("target_herbs") or []) if h],
        target_formulas=[str(f) for f in (value.get("target_formulas") or []) if f],
        reason=value.get("reason"),
    )


def _parse_population(value: Any) -> PopulationRule | None:
    if not isinstance(value, dict):
        return None
    if not value.get("population"):
        return None
    return PopulationRule(
        population=str(value["population"]),
        restriction_level=str(value.get("restriction_level") or "慎用"),
        target_herbs=[str(h) for h in (value.get("target_herbs") or []) if h],
        condition_text=value.get("condition_text"),
    )


def _clean_attributes(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items() if v not in (None, "", [], {})}
    return {}


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ── Section 组块（结构驱动）────────────────────────────────────────────────


@dataclass(slots=True)
class SectionChunk:
    """单个 section 的抽取批次。"""

    section: SectionPlan
    texts: list[str]  # 本批包含的 block 文本
    pages: list[int | None]
    start: int = 0  # 本批首块在 section 内的序号（保证 chunk_key 唯一）
    chunk_key: str = ""  # section_id + 起始序号 + 内容 hash


def split_section_batches(
    section: SectionPlan, blocks_by_id: dict[str, dict[str, Any]]
) -> list[SectionChunk]:
    """将章节 blocks 组块为 LLM 批次。

    结构驱动原则：
      - 表格块（drawing）自成批次（完整表优先）
      - 关系组章节（十八反/十九畏）整章一批（禁止切断）
      - 普通文本按字符上限分批
    """
    texts: list[str] = []
    pages: list[int | None] = []
    for block_id in section.block_ids:
        block = blocks_by_id.get(block_id)
        if block is None:
            continue
        texts.append(str(block.get("content") or ""))
        positions = block.get("positions") or []
        try:
            pages.append(int(str(positions[0].get("anchor"))) if positions else None)
        except (ValueError, TypeError):
            pages.append(None)

    if not texts:
        return []

    is_relation = section.extractor == "InteractionExtractor"
    if is_relation:
        return [_make_chunk(section, texts, pages, start=0)]

    # 表格块独立成批；文本块按上限聚合
    batches: list[SectionChunk] = []
    current: list[str] = []
    current_pages: list[int | None] = []
    current_start = 0
    for idx, text in enumerate(texts):
        is_table = "<drawing" in text
        if is_table:
            if current:
                batches.append(_make_chunk(section, current, current_pages,
                                           start=current_start))
                current, current_pages = [], []
            batches.append(_make_chunk(section, [text], [pages[idx]], start=idx))
            continue
        if not current:
            current_start = idx
        current.append(text)
        current_pages.append(pages[idx])
        if sum(len(t) for t in current) >= MAX_SECTION_CHARS_PER_CALL:
            batches.append(_make_chunk(section, current, current_pages,
                                       start=current_start))
            current, current_pages = [], []
    if current:
        batches.append(_make_chunk(section, current, current_pages,
                                   start=current_start))
    return batches


def _make_chunk(
    section: SectionPlan, texts: list[str], pages: list[int | None], *, start: int
) -> SectionChunk:
    """构造批次；chunk_key = section_id + 起始序号 + 内容 hash（防同内容碰撞）。"""
    content_hash = hashlib.sha1("|".join(texts).encode()).hexdigest()[:8]
    return SectionChunk(
        section=section,
        texts=texts,
        pages=pages,
        start=start,
        chunk_key=f"{section.section_id}:{start}:{content_hash}",
    )


# ── 主入口 ──────────────────────────────────────────────────────────────────


async def extract_records(
    *,
    book_id: str,
    blocks_path: Path,
    plan_path: Path,
    cache_dir: Path,
    client: UnifiedLLMClient,
    registry: PromptTemplateRegistry | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_progress: Callable[[ExtractionProgress], None] | None = None,
) -> list[SourceRecord]:
    """按抽取计划执行 Model A，返回全部 SourceRecord。

    Args:
        book_id: 书籍 ID
        blocks_path: 01_parse/blocks.jsonl
        plan_path: 03_plan/extraction_plan.json
        cache_dir: 批级缓存目录（断点续传）
        client: 抽取模型客户端
        on_progress: 进度回调

    Returns:
        SourceRecord 列表（顺序 = plan 顺序）。

    Raises:
        ExtractionError: 有批次在重试后仍失败（成功批次已缓存，可 resume 续跑）。
    """
    registry = registry or PromptTemplateRegistry()
    plan_data = load_artifact(plan_path)
    sections = [SectionPlan.model_validate(s) for s in plan_data.get("sections", [])]
    planned = [s for s in sections if not s.skip_reason]

    blocks = load_blocks(blocks_path)
    blocks_by_id = {str(b.get("blockid")): b for b in blocks}

    # 组块
    batches: list[SectionChunk] = []
    for sec in planned:
        batches.extend(split_section_batches(sec, blocks_by_id))

    if not batches:
        logger.warning("[Extractor] 无可抽取批次")
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    total = len(batches)

    async def _run_batch(batch: SectionChunk) -> list[SourceRecord]:
        cached = _load_batch_cache(cache_dir, batch.chunk_key,
                                   prompt_version=batch.section.prompt_version)
        if cached is not None:
            return cached
        try:
            result = await client.complete_json(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=registry.load("a", batch.section.template).render(
                    json.dumps(_build_chunks_payload(batch), ensure_ascii=False)
                ),
                semaphore=semaphore,
            )
        except LLMError as exc:
            # 输出截断 → 对半拆分重试（模板不变，文本减半）
            if is_truncation_error(exc) and len(batch.texts) > 1:
                logger.warning(
                    "[Extractor] 批次输出截断，对半拆分重试: %s", batch.chunk_key
                )
                mid = len(batch.texts) // 2
                left = _make_chunk(batch.section, batch.texts[:mid], batch.pages[:mid],
                                   start=batch.start)
                right = _make_chunk(batch.section, batch.texts[mid:], batch.pages[mid:],
                                    start=batch.start + mid)
                return (await _run_batch(left)) + (await _run_batch(right))
            raise
        raw_records = result.get("records") or []
        if not isinstance(raw_records, list):
            raw_records = []
        parsed = [
            _parse_record(r, book_id=book_id, seq=0, section=batch.section)
            for r in raw_records
            if isinstance(r, dict)
        ]
        _save_batch_cache(cache_dir, batch.chunk_key, parsed,
                          prompt_version=batch.section.prompt_version)
        return parsed

    # 并发执行：gather 保序；单批失败不中断其他批次；并发度由 semaphore 控制
    done_count = 0
    record_count = 0

    async def _tracked(batch: SectionChunk) -> list[SourceRecord]:
        nonlocal done_count, record_count
        result = await _run_batch(batch)
        done_count += 1
        record_count += len(result)
        if on_progress:
            on_progress(ExtractionProgress(current=done_count, total=total,
                                           records=record_count))
        return result

    gathered = await asyncio.gather(
        *(_tracked(b) for b in batches), return_exceptions=True
    )

    records: list[SourceRecord] = []
    failures: list[str] = []
    for batch, result in zip(batches, gathered):
        if isinstance(result, BaseException):
            failures.append(f"{batch.chunk_key}: {result}")
        else:
            records.extend(result)

    if failures:
        raise ExtractionError(
            f"{len(failures)}/{len(batches)} 批次抽取失败"
            f"（成功批次已缓存，可 resume 续跑）: {'; '.join(failures[:5])}"
        )

    # 全局序号重排
    for seq, rec in enumerate(records, 1):
        rec.record_id = f"SR-{book_id[:8]}-{seq:06d}"

    logger.info(
        "[Extractor] 完成: %d 批次 → %d 条记录 (calls=%d, tokens_in=%d, tokens_out=%d)",
        len(batches), len(records), client.usage.calls,
        client.usage.input_tokens, client.usage.output_tokens,
    )
    return records


def _build_chunks_payload(batch: SectionChunk) -> list[dict[str, Any]]:
    """构造注入模板的文本块列表（含页码与章节上下文）。"""
    items: list[dict[str, Any]] = []
    for text, page in zip(batch.texts, batch.pages):
        item: dict[str, Any] = {"text": text}
        if page is not None:
            item["page"] = page
        items.append(item)
    # 章节上下文（帮助 LLM 定位）
    ctx = {
        "section_id": batch.section.section_id,
        "section_title": batch.section.title,
        "heading_path": batch.section.heading_path,
        "extractor": batch.section.extractor,
    }
    return [{"context": ctx, "blocks": items}]


def _load_batch_cache(
    cache_dir: Path, chunk_key: str, *, prompt_version: str
) -> list[SourceRecord] | None:
    """读取批级缓存；模板版本不一致视为失效（需重抽）。"""
    path = cache_dir / f"batch_{_safe_key(chunk_key)}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("prompt_version") != prompt_version:
            return None
        return [SourceRecord.model_validate(r) for r in data.get("records", [])]
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def _save_batch_cache(
    cache_dir: Path, chunk_key: str, records: list[SourceRecord], *, prompt_version: str
) -> None:
    path = cache_dir / f"batch_{_safe_key(chunk_key)}.json"
    payload = {
        "prompt_version": prompt_version,
        "records": [r.model_dump(mode="json") for r in records],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _safe_key(key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", key)[:80]
