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
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from tcm_extract_engine.cab.planner import PromptTemplateRegistry, SectionPlan
from tcm_extract_engine.cab.profiler import load_blocks
from tcm_extract_engine.core.artifacts import load_artifact
from tcm_extract_engine.core.llm import (
    ImageContent,
    LLMError,
    TextContent,
    UnifiedLLMClient,
    is_truncation_error,
)
from tcm_extract_engine.core.records import (
    GenerationRef,
    ImageRef,
    SourceRecord,
    SourceRef,
    stable_record_id,
)
from tcm_extract_engine.core.tables import parse_table
from tcm_extract_engine.core.vision import ImageLoader
from tcm_extract_engine.domains import DomainPack, get_domain_pack

logger = logging.getLogger(__name__)

# 单次 LLM 调用的 section 文本上限（字符）
MAX_SECTION_CHARS_PER_CALL = 24_000

# A 阶段系统提示（与模板内的角色定位呼应）
_SYSTEM_PROMPT = "你是忠实的信息抽取引擎。严格按指令输出 JSON。"

# 多模态图片批次固定使用的模板：图片块可能内嵌在非 FIGURE 章节
# （如 STRUCTURED_TABLE 章节里的流程图），此时 batch.section.template 是章节模板
# （herb_table.md），必须与图片抽取解耦，统一走 figure.md。
FIGURE_TEMPLATE = "figure.md"


class ExtractionError(Exception):
    """抽取阶段错误。"""


@dataclass(slots=True)
class ExtractionProgress:
    """抽取进度回调。"""

    current: int
    total: int
    records: int


# ── 记录解析：LLM 输出 → SourceRecord ───────────────────────────────────────
#
# 解析逻辑收编于领域包 domains/tcm/parse.py；本模块保留旧签名 shim
# （tests 依赖 _parse_record/_as_str_list 的导入面）。


def _parse_record(
    raw: dict[str, Any],
    *,
    book_id: str,
    seq: int,
    section: SectionPlan,
    pack: DomainPack | None = None,
) -> SourceRecord:
    """将 LLM 输出的单条记录解析为 SourceRecord（委托领域包，宽松容错）。"""
    return (pack or get_domain_pack()).parse_record(
        raw,
        book_id=book_id,
        seq=seq,
        extractor=section.extractor,
        page_fallback=section.page_start,
        heading_path=section.heading_path,
        title=section.title,
    )


# ── Section 组块（结构驱动）────────────────────────────────────────────────


@dataclass(slots=True)
class SectionChunk:
    """单个 section 的抽取批次。"""

    section: SectionPlan
    texts: list[str]  # 本批包含的 block 文本
    pages: list[int | None]
    block_ids: list[str] = field(default_factory=list)  # 与 texts 对齐的 blockid（溯源用）
    start: int = 0  # 本批首块在 section 内的序号（保证 chunk_key 唯一）
    chunk_key: str = ""  # section_id + 起始序号 + 内容 hash
    passthrough: bool = False  # 表格直通批次：不经 LLM，原表逐行透传
    table_semantic: bool = False
    multimodal: bool = False  # 多模态批次：含图片，需调用 vision model
    image_paths: list[str] = field(default_factory=list)  # 图片路径列表（multimodal=True 时有效）


def split_section_batches(
    section: SectionPlan, blocks_by_id: dict[str, dict[str, Any]], *, pack: DomainPack | None = None
) -> list[SectionChunk]:
    """将章节 blocks 组块为 LLM 批次。

    结构驱动原则：
      - 表格块（drawing）自成直通批次（完整表优先，不经 LLM）
      - 图片块（drawing + asset_path）自成多模态批次（调用 vision model）
      - 关系组章节（十八反/十九畏）文本整章一批（禁止切断）
      - 普通文本按字符上限分批
    """
    texts: list[str] = []
    pages: list[int | None] = []
    ids: list[str] = []
    asset_paths: list[str] = []  # 图片路径（与 texts 对齐）
    is_drawing: list[bool] = []  # 是否为 drawing 块
    is_table: list[bool] = []  # 是否为表格块

    for block_id in section.block_ids:
        block = blocks_by_id.get(block_id)
        if block is None:
            continue
        texts.append(str(block.get("content") or ""))
        ids.append(str(block_id))
        positions = block.get("positions") or []
        try:
            pages.append(int(str(positions[0].get("anchor"))) if positions else None)
        except (ValueError, TypeError):
            pages.append(None)
        # 多模态元数据
        asset_path = str(block.get("asset_path") or "")
        asset_paths.append(asset_path)
        block_format = str(block.get("format") or "")
        is_draw = block_format == "drawing" or "<drawing" in str(block.get("content") or "")
        is_drawing.append(is_draw)
        # 表格判定：drawing 块且内容含表格标签（<table> 或 Markdown 表格）
        content = str(block.get("content") or "")
        is_tbl = is_draw and ("<table" in content.lower() or "|---" in content or "|---" in content)
        is_table.append(is_tbl)

    if not texts:
        return []

    is_relation = (pack or get_domain_pack()).chunk_strategies.get(
        section.extractor
    ) == "whole_section"

    # 表格块独立成直通批次；图片块独立成多模态批次；文本块按上限聚合
    batches: list[SectionChunk] = []
    table_chunks: list[SectionChunk] = []
    figure_chunks: list[SectionChunk] = []  # 多模态图片批次
    current: list[str] = []
    current_pages: list[int | None] = []
    current_ids: list[str] = []
    current_start = 0

    for idx, text in enumerate(texts):
        # 表格直通批次
        if is_table[idx]:
            table_chunks.append(
                _make_chunk(section, [text], [pages[idx]], [ids[idx]], start=idx, passthrough=True)
            )
            continue
        # 图片多模态批次（drawing 块且有 asset_path，非表格）
        if is_drawing[idx] and asset_paths[idx]:
            figure_chunks.append(
                _make_chunk(
                    section,
                    [text],
                    [pages[idx]],
                    [ids[idx]],
                    start=idx,
                    multimodal=True,
                    image_paths=[asset_paths[idx]],
                )
            )
            continue
        # 普通文本块
        if not current:
            current_start = idx
        current.append(text)
        current_pages.append(pages[idx])
        current_ids.append(ids[idx])
        if not is_relation and sum(len(t) for t in current) >= MAX_SECTION_CHARS_PER_CALL:
            batches.append(
                _make_chunk(section, current, current_pages, current_ids, start=current_start)
            )
            current, current_pages, current_ids = [], [], []
    if current:
        batches.append(
            _make_chunk(section, current, current_pages, current_ids, start=current_start)
        )
    # 输出顺序：表格直通 → 图片多模态 → 文本批次（稳定可复现）
    return table_chunks + figure_chunks + batches


def _make_chunk(
    section: SectionPlan,
    texts: list[str],
    pages: list[int | None],
    block_ids: list[str],
    *,
    start: int,
    passthrough: bool = False,
    multimodal: bool = False,
    image_paths: list[str] | None = None,
) -> SectionChunk:
    """构造批次；chunk_key = section_id + 起始序号 + 内容 hash（防同内容碰撞）。"""
    content_hash = hashlib.sha1("|".join(texts).encode()).hexdigest()[:8]
    return SectionChunk(
        section=section,
        texts=texts,
        pages=pages,
        block_ids=block_ids,
        start=start,
        chunk_key=f"{section.section_id}:{start}:{content_hash}",
        passthrough=passthrough,
        multimodal=multimodal,
        image_paths=image_paths or [],
    )


# ── 主入口 ──────────────────────────────────────────────────────────────────


async def extract_records(
    *,
    book_id: str,
    blocks_path: Path,
    plan_path: Path,
    cache_dir: Path,
    client: UnifiedLLMClient,
    vision_client: UnifiedLLMClient | None = None,
    registry: PromptTemplateRegistry | None = None,
    semaphore: asyncio.Semaphore | None = None,
    on_progress: Callable[[ExtractionProgress], None] | None = None,
    pack: DomainPack | None = None,
    table_semantics: bool = True,
) -> list[SourceRecord]:
    """按抽取计划执行 Model A，返回全部 SourceRecord。

    Args:
        book_id: 书籍 ID
        blocks_path: 01_parse/blocks.jsonl
        plan_path: 03_plan/extraction_plan.json
        cache_dir: 批级缓存目录（断点续传）
        client: 抽取模型客户端（纯文本）
        vision_client: 视觉模型客户端（多模态图片批次）；未传时回退到 client
        on_progress: 进度回调

    Returns:
        SourceRecord 列表（顺序 = plan 顺序）。

    Raises:
        ExtractionError: 有批次在重试后仍失败（成功批次已缓存，可 resume 续跑）。
    """
    registry = registry or PromptTemplateRegistry()
    vision_client = vision_client or client
    run_id = uuid.uuid4().hex
    plan_data = load_artifact(plan_path)
    pack = pack or get_domain_pack(plan_data.get("domain_id", "tcm"))
    sections = [SectionPlan.model_validate(s) for s in plan_data.get("sections", [])]
    planned = [s for s in sections if not s.skip_reason]

    blocks = load_blocks(blocks_path)
    blocks_by_id = {str(b.get("blockid")): b for b in blocks}

    # 组块
    batches: list[SectionChunk] = []
    for sec in planned:
        batches.extend(split_section_batches(sec, blocks_by_id, pack=pack))

    if not batches:
        logger.warning("[Extractor] 无可抽取批次")
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    total = len(batches)

    async def _run_batch(batch: SectionChunk) -> list[SourceRecord]:
        if batch.passthrough:
            # 表格直通：确定性逐行透传，不送 LLM、不读写缓存
            raw_rows = _passthrough_records(batch, book_id, pack=pack)
            if not raw_rows:
                raise ExtractionError(f"表格解析为空，不能标记为成功: {batch.chunk_key}")
            if not table_semantics:
                return raw_rows
            extractor = (
                "TableKnowledgeExtractor"
                if pack.id == "general"
                else batch.section.extractor
                if batch.section.extractor in {"InteractionExtractor", "PregnancyRuleExtractor"}
                else "HerbTableExtractor"
            )
            template_name = pack.extractor_templates[extractor]
            semantic_section = batch.section.model_copy(
                update={
                    "extractor": extractor,
                    "template": template_name,
                    "prompt_version": registry.version_of("a", template_name),
                }
            )
            semantic = replace(
                batch,
                section=semantic_section,
                passthrough=False,
                table_semantic=True,
                chunk_key=batch.chunk_key + "-facts",
            )
            return raw_rows + await _run_batch(semantic)
        # 多模态图片批次的缓存版本随 figure.md（而非章节模板），
        # 保证修改 figure.md 时旧图片缓存能正确失效。
        is_figure = bool(batch.multimodal and batch.image_paths)
        prompt_version = (
            registry.version_of("a", pack.figure_template)
            if is_figure
            else registry.version_of("a", batch.section.template)
        )
        cached = _load_batch_cache(
            cache_dir,
            batch.chunk_key,
            prompt_version=prompt_version,
            domain_id=pack.id,
            pack_version=pack.version,
            request_identity=_request_identity(batch, vision_client if is_figure else client),
        )
        if cached is not None:
            _attribute_batch(cached, batch)  # 旧缓存也补齐 block 溯源
            return cached

        # 多模态批次：调用 vision model
        if is_figure:
            return await _run_multimodal_batch(
                batch,
                vision_client,
                registry,
                semaphore,
                book_id,
                cache_dir,
                pack=pack,
                run_id=run_id,
            )

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
                logger.warning("[Extractor] 批次输出截断，对半拆分重试: %s", batch.chunk_key)
                mid = len(batch.texts) // 2
                left = _make_chunk(
                    batch.section,
                    batch.texts[:mid],
                    batch.pages[:mid],
                    batch.block_ids[:mid],
                    start=batch.start,
                )
                right = _make_chunk(
                    batch.section,
                    batch.texts[mid:],
                    batch.pages[mid:],
                    batch.block_ids[mid:],
                    start=batch.start + mid,
                )
                return (await _run_batch(left)) + (await _run_batch(right))
            raise
        raw_records = _record_items(result)
        parsed = [
            _parse_record(r, book_id=book_id, seq=0, section=batch.section, pack=pack)
            for r in raw_records
            if isinstance(r, dict)
        ]
        for rec in parsed:
            rec.generation.template = batch.section.template
            rec.generation.prompt_version = prompt_version
            rec.generation.model = str(getattr(client, "model", ""))
            rec.generation.run_id = run_id
        _save_batch_cache(
            cache_dir,
            batch.chunk_key,
            parsed,
            prompt_version=prompt_version,
            domain_id=pack.id,
            pack_version=pack.version,
            request_identity=_request_identity(batch, client),
        )
        _attribute_batch(parsed, batch)
        return parsed

    # 并发执行：gather 保序；单批失败不中断其他批次；并发度由 semaphore 控制
    done_count = 0
    record_count = 0
    ledger: list[dict[str, Any]] = []

    async def _tracked(batch: SectionChunk) -> list[SourceRecord]:
        nonlocal done_count, record_count
        try:
            result = await _run_batch(batch)
        except Exception as exc:
            ledger.append(
                {
                    "chunk_key": batch.chunk_key,
                    "block_ids": batch.block_ids,
                    "status": "FAILED",
                    "error": str(exc),
                }
            )
            raise
        ledger.append(
            {
                "chunk_key": batch.chunk_key,
                "block_ids": batch.block_ids,
                "status": "DONE",
                "records": len(result),
            }
        )
        done_count += 1
        record_count += len(result)
        if on_progress:
            on_progress(ExtractionProgress(current=done_count, total=total, records=record_count))
        return result

    gathered = await asyncio.gather(*(_tracked(b) for b in batches), return_exceptions=True)

    from tcm_extract_engine.core.artifacts import _atomic_write_json

    _atomic_write_json(
        cache_dir.parent / "batch_ledger.json", sorted(ledger, key=lambda x: x["chunk_key"])
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

    # Content identity remains stable when unrelated batches change.
    for rec in records:
        rec.record_id = stable_record_id(rec)
    records = list({r.record_id: r for r in records}.values())

    logger.info(
        "[Extractor] 完成: %d 批次 → %d 条记录 (calls=%d, tokens_in=%d, tokens_out=%d)",
        len(batches),
        len(records),
        client.usage.calls,
        client.usage.input_tokens,
        client.usage.output_tokens,
    )
    return records


def _dedupe_headers(headers: list[str]) -> list[str]:
    """重复表头加序号后缀，避免 dict 折叠丢列。

    如「正名|别名|正名|别名」两药一行的表：不去重时后者覆盖前者，
    会让 subject_text（首列）与 attributes（末列）指向不同药物。
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw_name in headers:
        name = raw_name.strip()
        if not name:
            out.append(name)
            continue
        count = seen.get(name, 0) + 1
        seen[name] = count
        out.append(name if count == 1 else f"{name}{count}")
    return out


def _passthrough_records(
    batch: SectionChunk, book_id: str, *, pack: DomainPack | None = None
) -> list[SourceRecord]:
    """表格直通：原表逐行透传为 TABLE_ROW 记录（不经 LLM，逐字保真）。

    - 表头→列名映射进 attributes；首列作 subject_text。
    - 不做任何推断/补全：空单元格保留空串，剂量/动作不解析。
    - record_id 先占位，由 extract_records 全局重排覆盖。
    """
    pack = pack or get_domain_pack()
    section = batch.section
    records: list[SourceRecord] = []
    for text, page in zip(batch.texts, batch.pages):
        parsed = parse_table(text)
        headers = _dedupe_headers(parsed.headers)
        for row_index, row in enumerate(parsed.rows):
            # 表头去重后再映射，杜绝重复列名相互覆盖（raw_text 仍留原表头，逐字保真）
            attrs = {h: v for h, v in zip(headers, row) if h}
            first = next((c for c in row if c.strip()), "")
            raw = " | ".join([*parsed.headers, *row])
            records.append(
                SourceRecord(
                    record_id=f"SR-{book_id[:8]}-000000",
                    book_id=book_id,
                    record_type="TABLE_ROW",
                    domain_id=pack.id,
                    source=SourceRef(
                        page_start=page,
                        page_end=page,
                        section_path=section.heading_path or [section.title],
                        block_ids=list(batch.block_ids),
                        chunk_id=batch.chunk_key,
                    ),
                    raw_text=raw,
                    subject_text=first or None,
                    payload={
                        "attributes": attrs,
                        "headers": parsed.headers,
                        "cells": row,
                        "row_index": row_index,
                        "header_rows": parsed.header_rows,
                        "cell_origins": parsed.cell_origins[row_index]
                        if parsed.cell_origins
                        else [],
                        "warnings": parsed.warnings,
                    },
                    passthrough=True,
                    needs_review=bool(parsed.warnings),
                )
            )
    return records


def _attribute_batch(records: list[SourceRecord], batch: SectionChunk) -> None:
    """回填批次级 block 溯源：chunk_id = 批次键；block_ids 精确到命中块（命不中回退整批）。"""
    for rec in records:
        rec.source.chunk_id = batch.chunk_key
        spans = _evidence_spans(rec.raw_text, batch)
        exact = bool(spans)
        rec.source.evidence_spans = spans
        matched = list(dict.fromkeys(span["block_id"] for span in spans))
        structural = rec.record_type in {"TABLE_ROW", "FIGURE"} or batch.table_semantic
        rec.source.block_ids = matched if exact else (list(batch.block_ids) if structural else [])
        rec.source.context_block_ids = list(batch.block_ids)
        rec.source.location_status = "character" if exact else ("block" if structural else "context_only")
        pages = [page for bid, page in zip(batch.block_ids, batch.pages) if bid in rec.source.block_ids and page is not None]
        if pages:
            rec.source.page_start, rec.source.page_end = min(pages), max(pages)
        if not rec.source.block_ids:
            rec.needs_review = True
            if "原文无法精确匹配，定位仅到上下文" not in rec.ambiguities:
                rec.ambiguities.append("原文无法精确匹配，定位仅到上下文")
        rec.generation = rec.generation.model_copy(
            update={
                "extractor": "FigureExtractor" if batch.multimodal else batch.section.extractor,
                "template": rec.generation.template or batch.section.template,
                "prompt_version": rec.generation.prompt_version or batch.section.prompt_version,
                "modality": (
                    "legacy_unknown" if rec.generation.model == "legacy:unattributed"
                    else "image" if batch.multimodal
                    else "table" if batch.passthrough or batch.table_semantic else "text"
                ),
            }
        )


def _match_blocks(raw_text: str, batch: SectionChunk) -> list[str]:
    """把记录 raw_text 反查回具体 block。

    - 单块批次（包括表格直通）：直接返回该块。
    - 多块批次：取内容包含 raw_text 的块（可多个）；无法命中时回退整批 block_ids。
    """
    ids = list(batch.block_ids)
    if len(ids) <= 1:
        return ids
    needle = _norm_ws(raw_text)
    if not needle:
        return ids
    matched = [bid for bid, text in zip(ids, batch.texts) if needle in _norm_ws(text)]
    return matched or ids


def _norm_ws(text: str) -> str:
    """折叠空白以便子串匹配（LLM raw_text 与原块空白可能不一致）。"""
    return " ".join(str(text or "").split())


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
    cache_dir: Path,
    chunk_key: str,
    *,
    prompt_version: str,
    domain_id: str = "tcm",
    pack_version: str = "2",
    request_identity: str = "",
) -> list[SourceRecord] | None:
    """读取批级缓存；模板版本不一致视为失效（需重抽）。"""
    path = cache_dir / f"batch_{_safe_key(chunk_key)}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            data.get("record_schema_version") != "2"
            or data.get("domain_id") != domain_id
            or data.get("pack_version") != pack_version
        ):
            return None
        migration = data.get("legacy_migration") or {}
        if data.get("request_identity", "") != request_identity:
            if not (
                data.get("request_identity") == "legacy:unattributed"
                and migration.get("target_request_identity") == request_identity
                and migration.get("requires_review") is True
            ):
                return None
        if data.get("prompt_version") != prompt_version:
            return None
        records = [SourceRecord.model_validate(r) for r in data.get("records", [])]
        if migration and any(
            not r.needs_review or r.generation.model != "legacy:unattributed" for r in records
        ):
            return None
        return records
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def _save_batch_cache(
    cache_dir: Path,
    chunk_key: str,
    records: list[SourceRecord],
    *,
    prompt_version: str,
    domain_id: str = "tcm",
    pack_version: str = "2",
    request_identity: str = "",
) -> None:
    path = cache_dir / f"batch_{_safe_key(chunk_key)}.json"
    payload = {
        "prompt_version": prompt_version,
        "record_schema_version": "2",
        "domain_id": domain_id,
        "pack_version": pack_version,
        "request_identity": request_identity,
        "records": [r.model_dump(mode="json") for r in records],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def migrate_legacy_batch_cache(
    cache_dir: Path,
    batch: SectionChunk,
    *,
    book_id: str,
    registry: PromptTemplateRegistry,
    client: UnifiedLLMClient,
    pack: DomainPack,
) -> bool:
    """Explicit recovery of one validated legacy cache; never invent its model identity.

    The caller must opt into migration before resume. Original bytes are backed
    up. Only this source batch, unchanged template and target configuration may
    reuse the result, and all recovered records remain subject to review.
    """
    from tcm_extract_engine.core.artifacts import _atomic_write_json
    from tcm_extract_engine.core.migration import migrate_record

    path = cache_dir / f"batch_{_safe_key(batch.chunk_key)}.json"
    if not path.is_file():
        return False
    original = path.read_bytes()
    data = json.loads(original)
    if data.get("record_schema_version") == "2":
        return False
    template_name = pack.figure_template if batch.multimodal else batch.section.template
    prompt_version = registry.version_of("a", template_name)
    if data.get("prompt_version") != prompt_version:
        raise ValueError(f"缓存模板已变化，必须重抽: {path.name}")
    raw_records = _record_items(data)
    records = []
    image_reference_mismatches = 0
    for raw in raw_records:
        if raw.get("book_id") != book_id:
            raise ValueError(f"缓存书籍不匹配: {path.name}")
        record = SourceRecord.model_validate(migrate_record(raw, pack.id))
        image = record.payload.get("image_ref") or {}
        if batch.multimodal and image.get("asset_path") and image["asset_path"] not in batch.image_paths:
            image_reference_mismatches += 1
            record.ambiguities.append("旧缓存图片引用与来源批次不匹配，保留原值，须按原图复核")
        record.needs_review = True
        record.generation = GenerationRef(
            extractor="FigureExtractor" if batch.multimodal else batch.section.extractor,
            template=template_name, prompt_version=prompt_version,
            model="legacy:unattributed", modality="legacy_unknown",
        )
        record.ambiguities.append("旧缓存缺少模型身份信息，已兼容恢复，须复核后使用")
        records.append(record.model_dump(mode="json"))
    backup = path.with_name(path.name + ".legacy-v1.bak")
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError(f"旧缓存备份冲突，拒绝覆盖: {backup.name}")
    else:
        with backup.open("xb") as stream:
            stream.write(original)
    _atomic_write_json(path, {
        "prompt_version": prompt_version, "record_schema_version": "2",
        "domain_id": pack.id, "pack_version": pack.version,
        "request_identity": "legacy:unattributed", "records": records,
        "legacy_migration": {
            "target_request_identity": _request_identity(batch, client),
            "requires_review": True, "empty_output": not records,
            "image_reference_mismatches": image_reference_mismatches,
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "backup": backup.name,
            "original_metadata": {key: value for key, value in data.items() if key != "records"},
        },
    })
    return True


# ── 多模态批次处理（图片/图表）────────────────────────────────────────────────


async def _run_multimodal_batch(
    batch: SectionChunk,
    client: UnifiedLLMClient,
    registry: PromptTemplateRegistry,
    semaphore: asyncio.Semaphore | None,
    book_id: str,
    cache_dir: Path,
    *,
    pack: DomainPack,
    run_id: str = "",
) -> list[SourceRecord]:
    """处理多模态批次：加载图片 → 构建多模态消息 → 调用 vision model → 解析结果。

    降级策略：图片加载失败时，回退到纯文本模式（使用 MinerU 描述）。
    """
    # 1. 加载图片并构建多模态内容
    user_contents: list[TextContent | ImageContent] = []
    loaded_images = 0

    for img_path in batch.image_paths:
        try:
            data_url = ImageLoader.to_data_url(img_path, max_size=1024)
            user_contents.append(ImageContent(image_url={"url": data_url}))
            loaded_images += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FigureExtractor] 图片加载失败，降级为文本: %s (%s)", img_path, exc)

    # 2. 添加文本内容（MinerU 描述 + 章节上下文）
    # 固定用 figure.md：图片块可能内嵌在非 FIGURE 章节，batch.section.template
    # 会是章节模板（如 herb_table.md），对图片抽取出错（返回空记录）。
    text_payload = json.dumps(_build_chunks_payload(batch), ensure_ascii=False)
    template = registry.load("a", pack.figure_template)
    rendered_prompt = template.render(text_payload)
    user_contents.insert(0, TextContent(text=rendered_prompt))

    # 3. 调用 LLM（多模态或降级为纯文本）
    try:
        if loaded_images > 0:
            # 多模态调用
            result = await client.complete_multimodal_json(
                system_prompt=_SYSTEM_PROMPT,
                user_contents=user_contents,
                semaphore=semaphore,
            )
        else:
            raise ExtractionError(f"图片无法加载，禁止用描述冒充视觉抽取: {batch.image_paths}")
    except LLMError as exc:
        logger.error("[FigureExtractor] 多模态调用失败: %s (%s)", batch.chunk_key, exc)
        raise

    # 4. 解析结果
    raw_records = _record_items(result)

    parsed = [
        _parse_record(r, book_id=book_id, seq=0, section=batch.section, pack=pack)
        for r in raw_records
        if isinstance(r, dict)
    ]

    # 4.1 回填图片真实路径（LLM 无从得知 asset_path，由批次元数据注入）
    asset = batch.image_paths[0] if batch.image_paths else ""
    for rec in parsed:
        if rec.record_type != "FIGURE":
            continue
        image_ref = rec.payload.get("image_ref") or ImageRef(
            asset_path=asset, figure_type="其他"
        ).model_dump(mode="json")
        image_ref["asset_path"] = asset
        rec.payload["image_ref"] = image_ref
        rec.payload = pack.validate_payload(rec.record_type, rec.payload)
        rec.generation.template = pack.figure_template
        rec.generation.prompt_version = template.version
        rec.generation.model = str(getattr(client, "model", ""))
        rec.generation.run_id = run_id

    # 5. 缓存与溯源（版本随 figure.md，与 _run_batch 的缓存失效判定一致）
    _save_batch_cache(
        cache_dir,
        batch.chunk_key,
        parsed,
        prompt_version=template.version,
        domain_id=pack.id,
        pack_version=pack.version,
        request_identity=_request_identity(batch, client),
    )
    _attribute_batch(parsed, batch)

    logger.info(
        "[FigureExtractor] 批次完成: %s (图片=%d, 记录=%d)",
        batch.chunk_key,
        loaded_images,
        len(parsed),
    )
    return parsed


def _safe_key(key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", key)[:80]


def _request_identity(batch: SectionChunk, client: UnifiedLLMClient) -> str:
    images = [(str(p), ImageLoader.image_hash(p)) for p in batch.image_paths]
    basis = {
        "model": str(getattr(client, "cache_identity", getattr(client, "model", ""))),
        "images": images,
        "table_semantic": batch.table_semantic,
    }
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()


def _record_items(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict) or "records" not in result:
        raise ExtractionError("模型输出缺少 records 数组")
    records = result["records"]
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ExtractionError("模型 records 必须是对象数组；无事实应显式返回 []")
    return records


def _evidence_spans(raw_text: str, batch: SectionChunk) -> list[dict[str, Any]]:
    needle = "".join(raw_text.split())
    if not needle:
        return []
    chars: list[str] = []
    offsets: list[tuple[int, int]] = []
    for block_index, text in enumerate(batch.texts):
        for offset, char in enumerate(text):
            if not char.isspace():
                chars.append(char)
                offsets.append((block_index, offset))
    start = "".join(chars).find(needle)
    if start < 0:
        return []
    ranges: dict[int, list[int]] = {}
    for block_index, offset in offsets[start:start+len(needle)]:
        ranges.setdefault(block_index, []).append(offset)
    return [{"block_id": batch.block_ids[index], "char_start": min(positions), "char_end": max(positions)+1,
             "quote": batch.texts[index][min(positions):max(positions)+1],
             "text_version": hashlib.sha256(batch.texts[index].encode()).hexdigest()}
            for index, positions in ranges.items()]
