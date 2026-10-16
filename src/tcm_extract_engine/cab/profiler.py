"""Document Profiling：解析产物 → 章节地图 + 内容类型分型。

目标不是抽知识，而是回答"这本书是什么结构"：
  - 哪些章节值得抽（值得抽 → 推荐 extractor）
  - 哪些章节不用抽（目录/前言/参考文献 → REFERENCE_ONLY）

内容分类与路由规则收编于领域包 domains/tcm/classify.py；
本模块 shim 重导出（保持 app/tests 导入面不变）。
输入：01_parse/blocks.jsonl；输出：02_profile/document_profile.json。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from tcm_extract_engine.core.document import DocumentType
from tcm_extract_engine.domains import DomainPack, get_domain_pack

logger = logging.getLogger(__name__)


class SectionProfile(BaseModel):
    """单个章节的画像。"""

    section_id: str  # sec-<idx4>
    title: str
    heading_path: list[str] = Field(default_factory=list)
    content_type: str
    block_ids: list[str] = Field(default_factory=list)
    block_count: int = 0
    char_count: int = 0
    table_count: int = 0  # 含 drawing 表格块的章节 >0
    page_start: int | None = None
    page_end: int | None = None
    sample_text: str = ""  # 前 300 字采样（供 Planner 参考）

    @property
    def worth_extracting(self) -> bool:
        return self.content_type not in ("REFERENCE_ONLY", "IRRELEVANT")


class DocumentProfile(BaseModel):
    """全书画像（02_profile/document_profile.json 主体）。"""

    document_type: DocumentType = DocumentType.UNKNOWN
    domain_id: str = "tcm"
    domain: str = ""  # 领域包 profile_domain（build_profile 注入）
    total_blocks: int = 0
    total_pages: int = 0
    sections: list[SectionProfile] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def legacy_domain(cls, value: Any) -> Any:
        if isinstance(value, dict) and "domain_id" not in value:
            from tcm_extract_engine.domains import registered_domains
            domain = value.get("domain")
            value = {**value, "domain_id": domain if domain in registered_domains() else "tcm"}
        return value

    @property
    def extractable_sections(self) -> list[SectionProfile]:
        return [s for s in self.sections if s.worth_extracting]

    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.sections:
            counts[s.content_type] = counts.get(s.content_type, 0) + 1
        return counts


# ── 主入口 ──────────────────────────────────────────────────────────────────


def load_blocks(blocks_path: Path) -> list[dict[str, Any]]:
    """读取 blocks.jsonl（跳过首行 meta）。"""
    if not blocks_path.is_file():
        raise FileNotFoundError(f"blocks 文件不存在: {blocks_path}")
    blocks: list[dict[str, Any]] = []
    with blocks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "content":
                blocks.append(item)
    return blocks


def build_profile(
    blocks_path: Path, *, book_name: str = "", pack: DomainPack | None = None
) -> DocumentProfile:
    """从 MinerU blocks 构建全书画像。

    按 heading_path（parent_headings + heading）聚合为章节；
    每个章节统计块数/字数/表格数/页码范围并分类。
    """
    pack = pack or get_domain_pack()
    blocks = load_blocks(blocks_path)
    sections = _aggregate_sections(blocks, pack=pack)
    doc_type = DocumentType(pack.infer_document_type(book_name))

    profile = DocumentProfile(
        document_type=doc_type,
        domain=pack.profile_domain,
        domain_id=pack.id,
        total_blocks=len(blocks),
        total_pages=_max_page(blocks),
        sections=sections,
    )
    logger.info(
        "[Profiler] %s: %d blocks → %d sections (%s), 类型分布 %s",
        book_name or blocks_path.parent.name,
        len(blocks),
        len(sections),
        doc_type.value,
        profile.type_counts(),
    )
    return profile


def _aggregate_sections(blocks: list[dict[str, Any]], *, pack: DomainPack) -> list[SectionProfile]:
    """按完整 heading_path 聚合相邻块为章节。"""
    sections: list[SectionProfile] = []
    index_by_key: dict[tuple[str, ...], int] = {}

    for block in blocks:
        heading = str(block.get("heading") or "")
        parents = [str(p) for p in (block.get("parent_headings") or [])]
        # 章节键 = 完整标题路径（level>0 时有效；无标题正文归入最近章节）
        key = tuple((*parents, heading)) if heading else _current_key(sections)
        if key is None or not any(key):
            key = ("",)  # 文档开头无标题内容
        if key not in index_by_key:
            index_by_key[key] = len(sections)
            sections.append(
                SectionProfile(
                    section_id=f"sec-{len(sections):04d}",
                    title=heading or (parents[-1] if parents else "（无标题）"),
                    heading_path=list(key),
                    content_type="KNOWLEDGE"
                    if pack.id == "general"
                    else "NARRATIVE_RULE",  # 分类后回填
                )
            )
        sec = sections[index_by_key[key]]
        content = str(block.get("content") or "")
        sec.block_ids.append(str(block.get("blockid") or ""))
        sec.block_count += 1
        sec.char_count += len(content)
        if str(block.get("format")) == "drawing" or "<drawing" in content:
            sec.table_count += 1
        page = _block_page(block)
        if page is not None:
            if sec.page_start is None or page < sec.page_start:
                sec.page_start = page
            if sec.page_end is None or page > sec.page_end:
                sec.page_end = page
        # 采样文本：取前若干块拼 300 字
        if len(sec.sample_text) < 300:
            sec.sample_text = (sec.sample_text + "\n" + content).strip()[:300]

    # 分类（用标题路径 + 表格数 + 采样文本；父级上下文参与路由，利用大纲层级）
    for sec in sections:
        sec.content_type = pack.route_section(
            sec.title,
            table_count=sec.table_count,
            body_text=sec.sample_text,
            heading_path=sec.heading_path,
        )
    return sections


def _current_key(sections: list[SectionProfile]) -> tuple[str, ...] | None:
    """无标题块归入最近一个章节。"""
    if not sections:
        return None
    last = sections[-1]
    return tuple(last.heading_path)


def _block_page(block: dict[str, Any]) -> int | None:
    """从 positions[0].anchor 提取页码（anchor 为 1-based 页号字符串）。"""
    positions = block.get("positions") or []
    if not positions:
        return None
    try:
        return int(str(positions[0].get("anchor")))
    except (ValueError, TypeError, AttributeError):
        return None


def _max_page(blocks: list[dict[str, Any]]) -> int:
    pages = [p for p in (_block_page(b) for b in blocks) if p is not None]
    return max(pages) if pages else 0
