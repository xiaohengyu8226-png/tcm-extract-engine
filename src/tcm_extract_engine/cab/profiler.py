"""Document Profiling：解析产物 → 章节地图 + 内容类型分型。

目标不是抽知识，而是回答"这本书是什么结构"：
  - 哪些章节值得抽（值得抽 → 推荐 extractor）
  - 哪些章节不用抽（目录/前言/参考文献 → REFERENCE_ONLY）

实现为确定性规则优先（零成本、可回归），LLM 仅在规则不确定时可选介入。
输入：01_parse/blocks.jsonl；输出：02_profile/document_profile.json。
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """章节内容类型；决定 recommended_extractor。"""

    NARRATIVE_RULE = "NARRATIVE_RULE"          # 叙述性规则文本
    STRUCTURED_TABLE = "STRUCTURED_TABLE"      # 结构化表格（附录A/B 等）
    ENTITY_LIST = "ENTITY_LIST"               # 实体清单
    ENTITY_GROUP_LIST = "ENTITY_GROUP_LIST"   # 分组实体清单（妊娠禁用）
    RELATION_PAIR = "RELATION_PAIR"           # 两两关系
    RELATION_GROUP = "RELATION_GROUP"         # 关系组（十八反/十九畏）
    HERB_MONOGRAPH = "HERB_MONOGRAPH"         # 药物专论
    CASE_RECORD = "CASE_RECORD"               # 医案
    FORMULA_RECORD = "FORMULA_RECORD"         # 方剂
    REFERENCE_ONLY = "REFERENCE_ONLY"         # 仅参考
    IRRELEVANT = "IRRELEVANT"                 # 无关


class DocumentType(str, Enum):
    """整本书的文档类型。"""

    STANDARD = "STANDARD"        # 规范/标准（如《中药饮片临床应用规范》）
    HANDBOOK = "HANDBOOK"        # 手册（禁忌手册/毒性手册）
    MONOGRAPH = "MONOGRAPH"      # 专著
    TEXTBOOK = "TEXTBOOK"        # 教材
    CASE_COLLECTION = "CASE_COLLECTION"  # 医案集
    UNKNOWN = "UNKNOWN"


# ── 章节标题 → (ContentType, extractor) 路由规则 ─────────────────────────────

# 规则表：标题关键词 → 内容类型（按优先级从上到下匹配）
_SECTION_RULES: list[tuple[str, ContentType]] = [
    # 配伍禁忌（最高优先级：十八反/十九畏是关系组，不能被 chunk 切断）
    ("十八反", ContentType.RELATION_GROUP),
    ("十九畏", ContentType.RELATION_GROUP),
    ("配伍禁忌", ContentType.RELATION_GROUP),
    ("相畏", ContentType.RELATION_GROUP),
    ("相反", ContentType.RELATION_GROUP),
    # 妊娠
    ("妊娠禁用", ContentType.ENTITY_GROUP_LIST),
    ("妊娠慎用", ContentType.ENTITY_GROUP_LIST),
    ("妊娠禁忌", ContentType.ENTITY_GROUP_LIST),
    ("孕妇禁", ContentType.ENTITY_GROUP_LIST),
    ("孕妇慎", ContentType.ENTITY_GROUP_LIST),
    ("孕期", ContentType.ENTITY_GROUP_LIST),
    ("哺乳", ContentType.ENTITY_GROUP_LIST),
    # 毒性药物表格
    ("有毒", ContentType.STRUCTURED_TABLE),
    ("毒性中药", ContentType.STRUCTURED_TABLE),
    ("医疗用毒性", ContentType.STRUCTURED_TABLE),
    ("毒副作用", ContentType.HERB_MONOGRAPH),
    # 剂量/用法
    ("剂量", ContentType.NARRATIVE_RULE),
    ("用量", ContentType.NARRATIVE_RULE),
    ("用法用量", ContentType.STRUCTURED_TABLE),
    ("煎", ContentType.NARRATIVE_RULE),
    ("炮制", ContentType.NARRATIVE_RULE),
    # 特殊人群
    ("儿童", ContentType.NARRATIVE_RULE),
    ("老年", ContentType.NARRATIVE_RULE),
    ("小儿", ContentType.NARRATIVE_RULE),
    ("肝功能", ContentType.NARRATIVE_RULE),
    ("肾功能", ContentType.NARRATIVE_RULE),
    ("过敏", ContentType.NARRATIVE_RULE),
    # 参考性内容
    ("目录", ContentType.IRRELEVANT),
    ("前言", ContentType.IRRELEVANT),
    ("序言", ContentType.IRRELEVANT),
    ("凡例", ContentType.REFERENCE_ONLY),
    ("参考文献", ContentType.IRRELEVANT),
    ("附录", ContentType.STRUCTURED_TABLE),  # 兜底：附录多为表格
]

# 高语义类型：表格存在时也不被 STRUCTURED_TABLE 优先覆盖。
# 十八反/十九畏、妊娠禁忌目录本身就是表格，降级会丢专用抽取器路由；
# IRRELEVANT 结构页（目录/前言/参考文献）不因含对齐文本而被当表格抽取
_TABLE_PRIORITY_EXEMPT: frozenset[ContentType] = frozenset({
    ContentType.RELATION_GROUP,
    ContentType.RELATION_PAIR,
    ContentType.ENTITY_GROUP_LIST,
    ContentType.IRRELEVANT,
})

# 文档类型推断（书名/文件名关键词）
_DOC_TYPE_RULES: list[tuple[str, DocumentType]] = [
    ("规范", DocumentType.STANDARD),
    ("标准", DocumentType.STANDARD),
    ("指南", DocumentType.STANDARD),
    ("手册", DocumentType.HANDBOOK),
    ("教材", DocumentType.TEXTBOOK),
    ("医案", DocumentType.CASE_COLLECTION),
    ("临证", DocumentType.CASE_COLLECTION),
    ("经验集", DocumentType.CASE_COLLECTION),
]


class SectionProfile(BaseModel):
    """单个章节的画像。"""

    section_id: str  # sec-<idx4>
    title: str
    heading_path: list[str] = Field(default_factory=list)
    content_type: ContentType
    block_ids: list[str] = Field(default_factory=list)
    block_count: int = 0
    char_count: int = 0
    table_count: int = 0  # 含 drawing 表格块的章节 >0
    page_start: int | None = None
    page_end: int | None = None
    sample_text: str = ""  # 前 300 字采样（供 Planner 参考）

    @property
    def worth_extracting(self) -> bool:
        return self.content_type not in (ContentType.REFERENCE_ONLY, ContentType.IRRELEVANT)


class DocumentProfile(BaseModel):
    """全书画像（02_profile/document_profile.json 主体）。"""

    document_type: DocumentType = DocumentType.UNKNOWN
    domain: str = "TCM_PRESCRIPTION_SAFETY"
    total_blocks: int = 0
    total_pages: int = 0
    sections: list[SectionProfile] = Field(default_factory=list)

    @property
    def extractable_sections(self) -> list[SectionProfile]:
        return [s for s in self.sections if s.worth_extracting]

    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.sections:
            counts[s.content_type.value] = counts.get(s.content_type.value, 0) + 1
        return counts


# ── 推荐 extractor（ContentType → extractor 名，与 plan 阶段约定）────────────

CONTENT_TYPE_TO_EXTRACTOR: dict[ContentType, str] = {
    ContentType.NARRATIVE_RULE: "GeneralRuleExtractor",
    ContentType.STRUCTURED_TABLE: "HerbTableExtractor",
    ContentType.ENTITY_LIST: "HerbTableExtractor",
    ContentType.ENTITY_GROUP_LIST: "PregnancyRuleExtractor",
    ContentType.RELATION_PAIR: "InteractionExtractor",
    ContentType.RELATION_GROUP: "InteractionExtractor",
    ContentType.HERB_MONOGRAPH: "GeneralRuleExtractor",
    ContentType.CASE_RECORD: "GeneralRuleExtractor",  # 医案不属当前安全审方重点
    ContentType.FORMULA_RECORD: "GeneralRuleExtractor",
    ContentType.REFERENCE_ONLY: "",  # 不抽取
    ContentType.IRRELEVANT: "",  # 不抽取
}


def classify_section(title: str, *, table_count: int = 0, body_text: str = "") -> ContentType:
    """按内容特征给章节分类；优先级：高语义关键词 > 表格 > 其他关键词 > 正文措辞。

    有表格时 STRUCTURED_TABLE 优先于叙述性关键词（如「煎」），
    避免「需特殊煎煮中药目录」这类结构对照表被路由去 GeneralRuleExtractor。

    Args:
        title: 章节 heading 文本
        table_count: 章节内表格块数量
        body_text: 章节正文采样（表格缺失时用于兜底判断）
    """
    keyword_type: ContentType | None = None
    for keyword, ctype in _SECTION_RULES:
        if keyword in title:
            keyword_type = ctype
            break
    if keyword_type is not None and keyword_type in _TABLE_PRIORITY_EXEMPT:
        return keyword_type
    # 有表格：表格优先于普通关键词；再落空才看措辞
    if table_count > 0:
        return ContentType.STRUCTURED_TABLE
    if keyword_type is not None:
        return keyword_type
    if _looks_like_rule_text(body_text):
        return ContentType.NARRATIVE_RULE
    return ContentType.REFERENCE_ONLY


_RULE_HINT_RE = re.compile(r"禁用|忌用|不宜|慎用|不得|严禁|不得超过|最大剂量|极量")


def _looks_like_rule_text(text: str) -> bool:
    """正文含规则性措辞密度判断（采样前 1000 字）。"""
    sample = text[:1000]
    if not sample:
        return False
    return len(_RULE_HINT_RE.findall(sample)) >= 2


def infer_document_type(book_name: str) -> DocumentType:
    """根据书名推断文档类型。"""
    for keyword, dtype in _DOC_TYPE_RULES:
        if keyword in book_name:
            return dtype
    return DocumentType.UNKNOWN


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


def build_profile(blocks_path: Path, *, book_name: str = "") -> DocumentProfile:
    """从 MinerU blocks 构建全书画像。

    按 heading_path（parent_headings + heading）聚合为章节；
    每个章节统计块数/字数/表格数/页码范围并分类。
    """
    blocks = load_blocks(blocks_path)
    sections = _aggregate_sections(blocks)
    doc_type = infer_document_type(book_name)

    profile = DocumentProfile(
        document_type=doc_type,
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


def _aggregate_sections(blocks: list[dict[str, Any]]) -> list[SectionProfile]:
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
                    content_type=ContentType.NARRATIVE_RULE,  # 分类后回填
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

    # 分类（用标题 + 表格数 + 采样文本）
    for sec in sections:
        sec.content_type = classify_section(
            sec.title, table_count=sec.table_count, body_text=sec.sample_text
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
