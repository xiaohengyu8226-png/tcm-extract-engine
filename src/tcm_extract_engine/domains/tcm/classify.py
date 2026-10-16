"""TCM 域内容分类与路由（收编自 cab/profiler.py + cab/planner.py）。

骨架（profiler 聚合 / planner 方案构建）通过本模块取得分类与路由规则；
枚举值一经落盘（document_profile.json）即不可更改。
"""

from __future__ import annotations

import re
from enum import Enum

from tcm_extract_engine.core.document import DocumentType


class ContentType(str, Enum):
    """章节内容类型；决定 recommended_extractor。"""

    NARRATIVE_RULE = "NARRATIVE_RULE"  # 叙述性规则文本
    STRUCTURED_TABLE = "STRUCTURED_TABLE"  # 结构化表格（附录A/B 等）
    ENTITY_LIST = "ENTITY_LIST"  # 实体清单
    ENTITY_GROUP_LIST = "ENTITY_GROUP_LIST"  # 分组实体清单（妊娠禁用）
    RELATION_PAIR = "RELATION_PAIR"  # 两两关系
    RELATION_GROUP = "RELATION_GROUP"  # 关系组（十八反/十九畏）
    HERB_MONOGRAPH = "HERB_MONOGRAPH"  # 药物专论
    CASE_RECORD = "CASE_RECORD"  # 医案
    FORMULA_RECORD = "FORMULA_RECORD"  # 方剂
    FIGURE = "FIGURE"  # 图片/图表（药材图/流程图/数据图）
    REFERENCE_ONLY = "REFERENCE_ONLY"  # 仅参考
    IRRELEVANT = "IRRELEVANT"  # 无关


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
    # 图片/图表（多模态）
    ("图", ContentType.FIGURE),
    ("照片", ContentType.FIGURE),
    ("显微", ContentType.FIGURE),
    ("色谱", ContentType.FIGURE),
    ("流程图", ContentType.FIGURE),
    ("示意图", ContentType.FIGURE),
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
# IRRELEVANT 结构页（目录/前言/参考文献）不因含对齐文本而被当表格抽取；
# FIGURE 图片章节可能包含表格截图，但应优先走多模态视觉抽取
_TABLE_PRIORITY_EXEMPT: frozenset[ContentType] = frozenset(
    {
        ContentType.RELATION_GROUP,
        ContentType.RELATION_PAIR,
        ContentType.ENTITY_GROUP_LIST,
        ContentType.IRRELEVANT,
        ContentType.FIGURE,
    }
)

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


# ── 推荐 extractor（ContentType → extractor 名，与 plan 阶段约定）────────────

CONTENT_TYPE_TO_EXTRACTOR: dict[ContentType, str] = {
    ContentType.NARRATIVE_RULE: "GeneralRuleExtractor",
    ContentType.STRUCTURED_TABLE: "HerbTableExtractor",
    ContentType.ENTITY_LIST: "HerbTableExtractor",
    ContentType.ENTITY_GROUP_LIST: "PregnancyRuleExtractor",
    ContentType.RELATION_PAIR: "InteractionExtractor",
    ContentType.RELATION_GROUP: "InteractionExtractor",
    ContentType.HERB_MONOGRAPH: "HerbFactsExtractor",  # 药物专论：性味归经/功效/药理
    ContentType.CASE_RECORD: "HerbFactsExtractor",  # 医案：主治/方剂佐证（不再丢失证据）
    ContentType.FORMULA_RECORD: "GeneralRuleExtractor",
    ContentType.FIGURE: "FigureExtractor",  # 图片/图表：多模态视觉抽取
    ContentType.REFERENCE_ONLY: "",  # 不抽取
    ContentType.IRRELEVANT: "",  # 不抽取
}


def classify_section(
    title: str,
    *,
    table_count: int = 0,
    body_text: str = "",
    heading_path: list[str] | None = None,
) -> ContentType:
    """按内容特征给章节分类；优先级：高语义关键词 > 表格 > 其他关键词 > 正文措辞。

    有表格时 STRUCTURED_TABLE 优先于叙述性关键词（如「煎」），
    避免「需特殊煎煮中药目录」这类结构对照表被路由去 GeneralRuleExtractor。

    Args:
        title: 章节 heading 文本
        table_count: 章节内表格块数量
        body_text: 章节正文采样（表格缺失时用于兜底判断）
        heading_path: 章节的完整标题路径（含父级，可重复含 title）。大纲修复后
            父级上下文（如「第X章 禁忌」）可参与路由，避免只看叶子标题时
            关键词落空；关键词依次在 title → 父级中查找，规则表顺序即优先级。
    """
    haystacks: list[str] = [title]
    if heading_path:
        haystacks.extend(p for p in heading_path if p and p != title)
    keyword_type: ContentType | None = None
    for keyword, ctype in _SECTION_RULES:
        if any(keyword in h for h in haystacks):
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
    # 止血兜底：未命中关键词、无表格、规则词密度不足的章节，不再判 REFERENCE_ONLY
    # 静默丢弃，改判叙述性规则走通用抽取器（general_rule.md）。旧行为会把非中医书
    # （如西医临床正文）整章丢掉——这是胃癌书 12/15 章节丢失的真凶。噪声记录交由
    # 确定性校验 + B 核验 + 人工审核门兜底。显式 IRRELEVANT（目录/前言/参考文献）
    # 与 REFERENCE_ONLY（凡例）已在上面 keyword_type 分支返回，丢弃语义不受影响。
    return ContentType.NARRATIVE_RULE


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


# extractor 名 → A 模板文件名
EXTRACTOR_TEMPLATES: dict[str, str] = {
    "GeneralRuleExtractor": "general_rule.md",
    "HerbTableExtractor": "herb_table.md",
    "InteractionExtractor": "interaction_group.md",
    "PregnancyRuleExtractor": "pregnancy_list.md",
    "HerbFactsExtractor": "herb_facts.md",
    "FigureExtractor": "figure.md",  # 多模态图片抽取
}


def chunk_strategy(extractor: str) -> str:
    """章节组块策略。

    - "whole_section"：关系组章节文本整章一批，禁止 chunk 切断十八反/十九畏；
    - "split"：普通文本按字符上限分批。
    """
    return "whole_section" if extractor == "InteractionExtractor" else "split"
