"""统一 SourceRecord / SafetyRecord Schema。

SourceRecord = 原始文献事实（Model A 的忠实抽取产物）。
SafetyRule   = 后续发布规则（RuleCompiler 从已审核记录编译）。

两者禁止混为一谈：SourceRecord 永不携带发布状态，
SafetyRule 永远可以回溯到 source_record_ids。

record_type 分型（对应不同 extractor）：
  GENERAL_RULE        叙述性规则（正文 4.x 等）
  HERB_TABLE          药物表格行（附录A/B/F 等）
  INTERACTION         配伍禁忌（十八反/十九畏 关系组）
  PREGNANCY           妊娠禁用/慎用
  DOSAGE              剂量规范
  USAGE               特殊煎服法/炮制要求
  POPULATION          特殊人群（儿童/老年/肝肾）
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class RecordType(str, enum.Enum):
    """SourceRecord 分型；决定使用哪个 extractor 与校验规则。"""

    GENERAL_RULE = "GENERAL_RULE"
    HERB_TABLE = "HERB_TABLE"
    INTERACTION = "INTERACTION"
    PREGNANCY = "PREGNANCY"
    DOSAGE = "DOSAGE"
    USAGE = "USAGE"
    POPULATION = "POPULATION"


class SourceRef(BaseModel):
    """文献来源定位（可追溯到 PDF 物理页与 MinerU block）。"""

    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    chunk_id: str | None = None


class HerbDose(BaseModel):
    """剂量区间（含单位与频次）。"""

    min_dose: float | None = None
    max_dose: float | None = None
    unit: str = "g"
    per_use: bool = False  # True=单次剂量；False=日剂量
    frequency: str | None = None  # 如 "一日2次"
    special_note: str | None = None  # 先煎/后下/包煎等


class InteractionMember(BaseModel):
    """配伍关系组成员（十八反/十九畏）。"""

    herb_name: str
    processed_form: str | None = None  # 炮制品，如 生马钱子/制马钱子
    role: str = "member"  # anchor | member（十九畏有方向性）


class InteractionGroup(BaseModel):
    """一条完整的配伍禁忌关系组（禁止被 chunk 切断）。"""

    group_name: str | None = None  # 如 "十八反-甘草组"
    interaction_type: str  # 十八反 | 十九畏 | 其他配伍禁忌
    anchor: InteractionMember
    members: list[InteractionMember] = Field(default_factory=list)
    direction: str | None = None  # 十九畏方向：A畏B


class PregnancyRule(BaseModel):
    """妊娠期用药规则。"""

    restriction_level: str  # 禁用 | 忌用 | 不宜 | 慎用
    target_herbs: list[str] = Field(default_factory=list)
    target_formulas: list[str] = Field(default_factory=list)
    reason: str | None = None  # 如 "可致胎儿畸形"


class PopulationRule(BaseModel):
    """特殊人群规则（儿童/老年/肝肾功能异常/过敏）。"""

    population: str  # 儿童 | 老年人 | 肝功能不全 | 肾功能不全 | 过敏体质 ...
    restriction_level: str  # 禁用 | 忌用 | 不宜 | 慎用 | 需监测
    target_herbs: list[str] = Field(default_factory=list)
    condition_text: str | None = None


class ActionLevel(str, enum.Enum):
    """处置动作强度（原文优先，禁止推断升级）。"""

    PROHIBIT = "PROHIBIT"  # 禁用/不得/严禁
    AVOID = "AVOID"  # 忌用/不宜
    CAUTION = "CAUTION"  # 慎用
    MONITOR = "MONITOR"  # 需监测/需医师确认
    INFO = "INFO"  # 提示性信息


class SourceRecord(BaseModel):
    """原始文献事实（Model A 输出、B 核验、人工审核的最小单元）。

    设计原则：
      - 忠实抽取：字段值必须有 raw_text 支撑，禁止推理补充
      - 分型存储：record_type 决定 attributes 里的结构化载荷
      - 可追溯：source + raw_text + block_ids 构成定位链
    """

    record_id: str  # SR-<book_id8>-<seq6>
    book_id: str
    record_type: RecordType

    source: SourceRef = Field(default_factory=SourceRef)
    raw_text: str = ""  # 支撑本记录的原文片段（必填，校验层检查）

    # 主体字段（按 record_type 选择性填充）
    herb_name: str | None = None  # HERB_TABLE / DOSAGE / USAGE 主键
    subject_text: str | None = None  # 规则主语描述（GENERAL_RULE）
    toxicity_level: str | None = None  # 大毒 | 有毒 | 小毒 | 无毒
    dose: HerbDose | None = None
    interaction: InteractionGroup | None = None
    pregnancy: PregnancyRule | None = None
    population: PopulationRule | None = None
    action: ActionLevel | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)  # 类型化附加字段

    # 质量与审核标记（由 validate/verify 阶段更新，不由 A 填写）
    confidence: float | None = None
    needs_review: bool = False
    ambiguities: list[str] = Field(default_factory=list)

    def has_evidence(self) -> bool:
        """source validation：记录必须有原文支撑。"""
        return bool(self.raw_text.strip())


class SafetyRule(BaseModel):
    """发布层规则（RuleCompiler 从 ReviewedRecord 编译，非 A 直接产出）。"""

    rule_id: str
    rule_type: RecordType
    targets: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    actions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)

    source_record_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    authority_level: str | None = None  # 药典/规范/教材/手册
    rule_version: str = "1"
    status: str = "DRAFT"  # DRAFT | APPROVED | PUBLISHED

    executability: str = "DIRECT_WITH_MASTER_DATA"  # DIRECT | DIRECT_WITH_MASTER_DATA | REQUIRES_EXTERNAL_KNOWLEDGE | HUMAN_JUDGMENT_REQUIRED


# ── 语义约束（供确定性校验使用）────────────────────────────────────────────


# 原文措辞 → 允许的 action 上限（原文"慎用"不得升级为 PROHIBIT）
_ACTION_CEILING: dict[str, ActionLevel] = {
    "禁用": ActionLevel.PROHIBIT,
    "禁止": ActionLevel.PROHIBIT,
    "不得": ActionLevel.PROHIBIT,
    "严禁": ActionLevel.PROHIBIT,
    "忌用": ActionLevel.AVOID,
    "不宜": ActionLevel.AVOID,
    "不相": ActionLevel.AVOID,  # 如"不宜与...同用"
    "慎用": ActionLevel.CAUTION,
    "谨慎": ActionLevel.CAUTION,
}

_ACTION_RANK: dict[ActionLevel, int] = {
    ActionLevel.INFO: 0,
    ActionLevel.MONITOR: 1,
    ActionLevel.CAUTION: 2,
    ActionLevel.AVOID: 3,
    ActionLevel.PROHIBIT: 4,
}


def action_ceiling(raw_text: str) -> ActionLevel | None:
    """根据原文措辞推断 action 上限；无匹配返回 None。"""
    best: ActionLevel | None = None
    for keyword, level in _ACTION_CEILING.items():
        if keyword in raw_text:
            if best is None or _ACTION_RANK[level] > _ACTION_RANK[best]:
                best = level
    return best


def action_exceeds_ceiling(record: SourceRecord) -> bool:
    """语义约束校验：action 强度是否超过原文措辞允许的上限。"""
    if record.action is None or not record.raw_text:
        return False
    ceiling = action_ceiling(record.raw_text)
    if ceiling is None:
        return False
    return _ACTION_RANK[record.action] > _ACTION_RANK[ceiling]
