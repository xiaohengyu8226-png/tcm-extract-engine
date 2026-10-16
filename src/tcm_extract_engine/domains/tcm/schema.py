"""TCM 域记录 schema（收编自 core/records.py）：分型、载荷模型、语义天花板。

SourceRecord 骨架（record_id/book_id/source/raw_text/passthrough 等）留在
core/records.py；本模块只承载域载荷与措辞强度约束，由骨架 shim 重导出，
保持全库 ``from tcm_extract_engine.core.records import ...`` 导入面不变。
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tcm_extract_engine.core.records import ImageRef, SourceRecord
from tcm_extract_engine.core.records import SafetyRule as SafetyRule


class RecordType(str, enum.Enum):
    """SourceRecord 分型；决定使用哪个 extractor 与校验规则。"""

    GENERAL_RULE = "GENERAL_RULE"
    HERB_TABLE = "HERB_TABLE"
    TABLE_ROW = "TABLE_ROW"  # 表格直通：原表逐行透传，不经 LLM
    INTERACTION = "INTERACTION"
    PREGNANCY = "PREGNANCY"
    DOSAGE = "DOSAGE"
    USAGE = "USAGE"
    POPULATION = "POPULATION"
    HERB_FACTS = "HERB_FACTS"  # 药物专论事实（性味归经/功效/药理）
    FIGURE = "FIGURE"  # 图片/图表（多模态视觉抽取）


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


class HerbFactVariant(BaseModel):
    """炮制品变体（生/制/炒等）的差异事实（细粒度复合字段）。

    基原条目承载共性事实；变体只记录与基原不同的部分
    （如 生附子 大毒 vs 制附子 有毒），避免字段整体复制。
    """

    variant_name: str  # 如 "生附子" / "制附子"
    nature_flavor: str | None = None  # 该变体性味（原文表述，如 "辛、热；大毒"）
    channels: list[str] = Field(default_factory=list)  # 归经（与基原不同才填）
    efficacy: list[str] = Field(default_factory=list)  # 功效（与基原不同才填）
    pharmacology: list[str] = Field(default_factory=list)  # 药理（与基原不同才填）
    toxicity: str | None = None  # 大毒 | 有毒 | 小毒 | 无毒
    notes: str | None = None  # 与基原差异说明（炮制方法/用法禁忌）


class HerbFacts(BaseModel):
    """药物专论事实（herb monograph facts）：性味归经/功效/药理。

    路由自 HERB_MONOGRAPH / CASE_RECORD 章节（herb_facts.md 模板族）；
    variants 承载炮制品变体的细粒度差异事实。
    """

    herb_name: str
    nature: str | None = None  # 四气：寒 | 热 | 温 | 凉 | 平（可带 微/大 前缀）
    flavors: list[str] = Field(default_factory=list)  # 五味：酸苦甘辛咸涩淡
    channels: list[str] = Field(default_factory=list)  # 归经（十二经 + 心包）
    efficacy: list[str] = Field(default_factory=list)  # 功效
    pharmacology: list[str] = Field(default_factory=list)  # 药理作用
    toxicity: str | None = None  # 大毒 | 有毒 | 小毒 | 无毒
    clinical_notes: list[str] = Field(default_factory=list)  # 主治/临床应用
    formulas: list[str] = Field(default_factory=list)  # 相关方剂（医案佐证）
    variants: list[HerbFactVariant] = Field(default_factory=list)  # 炮制品变体


class ActionLevel(str, enum.Enum):
    """处置动作强度（原文优先，禁止推断升级）。"""

    PROHIBIT = "PROHIBIT"  # 禁用/不得/严禁
    AVOID = "AVOID"  # 忌用/不宜
    CAUTION = "CAUTION"  # 慎用
    MONITOR = "MONITOR"  # 需监测/需医师确认
    INFO = "INFO"  # 提示性信息


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


def action_exceeds_ceiling(record: Any) -> bool:
    """语义约束校验：action 强度是否超过原文措辞允许的上限。

    record 为 SourceRecord 骨架（骨架 shim 场景），只读 action/raw_text。
    """
    if record.action is None or not record.raw_text:
        return False
    ceiling = action_ceiling(record.raw_text)
    if ceiling is None:
        return False
    return _ACTION_RANK[record.action] > _ACTION_RANK[ceiling]


class TCMPayload(BaseModel):
    """Typed TCM view of the complete envelope payload, never core fields."""

    model_config = ConfigDict(extra="forbid")
    herb_name: str | None = None
    toxicity_level: str | None = None
    dose: HerbDose | None = None
    interaction: InteractionGroup | None = None
    pregnancy: PregnancyRule | None = None
    population: PopulationRule | None = None
    herb_facts: HerbFacts | None = None
    image_ref: ImageRef | None = None
    action: ActionLevel | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class TCMRecord(SourceRecord):
    """Domain-local facade for typed algorithms and legacy Python constructors.

    Serialized artifacts must load through strict core SourceRecord. This facade
    accepts legacy keyword arguments only at the domain boundary, never on disk.
    """

    def __init__(self, **data: Any) -> None:
        payload = dict(data.pop("payload", {}))
        for key in TCMPayload.model_fields:
            if key in data:
                value = data.pop(key)
                payload[key] = (
                    value.model_dump(mode="json") if isinstance(value, BaseModel) else value
                )
        data["payload"] = payload
        super().__init__(**data)

    def __getattr__(self, name: str) -> Any:
        if name in TCMPayload.model_fields:
            if self.record_type == "TABLE_ROW":
                return self.payload.get(name, {} if name == "attributes" else None)
            return getattr(TCMPayload.model_validate(self.payload), name)
        raise AttributeError(name)


def view(record: SourceRecord) -> TCMRecord:
    return record if isinstance(record, TCMRecord) else TCMRecord(**record.model_dump(mode="json"))
