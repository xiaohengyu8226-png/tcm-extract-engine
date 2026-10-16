"""TCM 域 LLM 输出解析（收编自 cab/extractors.py）：raw JSON → SourceRecord。

入口 :func:`parse_record` 接收结构化的章节上下文参数（不依赖 cab 的 SectionPlan，
保持包 → 骨架单向依赖）。宽松容错：字段级降级。
"""

from __future__ import annotations

import re
from typing import Any

from tcm_extract_engine.domains.tcm.schema import (
    ActionLevel,
    HerbDose,
    HerbFacts,
    HerbFactVariant,
    ImageRef,
    InteractionGroup,
    InteractionMember,
    PopulationRule,
    PregnancyRule,
    RecordType,
)
from tcm_extract_engine.domains.tcm.schema import TCMRecord as SourceRecord

_RECORD_TYPE_MAP: dict[str, RecordType] = {t.value: t for t in RecordType}

_ACTION_MAP: dict[str, ActionLevel] = {
    "PROHIBIT": ActionLevel.PROHIBIT,
    "AVOID": ActionLevel.AVOID,
    "CAUTION": ActionLevel.CAUTION,
    "MONITOR": ActionLevel.MONITOR,
    "INFO": ActionLevel.INFO,
}


def parse_record(
    raw: dict[str, Any],
    *,
    book_id: str,
    seq: int,
    extractor: str,
    page_fallback: int | None,
    heading_path: list[str],
    title: str,
) -> SourceRecord:
    """将 LLM 输出的单条记录解析为 SourceRecord（宽松容错：字段级降级）。"""
    if raw.get("payload") is not None:
        from tcm_extract_engine.domains.tcm.schema import TCMPayload
        payload = TCMPayload.model_validate(raw["payload"]).model_dump(mode="json")
        for key in payload:
            if key in raw and payload[key] != raw[key]:
                raise ValueError(f"载荷与顶层字段冲突: {key}")
        raw = {**payload, **{k: v for k, v in raw.items() if k != "payload"}}
    record_type = _RECORD_TYPE_MAP.get(str(raw.get("record_type") or "").upper())
    if raw.get("record_type") and record_type is None:
        raise ValueError(f"未知 TCM record_type: {raw['record_type']}")
    if record_type is None:
        # 按 extractor 推断默认类型
        record_type = _infer_type_from_extractor(extractor)

    page_hint = raw.get("page_hint") or page_fallback
    page: int | None = None
    if page_hint is not None:
        try:
            page = int(str(page_hint))
        except (ValueError, TypeError):
            page = page_fallback

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
    herb_facts = _parse_herb_facts(raw.get("herb_facts"))
    image_ref = _parse_image_ref(raw.get("image_ref"))

    # 兼容旧模板输出：LLM 将类型化载荷放进 attributes（顶层字段缺失时回退）
    attributes = _clean_attributes(raw.get("attributes"))
    if pregnancy is None and record_type == RecordType.PREGNANCY:
        pregnancy = _parse_pregnancy(attributes)
    if population is None and record_type == RecordType.POPULATION:
        population = _parse_population(attributes)
    if herb_facts is None and record_type == RecordType.HERB_FACTS:
        herb_facts = _parse_herb_facts(attributes)
    if image_ref is None and record_type == RecordType.FIGURE:
        image_ref = _parse_image_ref(attributes)

    action = _ACTION_MAP.get(str(raw.get("action") or "").upper())

    return SourceRecord(
        record_id=f"SR-{book_id[:8]}-{seq:06d}",
        book_id=book_id,
        record_type=record_type,
        source={
            "page_start": page,
            "page_end": page,
            "section_path": heading_path or [title],
        },
        raw_text=str(raw.get("raw_text") or ""),
        herb_name=raw.get("herb_name") or None,
        subject_text=raw.get("subject_text") or None,
        toxicity_level=raw.get("toxicity_level") or None,
        dose=dose,
        interaction=interaction,
        pregnancy=pregnancy,
        population=population,
        herb_facts=herb_facts,
        image_ref=image_ref,
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
        "HerbFactsExtractor": RecordType.HERB_FACTS,
        "FigureExtractor": RecordType.FIGURE,
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
        target_herbs=_as_str_list(value.get("target_herbs")),
        target_formulas=_as_str_list(value.get("target_formulas")),
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
        target_herbs=_as_str_list(value.get("target_herbs")),
        condition_text=value.get("condition_text"),
    )


def _parse_herb_facts(value: Any) -> HerbFacts | None:
    if not isinstance(value, dict):
        return None
    herb_name = str(value.get("herb_name") or "").strip()
    if not herb_name:
        return None
    return HerbFacts(
        herb_name=herb_name,
        nature=value.get("nature") or None,
        flavors=_as_str_list(value.get("flavors")),
        channels=_as_str_list(value.get("channels")),
        efficacy=_as_str_list(value.get("efficacy")),
        pharmacology=_as_str_list(value.get("pharmacology")),
        toxicity=value.get("toxicity") or None,
        clinical_notes=_as_str_list(value.get("clinical_notes")),
        formulas=_as_str_list(value.get("formulas")),
        variants=[
            v for v in (_parse_variant(v) for v in value.get("variants") or []) if v is not None
        ],
    )


def _parse_image_ref(value: Any) -> ImageRef | None:
    """解析图片多模态载荷（asset_path 由抽取批次回填，此处可为空）。"""
    if not isinstance(value, dict):
        return None
    extracted = value.get("extracted_data")
    return ImageRef(
        asset_path=str(value.get("asset_path") or ""),
        figure_type=str(value.get("figure_type") or "其他"),
        description=str(value.get("description") or ""),
        extracted_data=extracted if isinstance(extracted, dict) else {},
        confidence=_to_float(value.get("confidence")) or 0.0,
        ocr_text=str(value.get("ocr_text") or ""),
    )


def _parse_variant(value: Any) -> HerbFactVariant | None:
    if not isinstance(value, dict):
        return None
    variant_name = str(value.get("variant_name") or "").strip()
    if not variant_name:
        return None
    return HerbFactVariant(
        variant_name=variant_name,
        nature_flavor=value.get("nature_flavor") or None,
        channels=_as_str_list(value.get("channels")),
        efficacy=_as_str_list(value.get("efficacy")),
        pharmacology=_as_str_list(value.get("pharmacology")),
        toxicity=value.get("toxicity") or None,
        notes=value.get("notes") or None,
    )


# target_herbs/target_formulas 分隔符：LLM 时而返回字符串而非数组，
# 直接迭代会把「芒硝、甘遂」逐字拆成 ["芒","硝","、","甘","遂"]。
_LIST_DELIMS = re.compile(r"[、，,;；/／\n]+")


def _as_str_list(value: Any) -> list[str]:
    """把 LLM 返回的目标药物/方剂归一为非空字符串列表。

    - list/tuple：逐项归一（项内若夹带分隔符同样切开），去空白、丢空项。
    - str：按中英文分隔符切分，杜绝逐字符拆散。
    - 其他（None/dict/数字）：空列表。
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in _LIST_DELIMS.split(value) if p.strip()]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.extend(p.strip() for p in _LIST_DELIMS.split(item) if p.strip())
            elif item not in (None, ""):
                out.append(str(item).strip())
        return [s for s in out if s]
    return []


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
