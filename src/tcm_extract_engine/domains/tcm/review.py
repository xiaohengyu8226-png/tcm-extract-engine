from __future__ import annotations

import json
from typing import Any

from tcm_extract_engine.core.records import SourceRecord
from tcm_extract_engine.domains.tcm.schema import view


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "、".join(str(v) for v in value if v not in (None, ""))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _dose_text(record: SourceRecord) -> str:
    dose = view(record).dose
    if dose is None:
        return ""
    parts: list[str] = []
    if dose.min_dose is not None and dose.max_dose is not None:
        parts.append(f"{dose.min_dose:g}~{dose.max_dose:g}{dose.unit}")
    elif dose.max_dose is not None:
        parts.append(f"≤{dose.max_dose:g}{dose.unit}")
    elif dose.min_dose is not None:
        parts.append(f"≥{dose.min_dose:g}{dose.unit}")
    if dose.per_use:
        parts.append("（单次剂量）")
    if dose.frequency:
        parts.append(f"频次：{dose.frequency}")
    if dose.special_note:
        parts.append(f"备注：{dose.special_note}")
    return "；".join(parts)


def _structured_content(record: SourceRecord) -> str:
    """把类型化载荷压平成医生可读的一列文本。"""
    record = view(record)
    parts: list[str] = []
    if record.interaction is not None:
        group = record.interaction
        head = group.interaction_type
        if group.group_name:
            head += f"（{group.group_name}）"
        anchor = group.anchor.herb_name + (
            f"（{group.anchor.processed_form}）" if group.anchor.processed_form else ""
        )
        members = "、".join(
            m.herb_name + (f"（{m.processed_form}）" if m.processed_form else "")
            for m in group.members
        )
        relation = f"{anchor} 畏 {members}" if group.direction else f"{anchor} 反/畏 {members}"
        parts.append(f"{head}：{relation}")
    if record.pregnancy is not None:
        rule = record.pregnancy
        targets = "、".join([*rule.target_herbs, *rule.target_formulas])
        seg = f"限制级别：{rule.restriction_level}"
        if targets:
            seg += f"；对象：{targets}"
        if rule.reason:
            seg += f"；原因：{rule.reason}"
        parts.append(seg)
    if record.population is not None:
        rule = record.population
        seg = f"人群：{rule.population}；限制：{rule.restriction_level}"
        if rule.target_herbs:
            seg += f"；药物：{'、'.join(rule.target_herbs)}"
        if rule.condition_text:
            seg += f"；条件：{rule.condition_text}"
        parts.append(seg)
    if record.herb_facts is not None:
        facts = record.herb_facts
        segs: list[str] = []
        if facts.nature:
            segs.append(f"性：{facts.nature}")
        if facts.flavors:
            segs.append(f"味：{'、'.join(facts.flavors)}")
        if facts.channels:
            segs.append(f"归经：{'、'.join(facts.channels)}")
        if facts.efficacy:
            segs.append(f"功效：{'、'.join(facts.efficacy)}")
        if facts.pharmacology:
            segs.append(f"药理：{'、'.join(facts.pharmacology)}")
        if facts.toxicity:
            segs.append(f"毒性：{facts.toxicity}")
        if facts.clinical_notes:
            segs.append(f"主治：{'、'.join(facts.clinical_notes)}")
        if facts.formulas:
            segs.append(f"方剂：{'、'.join(facts.formulas)}")
        for variant in facts.variants:
            vsegs = [f"{variant.variant_name}"]
            if variant.nature_flavor:
                vsegs.append(f"性味：{variant.nature_flavor}")
            if variant.channels:
                vsegs.append(f"归经：{'、'.join(variant.channels)}")
            if variant.efficacy:
                vsegs.append(f"功效：{'、'.join(variant.efficacy)}")
            if variant.pharmacology:
                vsegs.append(f"药理：{'、'.join(variant.pharmacology)}")
            if variant.toxicity:
                vsegs.append(f"毒性：{variant.toxicity}")
            if variant.notes:
                vsegs.append(variant.notes)
            segs.append("／".join(vsegs))
        parts.append("；".join(segs))
    dose = _dose_text(record)
    if dose:
        parts.append(dose)
    if record.toxicity_level:
        parts.append(f"毒性：{record.toxicity_level}")
    if record.record_type == "USAGE" and record.attributes.get("usage_note"):
        parts.append(_text(record.attributes["usage_note"]))
    elif record.attributes:
        extras = {k: v for k, v in record.attributes.items() if k != "usage_note"}
        if extras:
            note = "；".join(f"{k}={_text(v)}" for k, v in extras.items() if _text(v))
            if note:
                parts.append(note[:300])
    return "；".join(parts)


def _subject(record: SourceRecord) -> str:
    record = view(record)
    if record.herb_name:
        return record.herb_name
    if record.interaction is not None:
        return record.interaction.anchor.herb_name
    if record.pregnancy is not None:
        return "、".join([*record.pregnancy.target_herbs, *record.pregnancy.target_formulas])
    if record.population is not None:
        return record.population.population
    return _text(record.subject_text)


def _related_herbs(record: SourceRecord) -> str:
    record = view(record)
    if record.interaction is not None:
        return "、".join(m.herb_name for m in record.interaction.members)
    if record.pregnancy is not None:
        return "、".join(record.pregnancy.target_herbs)
    if record.population is not None:
        return "、".join(record.population.target_herbs)
    return ""
