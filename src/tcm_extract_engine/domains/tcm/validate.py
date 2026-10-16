"""TCM 域确定性校验规则（收编自 cab/validator.py 的域相关部分）。

骨架 cab/validator.py 通过领域包协议调用本模块的入口：
  validate_record        单记录域校验（SCHEMA / ENTITY / SEMANTIC）
  coverage_member_count  妊娠清单 member_count 覆盖率
  coverage_table_rows    表格行数覆盖率
  dedup_key              按记录类型构造去重主键

通用不变量（NUMERIC 剂量数值卫生、SOURCE 溯源完备、重复检测框架）
留在骨架；剂量离群阈值由 manifest.dose_outlier_grams 注入骨架。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from tcm_extract_engine.core.validation import (
    IssueCategory,
    Severity,
    ValidationIssue,
)
from tcm_extract_engine.domains.tcm.schema import (
    _ACTION_RANK,
    ActionLevel,
    RecordType,
    action_ceiling,
    action_exceeds_ceiling,
    view,
)
from tcm_extract_engine.domains.tcm.schema import (
    TCMRecord as SourceRecord,
)

# 妊娠/人群 restriction_level → action 上限（措辞禁止升级）
_RESTRICTION_CEILING: dict[str, ActionLevel] = {
    "禁用": ActionLevel.PROHIBIT,
    "禁止": ActionLevel.PROHIBIT,
    "忌用": ActionLevel.AVOID,
    "不宜": ActionLevel.AVOID,
    "慎用": ActionLevel.CAUTION,
    "谨慎": ActionLevel.CAUTION,
    "需监测": ActionLevel.MONITOR,
}

_TOXICITY_LEVELS = {"大毒", "有毒", "小毒", "无毒"}

# 药物专论事实（herb_facts）受控词表：四气/五味/归经
_FLAVORS = {"酸", "苦", "甘", "辛", "咸", "涩", "淡"}
_NATURE_RE = re.compile(r"^[微大]?[寒热温凉平]$")
# 归经脏器（长名优先，避免「心」误匹配「心包」）
_CHANNEL_ORGANS = (
    "心包",
    "三焦",
    "大肠",
    "小肠",
    "膀胱",
    "肺",
    "胃",
    "脾",
    "肾",
    "胆",
    "肝",
    "心",
)


def validate_record(record: SourceRecord) -> list[ValidationIssue]:
    """单记录域校验：SCHEMA（分型↔载荷）+ ENTITY（实体完整）+ SEMANTIC（措辞天花板）。"""
    record = view(record)
    issues: list[ValidationIssue] = []
    issues.extend(_check_numeric(record))
    issues.extend(_check_schema(record))
    issues.extend(_check_entity(record))
    issues.extend(_check_semantic(record))
    return issues


def _check_schema(record: SourceRecord) -> list[ValidationIssue]:
    """记录结构与 record_type 一致性。"""
    issues: list[ValidationIssue] = []

    def add(message: str, severity: Severity = Severity.ERROR) -> None:
        issues.append(
            ValidationIssue(
                record_id=record.record_id,
                category=IssueCategory.SCHEMA,
                severity=severity,
                message=message,
            )
        )

    if record.record_type == RecordType.INTERACTION and record.interaction is None:
        add("INTERACTION 记录缺少 interaction 载荷")
    if record.record_type == RecordType.PREGNANCY and record.pregnancy is None:
        add("PREGNANCY 记录缺少 pregnancy 载荷")
    if record.record_type == RecordType.POPULATION and record.population is None:
        add("POPULATION 记录缺少 population 载荷")
    if record.record_type == RecordType.HERB_TABLE and not (record.herb_name or "").strip():
        add("HERB_TABLE 记录缺少 herb_name")
    if record.record_type == RecordType.DOSAGE and record.dose is None:
        add("DOSAGE 记录缺少 dose 载荷")
    if record.record_type == RecordType.USAGE and not (
        (record.herb_name or "").strip() or (record.subject_text or "").strip()
    ):
        add("USAGE 记录缺少 herb_name/subject_text", Severity.WARNING)
    if record.record_type == RecordType.HERB_FACTS and record.herb_facts is None:
        add("HERB_FACTS 记录缺少 herb_facts 载荷")
    # FIGURE 记录：必须有 image_ref 载荷
    if record.record_type == RecordType.FIGURE and record.image_ref is None:
        add("FIGURE 记录缺少 image_ref 载荷")
    return issues


def _check_entity(record: SourceRecord) -> list[ValidationIssue]:
    """实体完整性。"""
    issues: list[ValidationIssue] = []

    def add(message: str, severity: Severity = Severity.ERROR) -> None:
        issues.append(
            ValidationIssue(
                record_id=record.record_id,
                category=IssueCategory.ENTITY,
                severity=severity,
                message=message,
            )
        )

    interaction = record.interaction
    if interaction is not None:
        if not interaction.members:
            add("配伍禁忌关系组缺少成员（关系组禁止被拆分/遗漏）")
        anchor_name = interaction.anchor.herb_name
        if anchor_name and any(m.herb_name == anchor_name for m in interaction.members):
            add(f"anchor「{anchor_name}」同时出现在 members 中", Severity.WARNING)
        names = [m.herb_name for m in interaction.members if m.herb_name]
        duplicated = [n for n, c in Counter(names).items() if c > 1]
        if duplicated:
            add(f"关系组成员重复: {duplicated}", Severity.WARNING)

    pregnancy = record.pregnancy
    if pregnancy is not None:
        if not pregnancy.target_herbs and not pregnancy.target_formulas:
            add("妊娠规则缺少目标药物（target_herbs/target_formulas 均为空）")

    population = record.population
    if population is not None and not population.target_herbs:
        add("特殊人群规则缺少目标药物", Severity.WARNING)

    herb_facts = record.herb_facts
    if herb_facts is not None:
        if not herb_facts.herb_name.strip():
            add("药物专论事实缺少 herb_name")
        if herb_facts.nature and not _NATURE_RE.fullmatch(herb_facts.nature.strip()):
            add(f"四气「{herb_facts.nature}」非标准表述（寒热温凉平）", Severity.WARNING)
        bad_flavors = [f for f in herb_facts.flavors if f not in _FLAVORS]
        if bad_flavors:
            add(f"五味出现非常规值: {bad_flavors}", Severity.WARNING)
        bad_channels = [
            c
            for c in herb_facts.channels
            if not (c.endswith("经") and any(o in c for o in _CHANNEL_ORGANS))
        ]
        if bad_channels:
            add(f"归经出现非常规值: {bad_channels}", Severity.WARNING)
        if herb_facts.toxicity and herb_facts.toxicity not in _TOXICITY_LEVELS:
            add(
                f"toxicity「{herb_facts.toxicity}」非标准枚举（大毒|有毒|小毒|无毒）",
                Severity.WARNING,
            )
        if not any(
            (
                herb_facts.efficacy,
                herb_facts.pharmacology,
                herb_facts.clinical_notes,
                herb_facts.formulas,
            )
        ):
            add("药物专论事实无功效/药理/主治内容（疑似空抽取）", Severity.WARNING)
        variant_names = [v.variant_name for v in herb_facts.variants]
        duplicated = [n for n, c in Counter(variant_names).items() if c > 1]
        if duplicated:
            add(f"炮制品变体名重复: {duplicated}", Severity.WARNING)
        for variant in herb_facts.variants:
            if variant.toxicity and variant.toxicity not in _TOXICITY_LEVELS:
                add(
                    f"变体「{variant.variant_name}」toxicity「{variant.toxicity}」"
                    "非标准枚举（大毒|有毒|小毒|无毒）",
                    Severity.WARNING,
                )
            if not any(
                (
                    variant.nature_flavor,
                    variant.channels,
                    variant.efficacy,
                    variant.pharmacology,
                    variant.toxicity,
                    variant.notes,
                )
            ):
                add(
                    f"变体「{variant.variant_name}」无任何差异事实（疑似空变体）",
                    Severity.WARNING,
                )

    image_ref = record.image_ref
    if image_ref is not None:
        if not image_ref.asset_path.strip():
            add("FIGURE 记录 image_ref.asset_path 为空（无法回溯图片）", Severity.WARNING)
        if not 0.0 <= image_ref.confidence <= 1.0:
            add(
                f"image_ref.confidence「{image_ref.confidence}」超出 [0,1] 区间",
                Severity.WARNING,
            )
        if not (
            image_ref.description.strip() or image_ref.extracted_data or image_ref.ocr_text.strip()
        ):
            add("FIGURE 记录无任何视觉抽取内容（疑似空抽取）", Severity.WARNING)
    return issues


def _check_semantic(record: SourceRecord) -> list[ValidationIssue]:
    """语义天花板：措辞强度禁止升级。"""
    issues: list[ValidationIssue] = []

    def add(message: str, severity: Severity = Severity.ERROR) -> None:
        issues.append(
            ValidationIssue(
                record_id=record.record_id,
                category=IssueCategory.SEMANTIC,
                severity=severity,
                message=message,
            )
        )

    # 原文措辞 vs action（records.action_exceeds_ceiling）
    if action_exceeds_ceiling(record) and record.action is not None:
        ceiling = action_ceiling(record.raw_text)
        add(
            f"action={record.action.value} 超过原文措辞上限（"
            f"原文仅支撑 ≤{ceiling.value if ceiling else '?'}），疑似语义升级"
        )

    # restriction_level vs action 一致性（妊娠/人群）
    restriction: str | None = None
    if record.pregnancy is not None:
        restriction = record.pregnancy.restriction_level
    elif record.population is not None:
        restriction = record.population.restriction_level
    if restriction is not None and record.action is not None:
        ceiling = _RESTRICTION_CEILING.get(restriction)
        if ceiling is not None and _ACTION_RANK[record.action] > _ACTION_RANK[ceiling]:
            add(
                f"restriction_level「{restriction}」仅支撑 action≤{ceiling.value}，"
                f"实际 action={record.action.value}"
            )

    if record.toxicity_level is not None and record.toxicity_level not in _TOXICITY_LEVELS:
        add(
            f"toxicity_level「{record.toxicity_level}」非标准枚举（大毒|有毒|小毒|无毒）",
            Severity.WARNING,
        )
    return issues


def coverage_member_count(records: list[SourceRecord]) -> list[ValidationIssue]:
    """妊娠清单 member_count 属性 vs 实际 target_herbs 数量。"""
    issues: list[ValidationIssue] = []
    for original in records:
        record = view(original)
        if record.pregnancy is None:
            continue
        declared = record.attributes.get("member_count")
        if declared is None:
            continue
        try:
            declared_count = int(declared)
        except (TypeError, ValueError):
            issues.append(
                ValidationIssue(
                    record_id=record.record_id,
                    category=IssueCategory.COVERAGE,
                    severity=Severity.WARNING,
                    message=f"member_count 非数值: {declared!r}",
                )
            )
            continue
        actual = len(record.pregnancy.target_herbs)
        if declared_count != actual:
            issues.append(
                ValidationIssue(
                    record_id=record.record_id,
                    category=IssueCategory.COVERAGE,
                    severity=Severity.ERROR,
                    message=(
                        f"成员覆盖率不符：member_count={declared_count}，"
                        f"实际 target_herbs={actual}（疑似遗漏或重复成员）"
                    ),
                )
            )
    return issues


def coverage_table_rows(
    records: list[SourceRecord], blocks: list[dict[str, Any]]
) -> list[ValidationIssue]:
    """表格行数覆盖率：表格 block 数据行数 vs 该章节 HERB_TABLE 记录数。

    按 block.heading 与 record.source.section_path[-1] 对齐章节。
    WARNING 级（部分行可能为分组行/注释行，允许偏差）。
    """
    rows_by_section: Counter[str] = Counter()
    for block in blocks:
        content = str(block.get("content") or "")
        if "<drawing" not in content:
            continue
        heading = str(block.get("heading") or "").strip()
        if not heading:
            continue
        rows = _count_table_rows(content)
        if rows > 0:
            rows_by_section[heading] += rows

    if not rows_by_section:
        return []

    record_counts: Counter[str] = Counter()
    for record in records:
        if (
            record.record_type not in (RecordType.HERB_TABLE, RecordType.TABLE_ROW)
            or not record.source.section_path
        ):
            continue
        record_counts[record.source.section_path[-1]] += 1

    issues: list[ValidationIssue] = []
    for section, rows in rows_by_section.items():
        got = record_counts.get(section, 0)
        if got < rows:
            issues.append(
                ValidationIssue(
                    record_id="-",
                    category=IssueCategory.COVERAGE,
                    severity=Severity.WARNING,
                    message=(
                        f"表格覆盖率不足：章节「{section}」表格约 {rows} 数据行，"
                        f"仅抽取 {got} 条记录（疑似跳行/合并行）"
                    ),
                )
            )
    return issues


def _count_table_rows(content: str) -> int:
    """统计表格数据行数（HTML/Markdown 均剔除表头与分隔行）。"""
    if "<tr" in content:
        # 首行 <tr> 视为表头，数据行 = 总数 - 1
        return max(content.count("<tr") - 1, 0)
    lines = [ln.strip() for ln in content.splitlines() if ln.strip().startswith("|")]
    if not lines:
        return 0
    separators = sum(1 for ln in lines if re.fullmatch(r"\|[\s:|-]+\|?", ln))
    return max(len(lines) - separators - 1, 0)


def dedup_key(record: SourceRecord) -> tuple | None:
    """按记录类型构造去重主键。"""
    record = view(record)
    if record.record_type in (RecordType.HERB_TABLE, RecordType.DOSAGE, RecordType.USAGE):
        if record.herb_name:
            return (record.record_type, record.herb_name)
    elif record.record_type == RecordType.INTERACTION and record.interaction is not None:
        members = tuple(sorted(m.herb_name for m in record.interaction.members))
        return (record.interaction.anchor.herb_name, members)
    elif record.record_type == RecordType.PREGNANCY and record.pregnancy is not None:
        return (
            record.pregnancy.restriction_level,
            tuple(record.pregnancy.target_herbs),
        )
    elif record.record_type == RecordType.HERB_FACTS and record.herb_facts is not None:
        return (record.herb_facts.herb_name,)
    elif record.record_type == RecordType.FIGURE and record.image_ref is not None:
        # FIGURE 记录按图片路径去重（同一张图片不应重复抽取）
        return (RecordType.FIGURE, record.image_ref.asset_path)
    elif record.record_type in (RecordType.GENERAL_RULE, RecordType.POPULATION):
        if record.subject_text and record.action is not None:
            return (record.record_type, record.subject_text, record.action)
    return None


def _check_numeric(record: SourceRecord, outlier_grams: float = 100) -> list[ValidationIssue]:
    """剂量数值卫生（非负、min<=max、离群提示；离群阈值由领域包注入）。"""
    record = view(record)
    dose = record.dose
    if dose is None:
        return []
    issues: list[ValidationIssue] = []

    def add(message: str, severity: Severity = Severity.ERROR) -> None:
        issues.append(
            ValidationIssue(
                record_id=record.record_id,
                category=IssueCategory.NUMERIC,
                severity=severity,
                message=message,
            )
        )

    if dose.min_dose is not None and dose.min_dose < 0:
        add(f"min_dose 为负数: {dose.min_dose}")
    if dose.max_dose is not None and dose.max_dose < 0:
        add(f"max_dose 为负数: {dose.max_dose}")
    if dose.min_dose is not None and dose.max_dose is not None and dose.min_dose > dose.max_dose:
        add(f"min_dose ({dose.min_dose}) > max_dose ({dose.max_dose})")
    if dose.min_dose is not None and dose.max_dose is None:
        add("剂量只有下限无上限（疑似字段错位）", Severity.WARNING)
    if dose.max_dose is not None and dose.max_dose > outlier_grams:
        add(f"max_dose ({dose.max_dose}g) 超出常规范围，需人工复核", Severity.WARNING)
    return issues
