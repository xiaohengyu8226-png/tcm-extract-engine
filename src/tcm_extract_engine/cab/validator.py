"""确定性校验层（Validation）：A 的输出必须先过代码规则，才允许进入 B。

校验全部为确定性规则（只依赖记录本身 + 可选 blocks），不调用 LLM：
  SCHEMA    记录结构与 record_type 一致（INTERACTION 必有 interaction 载荷…）
  NUMERIC   剂量数值合法（非负、min<=max、离群值提示）
  SOURCE    可回溯（raw_text 非空、页码/block 定位）
  ENTITY    实体完整（关系组成员非空、妊娠目标非空、成员无重复）
  SEMANTIC  语义天花板（措辞强度禁止升级；restriction_level 与 action 一致）
  COVERAGE  覆盖率（member_count 比对、表格行数比对、疑似重复记录）

严重级别：
  ERROR   阻断进入 B（runner 将记录标记 needs_review 并暂停，或丢弃重抽）
  WARNING 放行但标记复核（人工抽查阶段重点看）
"""

from __future__ import annotations

import enum
import logging
import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from tcm_extract_engine.core.records import (
    _ACTION_RANK,
    ActionLevel,
    RecordType,
    SourceRecord,
    action_ceiling,
    action_exceeds_ceiling,
)

logger = logging.getLogger(__name__)


class Severity(str, enum.Enum):
    ERROR = "ERROR"      # 阻断进入 B
    WARNING = "WARNING"  # 放行但标记复核


class IssueCategory(str, enum.Enum):
    SCHEMA = "SCHEMA"
    NUMERIC = "NUMERIC"
    SOURCE = "SOURCE"
    ENTITY = "ENTITY"
    SEMANTIC = "SEMANTIC"
    COVERAGE = "COVERAGE"


class ValidationIssue(BaseModel):
    """单条校验问题。"""

    record_id: str
    category: IssueCategory
    severity: Severity
    message: str


class ValidationReport(BaseModel):
    """validate_records 的输出（05_validate/validation_report.json 主体）。"""

    total_records: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    @property
    def passed(self) -> bool:
        """无 ERROR 才允许进入 B。"""
        return self.error_count == 0

    def errors_for(self, record_id: str) -> list[ValidationIssue]:
        return [
            i for i in self.issues
            if i.severity == Severity.ERROR and i.record_id == record_id
        ]

    def summary(self) -> str:
        return (
            f"{self.total_records} 条记录: {self.error_count} ERROR / "
            f"{self.warning_count} WARNING → {'PASS' if self.passed else 'FAIL'}"
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

# 离群剂量上限（g；超过则提示人工复核，不阻断）
_DOSE_OUTLIER_GRAMS = 1000.0

_TOXICITY_LEVELS = {"大毒", "有毒", "小毒", "无毒"}


def validate_records(
    records: list[SourceRecord],
    *,
    blocks: list[dict[str, Any]] | None = None,
) -> ValidationReport:
    """对 A 输出做确定性校验。

    Args:
        records: Model A 产出的全部 SourceRecord。
        blocks: 解析 blocks（可选）；提供时启用表格行数覆盖率比对。

    Returns:
        ValidationReport；``passed`` 为 False 表示存在 ERROR，不应进入 B。
    """
    issues: list[ValidationIssue] = []
    for record in records:
        issues.extend(_check_schema(record))
        issues.extend(_check_numeric(record))
        issues.extend(_check_source(record))
        issues.extend(_check_entity(record))
        issues.extend(_check_semantic(record))
    issues.extend(_check_coverage_member_count(records))
    issues.extend(_check_coverage_duplicates(records))
    if blocks is not None:
        issues.extend(_check_coverage_table_rows(records, blocks))

    report = ValidationReport(total_records=len(records), issues=issues)
    logger.info("[Validator] %s", report.summary())
    return report


# ── 单记录校验 ───────────────────────────────────────────────────────────────


def _check_schema(record: SourceRecord) -> list[ValidationIssue]:
    """记录结构与 record_type 一致性。"""
    issues: list[ValidationIssue] = []

    def add(message: str, severity: Severity = Severity.ERROR) -> None:
        issues.append(ValidationIssue(
            record_id=record.record_id, category=IssueCategory.SCHEMA,
            severity=severity, message=message,
        ))

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
    return issues


def _check_numeric(record: SourceRecord) -> list[ValidationIssue]:
    """剂量数值合法性。"""
    dose = record.dose
    if dose is None:
        return []
    issues: list[ValidationIssue] = []

    def add(message: str, severity: Severity = Severity.ERROR) -> None:
        issues.append(ValidationIssue(
            record_id=record.record_id, category=IssueCategory.NUMERIC,
            severity=severity, message=message,
        ))

    if dose.min_dose is not None and dose.min_dose < 0:
        add(f"min_dose 为负数: {dose.min_dose}")
    if dose.max_dose is not None and dose.max_dose < 0:
        add(f"max_dose 为负数: {dose.max_dose}")
    if (
        dose.min_dose is not None
        and dose.max_dose is not None
        and dose.min_dose > dose.max_dose
    ):
        add(f"min_dose ({dose.min_dose}) > max_dose ({dose.max_dose})")
    if dose.min_dose is not None and dose.max_dose is None:
        add("剂量只有下限无上限（疑似字段错位）", Severity.WARNING)
    if dose.max_dose is not None and dose.max_dose > _DOSE_OUTLIER_GRAMS:
        add(f"max_dose ({dose.max_dose}g) 超出常规范围，需人工复核", Severity.WARNING)
    return issues


def _check_source(record: SourceRecord) -> list[ValidationIssue]:
    """可回溯性：原文支撑 + 物理定位。"""
    issues: list[ValidationIssue] = []
    if not record.raw_text.strip():
        issues.append(ValidationIssue(
            record_id=record.record_id, category=IssueCategory.SOURCE,
            severity=Severity.ERROR,
            message="raw_text 为空：记录无原文支撑，违反忠实抽取原则",
        ))
    if record.source.page_start is None and not record.source.block_ids:
        issues.append(ValidationIssue(
            record_id=record.record_id, category=IssueCategory.SOURCE,
            severity=Severity.WARNING,
            message="无页码且无 block_ids，无法回溯定位",
        ))
    return issues


def _check_entity(record: SourceRecord) -> list[ValidationIssue]:
    """实体完整性。"""
    issues: list[ValidationIssue] = []

    def add(message: str, severity: Severity = Severity.ERROR) -> None:
        issues.append(ValidationIssue(
            record_id=record.record_id, category=IssueCategory.ENTITY,
            severity=severity, message=message,
        ))

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
    return issues


def _check_semantic(record: SourceRecord) -> list[ValidationIssue]:
    """语义天花板：措辞强度禁止升级。"""
    issues: list[ValidationIssue] = []

    def add(message: str, severity: Severity = Severity.ERROR) -> None:
        issues.append(ValidationIssue(
            record_id=record.record_id, category=IssueCategory.SEMANTIC,
            severity=severity, message=message,
        ))

    # 原文措辞 vs action（records.action_exceeds_ceiling）
    if action_exceeds_ceiling(record):
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
            f"toxicity_level「{record.toxicity_level}」非标准枚举"
            "（大毒|有毒|小毒|无毒）", Severity.WARNING,
        )
    return issues


# ── 跨记录覆盖率校验 ─────────────────────────────────────────────────────────


def _check_coverage_member_count(records: list[SourceRecord]) -> list[ValidationIssue]:
    """妊娠清单 member_count 属性 vs 实际 target_herbs 数量。"""
    issues: list[ValidationIssue] = []
    for record in records:
        if record.pregnancy is None:
            continue
        declared = record.attributes.get("member_count")
        if declared is None:
            continue
        try:
            declared_count = int(declared)
        except (TypeError, ValueError):
            issues.append(ValidationIssue(
                record_id=record.record_id, category=IssueCategory.COVERAGE,
                severity=Severity.WARNING,
                message=f"member_count 非数值: {declared!r}",
            ))
            continue
        actual = len(record.pregnancy.target_herbs)
        if declared_count != actual:
            issues.append(ValidationIssue(
                record_id=record.record_id, category=IssueCategory.COVERAGE,
                severity=Severity.ERROR,
                message=(
                    f"成员覆盖率不符：member_count={declared_count}，"
                    f"实际 target_herbs={actual}（疑似遗漏或重复成员）"
                ),
            ))
    return issues


def _check_coverage_duplicates(records: list[SourceRecord]) -> list[ValidationIssue]:
    """疑似重复记录（相同主键出现多次）。"""
    issues: list[ValidationIssue] = []
    seen: dict[tuple, str] = {}
    for record in records:
        key = _dedup_key(record)
        if key is None:
            continue
        if key in seen:
            issues.append(ValidationIssue(
                record_id=record.record_id, category=IssueCategory.COVERAGE,
                severity=Severity.WARNING,
                message=f"疑似重复记录（与 {seen[key]} 主键相同），需去重或合并",
            ))
        else:
            seen[key] = record.record_id
    return issues


def _dedup_key(record: SourceRecord) -> tuple | None:
    """按记录类型构造去重主键。"""
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
    elif record.record_type in (RecordType.GENERAL_RULE, RecordType.POPULATION):
        if record.subject_text and record.action is not None:
            return (record.record_type, record.subject_text, record.action)
    return None


def _check_coverage_table_rows(
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
        if record.record_type != RecordType.HERB_TABLE or not record.source.section_path:
            continue
        record_counts[record.source.section_path[-1]] += 1

    issues: list[ValidationIssue] = []
    for section, rows in rows_by_section.items():
        got = record_counts.get(section, 0)
        if got < rows:
            issues.append(ValidationIssue(
                record_id="-",
                category=IssueCategory.COVERAGE,
                severity=Severity.WARNING,
                message=(
                    f"表格覆盖率不足：章节「{section}」表格约 {rows} 数据行，"
                    f"仅抽取 {got} 条记录（疑似跳行/合并行）"
                ),
            ))
    return issues


def _count_table_rows(content: str) -> int:
    """统计表格数据行数（HTML 按 <tr 计；Markdown 按 | 行计，剔除表头与分隔行）。"""
    if "<tr" in content:
        return content.count("<tr")
    lines = [ln.strip() for ln in content.splitlines() if ln.strip().startswith("|")]
    if not lines:
        return 0
    separators = sum(
        1 for ln in lines if re.fullmatch(r"\|[\s:|-]+\|?", ln)
    )
    return max(len(lines) - separators - 1, 0)
