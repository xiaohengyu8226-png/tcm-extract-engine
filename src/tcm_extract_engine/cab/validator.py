"""确定性校验层（Validation）：A 的输出必须先过代码规则，才允许进入 B。

校验全部为确定性规则（只依赖记录本身 + 可选 blocks），不调用 LLM。
骨架持有：
  SCHEMA/ENTITY/SEMANTIC  域规则 → 领域包 validate_record（见 domains/tcm/validate.py）
  NUMERIC   剂量数值卫生（非负、min<=max、离群值提示；阈值由包注入）
  SOURCE    可回溯（raw_text 非空、页码/block 定位）—— 溯源完备通用不变量
  COVERAGE  域覆盖率（member_count / 表格行数）→ 包；重复检测框架 → 骨架

严重级别：
  ERROR   阻断进入 B（runner 将记录标记 needs_review 并暂停，或丢弃重抽）
  WARNING 放行但标记复核（人工抽查阶段重点看）

信封模型（Severity/IssueCategory/ValidationIssue/ValidationReport）在
core/validation.py，本模块 shim 重导出（保持导入面不变）。
"""

from __future__ import annotations

import logging
from typing import Any

from tcm_extract_engine.core.records import SourceRecord
from tcm_extract_engine.core.validation import (
    IssueCategory,  # noqa: F401  # shim 重导出
    Severity,  # noqa: F401  # shim 重导出
    ValidationIssue,  # noqa: F401  # shim 重导出
    ValidationReport,  # noqa: F401  # shim 重导出
)
from tcm_extract_engine.domains import get_domain_pack

logger = logging.getLogger(__name__)


def validate_records(
    records: list[SourceRecord],
    *,
    blocks: list[dict[str, Any]] | None = None,
    pack: Any = None,
) -> ValidationReport:
    """对 A 输出做确定性校验。

    Args:
        records: Model A 产出的全部 SourceRecord。
        blocks: 解析 blocks（可选）；提供时启用表格行数覆盖率比对。

    Returns:
        ValidationReport；``passed`` 为 False 表示存在 ERROR，不应进入 B。
    """
    pack = pack or get_domain_pack(records[0].domain_id if records else "tcm")
    issues: list[ValidationIssue] = []
    for record in records:
        issues.extend(pack.validate_record(record))
        if record.domain_id != pack.id:
            raise ValueError("记录与领域包不匹配")
        issues.extend(_check_source(record))
    issues.extend(pack.coverage_member_count(records))
    issues.extend(_check_coverage_duplicates(records, pack=pack))
    if blocks is not None:
        issues.extend(pack.coverage_table_rows(records, blocks))

    report = ValidationReport(total_records=len(records), issues=issues)
    logger.info("[Validator] %s", report.summary())
    return report


# ── 骨架通用校验（域无关）────────────────────────────────────────────────────


def _check_source(record: SourceRecord) -> list[ValidationIssue]:
    """可回溯性：原文支撑 + 物理定位（溯源完备通用不变量）。"""
    issues: list[ValidationIssue] = []
    if not record.raw_text.strip():
        issues.append(
            ValidationIssue(
                record_id=record.record_id,
                category=IssueCategory.SOURCE,
                severity=Severity.ERROR,
                message="raw_text 为空：记录无原文支撑，违反忠实抽取原则",
            )
        )
    if record.source.location_status == "context_only":
        issues.append(ValidationIssue(record_id=record.record_id, category=IssueCategory.SOURCE,
            severity=Severity.WARNING, message="原文无法精确匹配，只保留上下文定位，需复核"))
    if record.source.page_start is None and not record.source.block_ids:
        issues.append(
            ValidationIssue(
                record_id=record.record_id,
                category=IssueCategory.SOURCE,
                severity=Severity.WARNING,
                message="无页码且无 block_ids，无法回溯定位",
            )
        )
    return issues


def _check_coverage_duplicates(records: list[SourceRecord], *, pack: Any) -> list[ValidationIssue]:
    """疑似重复记录（相同主键出现多次）；去重主键由领域包构造。"""
    issues: list[ValidationIssue] = []
    seen: dict[tuple, str] = {}
    for record in records:
        key = pack.dedup_key(record)
        if key is None:
            continue
        if key in seen:
            issues.append(
                ValidationIssue(
                    record_id=record.record_id,
                    category=IssueCategory.COVERAGE,
                    severity=Severity.WARNING,
                    message=f"疑似重复记录（与 {seen[key]} 主键相同），需去重或合并",
                )
            )
        else:
            seen[key] = record.record_id
    return issues
