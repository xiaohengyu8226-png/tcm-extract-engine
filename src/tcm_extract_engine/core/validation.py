"""校验信封模型（骨架）：严重级别 / 问题类别 / 单条问题 / 报告。

域规则由领域包（domains/*/validate.py）提供，产出 ValidationIssue；
骨架（cab/validator.py）持有框架、通用不变量（溯源完备）与编排。
本模块置于 core（骨架基础），使包 → 骨架的单向依赖不经过 cab 编排层。
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


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
