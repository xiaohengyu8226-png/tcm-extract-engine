"""中医自动化抽取与自反馈自优化引擎（C/A/B + GEPA 飞轮）。

在 Artifact-Driven 8 阶段流水线之上：
  - Book Manifest + 显式状态机（可断点恢复）
  - 分层 Artifact 工作目录（00_source ~ 08_publish）
  - SourceRecord / SafetyRule 统一 Schema
  - C/A/B 职责严格隔离（Profiler+Planner / Extractor / Validator+Verifier）
  - GEPA 反思式进化 + 跨书飞轮（自反馈自优化）

对外主入口：``tcm_extract_engine.runner.BookPipeline``。
"""

from tcm_extract_engine.cab.validator import (
    IssueCategory,
    Severity,
    ValidationIssue,
    ValidationReport,
    validate_records,
)
from tcm_extract_engine.cab.verifier import (
    CorrectionEntry,
    Verdict,
    VerificationReport,
    VerificationResult,
    verify_records,
)
from tcm_extract_engine.core.artifacts import (
    ArtifactHeader,
    artifact_hash,
    load_artifact,
    save_artifact,
)
from tcm_extract_engine.core.book import BookWorkspace
from tcm_extract_engine.core.manifest import BookManifest, BookStatus, StageRecord
from tcm_extract_engine.core.records import RecordType, SafetyRule, SourceRecord
from tcm_extract_engine.runner import BookPipeline

__all__ = [
    "ArtifactHeader",
    "BookManifest",
    "BookPipeline",
    "BookStatus",
    "BookWorkspace",
    "CorrectionEntry",
    "IssueCategory",
    "RecordType",
    "SafetyRule",
    "Severity",
    "SourceRecord",
    "StageRecord",
    "ValidationIssue",
    "ValidationReport",
    "Verdict",
    "VerificationReport",
    "VerificationResult",
    "artifact_hash",
    "load_artifact",
    "save_artifact",
    "validate_records",
    "verify_records",
]
