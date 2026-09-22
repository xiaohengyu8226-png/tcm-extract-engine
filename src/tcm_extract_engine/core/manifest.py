"""Book Manifest 与显式状态机。

每本书进入系统后生成 ``workspace/books/<book_id>/manifest.json``，
记录文件指纹、阶段产物与当前状态；幂等键为
``file_hash + pipeline_version + prompt_version``。

状态机（禁止只使用 done=true 之类的模糊状态）：

    PENDING → PARSING → PARSED → PROFILING → PROFILED
            → PLAN_GENERATED → WAITING_PLAN_CONFIRMATION → PLAN_CONFIRMED
            → EXTRACTING → EXTRACTED → VALIDATING → VALIDATED
            → VERIFYING → VERIFIED → WAITING_REVIEW → REVIEWED
            → PUBLISHED
    任意阶段失败 → FAILED（记录 error 信息，支持 resume）
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

PIPELINE_VERSION = "v0.3.0-artifact"

MANIFEST_FILENAME = "manifest.json"


class BookStatus(str, enum.Enum):
    """书籍处理状态机；迁移必须沿 STAGE_ORDER 前进。"""

    PENDING = "PENDING"
    PARSING = "PARSING"
    PARSED = "PARSED"
    PROFILING = "PROFILING"
    PROFILED = "PROFILED"
    PLAN_GENERATED = "PLAN_GENERATED"
    WAITING_PLAN_CONFIRMATION = "WAITING_PLAN_CONFIRMATION"
    PLAN_CONFIRMED = "PLAN_CONFIRMED"
    EXTRACTING = "EXTRACTING"
    EXTRACTED = "EXTRACTED"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    WAITING_REVIEW = "WAITING_REVIEW"
    REVIEWED = "REVIEWED"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"


# 状态机的合法前向迁移表（FAILED 允许从任意状态进入）
VALID_TRANSITIONS: dict[BookStatus, frozenset[BookStatus]] = {
    BookStatus.PENDING: frozenset({BookStatus.PARSING, BookStatus.FAILED}),
    BookStatus.PARSING: frozenset({BookStatus.PARSED, BookStatus.FAILED}),
    BookStatus.PARSED: frozenset({BookStatus.PROFILING, BookStatus.FAILED}),
    BookStatus.PROFILING: frozenset({BookStatus.PROFILED, BookStatus.FAILED}),
    BookStatus.PROFILED: frozenset({BookStatus.PLAN_GENERATED, BookStatus.FAILED}),
    BookStatus.PLAN_GENERATED: frozenset(
        {BookStatus.WAITING_PLAN_CONFIRMATION, BookStatus.FAILED}
    ),
    BookStatus.WAITING_PLAN_CONFIRMATION: frozenset(
        {BookStatus.PLAN_CONFIRMED, BookStatus.FAILED}
    ),
    BookStatus.PLAN_CONFIRMED: frozenset({BookStatus.EXTRACTING, BookStatus.FAILED}),
    BookStatus.EXTRACTING: frozenset({BookStatus.EXTRACTED, BookStatus.FAILED}),
    BookStatus.EXTRACTED: frozenset({BookStatus.VALIDATING, BookStatus.FAILED}),
    BookStatus.VALIDATING: frozenset({BookStatus.VALIDATED, BookStatus.FAILED}),
    BookStatus.VALIDATED: frozenset({BookStatus.VERIFYING, BookStatus.FAILED}),
    BookStatus.VERIFYING: frozenset({BookStatus.VERIFIED, BookStatus.FAILED}),
    BookStatus.VERIFIED: frozenset({BookStatus.WAITING_REVIEW, BookStatus.FAILED}),
    BookStatus.WAITING_REVIEW: frozenset({BookStatus.REVIEWED, BookStatus.FAILED}),
    BookStatus.REVIEWED: frozenset({BookStatus.PUBLISHED, BookStatus.FAILED}),
    BookStatus.PUBLISHED: frozenset(),
    BookStatus.FAILED: frozenset({BookStatus.PENDING}),
}

# 阶段名 → 完成时状态（用于 resume 定位）
STAGE_ORDER: tuple[str, ...] = (
    "parse",
    "profile",
    "plan",
    "extract",
    "validate",
    "verify",
    "review",
    "publish",
)


class StageRecord(BaseModel):
    """单阶段的产物登记。"""

    stage: str
    status: str = "PENDING"  # PENDING | RUNNING | DONE | FAILED | SKIPPED
    artifact: str | None = None  # 相对 book 目录的 Artifact 路径
    artifact_hash: str | None = None  # 上游 Artifact 内容 hash
    started_at: str | None = None
    finished_at: str | None = None
    records: int | None = None
    error: str | None = None

    def mark_running(self) -> None:
        self.status = "RUNNING"
        self.started_at = _now_iso()
        self.error = None

    def mark_done(self, *, artifact: str, artifact_hash: str, records: int | None = None) -> None:
        self.status = "DONE"
        self.artifact = artifact
        self.artifact_hash = artifact_hash
        self.finished_at = _now_iso()
        self.records = records

    def mark_failed(self, error: str) -> None:
        self.status = "FAILED"
        self.error = error[:2000]
        self.finished_at = _now_iso()


class BookManifest(BaseModel):
    """书籍工作目录的唯一权威状态文件。"""

    schema_version: str = "1"
    book_id: str
    file_name: str
    file_hash: str  # sha256:<hex>
    file_size: int = 0

    # 书目元信息（profile 阶段补充，先允许 null）
    source_type: str | None = None
    title: str | None = None
    edition: str | None = None
    publisher: str | None = None
    publish_year: str | None = None
    standard_no: str | None = None

    status: BookStatus = BookStatus.PENDING
    pipeline_version: str = PIPELINE_VERSION
    prompt_version: str | None = None

    created_at: str = Field(default_factory=lambda: _now_iso())
    updated_at: str = Field(default_factory=lambda: _now_iso())
    error: str | None = None

    stages: dict[str, StageRecord] = Field(default_factory=dict)

    def stage(self, name: str) -> StageRecord:
        """获取（或惰性创建）阶段记录。"""
        if name not in self.stages:
            if name not in STAGE_ORDER:
                raise ValueError(f"未知阶段名: {name}")
            self.stages[name] = StageRecord(stage=name)
        return self.stages[name]

    def transition(self, new_status: BookStatus) -> None:
        """按状态机校验并迁移状态。"""
        if new_status == self.status:
            return
        allowed = VALID_TRANSITIONS.get(self.status, frozenset())
        if new_status not in allowed:
            raise ValueError(f"非法状态迁移: {self.status.value} → {new_status.value}")
        self.status = new_status
        self.updated_at = _now_iso()

    def rewind(self, to_status: BookStatus) -> None:
        """显式回退状态（仅用于 rerun --from；下游阶段记录由调用方作废）。"""
        order = list(BookStatus)
        if order.index(to_status) >= order.index(self.status):
            raise ValueError(
                f"rewind 只能向后: {self.status.value} → {to_status.value}"
            )
        self.status = to_status
        self.error = None
        self.updated_at = _now_iso()

    def fail(self, error: str) -> None:
        """标记失败；FAILED 是唯一允许的跨阶段迁移。"""
        self.error = error[:2000]
        self.status = BookStatus.FAILED
        self.updated_at = _now_iso()

    def completed_stages(self) -> list[str]:
        """按 STAGE_ORDER 返回已完成阶段。"""
        return [
            name for name in STAGE_ORDER
            if self.stages.get(name) and self.stages[name].status == "DONE"
        ]

    def save(self, book_dir: Path) -> None:
        """原子写入 manifest.json。"""
        from .artifacts import _atomic_write_json

        self.updated_at = _now_iso()
        _atomic_write_json(book_dir / MANIFEST_FILENAME, self.model_dump(mode="json"))

    @classmethod
    def load(cls, book_dir: Path) -> BookManifest:
        """从 book 目录读取 manifest；不存在时抛 FileNotFoundError。"""
        import json

        path = book_dir / MANIFEST_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"manifest 不存在: {path}")
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
