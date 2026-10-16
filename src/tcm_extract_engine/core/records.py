"""Domain-independent, versioned records and evidence references."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

RECORD_SCHEMA_VERSION = "2"


class SourceRef(BaseModel):
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    context_block_ids: list[str] = Field(default_factory=list)
    chunk_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_spans: list[dict[str, Any]] = Field(default_factory=list)
    location_status: str = "unmatched"


class GenerationRef(BaseModel):
    extractor: str = ""
    template: str = ""
    prompt_version: str = ""
    modality: str = "text"
    model: str = ""
    run_id: str = ""


class ImageRef(BaseModel):
    """图片引用与抽取结果（多模态 FIGURE 记录专用）。

    承载视觉模型从图片中抽取的结构化信息，
    与 raw_text（MinerU 文本描述）互补。
    """

    asset_id: str | None = None
    asset_path: str  # 图片文件路径
    figure_type: str  # 药材形态图 | 图表 | 流程图 | 表格截图 | 显微图 | 其他
    description: str = ""  # LLM 生成的图片内容描述
    extracted_data: dict[str, Any] = Field(default_factory=dict)  # 结构化抽取结果
    confidence: float = 0.0  # 视觉模型置信度 (0-1)
    ocr_text: str = ""  # 图片中的文字（OCR 结果）


class FigurePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_ref: ImageRef
    attributes: dict[str, Any] = Field(default_factory=dict)


class TablePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    headers: list[str] = Field(default_factory=list)
    cells: list[str] = Field(default_factory=list)
    row_index: int = 0
    header_rows: list[list[str]] = Field(default_factory=list)
    cell_origins: list[tuple[int, int]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class SafetyRule(BaseModel):
    """发布层规则（RuleCompiler 从 ReviewedRecord 编译，非 A 直接产出）。"""

    rule_id: str
    rule_type: str
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


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: str = RECORD_SCHEMA_VERSION
    record_id: str
    book_id: str
    domain_id: str = "tcm"
    record_type: str
    source: SourceRef = Field(default_factory=SourceRef)
    generation: GenerationRef = Field(default_factory=GenerationRef)
    raw_text: str = ""
    subject_text: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    needs_review: bool = False
    ambiguities: list[str] = Field(default_factory=list)
    passthrough: bool = False

    @model_validator(mode="after")
    def check_payload(self) -> SourceRecord:
        if self.schema_version != RECORD_SCHEMA_VERSION:
            raise ValueError("旧记录格式，请先运行 migrate_records_to_envelope.py")
        from tcm_extract_engine.domains import get_domain_pack

        pack = get_domain_pack(self.domain_id)
        self.payload = pack.validate_payload(self.record_type, self.payload)
        json.dumps(self.payload, allow_nan=False)
        return self

    def has_evidence(self) -> bool:
        return bool(self.raw_text.strip())


def stable_record_id(record: SourceRecord) -> str:
    basis = {
        "book_id": record.book_id,
        "domain_id": record.domain_id,
        "type": record.record_type,
        "subject_text": record.subject_text,
        "payload": record.payload,
        "blocks": sorted(record.source.block_ids),
        "raw_text": record.raw_text,
    }
    digest = hashlib.sha256(
        json.dumps(basis, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"SR-{record.book_id[:12]}-{digest}"
