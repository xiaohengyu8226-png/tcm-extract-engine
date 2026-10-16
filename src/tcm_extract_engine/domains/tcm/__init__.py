"""TCM 领域包：中药饮片安全审方域知识（schema/分类/解析/校验/导出）。

manifest.json 是声明面（goal/标签/风险文案/适配字段），行为代码在本包各模块；
schema_spec.json 是记录 schema 规范（M5 引入，人闸审查面 + C0 起草面）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tcm_extract_engine.domains import AdaptationField, DomainPack, RiskRule
from tcm_extract_engine.domains.tcm.gepa_map import CATEGORY_ISSUE_MAP, VERDICT_ISSUE_MAP
from tcm_extract_engine.domains.tcm.parse import parse_record
from tcm_extract_engine.domains.tcm.validate import (
    coverage_member_count,
    coverage_table_rows,
    dedup_key,
    validate_record,
)

_MANIFEST = Path(__file__).parent / "manifest.json"
_SPEC = Path(__file__).parent / "schema_spec.json"


class TCMPack(DomainPack):
    """TCM 包：声明面从 manifest.json 装载，行为面由本包模块提供。"""

    def __init__(self) -> None:
        raw: dict[str, Any] = json.loads(_MANIFEST.read_text(encoding="utf-8"))
        self.id: str = raw["id"]
        self.version = "2"
        self.figure_template = "figure.md"
        from tcm_extract_engine.domains.tcm.classify import (
            CONTENT_TYPE_TO_EXTRACTOR,
            EXTRACTOR_TEMPLATES,
            classify_section,
            infer_document_type,
        )
        from tcm_extract_engine.domains.tcm.review import (
            _related_herbs,
            _structured_content,
            _subject,
        )

        self.route_section = classify_section
        self.infer_document_type = infer_document_type
        self.content_extractors = {str(k.value): v for k, v in CONTENT_TYPE_TO_EXTRACTOR.items()}
        self.extractor_templates = EXTRACTOR_TEMPLATES
        self.chunk_strategies = {
            k: ("whole_section" if k == "InteractionExtractor" else "split")
            for k in EXTRACTOR_TEMPLATES
        }
        self.flatten_for_review = _structured_content
        self.subject_of = _subject
        self.related_subjects = _related_herbs
        self.label: str = raw["label"]
        self.goal: str = raw["goal"]
        self.profile_domain: str = raw["profile_domain"]
        self.dose_outlier_grams: float = raw["dose_outlier_grams"]
        self.review_verdicts: frozenset[str] = frozenset(raw["review_verdicts"])
        self.verdict_issue_map = {
            **VERDICT_ISSUE_MAP,
            "PARTIAL": ("inconsistent", "medium", 0.9),
            "FIGURE_REVIEW": ("inconsistent", "high", 0.9),
        }
        self.category_issue_map = CATEGORY_ISSUE_MAP
        self.labels: dict[str, Any] = raw["labels"]
        self.schema_spec: dict[str, Any] = json.loads(_SPEC.read_text(encoding="utf-8"))
        self.risk_rules: list[RiskRule] = [RiskRule(**r) for r in raw.get("risk_rules", [])]
        self.adaptation_fields: list[AdaptationField] = [
            AdaptationField(**f) for f in raw.get("adaptation_fields", [])
        ]
        self.parse_record = parse_record
        self.validate_record = validate_record
        self.coverage_member_count = coverage_member_count
        self.coverage_table_rows = coverage_table_rows
        self.dedup_key = dedup_key

    def validate_payload(self, record_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        from tcm_extract_engine.core.records import TablePayload
        from tcm_extract_engine.domains.tcm.schema import RecordType, TCMPayload

        if record_type not in {v.value for v in RecordType}:
            raise ValueError(f"TCM 不支持 record_type={record_type}")
        if record_type == "TABLE_ROW":
            return TablePayload.model_validate(payload).model_dump(mode="json")
        return TCMPayload.model_validate(payload).model_dump(mode="json")

    def normalize_correction(self, corrected: dict[str, Any]) -> dict[str, Any]:
        from tcm_extract_engine.domains.tcm.schema import TCMPayload

        corrected = dict(corrected)
        legacy = {k: corrected.pop(k) for k in list(corrected) if k in TCMPayload.model_fields}
        if legacy:
            corrected["payload"] = {**corrected.get("payload", {}), **legacy}
        return corrected
