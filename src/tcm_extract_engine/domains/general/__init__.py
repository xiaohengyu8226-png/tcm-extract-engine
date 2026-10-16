from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tcm_extract_engine.domains import AdaptationField
from tcm_extract_engine.domains.general.classify import infer_document_type, route_section
from tcm_extract_engine.domains.general.parse import parse_record
from tcm_extract_engine.domains.general.validate import (
    coverage_member_count,
    coverage_table_rows,
    dedup_key,
    validate_record,
)


class GeneralPack:
    id = "general"
    version = "2"
    label = "通用知识"
    goal = "忠实、细粒度地抽取本书知识并保留条件和原文证据"
    profile_domain = "general"
    figure_template = "figure_general.md"
    dose_outlier_grams = 0.0
    review_verdicts = frozenset(
        {
            "PARTIAL",
            "UNSUPPORTED",
            "MISSING",
            "WRONG_DOSE",
            "WRONG_RELATION",
            "WRONG_NORMALIZATION",
            "OCR_UNCERTAIN",
            "FIGURE_REVIEW",
        }
    )
    verdict_issue_map = {
        v: ("unsupported" if v == "UNSUPPORTED" else "inconsistent", "high", 0.9)
        for v in review_verdicts
    }
    category_issue_map = {
        v: ("inconsistent", "high", 0.9)
        for v in ("SCHEMA", "SOURCE", "ENTITY", "SEMANTIC", "NUMERIC", "COVERAGE")
    }
    content_extractors = {"KNOWLEDGE": "KnowledgeExtractor", "IRRELEVANT": ""}
    extractor_templates = {
        "KnowledgeExtractor": "knowledge.md",
        "FigureExtractor": "figure_general.md",
        "TableKnowledgeExtractor": "knowledge_table.md",
    }
    chunk_strategies = {k: "split" for k in extractor_templates}
    risk_rules: list = []
    adaptation_fields = [
        AdaptationField("scope", "抽取范围"),
        AdaptationField("constraints", "证据与条件约束", True),
    ]
    route_section = staticmethod(route_section)
    infer_document_type = staticmethod(infer_document_type)
    parse_record = staticmethod(parse_record)
    validate_record = staticmethod(validate_record)
    dedup_key = staticmethod(dedup_key)
    coverage_member_count = staticmethod(coverage_member_count)
    coverage_table_rows = staticmethod(coverage_table_rows)

    def __init__(self) -> None:
        self.labels = {
            "record_type": {"KNOWLEDGE": "知识事实", "TABLE_ROW": "原表行", "FIGURE": "图片知识"},
            "action": {},
            "verdict": {},
            "content_type": {"KNOWLEDGE": "正文知识", "IRRELEVANT": "不抽取"},
            "extractor": self.extractor_templates,
            "template": {v: v for v in self.extractor_templates.values()},
            "manual_skip": "人工跳过",
        }
        self.schema_spec = json.loads((Path(__file__).parent / "schema_spec.json").read_text())

    def validate_payload(self, record_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        from tcm_extract_engine.core.records import FigurePayload, TablePayload
        from tcm_extract_engine.domains.general.schema import KnowledgeFact

        models = {"KNOWLEDGE": KnowledgeFact, "TABLE_ROW": TablePayload, "FIGURE": FigurePayload}
        if record_type not in models:
            raise ValueError(f"general 不支持 record_type={record_type}")
        normalized = models[record_type].model_validate(payload).model_dump(mode="json")
        if record_type == "FIGURE":
            from tcm_extract_engine.core.visual import VisualData
            normalized["image_ref"]["extracted_data"] = VisualData.model_validate(normalized["image_ref"]["extracted_data"]).model_dump(mode="json", by_alias=True)
        return normalized

    def flatten_for_review(self, record: Any) -> str:
        return json.dumps(record.payload, ensure_ascii=False)

    def subject_of(self, record: Any) -> str:
        return record.subject_text or record.payload.get("subject", "")

    def related_subjects(self, record: Any) -> str:
        return str(record.payload.get("object", ""))

    def normalize_correction(self, corrected: dict[str, Any]) -> dict[str, Any]:
        return dict(corrected)
