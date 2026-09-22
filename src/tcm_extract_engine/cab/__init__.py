"""C/A/B 三职责：Profiler+Planner(C) / Extractors(A) / Validator+Verifier(B)。"""

from tcm_extract_engine.cab.extractors import extract_records
from tcm_extract_engine.cab.planner import (
    PromptTemplate,
    PromptTemplateRegistry,
    SectionPlan,
    build_extraction_plan,
)
from tcm_extract_engine.cab.profiler import DocumentProfile, build_profile, load_blocks
from tcm_extract_engine.cab.validator import ValidationReport, validate_records
from tcm_extract_engine.cab.verifier import VerificationReport, verify_records

__all__ = [
    "DocumentProfile",
    "PromptTemplate",
    "PromptTemplateRegistry",
    "SectionPlan",
    "ValidationReport",
    "VerificationReport",
    "build_extraction_plan",
    "build_profile",
    "extract_records",
    "load_blocks",
    "validate_records",
    "verify_records",
]
