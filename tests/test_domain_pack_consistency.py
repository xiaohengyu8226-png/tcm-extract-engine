"""M5 一致性测试族：领域包声明面（schema_spec.json / manifest.json）↔ 实现同步。

设计意图：schema_spec.json 是「人闸审查面 + C0 起草面」，代码枚举是行为面。
二者必须同步演进——任何单侧改动（改枚举不改 spec、改标签不改 manifest、
新增抽取器不改模板表、收编丢模块）都会在此族内失败。

覆盖维度：
  A. 包完整性扫描（包文件齐备 + 各模块入口符号完整）
  B. schema_spec.json 与代码枚举/语义表逐项对账
  C. manifest 标签表与枚举对账（web/Excel 消费方随动）
  D. GEPA 映射键域合法 + 协议成员全部绑定 + 解析入口冒烟
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tcm_extract_engine.core.validation import IssueCategory
from tcm_extract_engine.domains import DomainPack, get_domain_pack
from tcm_extract_engine.domains.tcm import classify, gepa_map
from tcm_extract_engine.domains.tcm import parse as tcm_parse
from tcm_extract_engine.domains.tcm import schema as tcm_schema
from tcm_extract_engine.domains.tcm import validate as tcm_validate
from tcm_extract_engine.domains.tcm import verdict as tcm_verdict
from tcm_extract_engine.domains.tcm.schema import TCMRecord as SourceRecord

PACK_DIR = Path(tcm_schema.__file__).parent
PROMPTS_A = Path(__file__).resolve().parents[1] / "prompts" / "a"

EXPECTED_PACK_FILES = {
    "__init__.py",
    "manifest.json",
    "schema_spec.json",
    "schema.py",
    "classify.py",
    "parse.py",
    "validate.py",
    "verdict.py",
    "gepa_map.py",
}


def _pack() -> DomainPack:
    return get_domain_pack()


def _spec() -> dict[str, Any]:
    return _pack().schema_spec


# ── A. 包完整性扫描 ─────────────────────────────────────────────────────────


def test_pack_files_complete() -> None:
    """收编后的包目录必须齐备声明面 + 行为面文件。"""
    actual = {p.name for p in PACK_DIR.iterdir() if p.is_file() and not p.name.startswith(".")}
    missing = EXPECTED_PACK_FILES - actual
    assert not missing, f"包目录缺文件: {sorted(missing)}"


def test_pack_module_entrypoints() -> None:
    """各模块必须暴露骨架依赖的入口符号（收编是否丢了接口）。"""
    expected: dict[str, list[str]] = {
        "schema": [
            "RecordType",
            "ActionLevel",
            "SafetyRule",
            "action_ceiling",
            "action_exceeds_ceiling",
        ],
        "classify": [
            "ContentType",
            "DocumentType",
            "EXTRACTOR_TEMPLATES",
            "CONTENT_TYPE_TO_EXTRACTOR",
            "classify_section",
            "infer_document_type",
            "chunk_strategy",
        ],
        "parse": ["parse_record"],
        "validate": [
            "validate_record",
            "coverage_member_count",
            "coverage_table_rows",
            "dedup_key",
        ],
        "verdict": ["Verdict"],
        "gepa_map": ["VERDICT_ISSUE_MAP", "CATEGORY_ISSUE_MAP"],
    }
    modules = {
        "schema": tcm_schema,
        "classify": classify,
        "parse": tcm_parse,
        "validate": tcm_validate,
        "verdict": tcm_verdict,
        "gepa_map": gepa_map,
    }
    for mod_name, symbols in expected.items():
        missing = [s for s in symbols if not hasattr(modules[mod_name], s)]
        assert not missing, f"模块 {mod_name} 缺入口符号: {missing}"


# ── B. schema_spec.json ↔ 代码对账 ──────────────────────────────────────────


def test_spec_top_level_keys() -> None:
    expected = {
        "schema_version",
        "record_types",
        "payload_models",
        "action_levels",
        "action_keywords",
        "restriction_levels",
        "toxicity_levels",
        "verdicts",
        "content_types",
        "document_types",
        "extractor_templates",
        "chunk_strategies",
    }
    assert set(_spec()) == expected


def test_spec_record_types_match_enum() -> None:
    assert set(_spec()["record_types"]) == {t.value for t in tcm_schema.RecordType}


def test_spec_payload_models_cover_schema() -> None:
    """spec.payload_models 必须与 schema.py 中定义的 pydantic 模型一一对应。"""
    defined = {
        name
        for name, obj in vars(tcm_schema).items()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and getattr(obj, "__module__", "").endswith(".tcm.schema")
        and name != "TCMRecord"
    }
    assert set(_spec()["payload_models"]) == defined


def test_spec_payload_fields_match_models() -> None:
    for name, entry in _spec()["payload_models"].items():
        model = getattr(tcm_schema, name)
        assert set(entry["fields"]) == set(model.model_fields), f"{name} 字段表与模型不一致"


def test_spec_record_type_payload_refs_valid() -> None:
    for rt, entry in _spec()["record_types"].items():
        payload = entry.get("payload")
        if payload is None:
            continue
        assert payload in {*_spec()["payload_models"], "TablePayload"}, (
            f"{rt} 引用的载荷模型 {payload} 未在 payload_models 声明"
        )


def test_spec_action_levels_match_enum() -> None:
    assert set(_spec()["action_levels"]) == {a.value for a in tcm_schema.ActionLevel}


def test_spec_action_keywords_match_ceiling() -> None:
    ceiling = {k: v.value for k, v in tcm_schema._ACTION_CEILING.items()}
    assert ceiling == _spec()["action_keywords"]


def test_spec_restriction_levels_match_ceiling() -> None:
    ceiling = {k: v.value for k, v in tcm_validate._RESTRICTION_CEILING.items()}
    assert ceiling == _spec()["restriction_levels"]


def test_spec_toxicity_levels_match() -> None:
    assert set(_spec()["toxicity_levels"]) == tcm_validate._TOXICITY_LEVELS


def test_spec_verdicts_match_enum() -> None:
    assert set(_spec()["verdicts"]) == {v.value for v in tcm_verdict.Verdict}


def test_spec_content_types_match_routing() -> None:
    spec_ct = _spec()["content_types"]
    ct_enum = classify.ContentType
    assert set(spec_ct) == {c.value for c in ct_enum}
    for key, entry in spec_ct.items():
        expected_extractor = classify.CONTENT_TYPE_TO_EXTRACTOR[ct_enum(key)]
        assert entry["extractor"] == expected_extractor, (
            f"{key} 路由与 CONTENT_TYPE_TO_EXTRACTOR 不一致"
        )


def test_spec_document_types_match_enum() -> None:
    assert set(_spec()["document_types"]) == {d.value for d in classify.DocumentType}


def test_spec_extractor_templates_match() -> None:
    assert _spec()["extractor_templates"] == classify.EXTRACTOR_TEMPLATES


def test_spec_chunk_strategies_match() -> None:
    spec_strategies = _spec()["chunk_strategies"]
    assert set(spec_strategies) == set(classify.EXTRACTOR_TEMPLATES)
    for extractor, declared in spec_strategies.items():
        actual = classify.chunk_strategy(extractor)
        assert actual == declared, f"{extractor} 组块策略漂移"
        assert declared in {"whole_section", "split"}


def test_extractor_template_files_exist() -> None:
    for extractor, template in classify.EXTRACTOR_TEMPLATES.items():
        path = PROMPTS_A / template
        assert path.is_file(), f"{extractor} 的 A 模板缺失: {path}"


# ── C. manifest 标签表 ↔ 枚举对账 ───────────────────────────────────────────


def test_manifest_label_tables_cover_enums() -> None:
    labels = _pack().labels
    assert set(labels["record_type"]) == {t.value for t in tcm_schema.RecordType}
    assert set(labels["action"]) == {a.value for a in tcm_schema.ActionLevel}
    assert set(labels["content_type"]) == {c.value for c in classify.ContentType}
    assert set(labels["extractor"]) == set(classify.EXTRACTOR_TEMPLATES)
    assert set(labels["template"]) == set(classify.EXTRACTOR_TEMPLATES.values())


def test_manifest_required_label_groups() -> None:
    """web/Excel 消费方依赖的标签组必须存在（manual_skip 为标量标记）。"""
    required = {
        "extractor",
        "template",
        "content_type",
        "record_type",
        "action",
        "verdict",
        "manual_skip",
    }
    assert required <= set(_pack().labels)


def test_manifest_verdict_labels_subset() -> None:
    verdict_values = {v.value for v in tcm_verdict.Verdict}
    assert set(_pack().labels["verdict"]) <= verdict_values


def test_review_verdicts_valid() -> None:
    verdict_values = {v.value for v in tcm_verdict.Verdict}
    assert _pack().review_verdicts <= verdict_values


# ── D. GEPA 映射 / 协议绑定 / 入口冒烟 ───────────────────────────────────────


def test_gepa_verdict_map_keys_valid() -> None:
    verdict_values = {v.value for v in tcm_verdict.Verdict}
    assert set(gepa_map.VERDICT_ISSUE_MAP) <= verdict_values
    for issue_type, severity, confidence in gepa_map.VERDICT_ISSUE_MAP.values():
        assert issue_type and isinstance(issue_type, str)
        assert severity and isinstance(severity, str)
        assert 0.0 <= confidence <= 1.0


def test_gepa_category_map_keys_valid() -> None:
    categories = {c.value for c in IssueCategory}
    assert set(gepa_map.CATEGORY_ISSUE_MAP) <= categories
    for issue_type, severity, confidence in gepa_map.CATEGORY_ISSUE_MAP.values():
        assert issue_type and isinstance(issue_type, str)
        assert severity and isinstance(severity, str)
        assert 0.0 <= confidence <= 1.0


def test_protocol_members_all_bound() -> None:
    pack = _pack()
    missing = [name for name in DomainPack.__annotations__ if not hasattr(pack, name)]
    assert not missing, f"TCMPack 未绑定协议成员: {missing}"


def test_pack_declarations_nonempty() -> None:
    pack = _pack()
    for attr in ("id", "label", "goal", "profile_domain"):
        value = getattr(pack, attr)
        assert isinstance(value, str) and value.strip(), f"{attr} 为空"
    assert pack.dose_outlier_grams > 0


def test_parse_record_smoke() -> None:
    record = tcm_parse.parse_record(
        {
            "record_type": "GENERAL_RULE",
            "subject_text": "生附子",
            "action": "PROHIBIT",
            "raw_text": "生附子有毒，内服须炮制。",
        },
        book_id="consistency-check",
        seq=1,
        extractor="GeneralRuleExtractor",
        page_fallback=None,
        heading_path=["示例章"],
        title="示例章",
    )
    assert isinstance(record, SourceRecord)
    assert record.record_type == tcm_schema.RecordType.GENERAL_RULE
    assert record.action == tcm_schema.ActionLevel.PROHIBIT
    assert record.record_id == "SR-consiste-000001"


def test_risk_rules_min_counts_valid() -> None:
    content_types = {c.value for c in classify.ContentType}
    for rule in _pack().risk_rules:
        assert set(rule.min_counts) <= content_types, f"风险规则引用未知内容类型: {rule.min_counts}"
        assert rule.text.strip(), "风险规则文案为空"


def test_adaptation_fields_valid() -> None:
    fields = _pack().adaptation_fields
    assert fields, "适配字段声明为空"
    keys = [f.key for f in fields]
    assert len(keys) == len(set(keys)), f"适配字段 key 重复: {keys}"
    for field in fields:
        assert field.key.strip() and field.title.strip()
