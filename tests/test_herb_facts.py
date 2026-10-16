"""herb_facts 模板族（性味归经/功效/药理）扩展的单元测试。

覆盖：解析（含 attributes 回退 / extractor 推断）、确定性校验
（SCHEMA/ENTITY 受控词表）、去重主键、内容类型路由。
声明面与实现的一致性由 tests/test_domain_pack_consistency.py 保证。
"""

from __future__ import annotations

from tcm_extract_engine.core.validation import IssueCategory, Severity
from tcm_extract_engine.domains.tcm.classify import (
    CONTENT_TYPE_TO_EXTRACTOR,
    ContentType,
)
from tcm_extract_engine.domains.tcm.parse import parse_record
from tcm_extract_engine.domains.tcm.schema import RecordType
from tcm_extract_engine.domains.tcm.schema import TCMRecord as SourceRecord
from tcm_extract_engine.domains.tcm.validate import dedup_key, validate_record


def _parse(raw: dict, *, extractor: str = "HerbFactsExtractor") -> SourceRecord:
    return parse_record(
        raw,
        book_id="book0001",
        seq=1,
        extractor=extractor,
        page_fallback=None,
        heading_path=["附子"],
        title="附子",
    )


# ── 解析 ──────────────────────────────────────────────────────────────────


def test_parse_herb_facts_payload() -> None:
    record = _parse(
        {
            "record_type": "HERB_FACTS",
            "herb_name": "附子",
            "raw_text": "附子辛、甘，大热；有毒。归心、肾、脾经。",
            "herb_facts": {
                "herb_name": "附子",
                "nature": "大热",
                "flavors": ["辛", "甘"],
                "channels": ["心经", "肾经", "脾经"],
                "efficacy": ["回阳救逆", "散寒止痛"],
                "pharmacology": ["强心作用"],
                "toxicity": "有毒",
                "clinical_notes": ["亡阳虚脱"],
                "formulas": ["四逆汤"],
            },
        }
    )
    assert record.record_type == RecordType.HERB_FACTS
    facts = record.herb_facts
    assert facts is not None
    assert facts.herb_name == "附子"
    assert facts.nature == "大热"
    assert facts.flavors == ["辛", "甘"]
    assert facts.channels == ["心经", "肾经", "脾经"]
    assert facts.efficacy == ["回阳救逆", "散寒止痛"]
    assert facts.pharmacology == ["强心作用"]
    assert facts.toxicity == "有毒"
    assert facts.clinical_notes == ["亡阳虚脱"]
    assert facts.formulas == ["四逆汤"]


def test_parse_herb_facts_attribute_fallback() -> None:
    """顶层缺失时回退 attributes（兼容旧模板把载荷字段平铺进 attributes）。"""
    record = _parse(
        {
            "record_type": "HERB_FACTS",
            "herb_name": "人参",
            "raw_text": "人参甘、微苦，微温。",
            "attributes": {"herb_name": "人参", "nature": "微温"},
        }
    )
    assert record.herb_facts is not None
    assert record.herb_facts.herb_name == "人参"
    assert record.herb_facts.nature == "微温"


def test_parse_herb_facts_str_list_split() -> None:
    """LLM 返回逗号串而非数组时按分隔符切分。"""
    record = _parse(
        {
            "record_type": "HERB_FACTS",
            "raw_text": "附子功效回阳救逆、散寒止痛。",
            "herb_facts": {
                "herb_name": "附子",
                "efficacy": "回阳救逆、散寒止痛",
                "channels": "心经、肾经",
            },
        }
    )
    facts = record.herb_facts
    assert facts is not None
    assert facts.efficacy == ["回阳救逆", "散寒止痛"]
    assert facts.channels == ["心经", "肾经"]


def test_parse_herb_facts_variants() -> None:
    """variants 复合字段：变体只记录与基原不同的部分。"""
    record = _parse(
        {
            "record_type": "HERB_FACTS",
            "herb_name": "附子",
            "raw_text": "生附子大毒；制附子毒性减弱，辛、热。",
            "herb_facts": {
                "herb_name": "附子",
                "nature": "大热",
                "efficacy": ["回阳救逆"],
                "variants": [
                    {"variant_name": "生附子", "toxicity": "大毒"},
                    {
                        "variant_name": "制附子",
                        "nature_flavor": "辛、热",
                        "toxicity": "有毒",
                        "notes": "炮制后毒性减弱",
                    },
                ],
            },
        }
    )
    facts = record.herb_facts
    assert facts is not None
    assert len(facts.variants) == 2
    sheng, zhi = facts.variants
    assert sheng.variant_name == "生附子"
    assert sheng.toxicity == "大毒"
    assert sheng.nature_flavor is None  # 与基原相同不重复记录
    assert zhi.variant_name == "制附子"
    assert zhi.nature_flavor == "辛、热"
    assert zhi.toxicity == "有毒"
    assert zhi.notes == "炮制后毒性减弱"


def test_parse_herb_facts_variants_drop_invalid() -> None:
    """无 variant_name 或非 dict 的变体被静默丢弃。"""
    record = _parse(
        {
            "record_type": "HERB_FACTS",
            "raw_text": "原文",
            "herb_facts": {
                "herb_name": "附子",
                "variants": [
                    {"variant_name": "生附子", "toxicity": "大毒"},
                    {"toxicity": "小毒"},
                    "not-a-dict",
                ],
            },
        }
    )
    facts = record.herb_facts
    assert facts is not None
    assert [v.variant_name for v in facts.variants] == ["生附子"]


def test_infer_type_from_extractor() -> None:
    """缺 record_type 时按 HerbFactsExtractor 推断 HERB_FACTS。"""
    record = _parse(
        {
            "herb_facts": {"herb_name": "甘草"},
            "raw_text": "甘草性平，味甘。",
        }
    )
    assert record.record_type == RecordType.HERB_FACTS


# ── 校验 ──────────────────────────────────────────────────────────────────


def _record_with_facts(facts: dict | None, *, raw_text: str = "原文") -> SourceRecord:
    from tcm_extract_engine.domains.tcm.schema import HerbFacts

    payload = HerbFacts(**facts) if facts is not None else None
    return SourceRecord(
        record_id="SR-book0001-000001",
        book_id="book0001",
        record_type=RecordType.HERB_FACTS,
        raw_text=raw_text,
        herb_name=(facts or {}).get("herb_name"),
        herb_facts=payload,
    )


def test_validate_herb_facts_missing_payload() -> None:
    record = _record_with_facts(None)
    issues = validate_record(record)
    schema = [i for i in issues if i.category == IssueCategory.SCHEMA]
    assert any("缺少 herb_facts 载荷" in i.message for i in schema)
    assert all(i.severity == Severity.ERROR for i in schema)


def test_validate_herb_facts_ok() -> None:
    record = _record_with_facts(
        {
            "herb_name": "附子",
            "nature": "大热",
            "flavors": ["辛", "甘"],
            "channels": ["心经", "肾经"],
            "efficacy": ["回阳救逆"],
            "toxicity": "有毒",
        }
    )
    assert validate_record(record) == []


def test_validate_herb_facts_vocab_warnings() -> None:
    """受控词表外的四气/五味/归经/毒性只给 WARNING，不阻断。"""
    record = _record_with_facts(
        {
            "herb_name": "附子",
            "nature": "大燥",
            "flavors": ["辛", "辣"],
            "channels": ["心经", "肺"],
            "efficacy": ["回阳救逆"],
            "toxicity": "极毒",
        }
    )
    issues = validate_record(record)
    entity = [i for i in issues if i.category == IssueCategory.ENTITY]
    assert all(i.severity == Severity.WARNING for i in entity)
    joined = "；".join(i.message for i in entity)
    assert "大燥" in joined
    assert "辣" in joined
    assert "肺" in joined
    assert "极毒" in joined


def test_validate_herb_facts_empty_extraction() -> None:
    """只有药名而无任何事实内容 → 疑似空抽取 WARNING。"""
    record = _record_with_facts({"herb_name": "附子"})
    issues = validate_record(record)
    entity = [i for i in issues if i.category == IssueCategory.ENTITY]
    assert any("疑似空抽取" in i.message for i in entity)


def test_validate_herb_facts_variant_warnings() -> None:
    """变体名重复 / 毒性非标准 / 空变体 → WARNING。"""
    from tcm_extract_engine.domains.tcm.schema import HerbFacts, HerbFactVariant

    record = SourceRecord(
        record_id="SR-book0001-000001",
        book_id="book0001",
        record_type=RecordType.HERB_FACTS,
        raw_text="原文",
        herb_name="附子",
        herb_facts=HerbFacts(
            herb_name="附子",
            efficacy=["回阳救逆"],
            variants=[
                HerbFactVariant(variant_name="生附子", toxicity="极毒"),
                HerbFactVariant(variant_name="生附子", toxicity="大毒"),
                HerbFactVariant(variant_name="淡附片"),
            ],
        ),
    )
    issues = validate_record(record)
    entity = [i for i in issues if i.category == IssueCategory.ENTITY]
    joined = "；".join(i.message for i in entity)
    assert "重复" in joined
    assert "极毒" in joined
    assert "疑似空变体" in joined


# ── 去重 / 路由 ────────────────────────────────────────────────────────────


def test_dedup_key_herb_facts() -> None:
    record = _record_with_facts({"herb_name": "附子"})
    assert dedup_key(record) == ("附子",)


def test_routing_herb_monograph_and_case_record() -> None:
    assert CONTENT_TYPE_TO_EXTRACTOR[ContentType.HERB_MONOGRAPH] == "HerbFactsExtractor"
    assert CONTENT_TYPE_TO_EXTRACTOR[ContentType.CASE_RECORD] == "HerbFactsExtractor"
