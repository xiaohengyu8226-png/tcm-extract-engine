"""确定性校验层测试：Schema/Numeric/Source/Entity/Semantic/Coverage 六类规则。"""

from __future__ import annotations

from typing import Any

from tcm_extract_engine.cab.validator import (
    IssueCategory,
    Severity,
    ValidationReport,
    validate_records,
)
from tcm_extract_engine.domains.tcm.schema import (
    ActionLevel,
    HerbDose,
    InteractionGroup,
    InteractionMember,
    PregnancyRule,
    RecordType,
)
from tcm_extract_engine.domains.tcm.schema import TCMRecord as SourceRecord


def make_record(**overrides: Any) -> SourceRecord:
    """构造合法基线记录（无任何 issue），按需覆盖字段。"""
    defaults: dict[str, Any] = {
        "record_id": "SR-test0001-000001",
        "book_id": "test0001",
        "record_type": RecordType.GENERAL_RULE,
        "raw_text": "孕妇慎用红花。",
        "subject_text": "红花",
        "action": ActionLevel.CAUTION,
        "source": {"page_start": 12, "page_end": 12, "section_path": ["第四章"]},
    }
    defaults.update(overrides)
    return SourceRecord.model_validate(defaults)


def issue_tuples(report: ValidationReport) -> list[tuple[str, str]]:
    return [(i.category.value, i.severity.value) for i in report.issues]


# ── 基线：合法记录 ───────────────────────────────────────────────────────────


def test_valid_record_passes() -> None:
    report = validate_records([make_record()])
    assert report.passed
    assert report.issues == []
    assert report.total_records == 1
    assert "PASS" in report.summary()


# ── SCHEMA ──────────────────────────────────────────────────────────────────


def test_schema_interaction_missing_payload() -> None:
    record = make_record(record_type=RecordType.INTERACTION, interaction=None)
    report = validate_records([record])
    assert not report.passed
    assert (IssueCategory.SCHEMA.value, Severity.ERROR.value) in issue_tuples(report)


def test_schema_herb_table_missing_name() -> None:
    record = make_record(record_type=RecordType.HERB_TABLE, herb_name=None)
    report = validate_records([record])
    assert not report.passed
    assert any(
        i.category == IssueCategory.SCHEMA and "herb_name" in i.message for i in report.issues
    )


def test_schema_dosage_missing_dose() -> None:
    record = make_record(record_type=RecordType.DOSAGE, dose=None)
    report = validate_records([record])
    assert not report.passed


# ── NUMERIC ─────────────────────────────────────────────────────────────────


def test_numeric_min_greater_than_max() -> None:
    record = make_record(dose=HerbDose(min_dose=30, max_dose=3, unit="g"))
    report = validate_records([record])
    assert not report.passed
    assert any(
        i.category == IssueCategory.NUMERIC and "min_dose" in i.message for i in report.issues
    )


def test_numeric_negative_dose() -> None:
    record = make_record(dose=HerbDose(max_dose=-5, unit="g"))
    report = validate_records([record])
    assert not report.passed


def test_numeric_outlier_warning_only() -> None:
    record = make_record(dose=HerbDose(max_dose=2000, unit="g"))
    report = validate_records([record])
    assert report.passed  # WARNING 不阻断
    assert report.warning_count == 1


def test_numeric_min_without_max_warning() -> None:
    record = make_record(dose=HerbDose(min_dose=3, unit="g"))
    report = validate_records([record])
    assert report.passed
    assert report.warning_count == 1


# ── SOURCE ─────────────────────────────────────────────────────────────────


def test_source_empty_raw_text() -> None:
    record = make_record(raw_text="")
    report = validate_records([record])
    assert not report.passed
    assert any(
        i.category == IssueCategory.SOURCE and i.severity == Severity.ERROR for i in report.issues
    )


def test_source_no_location_warning() -> None:
    record = make_record(source={"section_path": ["第四章"]})
    report = validate_records([record])
    assert report.passed
    assert report.warning_count == 1


# ── ENTITY ─────────────────────────────────────────────────────────────────


def _interaction_record(members: list[InteractionMember]) -> SourceRecord:
    return make_record(
        record_type=RecordType.INTERACTION,
        raw_text="甘草反甘遂、大戟、芫花、海藻。",
        interaction=InteractionGroup(
            interaction_type="十八反",
            anchor=InteractionMember(herb_name="甘草", role="anchor"),
            members=members,
        ),
        action=ActionLevel.PROHIBIT,
    )


def test_entity_interaction_missing_members() -> None:
    record = _interaction_record([])
    report = validate_records([record])
    assert not report.passed
    assert any(i.category == IssueCategory.ENTITY and "成员" in i.message for i in report.issues)


def test_entity_interaction_duplicate_member_warning() -> None:
    members = [
        InteractionMember(herb_name="甘遂"),
        InteractionMember(herb_name="大戟"),
        InteractionMember(herb_name="甘遂"),  # 重复
    ]
    report = validate_records([_interaction_record(members)])
    assert report.passed
    assert report.warning_count == 1


def test_entity_pregnancy_missing_targets() -> None:
    record = make_record(
        record_type=RecordType.PREGNANCY,
        raw_text="孕妇禁用巴豆。",
        pregnancy=PregnancyRule(restriction_level="禁用", target_herbs=[]),
        action=ActionLevel.PROHIBIT,
    )
    report = validate_records([record])
    assert not report.passed


# ── SEMANTIC ───────────────────────────────────────────────────────────────


def test_semantic_action_upgrade_rejected() -> None:
    """原文「慎用」不得升级为 PROHIBIT。"""
    record = make_record(raw_text="孕妇慎用红花。", action=ActionLevel.PROHIBIT)
    report = validate_records([record])
    assert not report.passed
    assert any(i.category == IssueCategory.SEMANTIC and "上限" in i.message for i in report.issues)


def test_semantic_action_at_ceiling_allowed() -> None:
    record = make_record(raw_text="孕妇慎用红花。", action=ActionLevel.CAUTION)
    report = validate_records([record])
    assert report.passed


def test_semantic_restriction_ceiling() -> None:
    """妊娠 restriction_level=慎用 → action 不得为 PROHIBIT。"""
    record = make_record(
        record_type=RecordType.PREGNANCY,
        raw_text="妊娠期慎用大黄。",
        pregnancy=PregnancyRule(restriction_level="慎用", target_herbs=["大黄"]),
        action=ActionLevel.PROHIBIT,
    )
    report = validate_records([record])
    assert not report.passed
    assert any(
        i.category == IssueCategory.SEMANTIC and "restriction_level" in i.message
        for i in report.issues
    )


def test_semantic_nonstandard_toxicity_warning() -> None:
    record = make_record(toxicity_level="剧毒无比")
    report = validate_records([record])
    assert report.passed
    assert report.warning_count == 1


# ── COVERAGE ───────────────────────────────────────────────────────────────


def test_coverage_member_count_mismatch() -> None:
    record = make_record(
        record_type=RecordType.PREGNANCY,
        raw_text="妊娠禁用：巴豆、牵牛子、商陆。",
        pregnancy=PregnancyRule(
            restriction_level="禁用",
            target_herbs=["巴豆", "牵牛子"],  # 缺商陆
        ),
        attributes={"member_count": 3},
        action=ActionLevel.PROHIBIT,
    )
    report = validate_records([record])
    assert not report.passed
    assert any(
        i.category == IssueCategory.COVERAGE and "member_count" in i.message for i in report.issues
    )


def test_coverage_member_count_match_passes() -> None:
    record = make_record(
        record_type=RecordType.PREGNANCY,
        raw_text="妊娠禁用：巴豆、牵牛子。",
        pregnancy=PregnancyRule(restriction_level="禁用", target_herbs=["巴豆", "牵牛子"]),
        attributes={"member_count": 2},
        action=ActionLevel.PROHIBIT,
    )
    report = validate_records([record])
    assert report.passed


def test_coverage_duplicate_records() -> None:
    r1 = make_record(
        record_id="SR-test0001-000001",
        record_type=RecordType.HERB_TABLE,
        herb_name="制川乌",
        raw_text="制川乌 1.5～3g",
    )
    r2 = make_record(
        record_id="SR-test0001-000002",
        record_type=RecordType.HERB_TABLE,
        herb_name="制川乌",
        raw_text="制川乌 1.5～3g（重复行）",
    )
    report = validate_records([r1, r2])
    assert report.passed  # WARNING 不阻断
    assert report.warning_count == 1
    assert report.issues[0].record_id == "SR-test0001-000002"


def test_coverage_table_rows() -> None:
    """表格 3 数据行只抽出 1 条记录 → WARNING。"""
    blocks = [
        {
            "type": "content",
            "blockid": "tb1",
            "format": "drawing",
            "content": "<drawing id='tb-1'><table>"
            "<tr><td>药名</td></tr><tr><td>川乌</td></tr>"
            "<tr><td>草乌</td></tr><tr><td>附子</td></tr>"
            "</table></drawing>",
            "heading": "附录A 有毒中药",
            "parent_headings": [],
            "level": 2,
        },
    ]
    record = make_record(
        record_type=RecordType.HERB_TABLE,
        herb_name="川乌",
        raw_text="| 川乌 | 有毒 |",
        source={"page_start": 1, "section_path": ["附录A 有毒中药"]},
    )
    report = validate_records([record], blocks=blocks)
    assert report.passed  # WARNING 不阻断
    assert any(
        i.category == IssueCategory.COVERAGE and "表格覆盖率" in i.message for i in report.issues
    )


# ── Report 汇总 ────────────────────────────────────────────────────────────


def test_report_counts_and_errors_for() -> None:
    good = make_record()
    bad = make_record(record_id="SR-test0001-000002", raw_text="")
    report = validate_records([good, bad])
    assert report.total_records == 2
    assert report.error_count == 1
    assert report.errors_for(bad.record_id)
    assert report.errors_for(good.record_id) == []
    assert not report.passed
    assert "FAIL" in report.summary()
