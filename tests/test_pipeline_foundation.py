"""pipeline 基础模块测试：manifest 状态机 / artifact 信封 / SourceRecord / BookWorkspace。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcm_extract_engine.core.artifacts import (
    ArtifactHeader,
    artifact_hash,
    load_artifact,
    save_artifact,
)
from tcm_extract_engine.core.book import BookWorkspace, list_books, sha256_file
from tcm_extract_engine.core.manifest import (
    STAGE_ORDER,
    VALID_TRANSITIONS,
    BookManifest,
    BookStatus,
    StageRecord,
)
from tcm_extract_engine.core.records import (
    ActionLevel,
    InteractionGroup,
    InteractionMember,
    RecordType,
    SourceRecord,
    action_exceeds_ceiling,
)

# ── manifest / 状态机 ─────────────────────────────────────────────────────────


def test_status_order_forward() -> None:
    """正常路径 PENDING → PUBLISHED 全程可迁移。"""
    path = [
        BookStatus.PENDING,
        BookStatus.PARSING,
        BookStatus.PARSED,
        BookStatus.PROFILING,
        BookStatus.PROFILED,
        BookStatus.PLAN_GENERATED,
        BookStatus.WAITING_PLAN_CONFIRMATION,
        BookStatus.PLAN_CONFIRMED,
        BookStatus.EXTRACTING,
        BookStatus.EXTRACTED,
        BookStatus.VALIDATING,
        BookStatus.VALIDATED,
        BookStatus.VERIFYING,
        BookStatus.VERIFIED,
        BookStatus.WAITING_REVIEW,
        BookStatus.REVIEWED,
        BookStatus.PUBLISHED,
    ]
    for cur, nxt in zip(path, path[1:]):
        assert nxt in VALID_TRANSITIONS[cur], f"{cur} → {nxt} 应该合法"


def test_status_invalid_jump() -> None:
    """跳级迁移必须被拒绝（PENDING 不能直接到 EXTRACTED）。"""
    assert BookStatus.EXTRACTED not in VALID_TRANSITIONS[BookStatus.PENDING]
    m = _make_manifest()
    with pytest.raises(ValueError, match="非法状态迁移"):
        m.transition(BookStatus.EXTRACTED)


def test_any_status_can_fail() -> None:
    """任意状态都可以进入 FAILED。"""
    for status in BookStatus:
        if status in (BookStatus.FAILED, BookStatus.PUBLISHED):
            continue
        assert BookStatus.FAILED in VALID_TRANSITIONS.get(status, frozenset())


def test_manifest_fail_and_resume() -> None:
    m = _make_manifest()
    m.fail("测试错误")
    assert m.status == BookStatus.FAILED
    assert m.error == "测试错误"
    # FAILED 恢复后回到 PENDING（由 resume 逻辑推进到实际阶段）
    m.transition(BookStatus.PENDING)
    assert m.status == BookStatus.PENDING


def test_manifest_persist_roundtrip(tmp_path: Path) -> None:
    m = _make_manifest()
    m.stage("parse").mark_running()
    m.stage("parse").mark_done(
        artifact="01_parse/parse_summary.json",
        artifact_hash="sha256:abc",
        records=42,
    )
    m.transition(BookStatus.PARSING)
    m.transition(BookStatus.PARSED)
    m.save(tmp_path)

    loaded = BookManifest.load(tmp_path)
    assert loaded.book_id == m.book_id
    assert loaded.status == BookStatus.PARSED
    assert loaded.stage("parse").status == "DONE"
    assert loaded.stage("parse").records == 42


def test_completed_stages_order() -> None:
    m = _make_manifest()
    for stage in STAGE_ORDER[:3]:
        m.stage(stage).mark_running()
        m.stage(stage).mark_done(artifact=f"x/{stage}", artifact_hash="sha256:h")
    assert m.completed_stages() == ["parse", "profile", "plan"]


def test_stage_record_unknown_name() -> None:
    m = _make_manifest()
    with pytest.raises(ValueError, match="未知阶段名"):
        m.stage("nonexistent")


# ── artifacts ────────────────────────────────────────────────────────────────


def test_artifact_roundtrip(tmp_path: Path) -> None:
    header = ArtifactHeader(schema_version="1", book_id="abc123def456", task_id="run-1")
    path = tmp_path / "01_parse" / "parse_summary.json"
    digest = save_artifact(path, header=header, data={"block_count": 100, "pages": 20})

    assert digest.startswith("sha256:")
    loaded = load_artifact(path)
    assert loaded["header"]["book_id"] == "abc123def456"
    assert loaded["block_count"] == 100


def test_artifact_hash_stable() -> None:
    """内容相同 → hash 相同；内容不同 → hash 不同。"""
    a = artifact_hash({"records": [1, 2], "name": "测试"})
    b = artifact_hash({"name": "测试", "records": [1, 2]})  # 键序无关
    c = artifact_hash({"records": [1, 3], "name": "测试"})
    assert a == b
    assert a != c


def test_artifact_missing_header_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_artifact(bad)


# ── SourceRecord ──────────────────────────────────────────────────────────────


def test_source_record_minimal() -> None:
    rec = SourceRecord(
        record_id="SR-abc123de-000001",
        book_id="abc123def456",
        record_type=RecordType.GENERAL_RULE,
        raw_text="川乌不宜与半夏同用",
    )
    assert rec.has_evidence()
    assert rec.needs_review is False
    assert rec.dose is None


def test_source_record_dose_validation() -> None:
    from tcm_extract_engine.core.records import HerbDose

    rec = SourceRecord(
        record_id="SR-abc123de-000002",
        book_id="abc123def456",
        record_type=RecordType.DOSAGE,
        herb_name="附子",
        dose=HerbDose(min_dose=3.0, max_dose=15.0, unit="g"),
        raw_text="附子 3-15g",
    )
    assert rec.dose is not None
    assert rec.dose.min_dose == 3.0


def test_interaction_group_members() -> None:
    rec = SourceRecord(
        record_id="SR-abc123de-000003",
        book_id="abc123def456",
        record_type=RecordType.INTERACTION,
        interaction=InteractionGroup(
            interaction_type="十八反",
            anchor=InteractionMember(herb_name="甘草", role="anchor"),
            members=[
                InteractionMember(herb_name="甘遂"),
                InteractionMember(herb_name="大戟"),
                InteractionMember(herb_name="芫花"),
                InteractionMember(herb_name="海藻"),
            ],
        ),
        raw_text="甘草反甘遂、大戟、芫花、海藻",
    )
    assert rec.interaction is not None
    assert len(rec.interaction.members) == 4


# ── 语义约束：action 不得超出原文措辞 ────────────────────────────────────────


def test_action_ceiling_shenyong_not_prohibit() -> None:
    """原文"慎用"不得 action=PROHIBIT。"""
    rec = SourceRecord(
        record_id="SR-abc123de-000004",
        book_id="abc123def456",
        record_type=RecordType.PREGNANCY,
        action=ActionLevel.PROHIBIT,
        raw_text="妊娠期慎用大黄",
    )
    assert action_exceeds_ceiling(rec) is True


def test_action_ceiling_jinyong_ok() -> None:
    """原文"禁用"允许 action=PROHIBIT。"""
    rec = SourceRecord(
        record_id="SR-abc123de-000005",
        book_id="abc123def456",
        record_type=RecordType.PREGNANCY,
        action=ActionLevel.PROHIBIT,
        raw_text="妊娠期禁用巴豆霜",
    )
    assert action_exceeds_ceiling(rec) is False


def test_action_ceiling_bude_not_caution_downgrade() -> None:
    """原文"不得"（PROHIBIT 级）配 CACTION 不算越界（只查升级，不查降级）。"""
    rec = SourceRecord(
        record_id="SR-abc123de-000006",
        book_id="abc123def456",
        record_type=RecordType.GENERAL_RULE,
        action=ActionLevel.CAUTION,
        raw_text="不得内服",
    )
    assert action_exceeds_ceiling(rec) is False


# ── BookWorkspace ────────────────────────────────────────────────────────────


def test_workspace_create_idempotent(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 test content")

    books_root = tmp_path / "workspace" / "books"
    ws1 = BookWorkspace.create(books_root, pdf)
    ws2 = BookWorkspace.create(books_root, pdf)

    assert ws1.book_id == ws2.book_id
    manifest = ws1.load_manifest()
    assert manifest.file_hash == sha256_file(pdf)
    assert manifest.status == BookStatus.PENDING

    # 阶段目录全部就位
    for stage_dir in (
        "00_source", "01_parse", "02_profile", "03_plan",
        "04_extract", "05_validate", "06_verify", "07_review", "08_publish",
    ):
        assert (ws1.book_dir / stage_dir).is_dir(), f"缺少目录 {stage_dir}"


def test_workspace_invalid_book_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="book_id"):
        BookWorkspace(tmp_path, "BAD_ID!!")


def test_list_books(tmp_path: Path) -> None:
    books_root = tmp_path / "books"
    assert list_books(books_root) == []
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"pdf-a")
    ws = BookWorkspace.create(books_root, pdf)
    # 非法目录不参与列出
    (books_root / "notamanifest").mkdir()
    found = list_books(books_root)
    assert [w.book_id for w in found] == [ws.book_id]


# ── 工具 ─────────────────────────────────────────────────────────────────────


def _make_manifest() -> BookManifest:
    return BookManifest(
        book_id="abc123def456",
        file_name="测试.pdf",
        file_hash="sha256:" + "0" * 64,
        file_size=1024,
    )


def test_stage_record_lifecycle() -> None:
    r = StageRecord(stage="parse")
    assert r.status == "PENDING"
    r.mark_running()
    assert r.status == "RUNNING"
    r.mark_failed("boom")
    assert r.status == "FAILED" and r.error == "boom"
    r.mark_running()  # 失败重试
    r.mark_done(artifact="a.json", artifact_hash="sha256:x", records=3)
    assert r.status == "DONE" and r.records == 3
