"""Web 端点回归测试：蓝图人工编辑（PUT /schema）+ 纠错覆盖（apply-corrections）
+ 工作流回退/重跑语义（back 只回退状态，retry 才作废记录）。

不起 HTTP 服务，直接调用 app 模块里的端点协程（monkeypatch BOOKS_ROOT / _pipeline），
用真实产物读写路径验证行为。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from tcm_extract_engine.cab.planner import PromptTemplateRegistry
from tcm_extract_engine.cab.profiler import CONTENT_TYPE_TO_EXTRACTOR
from tcm_extract_engine.cab.verifier import CorrectionEntry, Verdict
from tcm_extract_engine.core.artifacts import (
    ArtifactHeader,
    load_artifact_records,
    save_artifact,
)
from tcm_extract_engine.core.book import BookWorkspace
from tcm_extract_engine.core.manifest import BookManifest, BookStatus, StageRecord
from tcm_extract_engine.core.records import RecordType, SourceRecord
from tcm_extract_engine.web import app as web

BOOK_ID = "testbook0001"

VALID_EXTRACTOR = sorted({v for v in CONTENT_TYPE_TO_EXTRACTOR.values() if v})[0]
TEMPLATE = "general_rule.md"


def _header() -> ArtifactHeader:
    return ArtifactHeader(schema_version="1", book_id=BOOK_ID, task_id="t-test")


@pytest.fixture()
def books_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "books"
    root.mkdir()
    monkeypatch.setattr(web, "BOOKS_ROOT", root)
    return root


def _make_manifest(books_root: Path, status: BookStatus) -> BookWorkspace:
    ws = BookWorkspace(books_root, BOOK_ID)
    ws.book_dir.mkdir(parents=True, exist_ok=True)
    m = BookManifest(
        book_id=BOOK_ID,
        file_name="t.md",
        file_hash="sha256:" + "0" * 64,
        status=status,
    )
    m.save(ws.book_dir)
    return ws


def _save_plan(ws: BookWorkspace) -> None:
    save_artifact(
        ws.stage_artifact("plan"),
        header=_header(),
        data={
            "document_type": "审核规范",
            "goal": "测试",
            "risk_points": [],
            "sections": [
                {
                    "section_id": "sec-0001",
                    "title": "第一章 剂量规范",
                    "extractor": VALID_EXTRACTOR,
                    "template": TEMPLATE,
                    "prompt_version": "v0",
                    "skip_reason": None,
                },
                {
                    "section_id": "sec-0002",
                    "title": "前言",
                    "extractor": "",
                    "template": "",
                    "prompt_version": "",
                    "skip_reason": "非正文",
                },
            ],
        },
    )


def _section(raw: dict[str, Any], sid: str) -> dict[str, Any]:
    return next(s for s in raw["sections"] if s["section_id"] == sid)


# ── PUT /schema：蓝图人工编辑 ────────────────────────────────────────────────


def test_update_schema_edits_routing_and_stamps_header(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)

    result = asyncio.run(
        web.update_schema(
            BOOK_ID,
            {
                "sections": [
                    {"section_id": "sec-0001", "extractor": "", "template": "", "skip": True},
                    {
                        "section_id": "sec-0002",
                        "extractor": VALID_EXTRACTOR,
                        "template": TEMPLATE,
                        "skip": False,
                    },
                ]
            },
        )
    )

    raw = json.loads(ws.stage_artifact("plan").read_text(encoding="utf-8"))
    s1, s2 = _section(raw, "sec-0001"), _section(raw, "sec-0002")
    assert s1["skip_reason"] == "人工标记不抽取"
    assert (s1["extractor"], s1["template"], s1["prompt_version"]) == ("", "", "")
    assert s2["skip_reason"] is None
    assert s2["extractor"] == VALID_EXTRACTOR
    assert s2["prompt_version"] == PromptTemplateRegistry().version_of("a", TEMPLATE)
    assert raw["header"]["extra"].get("plan_edited_at")
    # 返回值 = 最新 schema 视图，含可编辑元数据
    assert {e["section_id"] for e in result["editable_sections"]} == {"sec-0001", "sec-0002"}
    assert result["available_templates"] and VALID_EXTRACTOR in result["available_extractors"]
    skip_view = next(e for e in result["editable_sections"] if e["section_id"] == "sec-0001")
    assert skip_view["skip"] is True


def test_update_schema_rejects_bad_input(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)

    with pytest.raises(HTTPException) as ei:
        asyncio.run(
            web.update_schema(
                BOOK_ID,
                {"sections": [{"section_id": "sec-0001", "extractor": "NoSuchExtractor",
                              "template": TEMPLATE, "skip": False}]},
            )
        )
    assert ei.value.status_code == 400

    with pytest.raises(HTTPException) as eb:
        asyncio.run(
            web.update_schema(
                BOOK_ID,
                {"sections": [{"section_id": "sec-0001", "extractor": "",
                              "template": "", "skip": False}]},
            )
        )
    assert eb.value.status_code == 400  # 取消跳过必须给全抽取器 + 模板

    with pytest.raises(HTTPException) as et:
        asyncio.run(
            web.update_schema(
                BOOK_ID,
                {"sections": [{"section_id": "sec-0001", "extractor": VALID_EXTRACTOR,
                              "template": "not_exist.md", "skip": False}]},
            )
        )
    assert et.value.status_code == 400


def test_update_schema_rewinds_extract_when_confirmed(
    books_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _make_manifest(books_root, BookStatus.PLAN_CONFIRMED)
    _save_plan(ws)
    m = ws.load_manifest()
    m.stages["extract"] = StageRecord(stage="extract", status="DONE")
    m.save(ws.book_dir)

    calls: list[tuple[str, str]] = []

    class _FakePipe:
        def rewind(self, book_id: str, *, to_stage: str) -> None:
            calls.append((book_id, to_stage))

    monkeypatch.setattr(web, "_pipeline", _FakePipe())
    asyncio.run(
        web.update_schema(
            BOOK_ID,
            {"sections": [{"section_id": "sec-0001", "extractor": "", "template": "",
                           "skip": True}]},
        )
    )
    assert calls == [(BOOK_ID, "extract")]


def test_update_schema_no_rewind_before_confirmation(
    books_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)
    calls: list[str] = []

    class _FakePipe:
        def rewind(self, book_id: str, *, to_stage: str) -> None:
            calls.append(to_stage)

    monkeypatch.setattr(web, "_pipeline", _FakePipe())
    asyncio.run(
        web.update_schema(
            BOOK_ID,
            {"sections": [{"section_id": "sec-0001", "extractor": "", "template": "",
                           "skip": True}]},
        )
    )
    assert calls == []


# ── apply-corrections + verification：纠错覆盖 ──────────────────────────────


def _record(herb: str = "砒石") -> SourceRecord:
    return SourceRecord(
        record_id=f"SR-{BOOK_ID[:8]}-000001",
        book_id=BOOK_ID,
        record_type=RecordType.GENERAL_RULE,
        raw_text="砒石极毒，内服宜慎。",
        herb_name=herb,
        subject_text="砒石",
    )


def _save_verify(ws: BookWorkspace, record: SourceRecord, *entries: CorrectionEntry) -> None:
    save_artifact(
        ws.stage_artifact("verify"),
        header=_header(),
        data={"record_count": 1, "needs_review_count": 0},
        records=[record],
    )
    save_artifact(
        ws.stage_dir("verify") / "correction_log.jsonl",
        header=_header(),
        data={"correction_count": len(entries)},
        records=list(entries),
    )


def test_apply_corrections_overwrites_record_and_marks_applied(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_REVIEW)
    rec = _record()
    _save_verify(
        ws,
        rec,
        CorrectionEntry(
            record_id=rec.record_id,
            field="herb_name",
            original="砒石",
            suggested="砒霜（制）",
            reason="原文为制砒霜",
            verdict=Verdict.PARTIAL,
        ),
    )

    out = asyncio.run(web.apply_corrections(BOOK_ID))
    assert out == {"applied": 1, "failed": 0, "total": 1, "note": out["note"]}

    records = load_artifact_records(ws.stage_artifact("verify"), SourceRecord)
    assert records[0].herb_name == "砒霜（制）"
    assert records[0].raw_text == rec.raw_text  # 其余字段不动
    entries = load_artifact_records(
        ws.stage_dir("verify") / "correction_log.jsonl", CorrectionEntry
    )
    assert entries[0].applied is True
    assert entries[0].original == "砒石"  # 原值留档

    # 再次应用：幂等，无重复覆盖
    out2 = asyncio.run(web.apply_corrections(BOOK_ID))
    assert out2["applied"] == 0


def test_apply_corrections_invalid_suggestion_goes_manual(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_REVIEW)
    rec = _record()
    _save_verify(
        ws,
        rec,
        CorrectionEntry(
            record_id=rec.record_id,
            field="record_type",
            original="GENERAL_RULE",
            suggested="NOT_A_TYPE",
            verdict=Verdict.PARTIAL,
        ),
    )
    out = asyncio.run(web.apply_corrections(BOOK_ID))
    assert (out["applied"], out["failed"]) == (0, 1)
    records = load_artifact_records(ws.stage_artifact("verify"), SourceRecord)
    assert records[0].record_type is RecordType.GENERAL_RULE  # 未被脏写
    entries = load_artifact_records(
        ws.stage_dir("verify") / "correction_log.jsonl", CorrectionEntry
    )
    assert entries[0].applied is False


def test_get_verification_stats_and_chunks(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_REVIEW)
    rec = _record()
    _save_verify(
        ws,
        rec,
        CorrectionEntry(
            record_id=rec.record_id, field="herb_name",
            original="砒石", suggested="砒霜（制）", verdict=Verdict.PARTIAL,
        ),
    )
    save_artifact(
        ws.stage_artifact("validate"),
        header=_header(),
        data={"total_records": 1, "issues": []},
    )

    d = asyncio.run(web.get_verification(BOOK_ID))
    assert d["total"] == 1
    assert d["stats"] == {
        "total": 1,
        "pass_rate": 0.0,  # 有纠错建议的记录不计入通过
        "needs_review": 0,
        "correction_count": 1,
        "applied_count": 0,
    }
    assert d["records"][0]["record_id"] == rec.record_id
    assert d["chunks"] == []  # 无 blocks.jsonl 时为空而非报错

    asyncio.run(web.apply_corrections(BOOK_ID))
    d2 = asyncio.run(web.get_verification(BOOK_ID))
    assert d2["stats"]["applied_count"] == 1


# ── workflow/back vs retry：回退不作废记录，重跑才作废 ────────────────────


def _full_manifest(books_root: Path, status: BookStatus) -> BookWorkspace:
    ws = _make_manifest(books_root, status)
    m = ws.load_manifest()
    for stage in ("parse", "profile", "plan", "extract", "validate", "verify"):
        m.stages[stage] = StageRecord(stage=stage, status="DONE")
    m.save(ws.book_dir)
    return ws


def test_back_only_rewinds_status_and_keeps_stage_records(books_root: Path) -> None:
    ws = _full_manifest(books_root, BookStatus.WAITING_REVIEW)

    out = asyncio.run(web.workflow_op(BOOK_ID, "back"))
    assert out == {"status": "back", "stage": "extracted"}

    m = ws.load_manifest()
    assert m.status is BookStatus.EXTRACTED
    # 阶段记录全部保留（含被回退的 validate/verify）
    assert m.completed_stages() == ["parse", "profile", "plan", "extract", "validate", "verify"]
    assert web._view_stage(m) == "extracted"


def test_back_from_done_and_prompt_ready(books_root: Path) -> None:
    ws = _full_manifest(books_root, BookStatus.PUBLISHED)
    assert asyncio.run(web.workflow_op(BOOK_ID, "back"))["stage"] == "verified"
    assert ws.load_manifest().status is BookStatus.WAITING_REVIEW

    ws2 = _make_manifest(books_root, BookStatus.PLAN_CONFIRMED)
    m = ws2.load_manifest()
    for stage in ("parse", "profile", "plan"):
        m.stages[stage] = StageRecord(stage=stage, status="DONE")
    m.save(ws2.book_dir)
    assert asyncio.run(web.workflow_op(BOOK_ID, "back"))["stage"] == "prompt_ready"
    m2 = ws2.load_manifest()
    assert m2.status is BookStatus.WAITING_PLAN_CONFIRMATION
    assert m2.completed_stages() == ["parse", "profile", "plan"]  # plan 记录不作废


def test_view_stage_failed_falls_back_to_completed(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.FAILED)
    m = ws.load_manifest()
    m.stages["extract"] = StageRecord(stage="extract", status="DONE")
    m.save(ws.book_dir)
    assert web._view_stage(ws.load_manifest()) == "extracted"


def test_retry_rewinds_stage_records(books_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _full_manifest(books_root, BookStatus.VERIFIED)

    calls: list[tuple[str, str]] = []

    class _FakePipe:
        def rewind(self, book_id: str, *, to_stage: str) -> Any:
            calls.append((book_id, to_stage))
            return None

    monkeypatch.setattr(web, "_pipeline", _FakePipe())
    spawned: list[str] = []

    def _fake_spawn(book_id: str, coro: Any) -> None:
        coro.close()  # 不真跑后台重跑，只登记
        spawned.append(book_id)

    monkeypatch.setattr(web, "_spawn", _fake_spawn)

    out = asyncio.run(web.workflow_op(BOOK_ID, "retry-verify"))
    assert out["status"] == "started"
    assert spawned == [BOOK_ID]
    assert calls == [(BOOK_ID, "validate")]  # 重新校验才作废 validate 及下游

    # 真 rewind 后阶段记录确实被作废（产物文件不动）
    from tcm_extract_engine.runner import BookPipeline

    BookPipeline(books_root).rewind(BOOK_ID, to_stage="validate")
    m = ws.load_manifest()
    assert m.status is BookStatus.EXTRACTED
    assert m.completed_stages() == ["parse", "profile", "plan", "extract"]


# ── workflow/goto-<key>：点更早步骤节点仅回退查看，不作废任何记录 ───────────


def test_goto_only_rewinds_status_and_keeps_all_records(books_root: Path) -> None:
    ws = _full_manifest(books_root, BookStatus.WAITING_REVIEW)  # view "verified"

    out = asyncio.run(web.workflow_op(BOOK_ID, "goto-extract"))
    assert out == {"status": "goto", "stage": "extracted"}

    m = ws.load_manifest()
    assert m.status is BookStatus.EXTRACTED
    # 下游 validate/verify 阶段记录全部保留（与 retry 的关键区别：点节点不清空下游）
    assert m.completed_stages() == ["parse", "profile", "plan", "extract", "validate", "verify"]


def test_goto_multiple_earlier_steps_is_lossless(books_root: Path) -> None:
    ws = _full_manifest(books_root, BookStatus.PUBLISHED)  # view "done"

    assert asyncio.run(web.workflow_op(BOOK_ID, "goto-prompt"))["stage"] == "prompt_ready"
    assert ws.load_manifest().status is BookStatus.WAITING_PLAN_CONFIRMATION

    assert asyncio.run(web.workflow_op(BOOK_ID, "goto-parse"))["stage"] == "parsed"
    m = ws.load_manifest()
    assert m.status is BookStatus.PARSED
    assert m.completed_stages() == ["parse", "profile", "plan", "extract", "validate", "verify"]
