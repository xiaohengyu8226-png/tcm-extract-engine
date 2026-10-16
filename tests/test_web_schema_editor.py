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

from tcm_extract_engine.cab.planner import (
    BookAdaptedRegistry,
    PromptTemplateRegistry,
    adaptation_version,
)
from tcm_extract_engine.cab.verifier import CorrectionEntry, Verdict
from tcm_extract_engine.core.artifacts import (
    ArtifactHeader,
    load_artifact,
    load_artifact_records,
    save_artifact,
)
from tcm_extract_engine.core.book import ADAPTATION_FILENAME, BookWorkspace
from tcm_extract_engine.core.manifest import BookManifest, BookStatus, StageRecord
from tcm_extract_engine.domains.tcm.schema import RecordType
from tcm_extract_engine.domains.tcm.schema import TCMRecord as SourceRecord
from tcm_extract_engine.web import app as web

BOOK_ID = "testbook0001"

VALID_EXTRACTOR = "GeneralRuleExtractor"  # 显式命名，避免新增抽取器后索引漂移
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


# ── PUT /schema：蓝图人工调整（三态契约 + 冻结门）──────────────────────────────

TABLE_EXTRACTOR = "HerbTableExtractor"  # 显式命名，避免新增抽取器后索引漂移


def _save_profile(ws: BookWorkspace, cts: dict[str, str]) -> None:
    save_artifact(
        ws.stage_artifact("profile"),
        header=_header(),
        data={
            "document_type": "HANDBOOK",
            "total_blocks": 10,
            "sections": [
                {"section_id": sid, "title": sid, "content_type": ct} for sid, ct in cts.items()
            ],
        },
    )


def test_update_schema_mode_extract_sets_routing_and_versions(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)
    _save_profile(ws, {"sec-0001": "NARRATIVE_RULE"})

    result = asyncio.run(
        web.update_schema(
            BOOK_ID,
            {
                "sections": [
                    {"section_id": "sec-0001", "mode": "extract", "extractor": TABLE_EXTRACTOR}
                ]
            },
        )
    )

    raw = json.loads(ws.stage_artifact("plan").read_text(encoding="utf-8"))
    s1 = _section(raw, "sec-0001")
    assert s1["skip_reason"] is None
    assert s1["extractor"] == TABLE_EXTRACTOR
    assert s1["template"] == "herb_table.md"  # 模板由抽取器唯一确定，不再可自由搭配
    assert s1["prompt_version"] == PromptTemplateRegistry().version_of("a", "herb_table.md")
    assert raw["header"]["extra"].get("plan_edited_at")
    # manifest 合成版本同步重算
    assert ws.load_manifest().prompt_version
    view = next(e for e in result["editable_sections"] if e["section_id"] == "sec-0001")
    assert view["mode"] == "extract"
    assert view["extractor_label"] == "中药表格抽取"


def test_update_schema_rejects_bad_input(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)

    with pytest.raises(HTTPException) as ei:
        asyncio.run(
            web.update_schema(
                BOOK_ID,
                {
                    "sections": [
                        {
                            "section_id": "sec-0001",
                            "mode": "extract",
                            "extractor": "NoSuchExtractor",
                        }
                    ]
                },
            )
        )
    assert ei.value.status_code == 400

    with pytest.raises(HTTPException) as em:
        asyncio.run(
            web.update_schema(
                BOOK_ID,
                {"sections": [{"section_id": "sec-0001", "mode": "nonsense"}]},
            )
        )
    assert em.value.status_code == 400


def test_update_schema_mode_auto_restores_derived_routing(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)  # sec-0002 原为跳过
    _save_profile(ws, {"sec-0001": "REFERENCE_ONLY", "sec-0002": "NARRATIVE_RULE"})

    # 可抽取类型：恢复默认路由（无需手选抽取器/模板）
    asyncio.run(
        web.update_schema(
            BOOK_ID,
            {
                "sections": [
                    {"section_id": "sec-0002", "mode": "extract", "extractor": TABLE_EXTRACTOR}
                ]
            },
        )
    )
    asyncio.run(
        web.update_schema(
            BOOK_ID,
            {"sections": [{"section_id": "sec-0002", "mode": "auto"}]},
        )
    )
    raw = json.loads(ws.stage_artifact("plan").read_text(encoding="utf-8"))
    s2 = _section(raw, "sec-0002")
    assert s2["extractor"] == "GeneralRuleExtractor"  # NARRATIVE_RULE 派生
    assert s2["template"] == "general_rule.md"
    assert s2["skip_reason"] is None

    # 不可抽取类型：恢复规划器自动跳过
    asyncio.run(
        web.update_schema(
            BOOK_ID,
            {"sections": [{"section_id": "sec-0001", "mode": "auto"}]},
        )
    )
    raw = json.loads(ws.stage_artifact("plan").read_text(encoding="utf-8"))
    s1 = _section(raw, "sec-0001")
    assert s1["skip_reason"] == "content_type=REFERENCE_ONLY 不抽取"
    assert (s1["extractor"], s1["template"], s1["prompt_version"]) == ("", "", "")


def test_update_schema_mode_auto_requires_profile(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)  # 无章节画像

    with pytest.raises(HTTPException) as ea:
        asyncio.run(
            web.update_schema(
                BOOK_ID,
                {"sections": [{"section_id": "sec-0002", "mode": "auto"}]},
            )
        )
    assert ea.value.status_code == 400


def test_update_schema_mode_skip_marks_manual(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)

    asyncio.run(
        web.update_schema(
            BOOK_ID,
            {"sections": [{"section_id": "sec-0001", "mode": "skip"}]},
        )
    )
    raw = json.loads(ws.stage_artifact("plan").read_text(encoding="utf-8"))
    s1 = _section(raw, "sec-0001")
    assert s1["skip_reason"] == "人工标记不抽取"
    assert (s1["extractor"], s1["template"], s1["prompt_version"]) == ("", "", "")


def test_update_schema_frozen_after_confirmation_until_unlock(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.PLAN_CONFIRMED)
    _save_plan(ws)

    with pytest.raises(HTTPException) as ef:
        asyncio.run(
            web.update_schema(
                BOOK_ID,
                {"sections": [{"section_id": "sec-0001", "mode": "skip"}]},
            )
        )
    assert ef.value.status_code == 400

    # 「解锁编辑」= goto-prompt 无损回退到确认门
    assert asyncio.run(web.workflow_op(BOOK_ID, "goto-prompt"))["stage"] == "prompt_ready"
    asyncio.run(
        web.update_schema(
            BOOK_ID,
            {"sections": [{"section_id": "sec-0001", "mode": "skip"}]},
        )
    )
    raw = json.loads(ws.stage_artifact("plan").read_text(encoding="utf-8"))
    assert _section(raw, "sec-0001")["skip_reason"] == "人工标记不抽取"


def test_update_schema_invalidates_extract_when_already_extracted(
    books_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 解锁后（WAITING_PLAN_CONFIRMATION）但已抽取过的书
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)
    m = ws.load_manifest()
    m.stages["extract"] = StageRecord(stage="extract", status="DONE")
    m.save(ws.book_dir)

    calls: list[tuple[str, str]] = []

    class _FakePipe:
        def invalidate_from(self, book_id: str, stage: str) -> None:
            calls.append((book_id, stage))

    monkeypatch.setattr(web, "_pipeline", _FakePipe())
    asyncio.run(
        web.update_schema(
            BOOK_ID,
            {"sections": [{"section_id": "sec-0001", "mode": "skip"}]},
        )
    )
    assert calls == [(BOOK_ID, "extract")]


def test_update_schema_no_invalidate_before_extract(
    books_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)
    calls: list[str] = []

    class _FakePipe:
        def invalidate_from(self, book_id: str, stage: str) -> None:
            calls.append(stage)

    monkeypatch.setattr(web, "_pipeline", _FakePipe())
    asyncio.run(
        web.update_schema(
            BOOK_ID,
            {"sections": [{"section_id": "sec-0001", "mode": "skip"}]},
        )
    )
    assert calls == []


def test_update_schema_uses_adapted_registry_for_prompt_version(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)
    _save_profile(ws, {"sec-0001": "NARRATIVE_RULE"})
    _save_adaptation(ws)  # general_rule.md 适配块
    base_version = PromptTemplateRegistry().version_of("a", TEMPLATE)

    asyncio.run(
        web.update_schema(
            BOOK_ID,
            {
                "sections": [
                    {"section_id": "sec-0001", "mode": "extract", "extractor": VALID_EXTRACTOR}
                ]
            },
        )
    )
    raw = json.loads(ws.stage_artifact("plan").read_text(encoding="utf-8"))
    s1 = _section(raw, "sec-0001")
    adapted = (
        BookAdaptedRegistry(
            {TEMPLATE: "## 本书适配提示\n- 重点盯防：剂量"},
            prompts_root=PromptTemplateRegistry().prompts_root,
        )
        .load("a", TEMPLATE)
        .version
    )
    assert s1["prompt_version"] == adapted
    assert s1["prompt_version"] != base_version


def test_get_schema_exposes_mode_labels_and_derived(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)
    _save_profile(ws, {"sec-0001": "NARRATIVE_RULE", "sec-0002": "REFERENCE_ONLY"})

    s = asyncio.run(web.get_schema(BOOK_ID))
    views = {e["section_id"]: e for e in s["editable_sections"]}
    assert views["sec-0001"]["mode"] == "auto"
    assert views["sec-0001"]["content_type_label"] == "叙述性规则"
    assert views["sec-0001"]["derived"] == {
        "extractor": "GeneralRuleExtractor",
        "template": "general_rule.md",
        "skip": False,
    }
    assert views["sec-0002"]["skip_kind"] == "auto"
    assert s["extractor_labels"]["GeneralRuleExtractor"] == "通用规则抽取"


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
            field="/payload/herb_name",
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
    assert records[0].record_type == RecordType.GENERAL_RULE  # 未被脏写
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
            record_id=rec.record_id,
            field="/payload/herb_name",
            original="砒石",
            suggested="砒霜（制）",
            verdict=Verdict.PARTIAL,
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


# ── 每书适配块：get_prompt 展示 + regenerate 门控 ────────────────────────────


def _save_adaptation(
    ws: BookWorkspace,
    template: str = TEMPLATE,
    block: str = "## 本书适配提示\n- 重点盯防：剂量",
    degraded: bool = False,
) -> None:
    save_artifact(
        ws.stage_dir("plan") / ADAPTATION_FILENAME,
        header=_header(),
        data={
            "profile_hash": "sha256:abc",
            "adaptations": {
                template: {
                    "template": template,
                    "block": "" if degraded else block,
                    "version": "" if degraded else "v-adapt",
                    "structured": {},
                    "degraded": degraded,
                }
            },
        },
    )


def test_get_prompt_includes_adaptation_block(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)
    _save_adaptation(ws)

    d = asyncio.run(web.get_prompt(BOOK_ID))
    adapt = {a["template"]: a for a in d["adaptations"]}
    assert TEMPLATE in adapt
    assert "本书适配提示" in adapt[TEMPLATE]["block"]
    assert adapt[TEMPLATE]["degraded"] is False
    # 合成提示词同时含 base 正文与适配块
    assert "本书适配提示" in d["prompt"]
    assert d["editable"] is False


def test_get_prompt_without_adaptation_file(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)

    d = asyncio.run(web.get_prompt(BOOK_ID))
    assert all(a["block"] == "" for a in d["adaptations"])


def test_regenerate_adaptations_at_confirmation_gate(
    books_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)
    _save_adaptation(ws)

    calls: dict[str, Any] = {}

    class _FakePipe:
        def rewind(self, book_id: str, *, to_stage: str) -> None:
            calls["rewind"] = (book_id, to_stage)

    monkeypatch.setattr(web, "_pipeline", _FakePipe())
    spawned: list[str] = []

    def _fake_spawn(book_id: str, coro: Any) -> None:
        coro.close()
        spawned.append(book_id)

    monkeypatch.setattr(web, "_spawn", _fake_spawn)

    r = asyncio.run(web.regenerate_adaptations(BOOK_ID))
    assert r["status"] == "started"
    assert calls["rewind"] == (BOOK_ID, "plan")
    assert spawned == [BOOK_ID]
    # 适配缓存被删除（下次 plan 重跑会重新生成）
    assert not (ws.stage_dir("plan") / ADAPTATION_FILENAME).is_file()


def test_update_adaptations_edits_block_and_bumps_version(books_root: Path) -> None:
    """人工编辑适配块：block/version 更新 → plan prompt_version = hash(base+块)。"""
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)
    _save_adaptation(ws)
    base_version = PromptTemplateRegistry().version_of("a", TEMPLATE)

    result = asyncio.run(
        web.update_adaptations(
            BOOK_ID,
            {
                "adaptations": [
                    {"template": TEMPLATE, "block": "## 本书适配提示\n- 人工干预：括号炮制品保留"}
                ],
            },
        )
    )

    data = load_artifact(ws.stage_dir("plan") / ADAPTATION_FILENAME)
    a = data["adaptations"][TEMPLATE]
    assert a["human_edited"] is True
    assert a["block"].endswith("括号炮制品保留")
    assert a["version"] == adaptation_version(a["block"])

    registry = BookAdaptedRegistry(
        {TEMPLATE: a["block"]}, prompts_root=PromptTemplateRegistry().prompts_root
    )
    plan = load_artifact(ws.stage_artifact("plan"))
    sec = _section(plan, "sec-0001")
    assert sec["prompt_version"] == registry.load("a", TEMPLATE).version
    assert sec["prompt_version"] != base_version
    assert result["invalidated_downstream"] is False
    assert ws.load_manifest().prompt_version


def test_update_adaptations_rejects_unknown_template(books_root: Path) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_PLAN_CONFIRMATION)
    _save_plan(ws)
    _save_adaptation(ws)
    with pytest.raises(HTTPException):
        asyncio.run(
            web.update_adaptations(
                BOOK_ID,
                {
                    "adaptations": [{"template": "nope.md", "block": "x"}],
                },
            )
        )


def test_regenerate_adaptations_rejects_non_confirmation(books_root: Path) -> None:
    _make_manifest(books_root, BookStatus.EXTRACTED)
    with pytest.raises(HTTPException):
        asyncio.run(web.regenerate_adaptations(BOOK_ID))


def test_export_review_file_does_not_approve_or_publish(books_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _make_manifest(books_root, BookStatus.WAITING_REVIEW)
    manifest = ws.load_manifest()
    manifest.stage('verify').mark_done(artifact='06_verify/verified_records.jsonl', artifact_hash='h')
    manifest.save(ws.book_dir)
    monkeypatch.setattr(web, '_export_excel', lambda _: ws.book_dir / 'review.xlsx')
    result = asyncio.run(web.step_export(BOOK_ID))
    assert result['status'] == 'done'
    assert ws.load_manifest().status == BookStatus.WAITING_REVIEW
