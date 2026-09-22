"""Web GEPA 飞轮面板端点测试：只读概览（runs / trend 扫描）+ 人工晋升门。

不起 HTTP，直接调用 app 端点协程；monkeypatch GEPA_ROOT 到 tmp，并把
prompts_root 重定向到 tmp（避免写真实 prompts/）。
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tcm_extract_engine.cab import planner as planner_mod
from tcm_extract_engine.core.artifacts import ArtifactHeader, save_artifact
from tcm_extract_engine.gepa.runner import (
    TrendRecord,
    append_trend,
    ledger_path_for,
)
from tcm_extract_engine.web import app as web

RUN_ID = "20260101T000000Z_abcdef"
TEMPLATE = "general_rule.md"
BOOK = "testbook0001"
BLOCK = "必须标注炮制方法"


@pytest.fixture()
def gepa_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "gepa"
    root.mkdir()
    monkeypatch.setattr(web, "GEPA_ROOT", root)
    return root


@pytest.fixture()
def prompts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    proot = tmp_path / "prompts"
    (proot / "a").mkdir(parents=True)
    (proot / "a" / TEMPLATE).write_text(f"# base\n{TEMPLATE} body\n", encoding="utf-8")
    # promote_constraint 内部 PromptTemplateRegistry().prompts_root 走 planner.project_paths.prompts
    monkeypatch.setattr(planner_mod, "project_paths", SimpleNamespace(prompts=proot))
    return proot


def _write_run(gepa_root: Path, *, improved: bool, best: float) -> Path:
    run_dir = gepa_root / BOOK / TEMPLATE / RUN_ID
    run_dir.mkdir(parents=True)
    save_artifact(
        run_dir / "run.json",
        header=ArtifactHeader(schema_version="1", book_id=BOOK, task_id="t"),
        data={
            "template": TEMPLATE,
            "run_id": RUN_ID,
            "baseline_fitness": 0.5,
            "best_fitness": best,
            "improved": improved,
            "total_generations": 2,
            "eval_calls": 4,
            "tokens": {"extract_input": 100, "extract_output": 50},
            "best_constraint": BLOCK,
        },
    )
    (run_dir / "best_constraint.md").write_text(BLOCK + "\n", encoding="utf-8")
    return run_dir


def test_overview_lists_runs_and_summary(gepa_root: Path) -> None:
    _write_run(gepa_root, improved=True, best=0.8)
    d = asyncio.run(web.gepa_overview())
    assert d["enabled"] is True
    assert d["summary"]["runs_total"] == 1
    assert d["summary"]["improved_runs"] == 1
    assert d["summary"]["promote_total"] == 0
    assert d["summary"]["latest_best"] == 0.8
    run = d["runs"][0]
    assert run["template"] == TEMPLATE
    assert run["book_ids"] == [BOOK]
    assert run["total_tokens"] == 150  # extract_input + extract_output
    assert run["promoted"] is False


def test_overview_marks_promoted_run(gepa_root: Path) -> None:
    _write_run(gepa_root, improved=True, best=0.8)
    append_trend(
        ledger_path_for(gepa_root),
        TrendRecord(
            ts="2026-01-01T00:00:00+00:00",
            kind="promote",
            template=TEMPLATE,
            run_id=RUN_ID,
            promoted=True,
            reviewer="rev",
        ),
    )
    d = asyncio.run(web.gepa_overview())
    assert d["summary"]["promote_total"] == 1
    assert d["runs"][0]["promoted"] is True


def test_overview_disabled_when_no_dir(gepa_root: Path) -> None:
    shutil.rmtree(gepa_root)
    d = asyncio.run(web.gepa_overview())
    assert d["enabled"] is False
    assert d["runs"] == []


def test_promote_dry_run_then_write(gepa_root: Path, prompts_root: Path) -> None:
    _write_run(gepa_root, improved=True, best=0.8)
    rel = f"{BOOK}/{TEMPLATE}/{RUN_ID}"
    tmpl_file = prompts_root / "a" / TEMPLATE

    dry = asyncio.run(
        web.gepa_promote({"run_dir": rel, "dry_run": True, "reviewer": "rev"})
    )
    assert dry["dry_run"] is True
    assert dry["version_bumped"] is True
    assert "GEPA:PROMOTED" not in tmpl_file.read_text(encoding="utf-8")  # 预览不落盘

    real = asyncio.run(
        web.gepa_promote({"run_dir": rel, "dry_run": False, "reviewer": "rev"})
    )
    assert real["changed"] is True and real["version_bumped"] is True
    text = tmpl_file.read_text(encoding="utf-8")
    assert "GEPA:PROMOTED:START" in text and BLOCK in text

    # 幂等：再晋升同一块 → 无变化
    again = asyncio.run(
        web.gepa_promote({"run_dir": rel, "dry_run": False, "reviewer": "rev"})
    )
    assert again["changed"] is False


def test_promote_rejects_path_traversal(gepa_root: Path) -> None:
    with pytest.raises(HTTPException) as exc:
        asyncio.run(web.gepa_promote({"run_dir": "../../etc"}))
    assert exc.value.status_code in (400, 404)


def test_promote_requires_run_json(gepa_root: Path) -> None:
    with pytest.raises(HTTPException) as exc:
        asyncio.run(web.gepa_promote({"run_dir": f"{BOOK}/{TEMPLATE}/nope"}))
    assert exc.value.status_code == 404
