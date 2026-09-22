"""引擎 CLI 冒烟测试（book / gepa 子命令）。"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tcm_extract_engine.cli import app

runner = CliRunner()


# ── book 子命令（Artifact-Driven 流水线）──────────────────────────────────────


def test_book_help() -> None:
    result = runner.invoke(app, ["book", "--help"])
    assert result.exit_code == 0
    for verb in ("run", "resume", "rerun", "retry", "confirm", "status", "list"):
        assert verb in result.output


def test_book_list_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["book", "list", "--books-root", str(tmp_path)])
    assert result.exit_code == 0
    assert "暂无书籍工作目录" in result.output


def test_book_run_missing_source(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["book", "run", str(tmp_path / "不存在.md"),
              "--books-root", str(tmp_path / "books")],
    )
    assert result.exit_code == 1
    assert "路径不存在" in result.output


def test_book_status_unknown_book(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["book", "status", "0" * 12, "--books-root", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "书籍不存在" in result.output


def test_book_resume_unknown_book(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["book", "resume", "0" * 12, "--books-root", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "书籍不存在" in result.output


# ── gepa 子命令（提示词进化）──────────────────────────────────────────────────


def test_gepa_help() -> None:
    result = runner.invoke(app, ["gepa", "--help"])
    assert result.exit_code == 0
    for verb in ("evolve", "evolve-multi", "evolve-all", "promote", "runs", "trend"):
        assert verb in result.output


def test_gepa_evolve_all_help() -> None:
    result = runner.invoke(app, ["gepa", "evolve-all", "--help"])
    assert result.exit_code == 0
    assert "--merge" in result.output
    assert "--template" in result.output


def test_gepa_trend_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["gepa", "trend", "--out-root", str(tmp_path / "gepa")])
    assert result.exit_code == 0
    assert "暂无趋势记录" in result.output


def test_gepa_runs_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["gepa", "runs", "--out-root", str(tmp_path / "gepa")])
    assert result.exit_code == 0
    assert "暂无进化产物" in result.output


def test_gepa_evolve_missing_book(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["gepa", "evolve", "0" * 12,
              "--books-root", str(tmp_path / "books"),
              "--out-root", str(tmp_path / "gepa")],
    )
    assert result.exit_code == 1


def test_gepa_promote_missing_run_dir(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["gepa", "promote", str(tmp_path / "nope")],
    )
    assert result.exit_code == 1
    assert "产物目录不存在" in result.output


def test_gepa_promote_dry_run(tmp_path: Path) -> None:
    from tcm_extract_engine.cab.planner import CHUNK_PLACEHOLDER
    from tcm_extract_engine.core.artifacts import ArtifactHeader, save_artifact

    # 造 run 产物
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    save_artifact(
        run_dir / "run.json",
        header=ArtifactHeader(schema_version="1", book_id="a" * 12, task_id="t"),
        data={"template": "general_rule.md", "best_constraint": "禁止臆测剂量。"},
    )
    (run_dir / "best_constraint.md").write_text("禁止臆测剂量。\n", encoding="utf-8")

    # 造 prompts 根
    a_dir = tmp_path / "prompts" / "a"
    a_dir.mkdir(parents=True)
    tpl = a_dir / "general_rule.md"
    tpl.write_text(f"# 模板\n禁止推理。\n\n{CHUNK_PLACEHOLDER}\n", encoding="utf-8")
    before = tpl.read_text(encoding="utf-8")

    result = runner.invoke(
        app, ["gepa", "promote", str(run_dir),
              "--prompts-root", str(tmp_path / "prompts"), "--dry-run"],
    )
    assert result.exit_code == 0
    assert "dry-run" in result.output
    assert "已晋升" in result.output
    assert tpl.read_text(encoding="utf-8") == before  # dry-run 未写盘
