"""导出脚本 scripts/export_book_review_excel.py 的回归测试。

覆盖本次「Excel 结果不好」的三处核心修复：
  1. 核验结论改读权威 per-run 工件，杜绝陈旧缓存按 record_id 撞车污染；
  2. 章节路径清洗掉被 MinerU 误当标题的整段表格 HTML；
  3. passthrough/表格直通记为「直通免核验」，统计页在缺报告时不崩。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_book_review_excel.py"


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_tcm_export_review", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── 章节路径 HTML 清洗 ──────────────────────────────────────────────────────


def test_render_section_path_collapses_table_html(mod: ModuleType) -> None:
    parts = [
        "第二章 中药饮片处方审核要点",
        "二、规范性审核要点",
        "<table><tr><td>正名</td><td>别名</td></tr><tr><td>大血藤</td></tr></table>",
    ]
    out = mod._render_section_path(parts)
    assert out == "第二章 中药饮片处方审核要点 / 二、规范性审核要点 / （表格）"
    assert "<table>" not in out and "<td>" not in out


def test_render_section_path_strips_inline_tags_and_dedup(mod: ModuleType) -> None:
    assert mod._render_section_path(["前言<b>x</b>"]) == "前言x"
    # 连续重复段折叠为一，避免刷屏
    assert mod._render_section_path(["第一章", "第一章", "正文"]) == "第一章 / 正文"


# ── 权威 verdict 源优先、忽略陈旧缓存 ─────────────────────────────────────────


def test_load_verify_results_prefers_authoritative_artifact(
    mod: ModuleType, tmp_path: Path
) -> None:
    book = tmp_path / "b1"
    _write(
        book / "06_verify" / "verification_results.jsonl",
        [
            '{"header": {"book_id": "b1"}}',
            json.dumps(
                {"record_id": "SR-1", "verdict": "SUPPORTED", "evidence_quote": "原文A"},
                ensure_ascii=False,
            ),
            json.dumps(
                {"record_id": "SR-2", "verdict": "PASSTHROUGH", "explanation": "表格直通"},
                ensure_ascii=False,
            ),
        ],
    )
    # 陈旧缓存：撞 SR-1 的号给错误 verdict，外加已不在当前记录集的孤儿 SR-9
    _write(
        book / "06_verify" / "cache" / "verify_stale.json",
        [
            json.dumps(
                {
                    "prompt_version": "x",
                    "response": {
                        "records": [
                            {
                                "record_id": "SR-1",
                                "verdict": "WRONG_NORMALIZATION",
                                "source_evidence": "陈旧",
                            },
                            {"record_id": "SR-9", "verdict": "WRONG_NORMALIZATION"},
                        ]
                    },
                },
                ensure_ascii=False,
            )
        ],
    )
    res = mod._load_verify_results(book, {})
    assert res["SR-1"]["verdict"] == "SUPPORTED"  # 权威工件胜出，而非陈旧 WRONG_NORMALIZATION
    assert res["SR-1"]["evidence"] == "原文A"
    assert res["SR-2"]["verdict"] == "PASSTHROUGH"
    assert "SR-9" not in res  # 孤儿陈旧 id 不再被注入


def test_load_verify_results_correction_from_log(mod: ModuleType, tmp_path: Path) -> None:
    book = tmp_path / "b2"
    _write(
        book / "06_verify" / "verification_results.jsonl",
        [
            '{"header": {"book_id": "b2"}}',
            json.dumps({"record_id": "SR-1", "verdict": "WRONG_RELATION"}, ensure_ascii=False),
        ],
    )
    _write(
        book / "06_verify" / "correction_log.jsonl",
        [
            '{"header": {"book_id": "b2"}}',
            json.dumps(
                {
                    "record_id": "SR-1",
                    "field": "anchor",
                    "original": "乌头",
                    "suggested": "草乌",
                    "reason": "锚点错配",
                },
                ensure_ascii=False,
            ),
        ],
    )
    res = mod._load_verify_results(book, {})
    assert "anchor：乌头 → 草乌" in res["SR-1"]["correction"]
    assert "锚点错配" in res["SR-1"]["correction"]


def test_load_verify_results_falls_back_to_cache_without_artifact(
    mod: ModuleType, tmp_path: Path
) -> None:
    book = tmp_path / "b3"
    _write(
        book / "06_verify" / "cache" / "verify_a.json",
        [
            json.dumps(
                {
                    "prompt_version": "p",
                    "response": {
                        "records": [
                            {"record_id": "SR-1", "verdict": "SUPPORTED", "source_evidence": "ev"}
                        ]
                    },
                },
                ensure_ascii=False,
            )
        ],
    )
    res = mod._load_verify_results(book, {})
    assert res["SR-1"]["verdict"] == "SUPPORTED"
    assert res["SR-1"]["evidence"] == "ev"


def test_load_verify_results_missing_everything_returns_empty(
    mod: ModuleType, tmp_path: Path
) -> None:
    assert mod._load_verify_results(tmp_path / "none", {}) == {}


# ── 统计页：缺报告不崩 + passthrough 计为直通免核验 ───────────────────────────


def test_stats_sheet_survives_missing_reports(mod: ModuleType, tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    from tcm_extract_engine.domains.tcm.schema import RecordType
    from tcm_extract_engine.domains.tcm.schema import TCMRecord as SourceRecord

    rec = SourceRecord(
        record_id="SR-1", book_id="b", record_type=RecordType.TABLE_ROW, passthrough=True
    )
    ws = openpyxl.Workbook().active
    mod._append_stats_sheet(ws, [rec], {}, tmp_path / "noBook")  # 无 validation_report / manifest
    values = {row[0]: row[1] for row in ws.iter_rows(values_only=True)}
    assert values["统计项"] == "数值"  # 表头正常
    assert values["核验 直通免核验"] == 1  # passthrough 计入直通，不虚报未核验
