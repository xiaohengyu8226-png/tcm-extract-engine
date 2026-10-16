"""表格直通通路测试：确定性解析 / 组块直通 / 免 LLM 抽取 / 免 B 核验 / 覆盖率 / 原表 sheet。"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from tcm_extract_engine.cab.extractors import extract_records, split_section_batches
from tcm_extract_engine.cab.planner import PromptTemplateRegistry, SectionPlan
from tcm_extract_engine.cab.validator import IssueCategory, validate_records
from tcm_extract_engine.cab.verifier import Verdict, verify_records
from tcm_extract_engine.core.artifacts import ArtifactHeader, save_artifact
from tcm_extract_engine.core.llm import LLMError, LLMUsage
from tcm_extract_engine.core.tables import parse_table
from tcm_extract_engine.domains.tcm.schema import RecordType
from tcm_extract_engine.domains.tcm.schema import TCMRecord as SourceRecord

TABLE_HTML = (
    "<drawing id='tb-1'><table>"
    "<tr><td>药名</td><td>毒性</td></tr>"
    "<tr><td>川乌</td><td>有大毒</td></tr>"
    "<tr><td>草乌</td><td>有大毒</td></tr>"
    "</table></drawing>"
)

TABLE_MD = "<drawing id='tb-2'>\n| 药名 | 剂量 |\n| --- | --- |\n| 附子 | 3～15g |\n</drawing>"


class FakeLLMClient:
    """空队列即抛错：用于断言直通路径零 LLM 调用。"""

    def __init__(self) -> None:
        self.model = "fake"
        self.usage = LLMUsage()
        self.queue: list[Any] = []
        self.calls: list[str] = []

    async def complete_json(
        self, *, system_prompt: str, user_prompt: str, semaphore: Any = None, **_: Any
    ) -> dict[str, Any]:
        self.calls.append(user_prompt)
        if not self.queue:
            raise LLMError("api", "队列为空（不应调用 LLM）")
        return self.queue.pop(0)


def _block(blockid: str, heading: str, page: int, content: str) -> dict[str, Any]:
    return {
        "type": "content",
        "blockid": blockid,
        "format": "drawing" if "<drawing" in content else "plain_text",
        "content": content,
        "heading": heading,
        "parent_headings": [],
        "level": 1,
        "positions": [{"type": "bbox", "anchor": str(page), "range": [0, 0, 1, 1]}],
    }


def _write_blocks(tmp_path: Path, blocks: list[dict[str, Any]]) -> Path:
    path = tmp_path / "blocks.jsonl"
    path.write_text(
        "\n".join(json.dumps(b, ensure_ascii=False) for b in blocks) + "\n",
        encoding="utf-8",
    )
    return path


def _section(
    block_ids: list[str],
    extractor: str = "HerbTableExtractor",
    template: str = "herb_table.md",
    title: str = "附录A 有毒中药",
) -> SectionPlan:
    registry = PromptTemplateRegistry()
    return SectionPlan(
        section_id="sec-1",
        title=title,
        heading_path=[title],
        extractor=extractor,
        template=template,
        prompt_version=registry.version_of("a", template),
        block_ids=block_ids,
        page_start=1,
        page_end=9,
    )


# ── 确定性解析 ───────────────────────────────────────────────────────────────


def test_parse_table_html() -> None:
    parsed = parse_table(TABLE_HTML)
    assert parsed.headers == ["药名", "毒性"]
    assert parsed.rows == [["川乌", "有大毒"], ["草乌", "有大毒"]]


def test_parse_table_markdown_skips_separator() -> None:
    parsed = parse_table(TABLE_MD)
    assert parsed.headers == ["药名", "剂量"]
    assert parsed.rows == [["附子", "3～15g"]]


def test_parse_table_normalizes_width() -> None:
    parsed = parse_table(
        "<drawing><table><tr><td>a</td><td>b</td></tr><tr><td>1</td></tr></table></drawing>"
    )
    assert parsed.rows == [["1", ""]]


# ── 组块：表格直通、关系组文本不切断 ─────────────────────────────────────────


def test_split_table_chunk_is_passthrough() -> None:
    blocks_by_id = {
        "t1": _block("t1", "附录A", 1, TABLE_HTML),
        "p1": _block("p1", "附录A", 1, "正文说明。"),
    }
    batches = split_section_batches(_section(["t1", "p1"]), blocks_by_id)
    assert len(batches) == 2
    assert batches[0].passthrough is True
    assert batches[1].passthrough is False


def test_split_relation_section_tables_also_passthrough() -> None:
    """关系组章节：表格块直通，文本整章一批不切断。"""
    blocks_by_id = {
        "t1": _block("t1", "附录C", 40, TABLE_HTML),
        "p1": _block("p1", "附录C", 40, "甘草反甘遂。"),
        "p2": _block("p2", "附录C", 41, "乌头反贝母。"),
    }
    sec = _section(
        ["t1", "p1", "p2"],
        extractor="InteractionExtractor",
        template="interaction_group.md",
        title="附录C 十八反",
    )
    batches = split_section_batches(sec, blocks_by_id)
    assert [b.passthrough for b in batches] == [True, False]
    assert len(batches[1].texts) == 2  # 文本整章不切断


# ── extract：表格免 LLM ─────────────────────────────────────────────────────


def test_extract_table_passthrough_skips_llm(tmp_path: Path) -> None:
    blocks = [
        {"type": "meta", "version": "1.0"},
        _block("t1", "附录A 有毒中药", 1, TABLE_HTML),
        _block("p1", "附录A 有毒中药", 1, "正文说明。"),
    ]
    blocks_path = _write_blocks(tmp_path, blocks)
    plan_path = tmp_path / "plan.json"
    save_artifact(
        plan_path,
        header=ArtifactHeader(schema_version="1", book_id="testbook01", task_id="t"),
        data={
            "document_type": "手册",
            "goal": "g",
            "risk_points": [],
            "sections": [_section(["t1", "p1"]).model_dump(mode="json")],
        },
    )
    client = FakeLLMClient()
    client.queue = [
        {
            "records": [
                {
                    "record_type": "HERB_TABLE",
                    "raw_text": "正文说明。",
                    "herb_name": "正文",
                }
            ]
        }
    ]

    records = asyncio.run(
        extract_records(
            book_id="testbook01",
            blocks_path=blocks_path,
            plan_path=plan_path,
            cache_dir=tmp_path / "cache",
            client=client,
            table_semantics=False,
        )
    )

    table_rows = [r for r in records if r.record_type == RecordType.TABLE_ROW]
    assert len(table_rows) == 2  # 两数据行逐行透传
    assert all(r.passthrough for r in table_rows)
    assert table_rows[0].payload["attributes"] == {"药名": "川乌", "毒性": "有大毒"}
    assert table_rows[0].subject_text == "川乌"
    # 表格直通记录溯源到那张 drawing 块（供后续入 Milvus）
    assert table_rows[0].source.block_ids == ["t1"]
    assert table_rows[0].source.chunk_id
    # 仅文本批次调用了一次 LLM；表格批次零调用
    assert len(client.calls) == 1


# ── verify：直通记录免 B 核验 ────────────────────────────────────────────────


def test_verify_passthrough_records_skip_llm(tmp_path: Path) -> None:
    blocks_path = _write_blocks(
        tmp_path,
        [
            {"type": "meta", "version": "1.0"},
            _block("t1", "附录A", 1, TABLE_HTML),
        ],
    )
    rec = SourceRecord(
        record_id="SR-testbook-000001",
        book_id="testbook01",
        record_type=RecordType.TABLE_ROW,
        raw_text="药名 | 毒性 | 川乌 | 有大毒",
        passthrough=True,
    )
    client = FakeLLMClient()  # 空队列：一旦送 B 即抛错

    report = asyncio.run(
        verify_records(
            records=[rec],
            blocks_path=blocks_path,
            client=client,
        )
    )

    assert client.calls == []
    assert report.results[0].verdict == Verdict.PASSTHROUGH
    assert rec.needs_review is False


# ── validator：覆盖率统计含 TABLE_ROW ────────────────────────────────────────


def _table_record(seq: int, herb: str) -> SourceRecord:
    return SourceRecord(
        record_id=f"SR-testbook-{seq:06d}",
        book_id="testbook01",
        record_type=RecordType.TABLE_ROW,
        raw_text=f"行{seq}",
        subject_text=herb,
        source={"section_path": ["附录A 有毒中药"]},
        passthrough=True,
    )


def test_validator_coverage_counts_table_row() -> None:
    blocks = [_block("t1", "附录A 有毒中药", 1, TABLE_HTML)]
    full = [_table_record(1, "川乌"), _table_record(2, "草乌")]
    report = validate_records(full, blocks=blocks)
    assert not any(i.category == IssueCategory.COVERAGE for i in report.issues)

    partial = [_table_record(1, "川乌")]
    report2 = validate_records(partial, blocks=blocks)
    assert any(i.category == IssueCategory.COVERAGE for i in report2.issues)


# ── 导出：合并「原始表格」sheet ─────────────────────────────────────────────


def test_export_raw_table_sheet(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_book_review_excel.py"
    spec = importlib.util.spec_from_file_location("_tcm_export", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    wb = openpyxl.Workbook()
    ws = wb.active
    mod._append_raw_table_sheet(ws, [_block("t1", "附录A 有毒中药", 1, TABLE_HTML)])

    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    assert rows[0][0] == "章节"
    # 第一段：标题行带章节/页/表ID，随后表头+数据行
    assert rows[1][0] == "附录A 有毒中药" and rows[1][3] == "表头"
    assert rows[2][3] == "数据1" and rows[2][4] == "川乌"
