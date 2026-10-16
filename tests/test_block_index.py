"""record→block 溯源索引导出（scripts/export_block_index.py）测试。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from tcm_extract_engine.core.artifacts import ArtifactHeader, save_artifact
from tcm_extract_engine.core.records import SourceRef
from tcm_extract_engine.domains.tcm.schema import RecordType
from tcm_extract_engine.domains.tcm.schema import TCMRecord as SourceRecord

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_block_index.py"


def _load_mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_tcm_block_index", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _block(
    blockid: str, content: str, *, fmt: str = "plain_text", page: int = 1, asset_path: str = ""
) -> dict:
    return {
        "type": "content",
        "blockid": blockid,
        "format": fmt,
        "content": content,
        "heading": "H",
        "parent_headings": [],
        "asset_path": asset_path,
        "positions": [{"type": "bbox", "anchor": str(page), "range": [0, 0, 1, 1]}],
    }


def _write_book(tmp_path: Path) -> Path:
    book = tmp_path / "book"
    (book / "01_parse").mkdir(parents=True)
    (book / "04_extract").mkdir(parents=True)
    blocks = [
        {"type": "meta", "version": "1.0"},
        _block("b1", "制川乌 1.5～3g，先煎。", page=12),
        _block(
            "t1",
            "<drawing><table><tr><td>川乌</td></tr></table></drawing>",
            fmt="drawing",
            page=5,
            asset_path="/abs/images/t1.jpg",
        ),
    ]
    (book / "01_parse" / "blocks.jsonl").write_text(
        "\n".join(json.dumps(b, ensure_ascii=False) for b in blocks) + "\n",
        encoding="utf-8",
    )
    header = ArtifactHeader(schema_version="1", book_id="x", task_id="t")
    records = [
        SourceRecord(
            record_id="SR-x-000001",
            book_id="x",
            record_type=RecordType.DOSAGE,
            source=SourceRef(
                page_start=12, section_path=["4.4 剂量"], block_ids=["b1"], chunk_id="sec:0:h"
            ),
            raw_text="制川乌 1.5～3g，先煎",
            herb_name="制川乌",
        ),
        SourceRecord(
            record_id="SR-x-000002",
            book_id="x",
            record_type=RecordType.TABLE_ROW,
            source=SourceRef(
                page_start=5, section_path=["附录A"], block_ids=["t1"], chunk_id="sec:1:h"
            ),
            raw_text="川乌 | 有大毒",
            passthrough=True,
        ),
        SourceRecord(
            record_id="SR-x-000003",
            book_id="x",
            record_type=RecordType.GENERAL_RULE,
            source=SourceRef(section_path=["前言"]),
            raw_text="无块记录",
        ),
    ]
    save_artifact(book / "04_extract" / "records.jsonl", header=header, data={}, records=records)
    return book


def test_build_block_index_single_block_id(tmp_path: Path) -> None:
    mod = _load_mod()
    rows = mod.build_block_index(_write_book(tmp_path))
    assert len(rows) == 3

    r1 = next(r for r in rows if r["record_id"] == "SR-x-000001")
    assert r1["block_id"] == "b1"
    assert r1["block_content"] == "制川乌 1.5～3g，先煎。"
    assert r1["block_format"] == "plain_text"
    assert r1["page"] == 12
    assert r1["herb_name"] == "制川乌"

    rt = next(r for r in rows if r["record_id"] == "SR-x-000002")
    assert rt["block_id"] == "t1"
    assert rt["block_format"] == "drawing"
    assert rt["asset_path"] == "/abs/images/t1.jpg"
    assert rt["passthrough"] is True

    # 无 block_ids 的记录 → block_id/content 留空（不丢行）
    r3 = next(r for r in rows if r["record_id"] == "SR-x-000003")
    assert r3["block_id"] == ""
    assert r3["block_content"] == ""
    assert r3["asset_path"] == ""


def test_export_writes_jsonl(tmp_path: Path) -> None:
    mod = _load_mod()
    book = _write_book(tmp_path)
    out = mod.export_block_index(book)
    assert out == book / "08_publish" / "block_index.jsonl"
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 3
    assert {"record_id", "block_id", "block_content", "page"} <= set(lines[0])


def test_prefers_verified_records(tmp_path: Path) -> None:
    mod = _load_mod()
    book = _write_book(tmp_path)
    (book / "06_verify").mkdir(parents=True)
    vrec = [
        SourceRecord(
            record_id="SR-v-000001",
            book_id="x",
            record_type=RecordType.DOSAGE,
            source=SourceRef(block_ids=["b1"], page_start=12),
            raw_text="制川乌",
        )
    ]
    save_artifact(
        book / "06_verify" / "verified_records.jsonl",
        header=ArtifactHeader(schema_version="1", book_id="x", task_id="t"),
        data={},
        records=vrec,
    )
    rows = mod.build_block_index(book)
    assert [r["record_id"] for r in rows] == ["SR-v-000001"]
