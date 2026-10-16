"""导出 record→block 溯源索引（供 Milvus 入库脚本直接读取）。

用法：
    python scripts/export_block_index.py --book-id 454ec4a03d1d

输出（默认）：
    workspace/books/<book_id>/08_publish/block_index.jsonl

形态（record 为中心，每行一条记录、单个 block_id）：
    {"record_id", "book_id", "record_type", "block_id", "block_content",
     "block_format", "asset_path", "page", "section_path", "subject_text",
     "herb_name", "raw_text", "needs_review", "passthrough"}

说明：
    - 只读工件，不修改 manifest；纯确定性 join（record.source.block_ids ↔ blocks.jsonl）。
    - 记录源优先取 06_verify/verified_records.jsonl（B 核验后的最终版），
      缺失时回退 04_extract/records.jsonl。
    - block_content 即待 embed 的原文（drawing 块为原始表格 HTML/Markdown）。
    - asset_path 为图/表/公式块的原始图片绝对路径（多模态入库用）；纯文本块为空。
    - 每条记录展开全部 block_id；命不中 block 时 block_id 留空并计数告警。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from tcm_extract_engine.cab.profiler import load_blocks
from tcm_extract_engine.core.artifacts import load_artifact_records
from tcm_extract_engine.core.records import SourceRecord

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspace" / "books"
OUTPUT_NAME = "block_index.jsonl"


def _load_records(book_dir: Path) -> list[SourceRecord]:
    """优先核验后记录，回退原始抽取记录。"""
    verified = book_dir / "06_verify" / "verified_records.jsonl"
    extracted = book_dir / "04_extract" / "records.jsonl"
    path = verified if verified.is_file() else extracted
    if not path.is_file():
        raise ValueError(f"未在 {book_dir} 找到抽取/核验记录")
    return load_artifact_records(path, SourceRecord)


def _block_page(block: dict[str, Any] | None) -> int | None:
    """块物理页：positions[0].anchor。"""
    if not block:
        return None
    positions = block.get("positions") or []
    if not positions:
        return None
    try:
        return int(str(positions[0].get("anchor")))
    except (ValueError, TypeError):
        return None


def build_block_index(book_dir: Path) -> list[dict[str, Any]]:
    """把记录按 source.block_ids join 回 blocks.jsonl，产出 record→block 映射行。"""
    records = _load_records(book_dir)
    blocks_path = book_dir / "01_parse" / "blocks.jsonl"
    blocks_by_id: dict[str, dict[str, Any]] = (
        {str(b.get("blockid")): b for b in load_blocks(blocks_path)}
        if blocks_path.is_file()
        else {}
    )

    rows: list[dict[str, Any]] = []
    unmatched = 0
    for rec in records:
        for block_id in rec.source.block_ids or [""]:
            block = blocks_by_id.get(block_id) if block_id else None
            if block_id and block is None:
                unmatched += 1
            page = rec.source.page_start
            if page is None:
                page = _block_page(block)
            rows.append(
                {
                    "record_id": rec.record_id,
                    "book_id": rec.book_id,
                    "record_type": rec.record_type,
                    "block_id": block_id,
                    "block_content": str((block or {}).get("content") or ""),
                    "block_format": str((block or {}).get("format") or ""),
                    "asset_path": str((block or {}).get("asset_path") or ""),
                    "page": page,
                    "section_path": list(rec.source.section_path),
                    "subject_text": rec.subject_text,
                    "herb_name": rec.payload.get("herb_name"),
                    "raw_text": rec.raw_text,
                    "needs_review": rec.needs_review,
                    "passthrough": rec.passthrough,
                }
            )
    if unmatched:
        logger.warning("[BlockIndex] %d 条记录的 block_id 在 blocks.jsonl 中缺失", unmatched)
    return rows


def export_block_index(book_dir: Path, output: Path | None = None) -> Path:
    """生成 block_index.jsonl，返回输出路径。"""
    rows = build_block_index(book_dir)
    if not rows:
        raise ValueError(f"{book_dir} 无可导出的记录")
    if output is None:
        output = book_dir / "08_publish" / OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 record→block 溯源索引（Milvus 入库用）")
    parser.add_argument("--book-id", required=True, help="书籍工作区 ID（workspace/books/<id>）")
    parser.add_argument("--workspace", default=str(WORKSPACE), help="书籍工作区根目录")
    parser.add_argument(
        "--output", default=None, help="输出 jsonl 路径（默认 08_publish/block_index.jsonl）"
    )
    args = parser.parse_args()

    book_dir = Path(args.workspace) / args.book_id
    if not book_dir.is_dir():
        raise FileNotFoundError(f"书籍工作区不存在: {book_dir}")
    output = export_block_index(book_dir, Path(args.output) if args.output else None)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
