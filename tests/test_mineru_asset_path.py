"""mineru 图片资产路径结构化（drawing 块 asset_path）测试。"""

from __future__ import annotations

from pathlib import Path

from tcm_extract_engine.parse.mineru import (
    _content_list_to_blocks,
    _find_asset_base,
    _resolve_asset_path,
)


def test_find_asset_base_locates_images_parent(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "doc" / "auto" / "images").mkdir(parents=True)
    assert _find_asset_base(raw) == raw / "doc" / "auto"


def test_find_asset_base_none_when_absent(tmp_path: Path) -> None:
    assert _find_asset_base(tmp_path / "nope") is None


def test_resolve_asset_path(tmp_path: Path) -> None:
    base = tmp_path / "auto"
    (base / "images").mkdir(parents=True)
    f = base / "images" / "a.jpg"
    f.write_bytes(b"x")
    assert _resolve_asset_path(base, "images/a.jpg") == str(f.resolve())
    assert _resolve_asset_path(base, "images/missing.jpg") == ""
    assert _resolve_asset_path(None, "images/a.jpg") == ""
    assert _resolve_asset_path(base, "") == ""


def test_content_list_to_blocks_sets_asset_path(tmp_path: Path) -> None:
    """image 块解析出图片绝对路径；无 img_path 的 table 块 asset_path 为空。"""
    base = tmp_path / "auto"
    (base / "images").mkdir(parents=True)
    img = base / "images" / "a.jpg"
    img.write_bytes(b"x")
    source_pdf = tmp_path / "book.pdf"
    content_list = [
        {"type": "image", "text": "", "img_path": "images/a.jpg",
         "image_caption": ["图1 药材性状"], "page_idx": 0, "bbox": [0, 0, 1, 1]},
        {"type": "table", "table_body": "<table><tr><td>x</td></tr></table>",
         "page_idx": 1, "bbox": [0, 0, 1, 1]},
    ]
    blocks = _content_list_to_blocks(source_pdf, content_list, asset_base=base)
    drawings = [b for b in blocks if b["format"] == "drawing"]
    assert len(drawings) == 2
    assert drawings[0]["asset_path"] == str(img.resolve())
    assert drawings[1]["asset_path"] == ""


def test_content_list_to_blocks_no_asset_base(tmp_path: Path) -> None:
    """未提供 asset_base（如 Markdown 直转）时 asset_path 一律为空，不报错。"""
    content_list = [
        {"type": "image", "text": "", "img_path": "images/a.jpg",
         "image_caption": ["图1"], "page_idx": 0, "bbox": [0, 0, 1, 1]},
    ]
    blocks = _content_list_to_blocks(tmp_path / "book.pdf", content_list)
    drawings = [b for b in blocks if b["format"] == "drawing"]
    assert drawings and drawings[0]["asset_path"] == ""
