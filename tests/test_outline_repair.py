"""大纲修复（cab/outline.py）单元测试。

覆盖：候选提取、健康闸门、LLM 修复、确定性重建、剔除与标题降级、
缓存幂等、优雅降级、blocks.jsonl 重写（meta 保留）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tcm_extract_engine.cab import outline as outline_mod
from tcm_extract_engine.cab.outline import (
    OutlineEntry,
    apply_outline,
    extract_candidates,
    hierarchy_needs_repair,
    repair_blocks,
    rewrite_blocks_file,
)
from tcm_extract_engine.cab.planner import PromptTemplateRegistry
from tcm_extract_engine.core.llm import LLMError


class FakeClient:
    """大纲修复测试用假客户端：顺序出队响应；空队列抛 LLMError（模拟降级）。"""

    def __init__(self, model: str = "fake-structure") -> None:
        self.model = model
        self.queue: list[Any] = []
        self.calls: list[str] = []

    async def complete_json(self, *, system_prompt: str, user_prompt: str, **_: Any) -> dict[str, Any]:
        self.calls.append(user_prompt)
        if not self.queue:
            raise LLMError("api", "FakeClient 队列为空")
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _payload_of(call: str) -> dict[str, Any]:
    """从渲染后的 prompt 中提取占位符处的 JSON 载荷（模板尾节后首个大括号起）。"""
    tail = call.rsplit("## 待修复标题流", 1)[1]
    return json.loads(tail[tail.index("{"):])


# ── 块构造工具 ───────────────────────────────────────────────────────────────


def _block(
    block_id: str,
    content: str,
    *,
    heading: str = "",
    level: int = 0,
    parents: list[str] | None = None,
    positions: list[dict[str, Any]] | None = None,
    fmt: str = "plain_text",
) -> dict[str, Any]:
    return {
        "block_id": block_id,
        "content": content,
        "heading": heading,
        "parent_headings": parents or [],
        "level": level,
        "positions": positions or [],
        "format": fmt,
    }


def _flat_blocks() -> list[dict[str, Any]]:
    """模拟 MinerU 平铺书：全部标题 text_level=2，父级只有书名。"""
    return [
        _block("b0", "# 中医药临床", heading="中医药临床", level=1),
        _block("b1", "# 第一章 绪论", heading="第一章 绪论", level=2, parents=["中医药临床"]),
        _block("b2", "绪论正文……", heading="第一章 绪论", level=0, parents=["中医药临床"]),
        _block("b3", "# 第一节 概述", heading="第一节 概述", level=2, parents=["中医药临床"]),
        _block("b4", "概述正文……", heading="第一节 概述", level=0, parents=["中医药临床"]),
        _block("b5", "# 第一章 禁忌", heading="第一章 禁忌", level=2, parents=["中医药临床"]),
        _block("b6", "禁忌正文……", heading="第一章 禁忌", level=0, parents=["中医药临床"]),
    ]


def _llm_blocks() -> list[dict[str, Any]]:
    """模拟无编号平铺书（书签/编号两条确定性路径都无法定级 → 必走 LLM）。"""
    return [
        _block("b0", "# 中医药临床", heading="中医药临床", level=1),
        _block("b1", "# 绪论", heading="绪论", level=2, parents=["中医药临床"]),
        _block("b2", "绪论正文……", heading="绪论", level=0, parents=["中医药临床"]),
        _block("b3", "# 概述", heading="概述", level=2, parents=["中医药临床"]),
        _block("b4", "概述正文……", heading="概述", level=0, parents=["中医药临床"]),
        _block("b5", "# 禁忌", heading="禁忌", level=2, parents=["中医药临床"]),
        _block("b6", "禁忌正文……", heading="禁忌", level=0, parents=["中医药临床"]),
    ]


def _llm_entries() -> list[dict[str, Any]]:
    return [
        {"level": 1, "keep": True, "title": "中医药临床"},
        {"level": 2, "keep": True, "title": "绪论"},
        {"level": 3, "keep": True, "title": "概述"},
        {"level": 2, "keep": True, "title": "禁忌"},
    ]


def _flat_entries() -> list[dict[str, Any]]:
    return [
        {"level": 1, "keep": True, "title": "中医药临床"},
        {"level": 2, "keep": True, "title": "第一章 绪论"},
        {"level": 3, "keep": True, "title": "第一节 概述"},
        {"level": 2, "keep": True, "title": "第一章 禁忌"},
    ]


def _entries(raw: list[dict[str, Any]]) -> list[OutlineEntry]:
    return [OutlineEntry(level=e["level"], keep=e["keep"], title=e["title"]) for e in raw]


# ── 候选提取 ─────────────────────────────────────────────────────────────────


def test_extract_candidates_with_page_and_height() -> None:
    blocks = [
        _block("b0", "# 甲", heading="甲", level=1, positions=[{"anchor": 3, "range": [10, 100, 200, 130]}]),
        _block("b1", "正文", heading="甲", level=0),
        _block("b2", "", heading="", level=2),  # 空 heading 跳过
        _block("b3", "# 乙", heading="乙", level=2, positions=[]),
    ]
    candidates = extract_candidates(blocks)
    assert [c["text"] for c in candidates] == ["甲", "乙"]
    assert candidates[0]["page"] == 3
    assert candidates[0]["height"] == 30  # round(130-100)
    assert candidates[1]["page"] is None
    assert candidates[1]["height"] is None


def test_extract_candidates_layout_features() -> None:
    """页级基准：行高比/缩进/居中偏移/宽比 客观计算，基准缺失时 None。"""
    blocks = [
        # 第 1 页：标题块（首行=标题跳过，后续行为正文基准）+ 居中大字标题
        _block(
            "b0", "# 章甲\n正文一\n正文二", heading="章甲", level=1,
            positions=[
                {"anchor": 1, "range": [200, 10, 400, 30]},
                {"anchor": 1, "range": [100, 50, 800, 70]},
                {"anchor": 1, "range": [100, 80, 900, 100]},
            ],
        ),
        _block("b2", "# 第一章 绪论", heading="第一章 绪论", level=2,
               positions=[{"anchor": 1, "range": [350, 200, 550, 240]}]),
        # 第 2 页：标题偏左缩进 + 无正文参照的页
        _block("b3", "# 一、概述", heading="一、概述", level=2,
               positions=[{"anchor": 2, "range": [50, 100, 250, 130]}]),
        _block("b4", "# 附录", heading="附录", level=2,
               positions=[{"anchor": 3, "range": [0, 0, 100, 20]}]),
    ]
    candidates = {c["text"]: c for c in extract_candidates(blocks)}

    chapter = candidates["第一章 绪论"]
    assert chapter["height"] == 40
    assert chapter["height_ratio"] == 2.0  # 40 / 正文行高 20
    assert chapter["indent"] == 250.0  # 350 - 100
    assert chapter["center_offset"] == 0.0  # 中心 450 == 页中线 450
    assert chapter["width_ratio"] == 0.22  # 200 / 900

    sub = candidates["一、概述"]
    # 全书级正文基准（来自第 1 页正文行）：行高比/缩进可用；页宽由标题行自身推断
    assert sub["height_ratio"] == 1.5  # 30 / 20
    assert sub["indent"] == -50.0  # 50 - 100
    assert sub["center_offset"] is not None

    appendix = candidates["附录"]
    # 第 3 页只有标题自身：页宽 = 标题宽 → 宽比 1.0、居中
    assert appendix["height_ratio"] == 1.0  # 20 / 20
    assert appendix["center_offset"] == 0.0
    assert appendix["width_ratio"] == 1.0


def test_hierarchy_needs_repair() -> None:
    flat = _flat_blocks()  # levels {1, 2}
    assert hierarchy_needs_repair(flat)
    healthy = _flat_blocks()
    healthy[3]["level"] = 3  # levels {1, 2, 3} → 已非退化
    assert not hierarchy_needs_repair(healthy)
    no_headings = [_block("b0", "纯正文") for _ in range(3)]
    assert not hierarchy_needs_repair(no_headings)


# ── 确定性重建 ───────────────────────────────────────────────────────────────


def test_apply_outline_rebuilds_hierarchy() -> None:
    blocks = _flat_blocks()
    repaired = apply_outline(blocks, _entries(_flat_entries()))
    by_id = {b["block_id"]: b for b in repaired}
    assert by_id["b1"]["level"] == 2
    assert by_id["b1"]["parent_headings"] == ["中医药临床"]
    # 第一节提升为 3 级：父级含章
    assert by_id["b3"]["level"] == 3
    assert by_id["b3"]["parent_headings"] == ["中医药临床", "第一章 绪论"]
    # 正文块父级同步
    assert by_id["b4"]["heading"] == "第一节 概述"
    assert by_id["b4"]["parent_headings"] == ["中医药临床", "第一章 绪论"]
    # 回到 2 级：栈弹出同级
    assert by_id["b5"]["level"] == 2
    assert by_id["b5"]["parent_headings"] == ["中医药临床"]
    assert by_id["b6"]["heading"] == "第一章 禁忌"
    assert by_id["b6"]["parent_headings"] == ["中医药临床"]


def test_apply_outline_drops_heading_and_strips_prefix() -> None:
    blocks = [
        _block("b0", "# 目 录\n第一章 绪论 … 1\n第一章 禁忌 … 12", heading="目 录", level=2),
        _block("b1", "# 第一章 绪论", heading="第一章 绪论", level=2),
        _block("b2", "正文", heading="第一章 绪论", level=0),
    ]
    entries = _entries([
        {"level": 0, "keep": False, "title": ""},
        {"level": 1, "keep": True, "title": "第一章 绪论"},
    ])
    repaired = apply_outline(blocks, entries)
    by_id = {b["block_id"]: b for b in repaired}
    dropped = by_id["b0"]
    assert dropped["level"] == 0
    assert dropped["heading"] == ""
    assert dropped["parent_headings"] == []
    # 标题行降级为正文：不再以 "#" 开头
    assert not dropped["content"].startswith("#")
    assert dropped["content"].splitlines()[0] == "目 录"
    # 后续标题照常（正文父级 = 栈[:-1]，一级标题下为空）
    assert by_id["b1"]["level"] == 1
    assert by_id["b2"]["heading"] == "第一章 绪论"
    assert by_id["b2"]["parent_headings"] == []


# ── repair_blocks 主入口 ─────────────────────────────────────────────────────


def test_repair_skips_healthy_hierarchy() -> None:
    blocks = _flat_blocks()
    blocks[3]["level"] = 3  # 三档层级 → 健康，跳过 LLM
    client = FakeClient()
    blocks_out, meta = asyncio.run(
        repair_blocks(blocks, client=client, registry=PromptTemplateRegistry())  # type: ignore[arg-type]
    )
    assert meta.source == "healthy-skip"
    assert not client.calls
    assert blocks_out is blocks


def test_repair_skips_when_no_candidates() -> None:
    blocks = [_block(f"b{i}", "正文", heading="", level=0) for i in range(3)]
    client = FakeClient()
    _, meta = asyncio.run(
        repair_blocks(blocks, client=client, registry=PromptTemplateRegistry())  # type: ignore[arg-type]
    )
    assert meta.source == "no-candidates"
    assert not client.calls


def test_repair_payload_carries_layout_context() -> None:
    """LLM 请求载荷 = {layout_context, candidates}，含全书版式基准。"""
    blocks = _llm_blocks()
    # 给正文行加行级 bbox，使基准可计算
    for block in blocks:
        if not block["level"]:
            block["positions"] = [{"anchor": 5, "range": [100, 10, 800, 30]}]
    for i, block in enumerate(blocks):
        if block["level"] and not block["positions"]:
            block["positions"] = [{"anchor": 5, "range": [300, 200, 500, 240]}]

    client = FakeClient()
    client.queue = [{"entries": _llm_entries()}]
    asyncio.run(
        repair_blocks(blocks, client=client, registry=PromptTemplateRegistry())  # type: ignore[arg-type]
    )
    assert len(client.calls) == 1
    # {{outline_input}} 是模板最后一部分：从「待修复标题流」之后提取纯 JSON
    payload = _payload_of(client.calls[0])
    ctx = payload["layout_context"]
    assert ctx["body_height_median"] == 20.0
    assert ctx["body_indent_median"] == 100.0
    assert ctx["page_width_median"] == 800.0
    assert ctx["heading_height_min"] == 40
    assert ctx["heading_height_max"] == 40
    assert payload["candidates"][0]["height_ratio"] == 2.0  # 40 / 20


def test_repair_success_and_cache_idempotency(tmp_path: Path) -> None:
    blocks = _llm_blocks()
    client = FakeClient()
    client.queue = [{"entries": _llm_entries()}]
    repaired, meta = asyncio.run(
        repair_blocks(
            blocks, client=client, registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )
    )
    assert meta.source == "llm"
    assert meta.kept == 4
    assert meta.dropped == 0
    assert meta.levels == {"1": 1, "2": 2, "3": 1}
    assert len(client.calls) == 1
    assert repaired[3]["level"] == 3  # 第二节提升为 3 级
    assert (tmp_path / "outline_repair.json").is_file()

    # 相同 blocks 二次调用：命中缓存，零 LLM 调用，重建结果一致
    client2 = FakeClient()
    repaired2, meta2 = asyncio.run(
        repair_blocks(
            _llm_blocks(), client=client2, registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )
    )
    assert meta2.source == "cache"
    assert not client2.calls
    assert repaired2 == repaired

    # 内容变化（换一本书的 blocks）→ 缓存失效，重新走 LLM
    client3 = FakeClient()
    client3.queue = [{"entries": _llm_entries()}]
    changed = _llm_blocks()
    changed[0]["heading"] = "中药禁忌学"
    changed[0]["content"] = "# 中药禁忌学"
    _, meta3 = asyncio.run(
        repair_blocks(
            changed, client=client3, registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )
    )
    assert meta3.source == "llm"
    assert len(client3.calls) == 1


def test_repair_degrades_on_llm_failure() -> None:
    blocks = _llm_blocks()
    client = FakeClient()  # 空队列 → LLMError
    blocks_out, meta = asyncio.run(
        repair_blocks(blocks, client=client, registry=PromptTemplateRegistry())  # type: ignore[arg-type]
    )
    assert meta.source == "degraded"
    assert blocks_out is blocks  # 原样返回，流水线不中断


def test_repair_degrades_on_invalid_output() -> None:
    blocks = _llm_blocks()
    client = FakeClient()
    client.queue = [{"entries": _llm_entries()[:2]}]  # 条目数不匹配 → 非法
    blocks_out, meta = asyncio.run(
        repair_blocks(blocks, client=client, registry=PromptTemplateRegistry())  # type: ignore[arg-type]
    )
    assert meta.source == "degraded"
    assert blocks_out is blocks


# ── 书签优先 ─────────────────────────────────────────────────────────────────


def _pdf_with_outline(tmp_path: Path) -> Path:
    """生成带内嵌书签大纲的 4 页空白 PDF（pypdf 写、pypdf 读，闭环验证）。"""
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(4):
        writer.add_blank_page(width=595, height=842)
    ch1 = writer.add_outline_item("第一章 绪论", 0)
    writer.add_outline_item("第一节 概述", 1, parent=ch1)
    writer.add_outline_item("第二节 方法", 1, parent=ch1)
    writer.add_outline_item("第二章 禁忌", 2)
    pdf = tmp_path / "outlined.pdf"
    with pdf.open("wb") as fh:
        writer.write(fh)
    return pdf


def _bookmark_blocks() -> list[dict[str, Any]]:
    """模拟平铺书：目录页 + 四个正文标题（层级全为 2），等待书签定级。"""
    return [
        _block("b0", "# 目录\n第一章 绪论 … 1", heading="目录", level=2,
               positions=[{"anchor": 1, "range": [0, 0, 300, 20]}]),
        _block("b1", "# 第一章 绪论", heading="第一章 绪论", level=2,
               positions=[{"anchor": 3, "range": [200, 10, 500, 40]}]),
        _block("b2", "# 第一节 概述", heading="第一节 概述", level=2,
               positions=[{"anchor": 3, "range": [200, 60, 500, 90]}]),
        _block("b3", "# 第二节 方法", heading="第二节 方法", level=2,
               positions=[{"anchor": 4, "range": [200, 10, 500, 40]}]),
        _block("b4", "# 第二章 禁忌", heading="第二章 禁忌", level=2,
               positions=[{"anchor": 5, "range": [200, 10, 500, 40]}]),
    ]


def _full_bookmarks() -> list[dict[str, Any]]:
    return [
        {"level": 1, "title": "第一章 绪论", "page": 3},
        {"level": 2, "title": "第一节 概述", "page": 3},
        {"level": 2, "title": "第二节 方法", "page": 4},
        {"level": 1, "title": "第二章 禁忌", "page": 5},
    ]


def test_load_pdf_outline_flattens_nested_tree(tmp_path: Path) -> None:
    outline = outline_mod.load_pdf_outline(_pdf_with_outline(tmp_path))
    assert [(o["level"], o["title"], o["page"]) for o in outline] == [
        (1, "第一章 绪论", 1),
        (2, "第一节 概述", 2),
        (2, "第二节 方法", 2),
        (1, "第二章 禁忌", 3),
    ]


def test_load_pdf_outline_degrades_gracefully(tmp_path: Path) -> None:
    assert outline_mod.load_pdf_outline(tmp_path / "missing.pdf") == []
    not_pdf = tmp_path / "book.txt"
    not_pdf.write_text("not a pdf", encoding="utf-8")
    assert outline_mod.load_pdf_outline(not_pdf) == []
    corrupt = tmp_path / "broken.pdf"
    corrupt.write_bytes(b"%PDF-1.4 not really")
    assert outline_mod.load_pdf_outline(corrupt) == []


def test_repair_from_bookmarks_assigns_levels_and_drops_frontmatter() -> None:
    candidates = [
        {"text": "目录", "page": 1, "height": 18},
        {"text": "第一章 绪论 …… 1", "page": 2, "height": 18},  # 目录条目带页码点线
        {"text": "第一章 绪论", "page": 3, "height": 38},
        {"text": "第一节 概述", "page": 3, "height": 38},
        {"text": "第二节 方法", "page": 4, "height": 38},
        {"text": "第二章 禁忌", "page": 5, "height": 38},
    ]
    entries = outline_mod._repair_from_bookmarks(candidates, _full_bookmarks())  # noqa: SLF001
    assert entries is not None
    assert [e.keep for e in entries] == [False, False, True, True, True, True]
    assert [e.level for e in entries] == [0, 0, 1, 2, 2, 1]
    assert entries[2].title == "第一章 绪论"  # 保留正文标题原文


def test_repair_from_bookmarks_prefers_exact_page_over_tolerance() -> None:
    """目录条目（差 1 页）与正文标题（同页）竞争同一书签 → 正文标题胜出。"""
    candidates = [
        {"text": "第一章 绪论", "page": 2, "height": 18},  # 目录条目（文本无页码）
        {"text": "第一章 绪论", "page": 3, "height": 38},  # 正文标题
        {"text": "第二章 禁忌", "page": 5, "height": 38},
        {"text": "第三章 剂量", "page": 6, "height": 38},
    ]
    bookmarks = [
        {"level": 1, "title": "第一章 绪论", "page": 3},
        {"level": 1, "title": "第二章 禁忌", "page": 5},
        {"level": 2, "title": "第三章 剂量", "page": 6},
    ]
    entries = outline_mod._repair_from_bookmarks(candidates, bookmarks)  # noqa: SLF001
    assert entries is not None
    assert entries[0].keep is False  # 目录条目降为正文
    assert entries[1].level == 1     # 正文标题拿到书签层级


def test_repair_from_bookmarks_rejects_partial_outlines() -> None:
    """书签没覆盖正文中部的标题（夹层未匹配）→ 放弃书签路径，交 LLM。"""
    candidates = [
        {"text": "第一章 绪论", "page": 3, "height": 38},
        {"text": "第一节 概述", "page": 3, "height": 38},
        {"text": "未收录小节", "page": 4, "height": 30},  # 书签里没有
        {"text": "第二章 禁忌", "page": 5, "height": 38},
    ]
    bookmarks = [
        {"level": 1, "title": "第一章 绪论", "page": 3},
        {"level": 2, "title": "第一节 概述", "page": 3},
        {"level": 1, "title": "第二章 禁忌", "page": 5},
    ]
    assert outline_mod._repair_from_bookmarks(candidates, bookmarks) is None  # noqa: SLF001


def test_repair_uses_bookmarks_without_llm() -> None:
    """书签充分 → 零 LLM 调用，source=bookmarks，层级/父级链确定性重建。"""
    blocks = _bookmark_blocks()
    client = FakeClient()  # 空队列：若误走 LLM 会降级，断言 source 区分两者
    repaired, meta = asyncio.run(
        repair_blocks(
            blocks, client=client, registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
            bookmarks=_full_bookmarks(),
        )
    )
    assert meta.source == "bookmarks"
    assert not client.calls
    by_id = {b["block_id"]: b for b in repaired}
    assert by_id["b1"]["level"] == 1
    assert by_id["b1"]["parent_headings"] == []
    assert by_id["b2"]["level"] == 2
    assert by_id["b2"]["parent_headings"] == ["第一章 绪论"]
    assert by_id["b4"]["level"] == 1
    assert by_id["b4"]["parent_headings"] == []
    assert by_id["b0"]["level"] == 0  # 目录降为正文
    assert not by_id["b0"]["content"].startswith("#")


def test_repair_falls_back_to_llm_when_bookmarks_insufficient() -> None:
    """书签残缺（条数不足）→ 编号也无从定级 → 落入 LLM 修复，不丢功能。"""
    blocks = _llm_blocks()
    client = FakeClient()
    client.queue = [{"entries": _llm_entries()}]
    _, meta = asyncio.run(
        repair_blocks(
            blocks, client=client, registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
            bookmarks=[{"level": 1, "title": "第一章 绪论", "page": 3}],
        )
    )
    assert meta.source == "llm"
    assert len(client.calls) == 1


# ── 编号优先 ─────────────────────────────────────────────────────────────────


def _numbering_blocks() -> list[dict[str, Any]]:
    """模拟中文编号体系的书：一、/（一）/1. 三级编号 + 前置目录页。"""
    return [
        _block("b0", "# 目录", heading="目录", level=2,
               positions=[{"anchor": 1, "range": [0, 0, 300, 20]}]),
        _block("b1", "# 一、总则", heading="一、总则", level=2),
        _block("b2", "总则正文……", heading="一、总则", level=0),
        _block("b3", "# （一）基本原则", heading="（一）基本原则", level=2),
        _block("b4", "原则正文……", heading="（一）基本原则", level=0),
        _block("b5", "# 1. 第一项", heading="1. 第一项", level=2),
        _block("b6", "第一项正文……", heading="1. 第一项", level=0),
        _block("b7", "# 二、分则", heading="二、分则", level=2),
        _block("b8", "分则正文……", heading="二、分则", level=0),
    ]


def test_detect_numbering_schemes() -> None:
    """编号族识别：中文编号各体系，含目录条目、无编号文本。"""
    cases = [
        ("第一篇 总论", ("part", 1)),
        ("第十二章 禁忌", ("chapter", 1)),
        ("第3章 剂量", ("chapter", 1)),
        ("第一节 概述", ("section", 1)),
        ("第十条 附则", ("clause", 1)),
        ("一、总则", ("han", 1)),
        ("二. 分则", ("han", 1)),
        ("（一）基本原则", ("han_paren", 1)),
        ("(二) 细则", ("han_paren", 1)),
        ("1. 第一项", ("arabic", 1)),
        ("1、第二项", ("arabic", 1)),
        ("1.1 子项", ("dot", 2)),
        ("1.1.1 孙项", ("dot", 3)),
        ("1.2．孙子项", ("dot", 2)),
        ("1.1、另一子项", ("dot", 2)),
        ("(1) 甲", ("arabic_paren", 1)),
        ("（2）乙", ("arabic_paren", 1)),
        ("附录A 十八反", ("appendix", 1)),
        ("附录一 药名索引", ("appendix", 1)),
        ("第一章 绪论 …… 12", ("chapter", 1)),  # 目录条目也识别编号，由 tocish 降级
        ("上篇 总论", None),
        ("前言", None),
        ("2024年版 修订说明", None),
        ("1.5g 用量", None),  # 数字后有单位，非编号
        ("本草纲目", None),
    ]
    for text, expected in cases:
        got = outline_mod._detect_numbering(text)  # noqa: SLF001
        assert got == expected, text


def test_repair_from_numbering_compresses_levels() -> None:
    """编号族按本书实际出现的族压缩成连续 1..N（不做跨书绝对层级假设）。"""
    candidates = [
        {"text": "目录", "page": 1, "height": 18},
        {"text": "第一章 绪论", "page": 3, "height": 38},
        {"text": "第一节 概述", "page": 3, "height": 38},
        {"text": "一、发病机理", "page": 4, "height": 38},
        {"text": "第二章 禁忌", "page": 5, "height": 38},
    ]
    entries = outline_mod._repair_from_numbering(candidates)  # noqa: SLF001
    assert entries is not None
    assert [e.keep for e in entries] == [False, True, True, True, True]
    assert [e.level for e in entries] == [0, 1, 2, 3, 1]


def test_repair_from_numbering_dotted_depth_relative() -> None:
    """只写 1.1.x 的书：点分十进制深度相对本书最浅深度归零，不空出第一层。"""
    candidates = [
        {"text": "前言", "page": 1, "height": 18},
        {"text": "1.1 甲", "page": 3, "height": 38},
        {"text": "1.2 乙", "page": 4, "height": 38},
        {"text": "1.2.1 乙一", "page": 4, "height": 38},
        {"text": "1.3 丙", "page": 5, "height": 38},
    ]
    entries = outline_mod._repair_from_numbering(candidates)  # noqa: SLF001
    assert entries is not None
    assert [e.level for e in entries] == [0, 1, 1, 2, 1]


def test_repair_from_numbering_drops_toc_entries() -> None:
    """目录条目（点线+页码后缀）降为正文，正文同款标题保留并定级。"""
    candidates = [
        {"text": "目录", "page": 1, "height": 18},
        {"text": "一、总则 …… 3", "page": 2, "height": 18},  # 目录条目
        {"text": "（一）原则 ····· 3", "page": 2, "height": 18},  # 目录条目
        {"text": "一、总则", "page": 3, "height": 38},
        {"text": "（一）基本原则", "page": 3, "height": 38},
        {"text": "二、分则", "page": 6, "height": 38},
    ]
    entries = outline_mod._repair_from_numbering(candidates)  # noqa: SLF001
    assert entries is not None
    assert [e.keep for e in entries] == [False, False, False, True, True, True]
    assert [e.level for e in entries] == [0, 0, 0, 1, 2, 1]


def test_repair_from_numbering_rejects_interleaved_unnumbered() -> None:
    """首个正文编号之后的未编号候选（体系不完整/正文行误标）→ 交 LLM。"""
    candidates = [
        {"text": "前言", "page": 1, "height": 18},
        {"text": "第一章 绪论", "page": 3, "height": 38},
        {"text": "概述", "page": 4, "height": 30},  # 夹在编号中间的未编号标题
        {"text": "第二章 禁忌", "page": 5, "height": 38},
        {"text": "第三章 剂量", "page": 6, "height": 38},
    ]
    assert outline_mod._repair_from_numbering(candidates) is None  # noqa: SLF001


def test_repair_from_numbering_requires_two_families() -> None:
    """只有一个编号族（无法定层级）或正文编号过少 → 交 LLM。"""
    single_family = [
        {"text": "第一章 绪论", "page": 3, "height": 38},
        {"text": "第二章 禁忌", "page": 5, "height": 38},
        {"text": "第三章 剂量", "page": 6, "height": 38},
        {"text": "第四章 附则", "page": 8, "height": 38},
    ]
    assert outline_mod._repair_from_numbering(single_family) is None  # noqa: SLF001
    too_few = [
        {"text": "第一章 绪论", "page": 3, "height": 38},
        {"text": "第一节 概述", "page": 3, "height": 38},
    ]
    assert outline_mod._repair_from_numbering(too_few) is None  # noqa: SLF001


def test_repair_uses_numbering_without_llm() -> None:
    """编号体系完整 → 零 LLM 调用，source=numbering，层级/父级链确定性重建。"""
    blocks = _numbering_blocks()
    client = FakeClient()  # 空队列：若误走 LLM 会降级，断言 source 区分两者
    repaired, meta = asyncio.run(
        repair_blocks(
            blocks, client=client, registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
        )
    )
    assert meta.source == "numbering"
    assert not client.calls
    by_id = {b["block_id"]: b for b in repaired}
    assert by_id["b0"]["level"] == 0  # 目录降为正文
    assert not by_id["b0"]["content"].startswith("#")
    assert by_id["b1"]["level"] == 1
    assert by_id["b1"]["parent_headings"] == []
    assert by_id["b3"]["level"] == 2
    assert by_id["b3"]["parent_headings"] == ["一、总则"]
    assert by_id["b5"]["level"] == 3
    assert by_id["b5"]["parent_headings"] == ["一、总则", "（一）基本原则"]
    assert by_id["b7"]["level"] == 1
    assert by_id["b7"]["parent_headings"] == []
    # 正文块随新层级重挂：一级标题下的正文父级为空，二级标题下含一级父
    assert by_id["b2"]["heading"] == "一、总则"
    assert by_id["b2"]["parent_headings"] == []
    assert by_id["b4"]["heading"] == "（一）基本原则"
    assert by_id["b4"]["parent_headings"] == ["一、总则"]


# ── 分批调用 ─────────────────────────────────────────────────────────────────


def test_split_candidates_respects_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(outline_mod, "BATCH_TOKEN_BUDGET", 50)
    cands = [
        {"text": "甲" * 40, "page": 1, "height": 30},
        {"text": "乙" * 40, "page": 1, "height": 30},
        {"text": "丙" * 40, "page": 2, "height": 30},
    ]
    batches = outline_mod._split_candidates(cands)  # noqa: SLF001
    # 每条 40+ 字 + 键开销 > 预算 50 → 每条独占一批
    assert [len(b) for b in batches] == [1, 1, 1]

    # 预算放宽后小候选合并为一批
    monkeypatch.setattr(outline_mod, "BATCH_TOKEN_BUDGET", 200)
    small = [{"text": f"第{i}条", "page": 1, "height": 30} for i in range(10)]
    assert len(outline_mod._split_candidates(small)) == 1  # noqa: SLF001


def test_prior_tail_keeps_last_decisions() -> None:
    entries = _entries([
        {"level": 1, "keep": True, "title": "章1"},
        {"level": 2, "keep": True, "title": "节1"},
        {"level": 0, "keep": False, "title": ""},
        {"level": 2, "keep": True, "title": "节2"},
        {"level": 3, "keep": True, "title": "条1"},
        {"level": 3, "keep": True, "title": "条2"},
    ])
    tail = outline_mod._prior_tail(entries)  # noqa: SLF001
    assert len(tail) == 5
    assert tail[0] == {"text": "节1", "level": 2, "keep": True}
    assert tail[-1] == {"text": "条2", "level": 3, "keep": True}
    assert outline_mod._prior_tail([]) == []  # noqa: SLF001


def test_repair_batches_with_prior_tail_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """预算=1 → 每条候选一批；批间载荷携带前批尾部决策保持层级连续。"""
    monkeypatch.setattr(outline_mod, "BATCH_TOKEN_BUDGET", 1)
    blocks = _llm_blocks()
    client = FakeClient()
    client.queue = [
        {"entries": [{"level": 1, "keep": True, "title": "中医药临床"}]},
        {"entries": [{"level": 2, "keep": True, "title": "绪论"}]},
        {"entries": [{"level": 3, "keep": True, "title": "概述"}]},
        {"entries": [{"level": 2, "keep": True, "title": "禁忌"}]},
    ]
    repaired, meta = asyncio.run(
        repair_blocks(blocks, client=client, registry=PromptTemplateRegistry())  # type: ignore[arg-type]
    )
    assert meta.source == "llm"
    assert meta.batches == 4
    assert len(client.calls) == 4

    first = _payload_of(client.calls[0])
    assert first["prior_tail"] == []
    assert [c["text"] for c in first["candidates"]] == ["中医药临床"]

    second = _payload_of(client.calls[1])
    assert second["prior_tail"] == [{"text": "中医药临床", "level": 1, "keep": True}]
    assert [c["text"] for c in second["candidates"]] == ["绪论"]

    by_id = {b["block_id"]: b for b in repaired}
    assert by_id["b3"]["level"] == 3
    assert by_id["b3"]["parent_headings"] == ["中医药临床", "绪论"]


def test_repair_batch_cache_reuses_successful_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """中途失败：已成功批入批级缓存；重试只补未完成批（不重复计费）。"""
    monkeypatch.setattr(outline_mod, "BATCH_TOKEN_BUDGET", 1)
    blocks = _llm_blocks()
    r1, r2, r3, r4 = (
        {"entries": [{"level": 1, "keep": True, "title": "中医药临床"}]},
        {"entries": [{"level": 2, "keep": True, "title": "绪论"}]},
        {"entries": [{"level": 3, "keep": True, "title": "概述"}]},
        {"entries": [{"level": 2, "keep": True, "title": "禁忌"}]},
    )

    # 第一次：第 3 批失败 → 整体降级，但前 2 批已缓存
    client1 = FakeClient()
    client1.queue = [r1, r2, LLMError("api", "第 3 批失败")]
    blocks_out, meta1 = asyncio.run(
        repair_blocks(
            blocks, client=client1, registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )
    )
    assert meta1.source == "degraded"
    assert blocks_out is blocks
    cached = json.loads((tmp_path / "outline_repair.json").read_text(encoding="utf-8"))
    assert len(cached["batches"]) == 2

    # 第二次：前 2 批命中缓存，只调用后 2 批 → 成功
    client2 = FakeClient()
    client2.queue = [r3, r4]
    repaired, meta2 = asyncio.run(
        repair_blocks(
            blocks, client=client2, registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
            cache_dir=tmp_path,
        )
    )
    assert meta2.source == "llm"
    assert meta2.batches == 4
    assert len(client2.calls) == 2
    by_id = {b["block_id"]: b for b in repaired}
    assert by_id["b3"]["level"] == 3
    cached2 = json.loads((tmp_path / "outline_repair.json").read_text(encoding="utf-8"))
    assert len(cached2["batches"]) == 4


# ── blocks.jsonl 重写 ────────────────────────────────────────────────────────


def test_rewrite_blocks_file_preserves_meta(tmp_path: Path) -> None:
    blocks_path = tmp_path / "blocks.jsonl"
    meta = {"type": "meta", "format": "lightrag", "blocks_builder_version": "v-test"}
    with blocks_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for b in _flat_blocks()[:2]:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")

    new_blocks = [_flat_blocks()[0], _flat_blocks()[5]]
    note = {"source": "llm", "candidates": 2}
    rewrite_blocks_file(blocks_path, new_blocks, meta_note=note)

    lines = blocks_path.read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])
    assert head["type"] == "meta"
    assert head["blocks_builder_version"] == "v-test"  # 原 meta 保留
    assert head["outline_repair"] == note  # 审计标记盖上
    assert [json.loads(line)["block_id"] for line in lines[1:]] == ["b0", "b5"]
