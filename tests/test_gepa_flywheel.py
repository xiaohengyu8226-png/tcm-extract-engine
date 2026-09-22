"""P4 测试：外圈飞轮（跨书 Pareto 进化 + 聚合评估）+ 质量趋势台账。

全部用 stub LLM 客户端，零 token、可重复。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tcm_extract_engine.core.artifacts import ArtifactHeader, load_artifact, save_artifact
from tcm_extract_engine.core.llm import LLMUsage
from tcm_extract_engine.gepa.engine import PromptIndividual
from tcm_extract_engine.gepa.flywheel import (
    AllTemplatesOutcome,
    evolve_across_books,
    evolve_all_templates,
    make_flywheel_evaluate_fn,
    plan_book_slices,
)
from tcm_extract_engine.gepa.runner import (
    TrendRecord,
    append_trend,
    ledger_path_for,
    load_trend,
)


class _Stub:
    """抽取/核验二合一 stub（按 system_prompt 分支）。"""

    def __init__(self, model: str) -> None:
        self.model = model
        self.usage = LLMUsage()
        self._cycle = ["SUPPORTED", "UNSUPPORTED", "WRONG_DOSE"]

    async def complete_json(
        self, *, system_prompt: str, user_prompt: str, semaphore: Any = None, **_: Any
    ) -> dict[str, Any]:
        self.usage.calls += 1
        self.usage.input_tokens += 8
        self.usage.output_tokens += 4
        if "抽取引擎" in system_prompt:
            return {"records": [
                {"record_type": "GENERAL_RULE", "subject_text": "附子",
                 "raw_text": "孕妇禁用附子，先煎三十分钟", "action": "PROHIBIT"}
                for _ in range(3)
            ]}
        ids = re.findall(r'"record_id":\s*"(SR-[^"]+)"', user_prompt)
        out = []
        for i, rid in enumerate(ids):
            verdict = self._cycle[i % len(self._cycle)]
            item: dict[str, Any] = {"record_id": rid, "verdict": verdict}
            if verdict != "SUPPORTED":
                item["issues"] = [{"field": "action", "problem": "与原文不符"}]
                item["needs_human_review"] = True
            out.append(item)
        return {"records": out}

    async def close(self) -> None:
        return None


class _ReflectStub:
    def __init__(self, block: str) -> None:
        self._block = block
        self.usage = LLMUsage()
        self.model = "reflect-stub"

    async def complete_json(self, *, system_prompt: str, user_prompt: str, **_: Any) -> dict[str, Any]:
        self.usage.calls += 1
        return {"constraint_block": self._block, "rationale": "跨书通用约束"}

    async def close(self) -> None:
        return None


def _make_book(books_root: Path, book_id: str, *, block_prefix: str) -> None:
    book_root = books_root / book_id
    (book_root / "01_parse").mkdir(parents=True, exist_ok=True)
    (book_root / "03_plan").mkdir(parents=True, exist_ok=True)
    b1, b2 = f"{block_prefix}1", f"{block_prefix}2"
    lines = [json.dumps({"type": "meta", "total": 2})]
    for bid, i in ((b1, 1), (b2, 2)):
        lines.append(json.dumps({
            "type": "content", "blockid": bid, "heading": "妊娠用药禁忌",
            "content": f"孕妇禁用附子第{i}条，先煎三十分钟。",
            "positions": [{"anchor": "10"}],
        }, ensure_ascii=False))
    (book_root / "01_parse/blocks.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    section = {
        "section_id": "s1", "title": "妊娠用药禁忌", "heading_path": ["妊娠用药禁忌"],
        "extractor": "GeneralRuleExtractor", "template": "general_rule.md",
        "prompt_version": "base", "block_ids": [b1, b2],
        "page_start": 10, "page_end": 10, "skip_reason": None,
    }
    save_artifact(
        book_root / "03_plan/extraction_plan.json",
        header=ArtifactHeader(schema_version="1", book_id=book_id, task_id=f"{book_id}:plan"),
        data={"goal": "测试", "document_type": "HANDBOOK", "sections": [section]},
    )


def _two_books(books_root: Path) -> list[str]:
    ids = ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
    _make_book(books_root, ids[0], block_prefix="a")
    _make_book(books_root, ids[1], block_prefix="b")
    return ids


# ── 切片 ────────────────────────────────────────────────────────────────────


def test_plan_book_slices_多书切片(tmp_path: Path) -> None:
    books_root = tmp_path / "books"
    ids = _two_books(books_root)
    slices = plan_book_slices(
        books_root=books_root, book_ids=ids, template="general_rule.md",
        out_dir=tmp_path / "slices",
    )
    assert len(slices) == 2
    assert {s.book_id for s in slices} == set(ids)
    for s in slices:
        assert s.blocks_path.is_file() and s.plan_path.is_file()


def test_plan_book_slices_跳过缺失书(tmp_path: Path) -> None:
    books_root = tmp_path / "books"
    ids = _two_books(books_root)
    slices = plan_book_slices(
        books_root=books_root, book_ids=[*ids, "cccccccccccc"],
        template="general_rule.md", out_dir=tmp_path / "slices",
    )
    assert len(slices) == 2  # 不存在的书被跳过


# ── 聚合评估 ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_flywheel_evaluate_fn_每本书一个分(tmp_path: Path) -> None:
    books_root = tmp_path / "books"
    ids = _two_books(books_root)
    slices = plan_book_slices(
        books_root=books_root, book_ids=ids, template="general_rule.md",
        out_dir=tmp_path / "slices",
    )
    evaluate_fn = make_flywheel_evaluate_fn(
        slices=slices, template="general_rule.md",
        extract_client=_Stub("e"), verify_client=_Stub("v"),
        work_dir_root=tmp_path / "cands", verify_sample=20,
    )
    seed = PromptIndividual(prompt_id="seed", extraction_prompt="", verification_prompt="")
    fr = await evaluate_fn(seed)
    assert set(fr.per_instance_scores) == set(ids)  # 每本书一个实例分
    assert all(0.0 <= v <= 1.0 for v in fr.per_instance_scores.values())
    assert 0.0 <= fr.fitness <= 1.0
    assert "aaaaaaaaaaaa" in fr.feedback and "bbbbbbbbbbbb" in fr.feedback


# ── 跨书进化 + 台账 ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evolve_across_books_落盘并记台账(tmp_path: Path) -> None:
    books_root = tmp_path / "books"
    ids = _two_books(books_root)
    out_root = tmp_path / "gepa"

    result = await evolve_across_books(
        book_ids=ids,
        template="general_rule.md",
        books_root=books_root,
        out_root=out_root,
        budget=4,
        population=2,
        generations=2,
        verify_sample=20,
        extract_client=_Stub("e"),
        verify_client=_Stub("v"),
        reflect_client=_ReflectStub("## 进化约束\n禁止语义升级。"),
    )

    assert set(result.book_id.split(",")) == set(ids)
    assert result.report_path.is_file() and result.best_constraint_path.is_file()
    assert result.baseline_fitness is not None
    assert result.eval_calls <= 4

    run = load_artifact(result.report_path)
    assert run["mode"] == "flywheel"
    assert set(run["book_ids"]) == set(ids)

    # 台账写入了一条 evolve 记录，含两本书
    records = load_trend(ledger_path_for(out_root))
    assert len(records) == 1
    assert records[0].kind == "evolve"
    assert set(records[0].book_ids) == set(ids)
    assert records[0].run_id == result.run_id


# ── 台账读写 ────────────────────────────────────────────────────────────────


def test_append_and_load_trend_容忍坏行(tmp_path: Path) -> None:
    ledger = ledger_path_for(tmp_path)
    append_trend(ledger, TrendRecord(ts="t1", kind="evolve", template="general_rule.md",
                                     book_ids=["x"], best_fitness=0.8, improved=True))
    append_trend(ledger, TrendRecord(ts="t2", kind="promote", template="general_rule.md",
                                     promoted=True, old_version="a", new_version="b"))
    # 追加一行坏数据
    with ledger.open("a", encoding="utf-8") as f:
        f.write("{不是合法json}\n")
    records = load_trend(ledger)
    assert len(records) == 2  # 坏行被跳过
    assert records[0].kind == "evolve" and records[1].kind == "promote"
    assert records[1].new_version == "b"


def test_load_trend_空文件(tmp_path: Path) -> None:
    assert load_trend(ledger_path_for(tmp_path)) == []


# ── P5: system-aware 合并 + 4 模板规模化批处理 ─────────────────────────────


@pytest.mark.asyncio
async def test_evolve_across_books_use_merge_可跑通(tmp_path: Path) -> None:
    books_root = tmp_path / "books"
    ids = _two_books(books_root)
    out_root = tmp_path / "gepa"

    result = await evolve_across_books(
        book_ids=ids,
        template="general_rule.md",
        books_root=books_root,
        out_root=out_root,
        budget=4,
        population=2,
        generations=2,
        use_merge=True,               # 启用 system-aware 合并（crossover_rate 默认 0.5）
        extract_client=_Stub("e"),
        verify_client=_Stub("v"),
        reflect_client=_ReflectStub("## 合并约束\n兼顾两书：剂量带单位且禁止语义升级。"),
    )
    run = load_artifact(result.report_path)
    assert run["header"]["extra"]["use_merge"] is True
    assert run["header"]["extra"]["crossover_rate"] == 0.5
    assert result.report_path.is_file()


@pytest.mark.asyncio
async def test_evolve_all_templates_汇总与跳过(tmp_path: Path) -> None:
    books_root = tmp_path / "books"
    ids = _two_books(books_root)
    out_root = tmp_path / "gepa"
    seen: list[tuple[str, bool]] = []

    def on_template(tpl: str, res: Any, skip: str | None) -> None:
        seen.append((tpl, res is not None))

    outcome = await evolve_all_templates(
        book_ids=ids,
        books_root=books_root,
        out_root=out_root,
        templates=["general_rule.md", "herb_table.md"],  # 后者书上无匹配章节
        budget=4,
        population=2,
        generations=2,
        extract_client=_Stub("e"),
        verify_client=_Stub("v"),
        reflect_client=_ReflectStub("## 进化约束\n禁止语义升级。"),
        on_template=on_template,
    )
    assert isinstance(outcome, AllTemplatesOutcome)
    assert [r.template for r in outcome.results] == ["general_rule.md"]
    assert "herb_table.md" in outcome.skipped  # 无匹配章节 → 跳过而非报错中断
    assert seen == [("general_rule.md", True), ("herb_table.md", False)]
    # 台账记录了成功模板的一条 evolve
    records = load_trend(ledger_path_for(out_root))
    assert len(records) == 1
    assert records[0].template == "general_rule.md"
