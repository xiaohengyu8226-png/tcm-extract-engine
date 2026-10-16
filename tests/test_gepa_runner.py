"""P3 测试：gepa_runner 的切片进化落盘 + 人工晋升（受管标记块 / version bump / 回滚）。

全部用 stub LLM 客户端，零 token、可重复。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tcm_extract_engine.cab.planner import CHUNK_PLACEHOLDER
from tcm_extract_engine.core.artifacts import ArtifactHeader, load_artifact, save_artifact
from tcm_extract_engine.core.llm import LLMUsage
from tcm_extract_engine.gepa.runner import (
    PROMOTE_END,
    PROMOTE_START,
    EvolveResult,
    evolve_template,
    load_run_best_constraint,
    promote_constraint,
)

BOOK_ID = "a1b2c3d4e5f6"


# ── stub 客户端 ──────────────────────────────────────────────────────────────


class _ExtractVerifyStub:
    """按 system_prompt 区分抽取/核验的 stub。"""

    def __init__(self, model: str) -> None:
        self.model = model
        self.usage = LLMUsage()
        self._cycle = ["SUPPORTED", "UNSUPPORTED", "WRONG_DOSE"]

    async def complete_json(
        self, *, system_prompt: str, user_prompt: str, semaphore: Any = None, **_: Any
    ) -> dict[str, Any]:
        self.usage.calls += 1
        self.usage.input_tokens += 10
        self.usage.output_tokens += 5
        if "抽取引擎" in system_prompt:
            return {"records": [
                {"record_type": "GENERAL_RULE", "subject_text": "附子",
                 "raw_text": "孕妇禁用附子，先煎三十分钟", "action": "PROHIBIT"}
                for _ in range(3)
            ]}
        import re
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
        return {"constraint_block": self._block, "rationale": "针对高频错误"}

    async def close(self) -> None:
        return None


# ── 合成书产物 ───────────────────────────────────────────────────────────────


def _make_book(books_root: Path, book_id: str = BOOK_ID) -> Path:
    book_root = books_root / book_id
    (book_root / "01_parse").mkdir(parents=True, exist_ok=True)
    (book_root / "03_plan").mkdir(parents=True, exist_ok=True)

    lines = [json.dumps({"type": "meta", "total": 2})]
    for i in (1, 2):
        lines.append(json.dumps({
            "type": "content", "blockid": f"b{i}", "heading": "妊娠用药禁忌",
            "content": f"孕妇禁用附子第{i}条，先煎三十分钟。",
            "positions": [{"anchor": "10"}],
        }, ensure_ascii=False))
    # 一个无关块（不应进切片）
    lines.append(json.dumps({
        "type": "content", "blockid": "zz", "heading": "其他", "content": "无关内容",
        "positions": [{"anchor": "11"}],
    }, ensure_ascii=False))
    (book_root / "01_parse/blocks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    section = {
        "section_id": "s1", "title": "妊娠用药禁忌", "heading_path": ["妊娠用药禁忌"],
        "extractor": "GeneralRuleExtractor", "template": "general_rule.md",
        "prompt_version": "base", "block_ids": ["b1", "b2"],
        "page_start": 10, "page_end": 10, "skip_reason": None,
    }
    other = {
        "section_id": "s2", "title": "跳过章", "heading_path": ["跳过"],
        "extractor": "", "template": "", "prompt_version": "",
        "block_ids": ["zz"], "page_start": 11, "page_end": 11, "skip_reason": "无模板",
    }
    save_artifact(
        book_root / "03_plan/extraction_plan.json",
        header=ArtifactHeader(schema_version="1", book_id=book_id, task_id=f"{book_id}:plan"),
        data={"goal": "测试", "document_type": "HANDBOOK", "sections": [section, other]},
    )
    return book_root


def _make_prompts(root: Path) -> Path:
    a_dir = root / "a"
    a_dir.mkdir(parents=True, exist_ok=True)
    base = f"# 抽取模板\n\n## 忠实性\n禁止推理。\n\n## 待抽取文本\n\n{CHUNK_PLACEHOLDER}\n"
    (a_dir / "general_rule.md").write_text(base, encoding="utf-8")
    return a_dir / "general_rule.md"


# ── evolve_template ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evolve_template_切片进化并落盘(tmp_path: Path) -> None:
    books_root = tmp_path / "books"
    _make_book(books_root)
    out_root = tmp_path / "gepa"

    result = await evolve_template(
        book_id=BOOK_ID,
        template="general_rule.md",
        books_root=books_root,
        out_root=out_root,
        budget=4,
        population=2,
        generations=2,
        verify_sample=20,
        extract_client=_ExtractVerifyStub("extract-stub"),
        verify_client=_ExtractVerifyStub("verify-stub"),
        reflect_client=_ReflectStub("## 进化约束\n剂量必须带单位。"),
    )

    assert isinstance(result, EvolveResult)
    assert result.book_id == BOOK_ID
    assert result.report_path.is_file()
    assert result.best_constraint_path.is_file()
    assert result.eval_calls <= 4
    assert result.baseline_fitness is not None
    assert 0.0 <= result.best_fitness <= 1.0

    # 切片只保留目标模板章节的块
    sliced_blocks = (result.run_dir / "slice/blocks.jsonl").read_text(encoding="utf-8")
    assert "b1" in sliced_blocks and "b2" in sliced_blocks
    assert "zz" not in sliced_blocks  # 跳过章的块被剔除

    # run.json 可读且带 header
    run = load_artifact(result.report_path)
    assert run["header"]["book_id"] == BOOK_ID
    assert run["template"] == "general_rule.md"
    assert "fitness_history" in run and "tokens" in run


@pytest.mark.asyncio
async def test_evolve_template_无匹配章节报错(tmp_path: Path) -> None:
    books_root = tmp_path / "books"
    _make_book(books_root)
    with pytest.raises(ValueError, match="无使用模板"):
        await evolve_template(
            book_id=BOOK_ID,
            template="herb_table.md",  # 该章节用的是 general_rule.md
            books_root=books_root,
            out_root=tmp_path / "gepa",
            extract_client=_ExtractVerifyStub("e"),
            verify_client=_ExtractVerifyStub("v"),
            reflect_client=_ReflectStub("x"),
        )


# ── load_run_best_constraint ────────────────────────────────────────────────


def test_load_run_best_constraint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    save_artifact(
        run_dir / "run.json",
        header=ArtifactHeader(schema_version="1", book_id=BOOK_ID, task_id="t"),
        data={"template": "general_rule.md", "best_constraint": "禁止臆测。"},
    )
    (run_dir / "best_constraint.md").write_text("禁止臆测剂量。\n", encoding="utf-8")
    block, template = load_run_best_constraint(run_dir)
    assert template == "general_rule.md"
    assert block == "禁止臆测剂量。"  # 优先读 best_constraint.md


# ── promote_constraint ──────────────────────────────────────────────────────


class TestPromote:
    def test_晋升写入受管块并bump版本(self, tmp_path: Path) -> None:
        path = _make_prompts(tmp_path)
        base_content = path.read_text(encoding="utf-8")
        res = promote_constraint(
            template="general_rule.md",
            constraint_block="## 进化约束\n禁止臆测剂量。",
            reviewer="alice",
            run_id="r1",
            prompts_root=tmp_path,
        )
        assert res.changed and res.version_bumped and not res.dry_run
        new_content = path.read_text(encoding="utf-8")
        assert PROMOTE_START in new_content and PROMOTE_END in new_content
        assert "禁止臆测剂量。" in new_content
        assert base_content.rstrip() in new_content  # 正文保留
        assert CHUNK_PLACEHOLDER in new_content       # 占位符完好
        assert res.new_version != res.old_version

    def test_重复晋升替换而非叠加(self, tmp_path: Path) -> None:
        path = _make_prompts(tmp_path)
        promote_constraint(template="general_rule.md", constraint_block="约束A",
                           run_id="r1", prompts_root=tmp_path)
        promote_constraint(template="general_rule.md", constraint_block="约束B",
                           run_id="r2", prompts_root=tmp_path)
        content = path.read_text(encoding="utf-8")
        assert content.count(PROMOTE_START) == 1
        assert "约束A" not in content and "约束B" in content

    def test_dry_run_不落盘(self, tmp_path: Path) -> None:
        path = _make_prompts(tmp_path)
        before = path.read_text(encoding="utf-8")
        res = promote_constraint(template="general_rule.md", constraint_block="约束X",
                                 prompts_root=tmp_path, dry_run=True)
        assert res.dry_run and res.changed and res.version_bumped
        assert path.read_text(encoding="utf-8") == before  # 未写盘

    def test_空约束块回滚到基线(self, tmp_path: Path) -> None:
        path = _make_prompts(tmp_path)
        base_content = path.read_text(encoding="utf-8")
        promote_constraint(template="general_rule.md", constraint_block="约束A",
                           run_id="r1", prompts_root=tmp_path)
        res = promote_constraint(template="general_rule.md", constraint_block="",
                                 prompts_root=tmp_path)
        content = path.read_text(encoding="utf-8")
        assert PROMOTE_START not in content and PROMOTE_END not in content
        assert "约束A" not in content
        assert content.rstrip() == base_content.rstrip()  # 回到基线正文
        assert res.version_bumped  # 相对上一次晋升，版本又变了（回到基线 hash）

    def test_模板不存在报错(self, tmp_path: Path) -> None:
        _make_prompts(tmp_path)
        with pytest.raises(FileNotFoundError):
            promote_constraint(template="nope.md", constraint_block="x",
                               prompts_root=tmp_path)


@pytest.mark.asyncio
async def test_evolution_rejects_partition_overlap_before_model_calls(tmp_path: Path) -> None:
    books = tmp_path/'books'
    _make_book(books)
    with pytest.raises(ValueError, match='不能重叠'):
        await evolve_template(book_id=BOOK_ID, template='general_rule.md', books_root=books,
            out_root=tmp_path/'out', section_ids=['s1'], development_section_ids=['s1'])


@pytest.mark.asyncio
async def test_default_optimization_excludes_reserved_sections(tmp_path: Path) -> None:
    books = tmp_path/'books'
    _make_book(books)
    # s1 is the only eligible section. Reserving it leaves no optimization data.
    with pytest.raises(ValueError, match='无使用模板'):
        await evolve_template(book_id=BOOK_ID, template='general_rule.md', books_root=books,
            out_root=tmp_path/'out', holdout_section_ids=['s1'])
