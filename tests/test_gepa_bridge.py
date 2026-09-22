"""P0 接线冒烟测试：gepa_bridge 的约束块注入、issue 映射、真实评估闭环。

全部用 stub LLM 客户端，零 token 成本、可重复。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tcm_extract_engine.cab.planner import CHUNK_PLACEHOLDER, PromptTemplateRegistry
from tcm_extract_engine.cab.validator import (
    IssueCategory,
    Severity,
    ValidationIssue,
    ValidationReport,
)
from tcm_extract_engine.cab.verifier import (
    CorrectionEntry,
    Verdict,
    VerificationReport,
    VerificationResult,
)
from tcm_extract_engine.core.artifacts import ArtifactHeader, save_artifact
from tcm_extract_engine.core.llm import LLMUsage
from tcm_extract_engine.core.records import SourceRecord
from tcm_extract_engine.gepa.bridge import (
    ConstrainedTemplateRegistry,
    EvalOutcome,
    build_feedback_text,
    evaluate_candidate,
    fold_outcome_to_fitness,
    make_evaluate_fn,
    make_reflective_proposer,
    make_system_aware_merger,
    verdict_to_issues,
)
from tcm_extract_engine.gepa.engine import (
    FitnessResult,
    GEPAConfig,
    GEPAEngine,
    IssueStats,
    PromptIndividual,
)

# ── 约束块注入 registry ──────────────────────────────────────────────────────


class TestConstrainedTemplateRegistry:
    def test_追加约束块改变版本但保留正文与占位符(self) -> None:
        base = PromptTemplateRegistry().load("a", "general_rule.md")
        constrained = ConstrainedTemplateRegistry(
            "general_rule.md", "## 进化约束\n禁止添加原文不存在的信息。"
        ).load("a", "general_rule.md")

        assert constrained.version != base.version
        assert CHUNK_PLACEHOLDER in constrained.content
        assert base.content.rstrip() in constrained.content  # 正文逐字不变
        assert "禁止添加原文不存在的信息。" in constrained.content

    def test_只影响目标模板(self) -> None:
        registry = ConstrainedTemplateRegistry("general_rule.md", "## 进化约束\nXXX")
        plain = PromptTemplateRegistry()
        other = registry.load("a", "herb_table.md")
        assert other.content == plain.load("a", "herb_table.md").content
        assert other.version == plain.load("a", "herb_table.md").version

    def test_空约束块等同基线(self) -> None:
        plain = PromptTemplateRegistry().load("a", "general_rule.md")
        empty = ConstrainedTemplateRegistry("general_rule.md", "  ").load(
            "a", "general_rule.md"
        )
        assert empty.version == plain.version


# ── verdict / validation → issues 映射 ──────────────────────────────────────


def _rec(rid: str) -> SourceRecord:
    return SourceRecord(
        record_id=rid, book_id="b" * 12, record_type="GENERAL_RULE", raw_text="原文"
    )


class TestVerdictToIssues:
    def test_verdict映射到issue类型与严重度(self) -> None:
        records = [_rec("SR-1"), _rec("SR-2"), _rec("SR-3")]
        verification = VerificationReport(total=3, results=[
            VerificationResult(record_id="SR-1", verdict=Verdict.SUPPORTED),
            VerificationResult(record_id="SR-2", verdict=Verdict.UNSUPPORTED),
            VerificationResult(record_id="SR-3", verdict=Verdict.WRONG_DOSE),
        ])
        issues = verdict_to_issues(records, verification, ValidationReport(total_records=3))
        types = {i["issue_type"] for i in issues}
        assert types == {"unsupported", "dosage_unit"}
        dose = next(i for i in issues if i["issue_type"] == "dosage_unit")
        assert dose["severity"] == "critical"
        assert dose["json_pointer"] == "/records/2"
        # SUPPORTED 不产问题
        assert all(i["json_pointer"] != "/records/0" for i in issues)

    def test_partial不计入问题(self) -> None:
        records = [_rec("SR-1")]
        verification = VerificationReport(total=1, results=[
            VerificationResult(record_id="SR-1", verdict=Verdict.PARTIAL),
        ])
        assert verdict_to_issues(records, verification, ValidationReport(total_records=1)) == []

    def test_校验error计入_warning忽略(self) -> None:
        records = [_rec("SR-1"), _rec("SR-2")]
        validation = ValidationReport(total_records=2, issues=[
            ValidationIssue(record_id="SR-1", category=IssueCategory.SOURCE,
                            severity=Severity.ERROR, message="缺原文"),
            ValidationIssue(record_id="SR-2", category=IssueCategory.COVERAGE,
                            severity=Severity.WARNING, message="覆盖率提示"),
        ])
        issues = verdict_to_issues(records, VerificationReport(total=2), validation)
        assert len(issues) == 1
        assert issues[0]["issue_type"] == "unsupported"
        assert issues[0]["json_pointer"] == "/records/0"


class TestFeedbackText:
    def test_反馈含verdict与修正建议(self) -> None:
        records = [_rec("SR-1")]
        verification = VerificationReport(
            total=1,
            results=[VerificationResult(
                record_id="SR-1", verdict=Verdict.UNSUPPORTED, explanation="action 无原文依据"
            )],
            corrections=[CorrectionEntry(
                record_id="SR-1", field="action", original="PROHIBIT",
                suggested="CAUTION", reason="原文为慎用", verdict=Verdict.UNSUPPORTED,
            )],
        )
        text = build_feedback_text(records, verification, ValidationReport(total_records=1))
        assert "UNSUPPORTED" in text
        assert "action 无原文依据" in text
        assert "修正建议" in text


# ── 真实评估闭环（stub client）──────────────────────────────────────────────


class _StubClient:
    """按 system_prompt 区分抽取/核验的 stub 客户端。"""

    def __init__(self, model: str, *, records_per_call: int = 3) -> None:
        self.model = model
        self.usage = LLMUsage()
        self._records_per_call = records_per_call
        self._verdict_cycle = ["SUPPORTED", "UNSUPPORTED", "WRONG_DOSE"]

    async def complete_json(
        self, *, system_prompt: str, user_prompt: str, semaphore: Any = None, **_: Any
    ) -> dict[str, Any]:
        self.usage.calls += 1
        self.usage.input_tokens += 10
        self.usage.output_tokens += 5
        if "抽取引擎" in system_prompt:
            return {"records": [
                {
                    "record_type": "GENERAL_RULE",
                    "subject_text": "附子",
                    "raw_text": "孕妇禁用附子，先煎三十分钟",
                    "action": "PROHIBIT",
                }
                for _ in range(self._records_per_call)
            ]}
        # 核验：从 user_prompt 回捞 record_id，循环给 verdict
        ids = re.findall(r'"record_id":\s*"(SR-[^"]+)"', user_prompt)
        out = []
        for i, rid in enumerate(ids):
            verdict = self._verdict_cycle[i % len(self._verdict_cycle)]
            item: dict[str, Any] = {"record_id": rid, "verdict": verdict}
            if verdict != "SUPPORTED":
                item["issues"] = [{"field": "action", "problem": "与原文不符"}]
                item["needs_human_review"] = True
            out.append(item)
        return {"records": out}


def _write_blocks(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"type": "meta", "total": 2})]
    for i in (1, 2):
        lines.append(json.dumps({
            "type": "content", "blockid": f"b{i}", "heading": "妊娠用药禁忌",
            "content": f"孕妇禁用附子第{i}条，先煎三十分钟。",
            "positions": [{"anchor": "10"}],
        }, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plan(path: Path, book_id: str) -> None:
    section = {
        "section_id": "s1", "title": "妊娠用药禁忌",
        "heading_path": ["妊娠用药禁忌"], "extractor": "GeneralRuleExtractor",
        "template": "general_rule.md", "prompt_version": "base",
        "block_ids": ["b1", "b2"], "page_start": 10, "page_end": 10,
    }
    save_artifact(
        path,
        header=ArtifactHeader(schema_version="1", book_id=book_id, task_id=f"{book_id}:plan"),
        data={"goal": "测试", "document_type": "HANDBOOK", "sections": [section]},
    )


@pytest.mark.asyncio
async def test_evaluate_candidate_闭环产出正确评估(tmp_path: Path) -> None:
    book_id = "a1b2c3d4e5f6"
    blocks_path = tmp_path / "blocks.jsonl"
    plan_path = tmp_path / "extraction_plan.json"
    _write_blocks(blocks_path)
    _write_plan(plan_path, book_id)

    outcome = await evaluate_candidate(
        book_id=book_id,
        blocks_path=blocks_path,
        plan_path=plan_path,
        target_template="general_rule.md",
        constraint_block="## 进化约束\n禁止添加原文不存在的信息。",
        extract_client=_StubClient("extract-stub"),
        verify_client=_StubClient("verify-stub"),
        work_dir=tmp_path / "cand",
    )

    assert isinstance(outcome, EvalOutcome)
    assert outcome.total_records == 3
    assert outcome.records_in_scope == 3
    # 3 条循环 SUPPORTED/UNSUPPORTED/WRONG_DOSE → 2 条有问题 → fitness < 1
    assert 0.0 < outcome.fitness.fitness < 1.0
    assert outcome.fitness.issue_stats.unsupported_count == 1
    assert outcome.fitness.issue_stats.dosage_unit_count == 1
    assert outcome.per_instance_scores == {book_id: outcome.fitness.fitness}
    assert "UNSUPPORTED" in outcome.feedback
    assert outcome.llm_calls >= 2


@pytest.mark.asyncio
async def test_evaluate_candidate_基线空约束块可跑(tmp_path: Path) -> None:
    book_id = "0f1e2d3c4b5a"
    blocks_path = tmp_path / "blocks.jsonl"
    plan_path = tmp_path / "extraction_plan.json"
    _write_blocks(blocks_path)
    _write_plan(plan_path, book_id)

    outcome = await evaluate_candidate(
        book_id=book_id,
        blocks_path=blocks_path,
        plan_path=plan_path,
        target_template="general_rule.md",
        constraint_block="",  # 基线
        extract_client=_StubClient("extract-stub"),
        verify_client=_StubClient("verify-stub"),
        work_dir=tmp_path / "baseline",
    )
    assert outcome.records_in_scope == 3
    assert 0.0 <= outcome.fitness.fitness <= 1.0


# ── P2: 折叠 / 反思变异 / 端到端小预算进化（stub，零 token）──────────────────


def _fit(score: float, **kw: Any) -> FitnessResult:
    return FitnessResult(
        fitness=score, pass_rate=score, penalty_rate=1.0 - score,
        missing_rate=0.0, hallucination_rate=0.0, issue_stats=IssueStats(), **kw,
    )


class _ReflectStub:
    """反思模块 stub：complete_json 返回约束块；可模拟异常。"""

    def __init__(self, block: str, *, rationale: str = "针对高频错误", boom: bool = False) -> None:
        self._block = block
        self._rationale = rationale
        self._boom = boom
        self.calls = 0

    async def complete_json(self, *, system_prompt: str, user_prompt: str, **_: Any) -> dict[str, Any]:
        self.calls += 1
        if self._boom:
            raise RuntimeError("反思服务不可用")
        return {"constraint_block": self._block, "rationale": self._rationale}


class _NullGateway:
    """注入 evaluate_fn 时不应被触碰的网关。"""

    async def generate(self, **_: Any) -> Any:  # pragma: no cover
        raise AssertionError("evaluate_fn 注入时不应调用网关")


def test_fold_outcome_补齐feedback与每实例分() -> None:
    outcome = EvalOutcome(
        fitness=_fit(0.7), feedback="FB", per_instance_scores={"bk": 0.7}, total_records=1,
    )
    folded = fold_outcome_to_fitness(outcome)
    assert folded.fitness == 0.7
    assert folded.feedback == "FB"
    assert folded.per_instance_scores == {"bk": 0.7}


class TestReflectiveProposer:
    @pytest.mark.asyncio
    async def test_反思产出新约束块(self) -> None:
        parent = PromptIndividual(prompt_id="p0", extraction_prompt="", verification_prompt="")
        fitness = _fit(0.5, feedback="[UNSUPPORTED] SR-1：action 无依据",
                       top_issues=["unsupported"])
        proposer = make_reflective_proposer(
            llm_client=_ReflectStub("## 进化约束\n禁止编造原文不存在的剂量。"),
            target_template="general_rule.md",
        )
        child = await proposer(parent, fitness)
        assert "禁止编造原文不存在的剂量" in child.extraction_prompt
        assert child.generation == 1
        assert child.parent_ids == ("p0",)
        assert child.mutation_applied.startswith("reflect")

    @pytest.mark.asyncio
    async def test_反思清洗围栏与占位符(self) -> None:
        parent = PromptIndividual(prompt_id="p0", extraction_prompt="", verification_prompt="")
        proposer = make_reflective_proposer(
            llm_client=_ReflectStub("```md\n## 约束\n{{chunk_json}}\n禁止臆测\n```"),
            target_template="general_rule.md",
        )
        child = await proposer(parent, _fit(0.4))
        assert CHUNK_PLACEHOLDER not in child.extraction_prompt
        assert "```" not in child.extraction_prompt
        assert "禁止臆测" in child.extraction_prompt

    @pytest.mark.asyncio
    async def test_反思失败回退静态变异(self) -> None:
        parent = PromptIndividual(prompt_id="p0", extraction_prompt="", verification_prompt="")
        proposer = make_reflective_proposer(
            llm_client=_ReflectStub("", boom=True), target_template="general_rule.md",
        )
        child = await proposer(parent, _fit(0.4))
        assert not child.mutation_applied.startswith("reflect")  # 走了 mutate_prompt

    @pytest.mark.asyncio
    async def test_反思无新意回退静态变异(self) -> None:
        parent = PromptIndividual(prompt_id="p0", extraction_prompt="SAME", verification_prompt="")
        proposer = make_reflective_proposer(
            llm_client=_ReflectStub("SAME"), target_template="general_rule.md",
        )
        child = await proposer(parent, _fit(0.4))
        assert not child.mutation_applied.startswith("reflect")


@pytest.mark.asyncio
async def test_端到端_单模板小预算_pareto反思进化(tmp_path: Path) -> None:
    """GEPAEngine + make_evaluate_fn + make_reflective_proposer + pareto，全 stub 跑通。"""
    book_id = "1a2b3c4d5e6f"
    blocks_path = tmp_path / "blocks.jsonl"
    plan_path = tmp_path / "extraction_plan.json"
    _write_blocks(blocks_path)
    _write_plan(plan_path, book_id)

    evaluate_fn = make_evaluate_fn(
        book_id=book_id,
        blocks_path=blocks_path,
        plan_path=plan_path,
        target_template="general_rule.md",
        extract_client=_StubClient("extract-stub"),
        verify_client=_StubClient("verify-stub"),
        work_dir_root=tmp_path / "cands",
    )
    proposer = make_reflective_proposer(
        llm_client=_ReflectStub("## 进化约束\n剂量必须带单位，禁止臆测配伍禁忌。"),
        target_template="general_rule.md",
    )
    config = GEPAConfig(
        population_size=2,
        max_generations=3,
        eval_sample_size=1,
        convergence_threshold=1.5,      # 不可达，靠预算停
        selection_strategy="pareto",
        crossover_rate=0.0,             # 走反思变异
        budget_metric_calls=4,          # 小预算：最多 4 次真实评估
    )
    engine = GEPAEngine(
        _NullGateway(), config, evaluate_fn=evaluate_fn, instruction_proposer=proposer,
    )
    seed = PromptIndividual(prompt_id="seed", extraction_prompt="", verification_prompt="")
    # evaluate_fn 忽略 chunks，占位 1 个即可绕过 optimize 的非空校验
    report = await engine.optimize(initial_individual=seed, sample_chunks=[None])

    assert report.total_generations >= 1
    assert engine._eval_calls <= 4
    assert 0.0 <= report.final_fitness <= 1.0


class TestSystemAwareMerger:
    @pytest.mark.asyncio
    async def test_合并产出兼顾双方的新块(self) -> None:
        pa = PromptIndividual(prompt_id="a", extraction_prompt="约束A：剂量带单位", verification_prompt="")
        pb = PromptIndividual(prompt_id="b", extraction_prompt="约束B：禁止臆测禁忌", verification_prompt="")
        merger = make_system_aware_merger(
            llm_client=_ReflectStub("约束A+B：剂量带单位且禁止臆测禁忌。", rationale="各取所长"),
            target_template="general_rule.md",
        )
        child = await merger(pa, pb, _fit(0.6), _fit(0.5))
        assert "剂量带单位且禁止臆测禁忌" in child.extraction_prompt
        assert child.generation == 1
        assert child.parent_ids == ("a", "b")
        assert child.mutation_applied.startswith("merge")

    @pytest.mark.asyncio
    async def test_合并清洗围栏与占位符(self) -> None:
        pa = PromptIndividual(prompt_id="a", extraction_prompt="约束A", verification_prompt="")
        pb = PromptIndividual(prompt_id="b", extraction_prompt="约束B", verification_prompt="")
        merger = make_system_aware_merger(
            llm_client=_ReflectStub("```md\n合并块\n{{chunk_json}}\n禁止臆测\n```"),
            target_template="general_rule.md",
        )
        child = await merger(pa, pb, _fit(0.5), _fit(0.5))
        assert CHUNK_PLACEHOLDER not in child.extraction_prompt
        assert "```" not in child.extraction_prompt
        assert "禁止臆测" in child.extraction_prompt

    @pytest.mark.asyncio
    async def test_相同父代抛错以回退crossover(self) -> None:
        pa = PromptIndividual(prompt_id="a", extraction_prompt="SAME", verification_prompt="")
        pb = PromptIndividual(prompt_id="b", extraction_prompt="SAME", verification_prompt="")
        merger = make_system_aware_merger(
            llm_client=_ReflectStub("不应被调用"), target_template="general_rule.md",
        )
        with pytest.raises(ValueError):
            await merger(pa, pb, _fit(0.5), _fit(0.5))

    @pytest.mark.asyncio
    async def test_合并空产出抛错以回退crossover(self) -> None:
        pa = PromptIndividual(prompt_id="a", extraction_prompt="约束A", verification_prompt="")
        pb = PromptIndividual(prompt_id="b", extraction_prompt="约束B", verification_prompt="")
        merger = make_system_aware_merger(
            llm_client=_ReflectStub(""), target_template="general_rule.md",
        )
        with pytest.raises(ValueError):
            await merger(pa, pb, _fit(0.5), _fit(0.5))

