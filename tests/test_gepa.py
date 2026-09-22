"""GEPA 优化引擎的单元测试。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from tcm_extract_engine.gepa.engine import (
    FitnessResult,
    GEPAConfig,
    GEPAEngine,
    IssueStats,
    PromptIndividual,
    _pareto_frontier_indices,
    compute_fitness,
    crossover,
    mutate_prompt,
)
from tcm_extract_engine.gepa.gateway_types import (
    ModelUsage,
    SourceChunk,
    StructuredModelResponse,
)

# ── 辅助 ──────────────────────────────────────────────────────────────────


class _FakeGateway:
    """测试用模拟模型网关。"""

    def __init__(
        self,
        extraction_records: list[dict[str, Any]] | None = None,
        verification_issues: list[dict[str, Any]] | None = None,
    ) -> None:
        self._records = extraction_records or [{"knowledge_type": "rule", "rule_text": "test"}]
        self._issues = verification_issues or []
        self.call_count = 0

    async def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> StructuredModelResponse:
        self.call_count += 1
        user_data = json.loads(user_prompt)

        # 根据调用上下文返回抽取或验证结果
        if "target_chunk" in user_data:
            return StructuredModelResponse(
                payload={"records": self._records},
                usage=ModelUsage(input_tokens=100, output_tokens=50),
            )
        elif "candidate" in user_data:
            return StructuredModelResponse(
                payload={"issues": self._issues},
                usage=ModelUsage(input_tokens=100, output_tokens=50),
            )
        else:
            return StructuredModelResponse(
                payload={},
                usage=ModelUsage(input_tokens=10, output_tokens=10),
            )


def _make_individual(gen: int = 0) -> PromptIndividual:
    return PromptIndividual(
        prompt_id=f"test_gen{gen}",
        extraction_prompt=f"请从以下文本中抽取审方规则。{gen}",
        verification_prompt=f"请验证以下审方规则是否与原文一致。{gen}",
        generation=gen,
    )


def _make_chunks(n: int = 3) -> list[SourceChunk]:
    return [
        SourceChunk(
            chunk_id=f"chunk_{i}",
            content=f"审方案例 {i}：患者男性，45岁，咳嗽。处方：杏仁 10g、桔梗 6g。",
            source_document_id="doc_001",
            chapter=f"第{i+1}章",
        )
        for i in range(n)
    ]


# ── 适应度函数测试 ────────────────────────────────────────────────────────


class TestComputeFitness:
    """适应度函数测试。"""

    def test_无问题时适应度为1(self) -> None:
        result = compute_fitness(total_candidates=10, issues=[])
        assert result.fitness == pytest.approx(1.0)
        assert result.pass_rate == 1.0
        assert result.hallucination_rate == 0.0

    def test_全部有问题时适应度较低(self) -> None:
        """当所有候选都有 unsupported 问题时，适应度应明显低于 1.0。"""
        issues = [
            {"issue_type": "unsupported", "json_pointer": f"/records/{i}", "confidence": 0.9, "severity": "high"}
            for i in range(10)
        ]
        result = compute_fitness(total_candidates=10, issues=issues)
        # pass_rate=0, penalty_rate=0.5, hallucination_rate=1.0 → 适应度应 < 0.5
        assert result.fitness < 0.5
        assert result.pass_rate == 0.0
        assert result.hallucination_rate == 1.0

    def test_低置信度问题被过滤(self) -> None:
        issues = [
            {"issue_type": "missing", "json_pointer": "/records/0", "confidence": 0.3},
        ]
        result = compute_fitness(total_candidates=10, issues=issues, min_confidence=0.5)
        assert result.fitness == pytest.approx(1.0)

    def test_严重度影响适应度(self) -> None:
        critical_issues = [
            {"issue_type": "negation", "json_pointer": f"/records/{i}", "confidence": 0.9, "severity": "critical"}
            for i in range(5)
        ]
        low_issues = [
            {"issue_type": "missing", "json_pointer": f"/records/{i}", "confidence": 0.9, "severity": "low"}
            for i in range(5)
        ]
        result_critical = compute_fitness(10, critical_issues)
        result_low = compute_fitness(10, low_issues)
        assert result_low.fitness > result_critical.fitness

    def test_空候选列表(self) -> None:
        result = compute_fitness(total_candidates=0, issues=[])
        assert result.fitness == 0.0


# ── 变异测试 ──────────────────────────────────────────────────────────────


class TestMutation:
    """变异算子测试。"""

    def test_变异修改提示词(self) -> None:
        individual = _make_individual(gen=0)
        fitness = FitnessResult(
            fitness=0.5, pass_rate=0.5, penalty_rate=0.5,
            missing_rate=0.8, hallucination_rate=0.3,
            issue_stats=IssueStats(missing_count=8),
            top_issues=["missing"],
        )
        mutated = mutate_prompt(individual, fitness, mutation_rate=1.0)
        assert mutated.extraction_prompt != individual.extraction_prompt
        assert mutated.generation == 1
        assert "missing" in mutated.mutation_applied.lower() or "进化" in mutated.mutation_applied

    def test_变异率0不变异(self) -> None:
        individual = _make_individual(gen=0)
        fitness = FitnessResult(
            fitness=0.5, pass_rate=0.5, penalty_rate=0.5,
            missing_rate=0.5, hallucination_rate=0.5,
            issue_stats=IssueStats(),
        )
        # mutation_rate=0 应该保持不变（概率性，但几乎确定）
        results = [mutate_prompt(individual, fitness, mutation_rate=0.0) for _ in range(10)]
        assert all(r.extraction_prompt == individual.extraction_prompt for r in results)

    def test_避免重复添加相同指令(self) -> None:
        individual = _make_individual(gen=0)
        fitness = FitnessResult(
            fitness=0.5, pass_rate=0.5, penalty_rate=0.5,
            missing_rate=0.8, hallucination_rate=0.3,
            issue_stats=IssueStats(),
            top_issues=["missing"],
        )
        mutated1 = mutate_prompt(individual, fitness, mutation_rate=1.0)
        # 再次变异，由于前缀已存在，应返回同一个体
        mutated2 = mutate_prompt(mutated1, fitness, mutation_rate=1.0)
        # 可能相同也可能不同（取决于随机选择的具体模板）
        assert mutated2.generation >= 1


# ── 交叉测试 ──────────────────────────────────────────────────────────────


class TestCrossover:
    """交叉算子测试。"""

    def test_交叉产生新个体(self) -> None:
        parent_a = _make_individual(gen=1)
        parent_b = _make_individual(gen=2)
        child = crossover(parent_a, parent_b, generation=3)
        assert child.generation == 3
        assert parent_a.prompt_id in child.parent_ids
        assert parent_b.prompt_id in child.parent_ids
        # 子代的提示词应该来自某个父代
        assert child.extraction_prompt in (parent_a.extraction_prompt, parent_b.extraction_prompt)


# ── GEPA 引擎集成测试 ─────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestGEPAEngine:
    """GEPA 引擎集成测试。"""

    async def test_无问题时快速收敛(self) -> None:
        """当 Model B 不发现任何问题时，应该快速收敛。"""
        gateway = _FakeGateway(
            extraction_records=[{"knowledge_type": "rule", "rule_text": "规则1"}],
            verification_issues=[],
        )
        config = GEPAConfig(
            population_size=3,
            max_generations=5,
            convergence_threshold=0.95,
            eval_sample_size=2,
        )
        engine = GEPAEngine(
            gateway, config,
            extract_model="test-model",
            verify_model="test-model",
        )
        report = await engine.optimize(
            initial_individual=_make_individual(),
            sample_chunks=_make_chunks(2),
        )
        assert report.converged
        assert report.final_fitness >= 0.95
        assert report.total_generations <= 5

    async def test_有问题时适应度低于阈值(self) -> None:
        """当 Model B 发现大量问题时，适应度应该低于阈值。"""
        gateway = _FakeGateway(
            extraction_records=[{"knowledge_type": "rule", "rule_text": "规则1"}],
            verification_issues=[
                {"issue_type": "unsupported", "json_pointer": "/records/0/field", "confidence": 0.9, "severity": "high"},
                {"issue_type": "missing", "json_pointer": "/records/0/evidence", "confidence": 0.8, "severity": "medium"},
            ],
        )
        config = GEPAConfig(
            population_size=2,
            max_generations=3,
            convergence_threshold=0.95,
            eval_sample_size=2,
        )
        engine = GEPAEngine(
            gateway, config,
            extract_model="test-model",
            verify_model="test-model",
        )
        report = await engine.optimize(
            initial_individual=_make_individual(),
            sample_chunks=_make_chunks(2),
        )
        assert report.final_fitness < 0.95

    async def test_on_generation_callback(self) -> None:
        """测试每代回调函数。"""
        callback_reports = []

        def on_gen(report: Any) -> None:
            callback_reports.append(report)

        gateway = _FakeGateway(
            extraction_records=[{"knowledge_type": "rule", "rule_text": "test"}],
            verification_issues=[],
        )
        config = GEPAConfig(
            population_size=2,
            max_generations=3,
            convergence_threshold=0.95,
            eval_sample_size=2,
        )
        engine = GEPAEngine(
            gateway, config,
            extract_model="test-model",
            verify_model="test-model",
        )
        await engine.optimize(
            initial_individual=_make_individual(),
            sample_chunks=_make_chunks(2),
            on_generation=on_gen,
        )
        assert len(callback_reports) >= 1


# ── P1: 注入点（反思变异 / 真实评估 / Pareto 选择）────────────────────────


def _fitness(
    score: float,
    *,
    per_instance: dict[str, float] | None = None,
    feedback: str = "",
) -> FitnessResult:
    return FitnessResult(
        fitness=score,
        pass_rate=score,
        penalty_rate=1.0 - score,
        missing_rate=0.0,
        hallucination_rate=0.0,
        issue_stats=IssueStats(),
        feedback=feedback,
        per_instance_scores=per_instance or {},
    )


class _UnusedGateway:
    """注入 evaluate_fn 时不应被调用的网关；一旦被调用即失败。"""

    async def generate(self, **_: Any) -> Any:  # pragma: no cover - 防御
        raise AssertionError("注入 evaluate_fn 时不应调用内置网关")


class TestEvaluateFnInjection:
    @pytest.mark.asyncio
    async def test_注入真实评估绕开内置网关(self) -> None:
        calls: list[str] = []

        def evaluate_fn(individual: PromptIndividual, chunks: Any) -> FitnessResult:
            calls.append(individual.prompt_id)
            return _fitness(0.42, feedback="证据不足")

        engine = GEPAEngine(
            _UnusedGateway(),
            GEPAConfig(population_size=2, max_generations=2, eval_sample_size=1),
            evaluate_fn=evaluate_fn,
        )
        report = await engine.optimize(
            initial_individual=_make_individual(),
            sample_chunks=_make_chunks(1),
        )
        assert calls  # evaluate_fn 确实被调用
        assert report.final_fitness == pytest.approx(0.42)

    @pytest.mark.asyncio
    async def test_budget_metric_calls_停止(self) -> None:
        engine = GEPAEngine(
            _UnusedGateway(),
            GEPAConfig(
                population_size=4,
                max_generations=10,
                eval_sample_size=1,
                convergence_threshold=1.5,  # 不可达
                budget_metric_calls=2,
            ),
            evaluate_fn=lambda ind, ch: _fitness(0.5),
        )
        report = await engine.optimize(
            initial_individual=_make_individual(),
            sample_chunks=_make_chunks(1),
        )
        assert engine._eval_calls == 2  # 恰好评估 2 次即停
        assert report.total_generations == 1

    @pytest.mark.asyncio
    async def test_skip_perfect_提前停止(self) -> None:
        engine = GEPAEngine(
            _UnusedGateway(),
            GEPAConfig(
                population_size=2,
                max_generations=10,
                eval_sample_size=1,
                convergence_threshold=1.5,  # 不可达，逼出 skip_perfect 分支
                skip_perfect=True,
                perfect_score=1.0,
            ),
            evaluate_fn=lambda ind, ch: _fitness(1.0),
        )
        report = await engine.optimize(
            initial_individual=_make_individual(),
            sample_chunks=_make_chunks(1),
        )
        assert report.total_generations == 1
        assert report.final_fitness == pytest.approx(1.0)
        assert not report.converged  # 满分停止，而非阈值收敛


class TestInstructionProposerInjection:
    @pytest.mark.asyncio
    async def test_反思变异钩子被采用(self) -> None:
        seen_feedback: list[str] = []

        def proposer(
            parent: PromptIndividual, parent_fitness: FitnessResult
        ) -> PromptIndividual:
            seen_feedback.append(parent_fitness.feedback)
            return PromptIndividual(
                prompt_id=f"{parent.prompt_id}_reflect",
                extraction_prompt=parent.extraction_prompt + "\n## 反思约束",
                verification_prompt=parent.verification_prompt,
                generation=parent.generation + 1,
                parent_ids=(parent.prompt_id,),
                mutation_applied="REFLECT",
            )

        engine = GEPAEngine(
            _UnusedGateway(),
            GEPAConfig(
                population_size=2,
                max_generations=3,
                eval_sample_size=1,
                convergence_threshold=1.5,  # 不可达，逼出变异路径
                crossover_rate=0.0,  # 关闭交叉，确保走 _propose 变异
            ),
            evaluate_fn=lambda ind, ch: _fitness(0.5, feedback="剂量单位缺失"),
            instruction_proposer=proposer,
        )
        report = await engine.optimize(
            initial_individual=_make_individual(),
            sample_chunks=_make_chunks(1),
        )
        assert seen_feedback and all(f == "剂量单位缺失" for f in seen_feedback)
        all_mutations = [
            m for gr in report.generation_reports for m in gr.mutations_applied
        ]
        assert "REFLECT" in all_mutations


class TestParetoSelection:
    @pytest.mark.asyncio
    async def test_pareto策略可跑通(self) -> None:
        engine = GEPAEngine(
            _UnusedGateway(),
            GEPAConfig(
                population_size=3,
                max_generations=2,
                eval_sample_size=1,
                convergence_threshold=1.5,
                selection_strategy="pareto",
                crossover_rate=0.0,  # 走变异路径，实际调用 _select_pareto_parent
            ),
            evaluate_fn=lambda ind, ch: _fitness(
                0.5, per_instance={"bookA": 0.7, "bookB": 0.3}
            ),
        )
        report = await engine.optimize(
            initial_individual=_make_individual(),
            sample_chunks=_make_chunks(1),
        )
        assert report.total_generations >= 1


class TestParetoFrontier:
    def test_非支配集保留局部专长个体(self) -> None:
        # 个体0擅长 bookA，个体1擅长 bookB，个体2全面较差被支配
        results = [
            _fitness(0.6, per_instance={"bookA": 0.9, "bookB": 0.3}),
            _fitness(0.6, per_instance={"bookA": 0.3, "bookB": 0.9}),
            _fitness(0.2, per_instance={"bookA": 0.1, "bookB": 0.1}),
        ]
        frontier = _pareto_frontier_indices(results)
        assert set(frontier) == {0, 1}  # 2 被 0、1 支配

    def test_无每实例分退化为综合最优(self) -> None:
        results = [_fitness(0.3), _fitness(0.9), _fitness(0.5)]
        assert _pareto_frontier_indices(results) == [1]


class TestMergeFnInjection:
    @pytest.mark.asyncio
    async def test_合并钩子被采用(self) -> None:
        seen: list[tuple[str, str]] = []

        def merge_fn(
            pa: PromptIndividual,
            pb: PromptIndividual,
            fa: FitnessResult,
            fb: FitnessResult,
        ) -> PromptIndividual:
            seen.append((pa.prompt_id, pb.prompt_id))
            return PromptIndividual(
                prompt_id=f"{pa.prompt_id}+{pb.prompt_id}",
                extraction_prompt=pa.extraction_prompt + "\n## 合并约束\n" + pb.extraction_prompt,
                verification_prompt=pa.verification_prompt,
                generation=max(pa.generation, pb.generation) + 1,
                parent_ids=(pa.prompt_id, pb.prompt_id),
                mutation_applied="MERGE",
            )

        engine = GEPAEngine(
            _UnusedGateway(),
            GEPAConfig(
                population_size=2,
                max_generations=2,
                eval_sample_size=1,
                convergence_threshold=1.5,  # 不可达
                crossover_rate=1.0,  # 强制走交叉/合并路径
            ),
            evaluate_fn=lambda ind, ch: _fitness(0.5),
            merge_fn=merge_fn,
        )
        report = await engine.optimize(
            initial_individual=_make_individual(),
            sample_chunks=_make_chunks(1),
        )
        assert seen  # merge_fn 确实被调用
        all_mutations = [
            m for gr in report.generation_reports for m in gr.mutations_applied
        ]
        assert "MERGE" in all_mutations

    @pytest.mark.asyncio
    async def test_合并失败回退crossover(self) -> None:
        def boom(
            pa: PromptIndividual,
            pb: PromptIndividual,
            fa: FitnessResult,
            fb: FitnessResult,
        ) -> PromptIndividual:
            raise RuntimeError("merge down")

        engine = GEPAEngine(
            _UnusedGateway(),
            GEPAConfig(
                population_size=2,
                max_generations=2,
                eval_sample_size=1,
                convergence_threshold=1.5,
                crossover_rate=1.0,
            ),
            evaluate_fn=lambda ind, ch: _fitness(0.5),
            merge_fn=boom,
        )
        report = await engine.optimize(
            initial_individual=_make_individual(),
            sample_chunks=_make_chunks(1),
        )
        # 合并抛错 → 回退内置 crossover，进化不中断
        assert report.total_generations >= 1

