"""GEPA — Genetic-Evolutionary Prompt Adaptation 优化引擎。

基于 Model B 验证反馈的进化式提示词优化：
1. 用当前提示词在样本 chunks 上运行 Model A 抽取 + Model B 验证
2. 计算适应度（通过率、严重度惩罚、遗漏率、幻觉率）
3. 选择精英、变异、交叉产生下一代提示词
4. 重复直到收敛或达到最大代数

与 knowledge/workflow.py 的 AsyncModelGateway 协议兼容。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tcm_extract_engine.gepa.gateway_types import AsyncModelGateway, SourceChunk

logger = logging.getLogger(__name__)


# ── 配置 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GEPAConfig:
    """GEPA 优化引擎配置。"""

    population_size: int = 8
    """每代提示词数量（含精英保留）。"""

    max_generations: int = 20
    """最大进化代数。"""

    convergence_threshold: float = 0.95
    """适应度收敛阈值，达到此值时停止进化。"""

    mutation_rate: float = 0.3
    """变异概率：每代中有多大比例的个体被变异。"""

    crossover_rate: float = 0.5
    """交叉概率。"""

    elite_ratio: float = 0.25
    """精英保留比例：适应度最高的个体直接进入下一代。"""

    eval_sample_size: int = 30
    """每代评估用的 chunk 样本数量。"""

    min_confidence: float = 0.5
    """Model B 输出问题低于此置信度的自动忽略。"""

    selection_strategy: str = "scalar"
    """父代选择策略："scalar"（精英+随机，默认，旧行为）或 "pareto"（Pareto 前沿采样）。"""

    budget_metric_calls: int | None = None
    """评估调用预算上限；None=不限（用 max_generations 停止）。达到即停。"""

    skip_perfect: bool = False
    """最优适应度达到 perfect_score 时提前停止进化（省预算）。"""

    perfect_score: float = 1.0
    """满分适应度（skip_perfect 的判定阈值）。"""


# ── 数据结构 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PromptIndividual:
    """一个提示词个体（候选提示词版本）。"""

    prompt_id: str
    """唯一标识。"""

    extraction_prompt: str
    """Model A 使用的抽取提示词。"""

    verification_prompt: str
    """Model B 使用的验证提示词。"""

    generation: int = 0
    """所属代数。"""

    parent_ids: tuple[str, ...] = ()
    """父代 ID（用于追溯进化链）。"""

    mutation_applied: str = ""
    """应用的变异操作描述。"""

    def fingerprint(self) -> str:
        """提示词内容的 SHA-256 指纹。"""
        combined = f"{self.extraction_prompt}\n---\n{self.verification_prompt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class IssueStats:
    """单代评估的问题统计。"""

    total_candidates: int = 0
    total_issues: int = 0
    unsupported_count: int = 0
    missing_count: int = 0
    misclassified_count: int = 0
    inconsistent_count: int = 0
    typo_or_ocr_count: int = 0
    negation_count: int = 0
    dosage_unit_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0


@dataclass(frozen=True, slots=True)
class FitnessResult:
    """适应度评估结果。"""

    fitness: float
    """综合适应度 [0, 1]。"""

    pass_rate: float
    """无问题候选比例。"""

    penalty_rate: float
    """严重度加权惩罚率。"""

    missing_rate: float
    """遗漏率。"""

    hallucination_rate: float
    """幻觉率。"""

    issue_stats: IssueStats
    """问题分布统计。"""

    top_issues: list[str] = field(default_factory=list)
    """高频问题类型（用于指导变异）。"""

    feedback: str = ""
    """富文本反馈（官方 GEPA 关键信号）：解释"为什么是这个分"，供反思式变异诊断。"""

    per_instance_scores: dict[str, float] = field(default_factory=dict)
    """每实例（如每本书/每章节）适应度分，用于 Pareto 前沿选择。"""


# 反思式变异钩子：给定父代 + 其适应度（含 feedback），返回变异后的新个体。
# 可为同步或异步函数；默认 None 时走内置静态模板 mutate_prompt。
ProposalFn = Callable[["PromptIndividual", "FitnessResult"], Any]

# 真实评估钩子：给定个体 + 评估样本，返回 FitnessResult。
# 可为同步或异步函数；默认 None 时走内置 self._evaluate（knowledge schema）。
EvaluateFn = Callable[["PromptIndividual", Sequence["SourceChunk"]], Any]

# system-aware 合并钩子（官方 GEPA 机制④）：给定两个父代 + 各自适应度，
# 合成一个兼顾双方优势的新个体。默认 None 时走内置 crossover。
MergeFn = Callable[
    ["PromptIndividual", "PromptIndividual", "FitnessResult", "FitnessResult"], Any
]


@dataclass(frozen=True, slots=True)
class GenerationReport:
    """单代进化报告。"""

    generation: int
    population_size: int
    best_fitness: float
    avg_fitness: float
    best_individual: PromptIndividual
    issue_stats: IssueStats
    mutations_applied: list[str]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class GEPAReport:
    """GEPA 完整优化报告。"""

    total_generations: int
    converged: bool
    final_fitness: float
    best_individual: PromptIndividual
    fitness_history: list[float]
    generation_reports: list[GenerationReport]
    total_elapsed_seconds: float
    total_model_calls: int

    def summary(self) -> str:
        """生成可读摘要。"""
        status = "已收敛" if self.converged else f"达到最大代数 ({self.total_generations})"
        return (
            f"GEPA 优化报告\n"
            f"  状态: {status}\n"
            f"  总代数: {self.total_generations}\n"
            f"  最终适应度: {self.final_fitness:.4f}\n"
            f"  适应度轨迹: {self.fitness_history[:5]}... → {self.fitness_history[-5:]}\n"
            f"  最优个体: {self.best_individual.prompt_id} (gen {self.best_individual.generation})\n"
            f"  总模型调用: {self.total_model_calls}\n"
            f"  总耗时: {self.total_elapsed_seconds:.1f}s\n"
        )


# ── 适应度函数 ─────────────────────────────────────────────────────────────


def compute_fitness(
    total_candidates: int,
    issues: list[dict[str, Any]],
    min_confidence: float = 0.5,
) -> FitnessResult:
    """
    评估一个提示词版本的质量，范围 [0, 1]。

    权重分配：
    - 40%: 通过率（无问题候选比例）
    - 30%: 严重度加权惩罚
    - 15%: 遗漏惩罚
    - 15%: 幻觉惩罚
    """
    if total_candidates <= 0:
        return FitnessResult(
            fitness=0.0, pass_rate=0.0, penalty_rate=0.0,
            missing_rate=0.0, hallucination_rate=0.0,
            issue_stats=IssueStats(),
        )

    # 过滤低置信度问题
    filtered = [i for i in issues if i.get("confidence", 0) >= min_confidence]

    # 统计问题类型分布
    type_counts = Counter(i.get("issue_type", "") for i in filtered)
    sev_counts = Counter(i.get("severity", "medium") for i in filtered)

    stats = IssueStats(
        total_candidates=total_candidates,
        total_issues=len(filtered),
        unsupported_count=type_counts.get("unsupported", 0),
        missing_count=type_counts.get("missing", 0),
        misclassified_count=type_counts.get("misclassified", 0),
        inconsistent_count=type_counts.get("inconsistent", 0),
        typo_or_ocr_count=type_counts.get("typo_or_ocr", 0),
        negation_count=type_counts.get("negation", 0),
        dosage_unit_count=type_counts.get("dosage_unit", 0),
        critical_count=sev_counts.get("critical", 0),
        high_count=sev_counts.get("high", 0),
        medium_count=sev_counts.get("medium", 0),
        low_count=sev_counts.get("low", 0),
    )

    # 受影响候选数（去重 json_pointer 定位到的具体候选）
    # json_pointer 格式为 /records/0/field 或 /records/0，
    # 使用 records/{index} 作为候选标识
    affected = set()
    for i in filtered:
        ptr = i.get("json_pointer", "")
        parts = ptr.split("/")
        if len(parts) >= 3:
            # /records/0/field → 候选标识 = "records/0"
            affected.add(f"{parts[1]}/{parts[2]}")
        elif len(parts) >= 2:
            affected.add(parts[1])
    candidates_with_issues = len(affected)

    pass_rate = 1.0 - (candidates_with_issues / total_candidates)

    # 严重度加权惩罚
    severity_weights = {"critical": 1.0, "high": 0.5, "medium": 0.2, "low": 0.1}
    penalty = sum(severity_weights.get(i.get("severity", "medium"), 0.2) for i in filtered)
    penalty_rate = penalty / total_candidates

    # 遗漏率
    missing_rate = stats.missing_count / total_candidates

    # 幻觉率
    hallucination_rate = stats.unsupported_count / total_candidates

    # 综合适应度
    fitness = (
        0.4 * pass_rate
        + 0.3 * max(0, 1.0 - penalty_rate)
        + 0.15 * max(0, 1.0 - missing_rate)
        + 0.15 * max(0, 1.0 - hallucination_rate)
    )
    fitness = max(0.0, min(1.0, fitness))

    # 高频问题类型（用于指导变异方向）
    top_issues = [t for t, _ in type_counts.most_common(3)]

    return FitnessResult(
        fitness=fitness,
        pass_rate=pass_rate,
        penalty_rate=penalty_rate,
        missing_rate=missing_rate,
        hallucination_rate=hallucination_rate,
        issue_stats=stats,
        top_issues=top_issues,
    )


# ── 变异算子 ──────────────────────────────────────────────────────────────

# 变异指令模板（根据 Model B 反馈的问题类型选择）
_MUTATION_TEMPLATES: dict[str, list[str]] = {
    "missing": [
        "在抽取知识时，必须完整提取原文中的所有关键信息，包括特殊煎煮要求（先煎、后下、包煎、另煎等）、禁忌条件和适用人群限制。",
        "案例类知识必须保留完整的分析链：处方描述 → 处方问题 → 处方分析 → 干预建议，不得省略任何环节。",
        "注意提取原文中所有涉及剂量限制、用法要求的内容，不要遗漏'每日X次'、'饭后服用'等细节。",
    ],
    "unsupported": [
        "禁止添加原文中不存在的信息。所有抽取的规则必须有原文证据（evidence_text），不得凭推理补充。",
        "如果原文未提及某个配伍禁忌或剂量限制，不要在抽取结果中添加。宁可漏抽也不要幻觉。",
        "所有 related_laws 字段的内容必须来自原文引用的法规名称，不得自行添加。",
    ],
    "misclassified": [
        "knowledge_type 分类标准：principle = 审方原则性描述（如审方流程、工作模式），rule = 具体可执行的审方规则（如'毛笔头应使用规范用名辛夷'）。",
        "category 分类必须准确：毒性中药相关规则 → 毒性中药审核，药物别名 → 规范性审核，儿童/老人/孕妇 → 特殊人群审核。",
    ],
    "negation": [
        "注意否定语义的准确理解：'不得超过'表示上限，'禁止'表示绝对不允许，'慎用'表示需要谨慎但可以使用，'忌用'/'禁用'表示不能使用。",
        "抽取时必须保留原文中的否定词（不得、禁止、不宜、忌用），不得将其转换为肯定表述。",
    ],
    "dosage_unit": [
        "剂量必须包含数值和单位（如 '3-6g'），不得遗漏单位或修改原始数值。",
        "毒性中药的最大用量必须准确保留（如：附子最大用量 15g，需先煎 30-60 分钟）。",
        "儿童剂量折算规则必须完整保留原文描述（如：按体重折算、按年龄折算）。",
    ],
    "typo_or_ocr": [
        "遇到疑似 OCR 错误（如'附于'可能是'附子'，'109'可能是'10g'），在 evidence_text 中标注 [疑似OCR错误] 并保留原文。",
        "药物名称以《中华人民共和国药典》规范用名为准，如发现别名（如毛笔头→辛夷），在 drug_aliases 中记录映射。",
    ],
    "inconsistent": [
        "抽取时注意与同章节其他规则的一致性。如果发现矛盾，在 evidence_text 中标注。",
    ],
    "general": [
        "提高抽取的精确度：每条规则只抽取原文明确描述的内容，evidence_text 必须是原文的精确片段。",
        "确保 severity 标注合理：涉及患者安全的（毒性、禁忌）→ critical/high，书写规范类 → medium/low。",
    ],
}


def mutate_prompt(
    individual: PromptIndividual,
    fitness_result: FitnessResult,
    mutation_rate: float,
) -> PromptIndividual:
    """对一个提示词个体应用变异操作。"""
    if random.random() > mutation_rate:
        return individual

    # 根据高频问题类型选择变异指令
    issue_types = fitness_result.top_issues or ["general"]
    selected_type = random.choice(issue_types)
    templates = _MUTATION_TEMPLATES.get(selected_type, _MUTATION_TEMPLATES["general"])
    mutation_instruction = random.choice(templates)

    # 变异策略：在抽取提示词末尾追加约束
    new_prompt = individual.extraction_prompt.rstrip()

    # 避免重复添加相同指令
    if mutation_instruction[:20] in new_prompt:
        return individual

    new_prompt += f"\n\n## 进化优化约束（第 {individual.generation + 1} 代）\n{mutation_instruction}\n"

    new_id = f"{individual.prompt_id}_mut{individual.generation + 1}"
    return PromptIndividual(
        prompt_id=new_id,
        extraction_prompt=new_prompt,
        verification_prompt=individual.verification_prompt,
        generation=individual.generation + 1,
        parent_ids=(individual.prompt_id,),
        mutation_applied=f"{selected_type}: {mutation_instruction[:50]}...",
    )


def crossover(
    parent_a: PromptIndividual,
    parent_b: PromptIndividual,
    generation: int,
) -> PromptIndividual:
    """交叉两个父代提示词：取 A 的抽取提示词 + B 的验证提示词（或反之）。"""
    # 简单策略：取适应度更好的抽取提示词 + 另一个的验证提示词
    # 这里随机选择方向
    if random.random() < 0.5:
        ext_prompt = parent_a.extraction_prompt
        ver_prompt = parent_b.verification_prompt
    else:
        ext_prompt = parent_b.extraction_prompt
        ver_prompt = parent_a.verification_prompt

    new_id = f"cross_{parent_a.prompt_id}_{parent_b.prompt_id}_g{generation}"
    return PromptIndividual(
        prompt_id=new_id,
        extraction_prompt=ext_prompt,
        verification_prompt=ver_prompt,
        generation=generation,
        parent_ids=(parent_a.prompt_id, parent_b.prompt_id),
        mutation_applied="crossover",
    )


# ── GEPA 引擎 ─────────────────────────────────────────────────────────────


class GEPAEngine:
    """GEPA 进化优化引擎。

    使用方式：

        engine = GEPAEngine(gateway, config)
        report = await engine.optimize(
            initial_individual=seed,
            sample_chunks=chunks[:30],
        )
        best_prompt = report.best_individual.extraction_prompt
    """

    def __init__(
        self,
        gateway: AsyncModelGateway,
        config: GEPAConfig | None = None,
        *,
        extract_model: str = "",
        verify_model: str = "",
        extract_params: Mapping[str, Any] | None = None,
        verify_params: Mapping[str, Any] | None = None,
        instruction_proposer: ProposalFn | None = None,
        evaluate_fn: EvaluateFn | None = None,
        merge_fn: MergeFn | None = None,
    ) -> None:
        self._gateway = gateway
        self._config = config or GEPAConfig()
        self._extract_model = extract_model
        self._verify_model = verify_model
        self._extract_params = extract_params or {}
        self._verify_params = verify_params or {}
        self._total_model_calls = 0
        # 可选注入点（默认 None → 走内置旧行为，保证零回归）
        self._instruction_proposer = instruction_proposer  # 反思式变异（官方 GEPA 机制 1）
        self._evaluate_fn = evaluate_fn                    # 真实评估（注入 pipeline 闭环）
        self._merge_fn = merge_fn                          # system-aware 合并（官方 GEPA 机制 4）
        self._eval_calls = 0                               # 评估调用计数（budget_metric_calls 停止）

    async def optimize(
        self,
        initial_individual: PromptIndividual,
        sample_chunks: Sequence[SourceChunk],
        *,
        extraction_system_prompt: str = "",
        verification_system_prompt: str = "",
        extraction_response_schema: Mapping[str, Any] | None = None,
        verification_response_schema: Mapping[str, Any] | None = None,
        on_generation: Any = None,
    ) -> GEPAReport:
        """运行完整的 GEPA 优化循环。

        Args:
            initial_individual: 初始提示词个体（通常来自 Model C）
            sample_chunks: 评估用的样本 chunks
            extraction_system_prompt: Model A 的系统提示词
            verification_system_prompt: Model B 的系统提示词
            extraction_response_schema: Model A 的输出 schema
            verification_response_schema: Model B 的输出 schema
            on_generation: 每代完成后的回调函数 (GenerationReport) -> None
        """
        start_time = time.monotonic()
        cfg = self._config

        # 截断样本
        eval_chunks = list(sample_chunks[:cfg.eval_sample_size])
        if not eval_chunks:
            raise ValueError("GEPA 需要至少 1 个评估样本 chunk")

        # 初始化种群
        population = [initial_individual]
        # 补充随机变异版本填满种群
        for i in range(1, cfg.population_size):
            mutated = mutate_prompt(initial_individual, FitnessResult(
                fitness=0.5, pass_rate=0.5, penalty_rate=0.5,
                missing_rate=0.5, hallucination_rate=0.5,
                issue_stats=IssueStats(),
            ), mutation_rate=1.0)
            population.append(mutated)

        fitness_history: list[float] = []
        generation_reports: list[GenerationReport] = []
        best_ever = initial_individual
        best_ever_fitness = 0.0

        for gen in range(cfg.max_generations):
            gen_start = time.monotonic()

            # 评估种群
            fitness_results: list[tuple[PromptIndividual, FitnessResult]] = []
            for individual in population:
                # 预算停止：达到 metric call 上限即中断本代评估
                if (
                    cfg.budget_metric_calls is not None
                    and self._eval_calls >= cfg.budget_metric_calls
                ):
                    break
                fitness = await self._evaluate_individual(
                    individual,
                    eval_chunks,
                    extraction_system_prompt=extraction_system_prompt,
                    verification_system_prompt=verification_system_prompt,
                    extraction_response_schema=extraction_response_schema,
                    verification_response_schema=verification_response_schema,
                )
                fitness_results.append((individual, fitness))

            if not fitness_results:
                # 预算耗尽，未评估任何个体
                break

            # 排序（适应度降序）
            fitness_results.sort(key=lambda x: x[1].fitness, reverse=True)
            best_in_gen = fitness_results[0]
            avg_fitness = sum(f.fitness for _, f in fitness_results) / len(fitness_results)

            if best_in_gen[1].fitness > best_ever_fitness:
                best_ever = best_in_gen[0]
                best_ever_fitness = best_in_gen[1].fitness

            fitness_history.append(best_in_gen[1].fitness)

            # 生成报告
            mutations_applied = [
                ind.mutation_applied for ind, _ in fitness_results if ind.mutation_applied
            ]
            gen_report = GenerationReport(
                generation=gen,
                population_size=len(population),
                best_fitness=best_in_gen[1].fitness,
                avg_fitness=avg_fitness,
                best_individual=best_in_gen[0],
                issue_stats=best_in_gen[1].issue_stats,
                mutations_applied=mutations_applied,
                elapsed_seconds=time.monotonic() - gen_start,
            )
            generation_reports.append(gen_report)

            if on_generation:
                await _maybe_await(on_generation(gen_report))

            logger.info(
                "[GEPA] Gen %d: best=%.4f avg=%.4f top_issues=%s",
                gen, best_in_gen[1].fitness, avg_fitness,
                best_in_gen[1].top_issues,
            )

            # 收敛检查
            if best_in_gen[1].fitness >= cfg.convergence_threshold:
                logger.info("[GEPA] 收敛于第 %d 代，适应度 %.4f", gen, best_in_gen[1].fitness)
                break

            # 满分提前停止（省预算，官方 GEPA skip_perfect_score）
            if cfg.skip_perfect and best_in_gen[1].fitness >= cfg.perfect_score:
                logger.info("[GEPA] 达到满分 %.4f，提前停止于第 %d 代", best_in_gen[1].fitness, gen)
                break

            # 预算停止：评估调用已达上限
            if (
                cfg.budget_metric_calls is not None
                and self._eval_calls >= cfg.budget_metric_calls
            ):
                logger.info(
                    "[GEPA] 评估预算耗尽 (%d calls)，停止于第 %d 代",
                    self._eval_calls, gen,
                )
                break

            # 生成下一代
            next_gen_size = cfg.population_size
            elite_count = max(1, int(next_gen_size * cfg.elite_ratio))
            next_population = [ind for ind, _ in fitness_results[:elite_count]]

            # 交叉（pareto 策略走前沿采样；注入 merge_fn 则为 system-aware 合并）
            while len(next_population) < next_gen_size:
                if random.random() < cfg.crossover_rate and len(fitness_results) >= 2:
                    if cfg.selection_strategy == "pareto":
                        pa_idx = self._select_pareto_parent(fitness_results)
                    else:
                        pa_idx = random.randint(0, elite_count)
                    pb_idx = random.randint(0, len(fitness_results) - 1)
                    pa, fa = fitness_results[pa_idx]
                    pb, fb = fitness_results[pb_idx]
                    child = await self._merge(pa, pb, fa, fb, gen + 1)
                    next_population.append(child)
                else:
                    break

            # 变异（pareto 策略走前沿采样；注入 instruction_proposer 则为反思式变异）
            while len(next_population) < next_gen_size:
                if cfg.selection_strategy == "pareto":
                    parent_idx = self._select_pareto_parent(fitness_results)
                else:
                    parent_idx = random.randint(0, elite_count - 1)
                parent, parent_fitness = fitness_results[parent_idx]
                child = await self._propose(parent, parent_fitness, cfg.mutation_rate)
                next_population.append(child)

            population = next_population[:next_gen_size]

        total_elapsed = time.monotonic() - start_time
        converged = best_ever_fitness >= cfg.convergence_threshold

        return GEPAReport(
            total_generations=len(generation_reports),
            converged=converged,
            final_fitness=best_ever_fitness,
            best_individual=best_ever,
            fitness_history=fitness_history,
            generation_reports=generation_reports,
            total_elapsed_seconds=total_elapsed,
            total_model_calls=self._total_model_calls,
        )

    async def _evaluate_individual(
        self,
        individual: PromptIndividual,
        chunks: Sequence[SourceChunk],
        *,
        extraction_system_prompt: str,
        verification_system_prompt: str,
        extraction_response_schema: Mapping[str, Any] | None,
        verification_response_schema: Mapping[str, Any] | None,
    ) -> FitnessResult:
        """评估分派：注入 evaluate_fn 时用真实 pipeline 闭环，否则走内置 schema 评估。

        每次评估计一次 metric call（供 budget_metric_calls 停止）。
        """
        self._eval_calls += 1
        if self._evaluate_fn is not None:
            result = self._evaluate_fn(individual, chunks)
            return await _resolve(result)
        return await self._evaluate(
            individual,
            chunks,
            extraction_system_prompt=extraction_system_prompt,
            verification_system_prompt=verification_system_prompt,
            extraction_response_schema=extraction_response_schema,
            verification_response_schema=verification_response_schema,
        )

    async def _propose(
        self,
        parent: PromptIndividual,
        parent_fitness: FitnessResult,
        mutation_rate: float,
    ) -> PromptIndividual:
        """变异分派：注入 instruction_proposer 时为反思式变异（读 feedback 诊断），否则静态模板。"""
        if self._instruction_proposer is not None:
            result = self._instruction_proposer(parent, parent_fitness)
            return await _resolve(result)
        return mutate_prompt(parent, parent_fitness, mutation_rate)

    async def _merge(
        self,
        pa: PromptIndividual,
        pb: PromptIndividual,
        fa: FitnessResult,
        fb: FitnessResult,
        generation: int,
    ) -> PromptIndividual:
        """交叉分派：注入 merge_fn 时为 system-aware 合并（官方 GEPA 机制 4），否则内置 crossover。

        merge_fn 失败或返回非法个体时回退 crossover，保证进化循环不中断。
        """
        if self._merge_fn is not None:
            try:
                child = await _resolve(self._merge_fn(pa, pb, fa, fb))
            except Exception:
                child = None
            if isinstance(child, PromptIndividual) and child.extraction_prompt.strip():
                return child
        return crossover(pa, pb, generation)

    def _select_pareto_parent(
        self,
        fitness_results: Sequence[tuple[PromptIndividual, FitnessResult]],
    ) -> int:
        """从 Pareto 前沿随机采样一个父代下标（官方 GEPA 机制 3）。

        前沿为空（如无每实例分且全被支配）时退化为随机选择。
        """
        frontier = _pareto_frontier_indices([f for _, f in fitness_results])
        if not frontier:
            return random.randint(0, len(fitness_results) - 1)
        return random.choice(frontier)

    async def _evaluate(
        self,
        individual: PromptIndividual,
        chunks: Sequence[SourceChunk],
        *,
        extraction_system_prompt: str,
        verification_system_prompt: str,
        extraction_response_schema: Mapping[str, Any] | None,
        verification_response_schema: Mapping[str, Any] | None,
    ) -> FitnessResult:
        """评估单个提示词个体的适应度。"""
        all_candidates: list[dict[str, Any]] = []
        all_issues: list[dict[str, Any]] = []

        ext_schema = extraction_response_schema or {
            "type": "object",
            "properties": {"records": {"type": "array"}},
        }
        ver_schema = verification_response_schema or {
            "type": "object",
            "properties": {"issues": {"type": "array"}},
        }

        # Step 1: 用 Model A 逐 chunk 抽取
        for chunk in chunks:
            user_prompt = json.dumps({
                "profile_instruction": individual.extraction_prompt,
                "target_chunk": {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "chapter": chunk.chapter,
                },
            }, ensure_ascii=False)

            try:
                resp = await self._gateway.generate(
                    model=self._extract_model,
                    system_prompt=extraction_system_prompt,
                    user_prompt=user_prompt,
                    response_schema=ext_schema,
                    parameters=self._extract_params,
                )
                self._total_model_calls += 1
                payload = resp.payload
                if isinstance(payload, str):
                    payload = json.loads(payload)
                records = payload.get("records", []) if isinstance(payload, dict) else []
                all_candidates.extend(records)
            except Exception:
                logger.warning("[GEPA] Model A 抽取失败: chunk=%s", chunk.chunk_id)
                continue

        if not all_candidates:
            return FitnessResult(
                fitness=0.0, pass_rate=0.0, penalty_rate=1.0,
                missing_rate=1.0, hallucination_rate=0.0,
                issue_stats=IssueStats(total_candidates=0),
            )

        # Step 2: 用 Model B 逐 candidate 验证
        for idx, candidate in enumerate(all_candidates):
            # 找到该 candidate 对应的 source chunk
            source_chunk = chunks[0]  # 简化：使用第一个 chunk 作为 source
            user_prompt = json.dumps({
                "profile_instruction": individual.verification_prompt,
                "candidate": candidate,
                "source": {
                    "content": source_chunk.content,
                    "chapter": source_chunk.chapter,
                },
            }, ensure_ascii=False)

            try:
                resp = await self._gateway.generate(
                    model=self._verify_model,
                    system_prompt=verification_system_prompt,
                    user_prompt=user_prompt,
                    response_schema=ver_schema,
                    parameters=self._verify_params,
                )
                self._total_model_calls += 1
                payload = resp.payload
                if isinstance(payload, str):
                    payload = json.loads(payload)
                issues = payload.get("issues", []) if isinstance(payload, dict) else []
                # 添加 record_index 和 severity 映射
                sev_map = {
                    "unsupported": "high", "missing": "medium",
                    "misclassified": "high", "inconsistent": "critical",
                    "typo_or_ocr": "critical", "negation": "critical",
                    "dosage_unit": "critical",
                }
                for issue in issues:
                    issue.setdefault("severity", sev_map.get(issue.get("issue_type", ""), "medium"))
                all_issues.extend(issues)
            except Exception:
                logger.warning("[GEPA] Model B 验证失败: candidate=%d", idx)
                continue

        return compute_fitness(len(all_candidates), all_issues, self._config.min_confidence)


async def _maybe_await(result: Any) -> None:
    """如果结果是 coroutine，await 它。"""
    if asyncio.iscoroutine(result):
        await result


async def _resolve(result: Any) -> Any:
    """如果结果是 coroutine/awaitable，await 后返回值；否则原样返回。

    供注入点（instruction_proposer / evaluate_fn）支持同步或异步实现。
    """
    if asyncio.iscoroutine(result):
        return await result
    return result


def _dominates(
    a: Mapping[str, float],
    b: Mapping[str, float],
    instances: Sequence[str],
) -> bool:
    """a 是否 Pareto 支配 b：所有实例 a>=b，且至少一个实例 a>b。"""
    ge = all(a.get(k, 0.0) >= b.get(k, 0.0) for k in instances)
    gt = any(a.get(k, 0.0) > b.get(k, 0.0) for k in instances)
    return ge and gt


def _pareto_frontier_indices(
    fitness_results: Sequence[FitnessResult],
) -> list[int]:
    """返回 Pareto 前沿（非支配集）个体下标。

    基于每实例适应度 per_instance_scores（官方 GEPA 机制 3）：
    一个候选只要在「某本书/某章节」上是当前最优，就保留在前沿，
    避免单标量最优淹没局部专长个体。

    无任何每实例分时，退化为综合适应度最优的单个个体。
    """
    scores = [dict(f.per_instance_scores) for f in fitness_results]
    if not any(scores):
        best = max(
            range(len(fitness_results)),
            key=lambda i: fitness_results[i].fitness,
        )
        return [best]

    instances = sorted({k for s in scores for k in s})
    frontier: list[int] = []
    for i, si in enumerate(scores):
        dominated = any(
            j != i and _dominates(scores[j], si, instances)
            for j in range(len(scores))
        )
        if not dominated:
            frontier.append(i)
    return frontier

