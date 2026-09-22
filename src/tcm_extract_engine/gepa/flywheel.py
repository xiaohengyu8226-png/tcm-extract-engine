"""GEPA 外圈飞轮（P4）：跨书进化 + 质量趋势聚合。

内圈（gepa_runner.evolve_template）在单书上进化模板；外圈把同一模板放到多本书的
切片上评估，用 per_instance_scores（每本书一个分）驱动 Pareto 跨书选择——胜出的
约束块必须对多本书整体更优，而非过拟合单本。

飞轮闭环：多书反馈 → 反思进化 → 人工晋升 → version bump 重刷 → 抽取质量↑ →
人工复核量↓ → 有余力复核更多书 → 更多反馈。台账（TrendRecord）记录每一圈的质量点。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tcm_extract_engine.core.artifacts import ArtifactHeader, save_artifact
from tcm_extract_engine.core.llm import UnifiedLLMClient
from tcm_extract_engine.gepa.bridge import (
    evaluate_candidate,
    make_reflective_proposer,
    make_system_aware_merger,
)
from tcm_extract_engine.gepa.engine import (
    GenerationReport,
    GEPAConfig,
    GEPAEngine,
    PromptIndividual,
    compute_fitness,
)
from tcm_extract_engine.gepa.runner import (
    EvolveResult,
    TrendRecord,
    _slice_book_artifacts,
    append_trend,
    ledger_path_for,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BookSlice:
    """一本书为进化切出的产物引用。"""

    book_id: str
    blocks_path: Path
    plan_path: Path


def plan_book_slices(
    *,
    books_root: Path,
    book_ids: list[str],
    template: str,
    out_dir: Path,
    section_ids_by_book: dict[str, list[str]] | None = None,
) -> list[BookSlice]:
    """为每本书切出「仅目标模板章节」的 blocks + plan，落到 out_dir/<book_id>/。"""
    slices: list[BookSlice] = []
    for book_id in book_ids:
        book_root = Path(books_root) / book_id
        if not book_root.is_dir():
            logger.warning("[Flywheel] 跳过不存在的书: %s", book_root)
            continue
        sec_ids = (section_ids_by_book or {}).get(book_id)
        slice_dir = Path(out_dir) / book_id
        blocks_path, plan_path, real_id = _slice_book_artifacts(
            book_root, template, sec_ids, slice_dir
        )
        slices.append(BookSlice(book_id=real_id, blocks_path=blocks_path, plan_path=plan_path))
    if not slices:
        raise ValueError(f"无可用书切片（template={template}, book_ids={book_ids}）")
    return slices


def make_flywheel_evaluate_fn(
    *,
    slices: list[BookSlice],
    template: str,
    extract_client: UnifiedLLMClient,
    verify_client: UnifiedLLMClient,
    work_dir_root: Path,
    verify_sample: int | None = 20,
) -> Any:
    """构造跨书 evaluate_fn：对每本书评估候选约束块，聚合成一个 FitnessResult。

    - 适应度：合并所有书的 issues + 记录数，用 compute_fitness 精确重算（全局 pass_rate）。
    - per_instance_scores：{book_id: 该书适应度} —— Pareto 跨书选择的输入。
    - feedback：按书分段拼接，供反思器看到「哪本书错在哪」。
    - 每候选按 fingerprint × book 分独立 work_dir → 跨代复用缓存（强缓存）。
    """
    from tcm_extract_engine.gepa.engine import FitnessResult

    async def evaluate(
        individual: PromptIndividual, chunks: Any = None
    ) -> FitnessResult:
        _ = chunks
        all_issues: list[dict[str, Any]] = []
        total_candidates = 0
        per_instance: dict[str, float] = {}
        feedback_parts: list[str] = []
        for sl in slices:
            work_dir = Path(work_dir_root) / sl.book_id / f"cand_{individual.fingerprint()}"
            outcome = await evaluate_candidate(
                book_id=sl.book_id,
                blocks_path=sl.blocks_path,
                plan_path=sl.plan_path,
                target_template=template,
                constraint_block=individual.extraction_prompt,
                extract_client=extract_client,
                verify_client=verify_client,
                work_dir=work_dir,
                verify_sample=verify_sample,
            )
            all_issues.extend(outcome.issues)
            total_candidates += outcome.records_in_scope
            per_instance[sl.book_id] = outcome.fitness.fitness
            if outcome.feedback:
                feedback_parts.append(f"【{sl.book_id}】\n{outcome.feedback}")

        aggregate = compute_fitness(total_candidates=total_candidates, issues=all_issues)
        return replace(
            aggregate,
            feedback="\n\n".join(feedback_parts),
            per_instance_scores=per_instance,
        )

    return evaluate


async def evolve_across_books(
    *,
    book_ids: list[str],
    template: str,
    books_root: Path,
    out_root: Path,
    section_ids_by_book: dict[str, list[str]] | None = None,
    population: int = 2,
    generations: int = 2,
    budget: int = 6,
    verify_sample: int | None = 20,
    with_baseline: bool = True,
    use_merge: bool = False,
    crossover_rate: float | None = None,
    extract_client: UnifiedLLMClient | None = None,
    verify_client: UnifiedLLMClient | None = None,
    reflect_client: UnifiedLLMClient | None = None,
    on_generation: Any = None,
) -> EvolveResult:
    """在多本书切片上跑 GEPA 进化（Pareto 跨书选择），落盘 + 记台账。

    use_merge=True 时启用 system-aware 合并（官方 GEPA 机制④）：把「各在不同书上
    取胜」的两个约束块由 LLM 合成一个兼顾双方的新块。合并复用 reflect_client。
    crossover_rate 未显式给出时，启用合并则默认 0.5、否则 0.0（P4 行为）。
    """
    from tcm_extract_engine.common.config import get_settings

    effective_crossover = (
        crossover_rate if crossover_rate is not None else (0.5 if use_merge else 0.0)
    )
    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{hashlib.sha256((','.join(book_ids) + ':' + template).encode()).hexdigest()[:6]}"
    )
    run_dir = Path(out_root) / "_multi" / template / run_id
    slices = plan_book_slices(
        books_root=books_root,
        book_ids=book_ids,
        template=template,
        out_dir=run_dir / "slices",
        section_ids_by_book=section_ids_by_book,
    )
    cands_dir = run_dir / "candidates"

    owned: list[UnifiedLLMClient] = []
    if extract_client is None or verify_client is None or reflect_client is None:
        settings = get_settings()
        if extract_client is None:
            extract_client = UnifiedLLMClient(settings.extract_llm)
            owned.append(extract_client)
        if verify_client is None:
            verify_client = UnifiedLLMClient(settings.verify_llm)
            owned.append(verify_client)
        if reflect_client is None:
            reflect_client = UnifiedLLMClient(settings.prompt_llm)
            owned.append(reflect_client)

    evaluate_fn = make_flywheel_evaluate_fn(
        slices=slices,
        template=template,
        extract_client=extract_client,
        verify_client=verify_client,
        work_dir_root=cands_dir,
        verify_sample=verify_sample,
    )
    proposer = make_reflective_proposer(
        llm_client=reflect_client, target_template=template
    )
    merger = (
        make_system_aware_merger(llm_client=reflect_client, target_template=template)
        if use_merge
        else None
    )
    config = GEPAConfig(
        population_size=population,
        max_generations=generations,
        eval_sample_size=1,
        convergence_threshold=1.5,
        selection_strategy="pareto",     # 跨书 Pareto：任一本书最优即留前沿
        crossover_rate=effective_crossover,
        budget_metric_calls=budget,
    )
    engine = GEPAEngine(
        None,
        config,
        evaluate_fn=evaluate_fn,
        instruction_proposer=proposer,
        merge_fn=merger,
    )

    seed = PromptIndividual(prompt_id="seed", extraction_prompt="", verification_prompt="")
    baseline_fitness: float | None = None
    gen_reports: list[GenerationReport] = []

    def _collect(rep: GenerationReport) -> None:
        gen_reports.append(rep)
        if on_generation is not None:
            on_generation(rep)

    try:
        if with_baseline:
            base_fr = await evaluate_fn(seed)   # 复用 seed 缓存目录，gen0 命中
            baseline_fitness = base_fr.fitness
        report = await engine.optimize(
            initial_individual=seed,
            sample_chunks=[None],
            on_generation=_collect,
        )
    finally:
        for client in owned:
            await client.close()

    best_block = report.best_individual.extraction_prompt.strip()
    improved = baseline_fitness is None or report.final_fitness > baseline_fitness + 1e-9

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_constraint.md"
    best_path.write_text(best_block + "\n", encoding="utf-8")

    tokens = {
        "extract_input": extract_client.usage.input_tokens,
        "extract_output": extract_client.usage.output_tokens,
        "extract_calls": extract_client.usage.calls,
        "verify_input": verify_client.usage.input_tokens,
        "verify_output": verify_client.usage.output_tokens,
        "verify_calls": verify_client.usage.calls,
        "reflect_input": reflect_client.usage.input_tokens,
        "reflect_output": reflect_client.usage.output_tokens,
        "reflect_calls": reflect_client.usage.calls,
    }
    total_tokens = sum(
        v for k, v in tokens.items() if k.endswith("_input") or k.endswith("_output")
    )
    report_path = run_dir / "run.json"
    save_artifact(
        report_path,
        header=ArtifactHeader(
            schema_version="1",
            book_id=",".join(sl.book_id for sl in slices),
            task_id=f"multi:gepa:{template}:{run_id}",
            model=extract_client.model,
            extra={
                "template": template,
                "run_id": run_id,
                "mode": "flywheel",
                "book_ids": [sl.book_id for sl in slices],
                "population": population,
                "generations": generations,
                "budget_metric_calls": budget,
                "verify_sample": verify_sample,
                "use_merge": use_merge,
                "crossover_rate": effective_crossover,
            },
        ),
        data={
            "template": template,
            "run_id": run_id,
            "mode": "flywheel",
            "book_ids": [sl.book_id for sl in slices],
            "baseline_fitness": baseline_fitness,
            "best_fitness": report.final_fitness,
            "improved": improved,
            "fitness_history": report.fitness_history,
            "total_generations": report.total_generations,
            "eval_calls": engine._eval_calls,
            "converged": report.converged,
            "elapsed_seconds": round(report.total_elapsed_seconds, 2),
            "best_prompt_id": report.best_individual.prompt_id,
            "best_mutation": report.best_individual.mutation_applied,
            "best_constraint": best_block,
            "tokens": tokens,
            "generations": [
                {
                    "generation": g.generation,
                    "best_fitness": g.best_fitness,
                    "avg_fitness": g.avg_fitness,
                    "mutations_applied": g.mutations_applied,
                }
                for g in gen_reports
            ],
        },
    )
    append_trend(
        ledger_path_for(out_root),
        TrendRecord(
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            kind="evolve",
            template=template,
            book_ids=[sl.book_id for sl in slices],
            baseline_fitness=baseline_fitness,
            best_fitness=report.final_fitness,
            improved=improved,
            eval_calls=engine._eval_calls,
            total_tokens=total_tokens,
            run_id=run_id,
        ),
    )
    logger.info(
        "[Flywheel] 跨书进化完成 template=%s books=%d run=%s baseline=%s best=%.4f improved=%s",
        template, len(slices), run_id, baseline_fitness, report.final_fitness, improved,
    )
    return EvolveResult(
        book_id=",".join(sl.book_id for sl in slices),
        template=template,
        run_id=run_id,
        run_dir=run_dir,
        report_path=report_path,
        best_constraint_path=best_path,
        baseline_fitness=baseline_fitness,
        best_fitness=report.final_fitness,
        fitness_history=report.fitness_history,
        best_constraint=best_block,
        best_mutation=report.best_individual.mutation_applied,
        improved=improved,
        eval_calls=engine._eval_calls,
        total_generations=report.total_generations,
        tokens=tokens,
    )


@dataclass(frozen=True, slots=True)
class AllTemplatesOutcome:
    """4 模板规模化批处理结果。"""

    results: list[EvolveResult]
    skipped: dict[str, str]
    """{模板名: 跳过原因}（如某模板在所给书上无匹配章节）。"""


async def evolve_all_templates(
    *,
    book_ids: list[str],
    books_root: Path,
    out_root: Path,
    templates: list[str] | None = None,
    section_ids_by_book: dict[str, list[str]] | None = None,
    population: int = 2,
    generations: int = 2,
    budget: int = 6,
    verify_sample: int | None = 20,
    with_baseline: bool = True,
    use_merge: bool = False,
    extract_client: UnifiedLLMClient | None = None,
    verify_client: UnifiedLLMClient | None = None,
    reflect_client: UnifiedLLMClient | None = None,
    on_template: Any = None,
) -> AllTemplatesOutcome:
    """对 4 个 A 模板依次跑跨书飞轮，汇总「模板 × baseline/best/是否建议晋升」。

    单个模板失败（如所给书无该模板章节）不阻断其余模板：记入 skipped 并继续。
    三个 LLM 客户端在批处理内只创建/复用一次，全部模板共享（用量累计）。
    on_template(template, result_or_none, skip_reason) 在每个模板结束时回调，供 CLI 实时渲染。
    """
    from tcm_extract_engine.cab.planner import EXTRACTOR_TEMPLATES
    from tcm_extract_engine.common.config import get_settings

    targets = templates or list(EXTRACTOR_TEMPLATES.values())
    owned: list[UnifiedLLMClient] = []
    if extract_client is None or verify_client is None or reflect_client is None:
        settings = get_settings()
        if extract_client is None:
            extract_client = UnifiedLLMClient(settings.extract_llm)
            owned.append(extract_client)
        if verify_client is None:
            verify_client = UnifiedLLMClient(settings.verify_llm)
            owned.append(verify_client)
        if reflect_client is None:
            reflect_client = UnifiedLLMClient(settings.prompt_llm)
            owned.append(reflect_client)

    results: list[EvolveResult] = []
    skipped: dict[str, str] = {}
    try:
        for template in targets:
            try:
                res = await evolve_across_books(
                    book_ids=book_ids,
                    template=template,
                    books_root=books_root,
                    out_root=out_root,
                    section_ids_by_book=section_ids_by_book,
                    population=population,
                    generations=generations,
                    budget=budget,
                    verify_sample=verify_sample,
                    with_baseline=with_baseline,
                    use_merge=use_merge,
                    extract_client=extract_client,
                    verify_client=verify_client,
                    reflect_client=reflect_client,
                )
            except Exception as exc:  # noqa: BLE001 - 单模板失败不阻断批处理
                reason = f"{type(exc).__name__}: {exc}"
                logger.warning("[Flywheel] 模板 %s 进化跳过：%s", template, reason)
                skipped[template] = reason
                if on_template is not None:
                    on_template(template, None, reason)
                continue
            results.append(res)
            if on_template is not None:
                on_template(template, res, None)
    finally:
        for client in owned:
            await client.close()
    return AllTemplatesOutcome(results=results, skipped=skipped)
