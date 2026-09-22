"""GEPA 进化 runner（P3）：把 P0 桥接 + P1/P2 注入点编排成可落盘、可晋升的流程。

两个动作，严格分离（守「人工确认后晋升」的门）：

  evolve_template —— 对「某本书 × 某个 A 模板」跑小预算 GEPA 进化，
      产物落到 workspace/gepa/<book>/<template>/<run_id>/：
        run.json          进化报告（baseline/best/轨迹/token）
        best_constraint.md 胜出约束块（promote 读它）
        slice/            本次进化用的切片 blocks+plan（可复现）
        candidates/       每候选独立缓存目录（强缓存、互不污染）
      *不碰* prompts/ 与 BookPipeline 主体。

  promote_constraint —— 人工门：把胜出约束块写入 prompts/a/<template> 的
      「受管标记块」内（<!-- GEPA:PROMOTED:START/END -->），base 正文逐字不动。
      写入 → 模板内容 hash 变 → PromptTemplate.version 变 →
      BookPipeline 依 prompt_version 幂等判定，对相关书 rerun extract 即「version bump 重刷」。
      重复晋升替换同一标记块（幂等、可回滚）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tcm_extract_engine.cab.planner import PromptTemplateRegistry
from tcm_extract_engine.core.artifacts import ArtifactHeader, load_artifact, save_artifact
from tcm_extract_engine.core.llm import UnifiedLLMClient
from tcm_extract_engine.gepa.bridge import (
    evaluate_candidate,
    make_evaluate_fn,
    make_reflective_proposer,
)
from tcm_extract_engine.gepa.engine import (
    GenerationReport,
    GEPAConfig,
    GEPAEngine,
    PromptIndividual,
)

logger = logging.getLogger(__name__)

PROMOTE_START = "<!-- GEPA:PROMOTED:START -->"
PROMOTE_END = "<!-- GEPA:PROMOTED:END -->"


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _template_version(content: str) -> str:
    """与 planner.PromptTemplate.version 一致：sha256(content)[:12]。"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


# ── 质量趋势台账（append-only JSONL）────────────────────────────────────────


@dataclass(slots=True)
class TrendRecord:
    """飞轮台账的一条记录：一次 evolve 或 promote 的质量快照。"""

    ts: str
    kind: str                       # evolve | promote
    template: str
    book_ids: list[str] = field(default_factory=list)
    baseline_fitness: float | None = None
    best_fitness: float | None = None
    improved: bool = False
    eval_calls: int = 0
    total_tokens: int = 0
    run_id: str = ""
    promoted: bool = False
    old_version: str = ""
    new_version: str = ""
    reviewer: str = ""


def ledger_path_for(out_root: Path) -> Path:
    """台账固定落在进化产物根目录下（跨书、跨模板共用一条时间线）。"""
    return Path(out_root) / "_trend.jsonl"


def append_trend(ledger_path: Path, record: TrendRecord) -> None:
    """追加一条台账记录（append-only，永不改写历史）。"""
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_trend(ledger_path: Path) -> list[TrendRecord]:
    """读取台账（容忍空文件/坏行）。"""
    ledger_path = Path(ledger_path)
    if not ledger_path.is_file():
        return []
    records: list[TrendRecord] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(TrendRecord(**json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            logger.warning("[GEPARunner] 跳过坏台账行: %s", line[:80])
    return records


# ── 切片：从真书产物切出「仅目标模板章节」的 blocks + plan ─────────────────────


def _slice_book_artifacts(
    book_root: Path,
    template: str,
    section_ids: list[str] | None,
    out_dir: Path,
) -> tuple[Path, Path, str]:
    """切出 blocks.jsonl + extraction_plan.json（仅目标章节），返回 (blocks, plan, book_id)。

    控成本的核心旋钮：只喂给 GEPA 与目标模板相关的章节，避免整本书全量抽取。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = load_artifact(book_root / "03_plan/extraction_plan.json")
    sections = plan.get("sections", [])
    if section_ids:
        wanted = set(section_ids)
        chosen = [s for s in sections if s.get("section_id") in wanted]
    else:
        chosen = [
            s for s in sections
            if s.get("template") == template and not s.get("skip_reason")
        ]
    if not chosen:
        raise ValueError(
            f"书中无使用模板 {template!r} 的可抽章节（book={book_root.name}）"
        )

    keep_blocks: set[str] = set()
    for sec in chosen:
        keep_blocks.update(str(b) for b in sec.get("block_ids", []))

    # 切 blocks：保留 meta 行 + 命中 block
    src_blocks = book_root / "01_parse/blocks.jsonl"
    out_blocks = out_dir / "blocks.jsonl"
    kept = 0
    with out_blocks.open("w", encoding="utf-8") as fout:
        for line in src_blocks.open(encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "meta":
                obj["total"] = len(keep_blocks)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            elif str(obj.get("blockid")) in keep_blocks:
                fout.write(line + "\n")
                kept += 1

    sliced_plan = dict(plan)
    sliced_plan["sections"] = chosen
    out_plan = out_dir / "extraction_plan.json"
    out_plan.write_text(
        json.dumps(sliced_plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    book_id = plan["header"]["book_id"]
    logger.info(
        "[GEPARunner] 切片 book=%s template=%s sections=%d blocks=%d",
        book_id, template, len(chosen), kept,
    )
    return out_blocks, out_plan, book_id


# ── evolve 结果容器 ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class EvolveResult:
    """一次 evolve 运行的结果与产物位置。"""

    book_id: str
    template: str
    run_id: str
    run_dir: Path
    report_path: Path
    best_constraint_path: Path
    baseline_fitness: float | None
    best_fitness: float
    fitness_history: list[float]
    best_constraint: str
    best_mutation: str
    improved: bool
    eval_calls: int
    total_generations: int
    tokens: dict[str, int] = field(default_factory=dict)


async def evolve_template(
    *,
    book_id: str,
    template: str,
    books_root: Path,
    out_root: Path,
    section_ids: list[str] | None = None,
    population: int = 2,
    generations: int = 2,
    budget: int = 4,
    verify_sample: int | None = 20,
    selection_strategy: str = "pareto",
    with_baseline: bool = True,
    extract_client: UnifiedLLMClient | None = None,
    verify_client: UnifiedLLMClient | None = None,
    reflect_client: UnifiedLLMClient | None = None,
    on_generation: Any = None,
) -> EvolveResult:
    """对「一本书 × 一个 A 模板」跑小预算 GEPA 进化并落盘产物。

    客户端可注入（测试用 stub）；未注入则按 settings 建真客户端并在结束时关闭。
    baseline（空约束块）复用 seed 的候选缓存目录 → gen0 命中缓存，几乎零额外成本。
    """
    from tcm_extract_engine.common.config import get_settings

    book_root = Path(books_root) / book_id
    if not book_root.is_dir():
        raise FileNotFoundError(f"书籍工作目录不存在: {book_root}")

    run_id = f"{_now_tag()}_{hashlib.sha256(f'{book_id}:{template}'.encode()).hexdigest()[:6]}"
    run_dir = Path(out_root) / book_id / template / run_id
    slice_dir = run_dir / "slice"
    blocks_path, plan_path, real_book_id = _slice_book_artifacts(
        book_root, template, section_ids, slice_dir
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
    evaluate_fn = make_evaluate_fn(
        book_id=real_book_id,
        blocks_path=blocks_path,
        plan_path=plan_path,
        target_template=template,
        extract_client=extract_client,
        verify_client=verify_client,
        work_dir_root=cands_dir,
        verify_sample=verify_sample,
    )
    proposer = make_reflective_proposer(
        llm_client=reflect_client, target_template=template
    )
    config = GEPAConfig(
        population_size=population,
        max_generations=generations,
        eval_sample_size=1,
        convergence_threshold=1.5,          # 不可达：靠 budget 停，轨迹完整可观察
        selection_strategy=selection_strategy,
        crossover_rate=0.0,                 # 关闭交叉，专注反思变异
        budget_metric_calls=budget,
    )
    engine = GEPAEngine(
        None,                               # 注入 evaluate_fn 后不使用内置网关
        config,
        evaluate_fn=evaluate_fn,
        instruction_proposer=proposer,
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
            # 复用 seed 候选缓存目录 → 与 gen0 的 seed 评估同一 work_dir，缓存命中
            seed_dir = cands_dir / f"cand_{seed.fingerprint()}"
            base_outcome = await evaluate_candidate(
                book_id=real_book_id,
                blocks_path=blocks_path,
                plan_path=plan_path,
                target_template=template,
                constraint_block="",
                extract_client=extract_client,
                verify_client=verify_client,
                work_dir=seed_dir,
                verify_sample=verify_sample,
            )
            baseline_fitness = base_outcome.fitness.fitness
        report = await engine.optimize(
            initial_individual=seed,
            sample_chunks=[None],           # evaluate_fn 忽略 chunks
            on_generation=_collect,
        )
    finally:
        for client in owned:
            await client.close()

    best_block = report.best_individual.extraction_prompt.strip()
    improved = baseline_fitness is None or report.final_fitness > baseline_fitness + 1e-9

    # 落盘产物
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
    report_path = run_dir / "run.json"
    save_artifact(
        report_path,
        header=ArtifactHeader(
            schema_version="1",
            book_id=real_book_id,
            task_id=f"{real_book_id}:gepa:{template}:{run_id}",
            model=extract_client.model,
            extra={
                "template": template,
                "run_id": run_id,
                "population": population,
                "generations": generations,
                "budget_metric_calls": budget,
                "verify_sample": verify_sample,
                "selection_strategy": selection_strategy,
            },
        ),
        data={
            "template": template,
            "run_id": run_id,
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
    logger.info(
        "[GEPARunner] evolve 完成 book=%s template=%s run=%s baseline=%s best=%.4f improved=%s",
        real_book_id, template, run_id, baseline_fitness, report.final_fitness, improved,
    )
    append_trend(
        ledger_path_for(out_root),
        TrendRecord(
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            kind="evolve",
            template=template,
            book_ids=[real_book_id],
            baseline_fitness=baseline_fitness,
            best_fitness=report.final_fitness,
            improved=improved,
            eval_calls=engine._eval_calls,
            total_tokens=sum(
                tokens[k] for k in tokens if k.endswith("_input") or k.endswith("_output")
            ),
            run_id=run_id,
        ),
    )
    return EvolveResult(
        book_id=real_book_id,
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


# ── promote：人工门，写入受管标记块 ──────────────────────────────────────────


@dataclass(slots=True)
class PromoteResult:
    """一次晋升的结果。"""

    template: str
    prompts_path: Path
    old_version: str
    new_version: str
    version_bumped: bool
    changed: bool
    dry_run: bool
    promoted_block: str


def load_run_best_constraint(run_dir: Path) -> tuple[str, str]:
    """从 evolve 产物读取胜出约束块 + template，返回 (constraint_block, template)。"""
    run_dir = Path(run_dir)
    report = load_artifact(run_dir / "run.json")
    template = report.get("template", "")
    best_path = run_dir / "best_constraint.md"
    if best_path.is_file():
        block = best_path.read_text(encoding="utf-8").strip()
    else:
        block = str(report.get("best_constraint", "")).strip()
    if not template:
        raise ValueError(f"run.json 缺少 template 字段: {run_dir}")
    return block, template


def promote_constraint(
    *,
    template: str,
    constraint_block: str,
    reviewer: str = "cli",
    run_id: str = "",
    prompts_root: Path | None = None,
    dry_run: bool = False,
    ledger_path: Path | None = None,
) -> PromoteResult:
    """把约束块写入 prompts/a/<template> 的受管标记块（base 正文逐字不动）。

    - 已存在标记块 → 整体替换（幂等，避免多次晋升叠加膨胀）。
    - 不存在 → 追加到文件末尾。
    - 空约束块 → 移除标记块（等价于回滚到基线模板）。
    - dry_run → 只计算新版本号与差异，不落盘。
    """
    root = Path(prompts_root) if prompts_root is not None else PromptTemplateRegistry().prompts_root
    path = root / "a" / template
    if not path.is_file():
        raise FileNotFoundError(f"目标模板不存在: {path}")

    content = path.read_text(encoding="utf-8")
    old_version = _template_version(content)
    block = (constraint_block or "").strip()

    if block:
        header = "## 进化约束（GEPA 晋升"
        meta = " · ".join(x for x in (run_id, reviewer) if x)
        managed_body = f"{header}{' · ' + meta if meta else ''}）\n{block}\n"
        managed = f"{PROMOTE_START}\n{managed_body}{PROMOTE_END}"
    else:
        managed = ""

    has_block = PROMOTE_START in content and PROMOTE_END in content
    if has_block:
        pattern = re.compile(
            re.escape(PROMOTE_START) + r".*?" + re.escape(PROMOTE_END) + r"\n?",
            re.DOTALL,
        )
        new_content = pattern.sub(managed + ("\n" if managed else ""), content)
    elif managed:
        new_content = content.rstrip("\n") + "\n\n" + managed + "\n"
    else:
        new_content = content

    new_version = _template_version(new_content)
    changed = new_content != content
    if changed and not dry_run:
        path.write_text(new_content, encoding="utf-8")
        logger.info(
            "[GEPARunner] 晋升 template=%s version %s→%s reviewer=%s run=%s",
            template, old_version, new_version, reviewer, run_id,
        )
        if ledger_path is not None:
            append_trend(
                ledger_path,
                TrendRecord(
                    ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    kind="promote",
                    template=template,
                    run_id=run_id,
                    reviewer=reviewer,
                    promoted=bool(block),
                    old_version=old_version,
                    new_version=new_version,
                ),
            )
    return PromoteResult(
        template=template,
        prompts_path=path,
        old_version=old_version,
        new_version=new_version,
        version_bumped=old_version != new_version,
        changed=changed,
        dry_run=dry_run,
        promoted_block=block,
    )
