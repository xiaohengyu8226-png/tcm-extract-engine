"""GEPA ↔ BookPipeline 桥接层（P0：真实评估注入）。

职责（严格限定，不改 GEPAEngine 与 BookPipeline 主体）：
  1. ConstrainedTemplateRegistry —— 把「候选追加约束块」注入到 *指定* A 模板，
     模板正文逐字不变（守 planner 的防漂移原则），render 时占位符照常替换。
  2. verdict_to_issues —— 把 B 的 VerificationReport + 确定性 ValidationReport
     映射为 GEPA compute_fitness 需要的 issue 结构（issue_type/severity/
     confidence/json_pointer），对齐 knowledge.gepa._MUTATION_TEMPLATES 的类型。
  3. evaluate_candidate —— 真实评估：用真 A 模板+extractor+validate+B verifier
     对「某本书 × 某个 A 模板 × 某段约束块」打分，产出 EvalOutcome
     （fitness + 富文本 feedback + 每实例分）。

P0 阶段只使用 knowledge.gepa 现有的 compute_fitness / FitnessResult，不改动 gepa.py；
feedback 与 per_instance_scores 先挂在 EvalOutcome 上，P1 再折叠进 FitnessResult。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from tcm_extract_engine.cab.extractors import extract_records
from tcm_extract_engine.cab.planner import (
    CHUNK_PLACEHOLDER,
    BookAdaptedRegistry,
    PromptTemplate,
    PromptTemplateRegistry,
    SectionPlan,
)
from tcm_extract_engine.cab.profiler import load_blocks
from tcm_extract_engine.cab.validator import ValidationReport, validate_records
from tcm_extract_engine.cab.verifier import VerificationReport, verify_records
from tcm_extract_engine.core.artifacts import load_artifact
from tcm_extract_engine.core.llm import UnifiedLLMClient
from tcm_extract_engine.core.records import SourceRecord
from tcm_extract_engine.domains import get_domain_pack
from tcm_extract_engine.gepa.engine import (
    EvaluateFn,
    FitnessResult,
    MergeFn,
    PromptIndividual,
    ProposalFn,
    compute_fitness,
    mutate_prompt,
)

logger = logging.getLogger(__name__)


# ── 约束块注入 registry ──────────────────────────────────────────────────────


class ConstrainedTemplateRegistry(PromptTemplateRegistry):
    """在指定 A 模板末尾追加「进化约束块」的模板注册表。

    - 只影响 role=="a" 且 name==target_template 的模板；其余（含 B/C）原样返回。
    - 模板正文逐字不变，约束块追加在末尾（CHUNK_PLACEHOLDER 之前/之后均可，
      占位符本身保持完整，render 照常替换）。
    - 每个候选约束块 → 内容 hash 变 → PromptTemplate.version 变，天然区分版本。
    """

    def __init__(
        self,
        target_template: str,
        constraint_block: str = "",
        *,
        prompts_root: Path | None = None,
        base_registry: PromptTemplateRegistry | None = None,
    ) -> None:
        super().__init__(prompts_root)
        self.base_registry = base_registry or PromptTemplateRegistry(prompts_root)
        self.target_template = target_template
        self.constraint_block = constraint_block.strip()

    def load(self, role: str, name: str) -> PromptTemplate:
        if role != "a" or name != self.target_template or not self.constraint_block:
            return self.base_registry.load(role, name)
        key = f"{role}/{name}|+constraint({hash(self.constraint_block) & 0xFFFF:04x})"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        base = self.base_registry.load(role, name)
        content = f"{base.content.rstrip()}\n\n{self.constraint_block}\n"
        template = PromptTemplate(name=base.name, path=base.path, content=content)
        self._cache[key] = template
        return template


# ── verdict / validation → GEPA issues ──────────────────────────────────────

# 信号映射（verdict/category → issue 三元组）由领域包声明（domains/tcm/gepa_map.py），
# 经 get_domain_pack() 读取；SUPPORTED / PARTIAL 不在映射内即不计入（PARTIAL 多为兜底，噪声大）。

# validate 的 ERROR 与 WARNING 都计问题，但 WARNING 弱化：严重度降一级、
# confidence 打折（仍高于 compute_fitness 的 min_confidence floor，故不被过滤掉）。
_SEVERITY_CATEGORY_CONFIDENCE_FLOOR = 0.5
_SEVERITY_DOWNGRADE = {"critical": "high", "high": "medium", "medium": "low", "low": "low"}
_WARNING_CONFIDENCE_FACTOR = 0.65


def verdict_to_issues(
    records_in_scope: Sequence[SourceRecord],
    verification: VerificationReport,
    validation: ValidationReport,
    *,
    pack: Any = None,
) -> list[dict[str, Any]]:
    """把 B 核验 + 确定性校验映射为 GEPA compute_fitness 的 issue 列表。

    json_pointer 采用 /records/{index} 形式，index = 记录在 records_in_scope 中的
    位置；compute_fitness 用它去重统计「受影响候选数」。
    """
    pack = pack or get_domain_pack(records_in_scope[0].domain_id if records_in_scope else "tcm")
    index_by_id = {r.record_id: i for i, r in enumerate(records_in_scope)}
    in_scope_ids = set(index_by_id)
    issues: list[dict[str, Any]] = []

    # 1) B 核验 verdict
    for result in verification.results:
        idx = index_by_id.get(result.record_id)
        if idx is None:
            continue
        mapped = pack.verdict_issue_map.get(result.verdict.value)
        if mapped is None:
            continue
        issue_type, severity, confidence = mapped
        issues.append(
            {
                "issue_type": issue_type,
                "severity": severity,
                "confidence": confidence,
                "json_pointer": f"/records/{idx}",
                "message": result.explanation or result.verdict.value,
            }
        )

    # 2) 确定性校验 ERROR + WARNING（WARNING 降一级严重度、confidence 打折，
    #    信号弱于 ERROR 但不再被完全忽略；同记录同类别去重，ERROR 优先保留）
    seen: set[tuple[int, str]] = set()
    ordered = sorted(
        validation.issues,
        key=lambda vi: 0 if vi.severity.value == "ERROR" else 1,
    )
    for vi in ordered:
        sev_name = vi.severity.value
        if sev_name not in ("ERROR", "WARNING"):
            continue
        idx = index_by_id.get(vi.record_id)
        if idx is None:
            continue
        mapped = pack.category_issue_map.get(vi.category.value)
        if mapped is None:
            continue
        issue_type, severity, confidence = mapped
        if sev_name == "WARNING":
            severity = _SEVERITY_DOWNGRADE.get(severity, "low")
            confidence = round(confidence * _WARNING_CONFIDENCE_FACTOR, 3)
        if confidence < _SEVERITY_CATEGORY_CONFIDENCE_FLOOR:
            continue
        dedup = (idx, issue_type)
        if dedup in seen:
            continue
        seen.add(dedup)
        issues.append(
            {
                "issue_type": issue_type,
                "severity": severity,
                "confidence": confidence,
                "json_pointer": f"/records/{idx}",
                "message": vi.message,
            }
        )

    _ = in_scope_ids  # 保留：未来按范围裁剪 verification 冗余结果时使用
    return issues


# ── 富文本反馈（喂给 P2 反思器）──────────────────────────────────────────────


def build_feedback_text(
    records_in_scope: Sequence[SourceRecord],
    verification: VerificationReport,
    validation: ValidationReport,
    *,
    max_items: int = 12,
) -> str:
    """把失败样本汇编成自然语言反馈（GEPA 反思式变异的核心素材）。

    官方 GEPA 的关键：metric 不止返回标量分，还要返回「为什么是这个分」的文本。
    这里把 B 的 verdict/explanation/evidence_quote + correction reason + 校验 ERROR
    拼成可读清单，供反思 LLM 诊断并产出针对性约束块。
    """
    record_by_id = {r.record_id: r for r in records_in_scope}
    lines: list[str] = []

    # B 核验失败项（优先严重 verdict）
    severity_order = {
        "UNSUPPORTED": 0,
        "WRONG_DOSE": 0,
        "WRONG_RELATION": 0,
        "WRONG_NORMALIZATION": 1,
        "MISSING": 1,
        "OCR_UNCERTAIN": 2,
    }
    failing = [
        r
        for r in verification.results
        if r.record_id in record_by_id and r.verdict.value in severity_order
    ]
    failing.sort(key=lambda r: severity_order.get(r.verdict.value, 9))

    corrections_by_record: dict[str, list[str]] = {}
    for c in verification.corrections:
        corrections_by_record.setdefault(c.record_id, []).append(
            f"{c.field}: {c.original!r} → {c.suggested!r}（{c.reason}）"
        )

    for r in failing[:max_items]:
        rec = record_by_id.get(r.record_id)
        snippet = rec.raw_text[:120] if rec else ""
        lines.append(
            f"- [{r.verdict.value}] {r.record_id}"
            + (f"：{r.explanation}" if r.explanation else "")
            + (f"｜原文「{snippet}」" if snippet else "")
        )
        for corr in corrections_by_record.get(r.record_id, [])[:3]:
            lines.append(f"    · 修正建议 {corr}")

    # 确定性校验 ERROR 汇总
    error_lines = [
        f"- [VALIDATE/{vi.category.value}] {vi.record_id}: {vi.message}"
        for vi in validation.issues
        if vi.severity.value == "ERROR"
    ]
    if error_lines:
        lines.append("确定性校验 ERROR：")
        lines.extend(error_lines[:max_items])

    # 确定性校验 WARNING 汇总（弱信号，排在 ERROR 之后，单独限量）
    warning_lines = [
        f"- [VALIDATE-WARN/{vi.category.value}] {vi.record_id}: {vi.message}"
        for vi in validation.issues
        if vi.severity.value == "WARNING"
    ]
    if warning_lines:
        lines.append("确定性校验 WARNING（弱信号，供参考）：")
        lines.extend(warning_lines[:max_items])

    if not lines:
        return "本批次未发现核验/校验问题（可能全部 SUPPORTED，或范围为空）。"
    return "\n".join(lines)


# ── 评估结果容器 ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class EvalOutcome:
    """单个候选（书 × 模板 × 约束块）的真实评估结果。"""

    fitness: FitnessResult
    feedback: str
    per_instance_scores: dict[str, float] = field(default_factory=dict)
    total_records: int = 0
    records_in_scope: int = 0
    evaluated_records: int = 0
    unassessed_records: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    """映射后的 GEPA issue 列表（跨书聚合时用于精确重算适应度）。"""


# ── 主入口：真实评估 ─────────────────────────────────────────────────────────


def _plan_sections(plan_path: Path) -> list[SectionPlan]:
    plan_data = load_artifact(plan_path)
    return [SectionPlan.model_validate(s) for s in plan_data.get("sections", [])]


def _scope_records(
    records: Sequence[SourceRecord],
    sections: Sequence[SectionPlan],
    target_template: str,
) -> list[SourceRecord]:
    """把打分范围限定到「使用了目标模板的章节」所产出的记录。

    依据 plan 中 template==target_template 的章节 block_ids，与记录 source.block_ids
    求交集；无法归属（无 block_ids）时回退为全部记录，保证 P0 冒烟不被空范围卡死。
    """
    return [r for r in records if not r.passthrough and r.generation.template == target_template]


async def evaluate_candidate(
    *,
    book_id: str,
    blocks_path: Path,
    plan_path: Path,
    target_template: str,
    constraint_block: str,
    extract_client: UnifiedLLMClient,
    verify_client: UnifiedLLMClient,
    work_dir: Path,
    registry: PromptTemplateRegistry | None = None,
    verify_sample: int | None = None,
    pack: Any = None,
    vision_client: UnifiedLLMClient | None = None,
    vision_verify_client: UnifiedLLMClient | None = None,
) -> EvalOutcome:
    """对「一本书 × 一个 A 模板 × 一段候选约束块」跑真实评估。

    流程：注入约束块 → extract_records（真 A 模板+extractor）→ validate（廉价预筛）
    → verify_records（B，可采样）→ verdict_to_issues → compute_fitness。

    Args:
        target_template: 被进化的 A 模板文件名（如 "general_rule.md"）。
        constraint_block: 候选追加约束块正文（空串 = 基线，模板正文原样）。
        work_dir: 候选专属缓存/中间产物目录（隔离，避免命中旧 prompt_version 缓存）。
        verify_sample: 仅对前 N 条范围内记录做 B 核验（省钱）；None = 全量。

    Returns:
        EvalOutcome（fitness + feedback + 每实例分 + token 用量）。
    """
    clients = list({id(c): c for c in [extract_client, verify_client, vision_client, vision_verify_client] if c is not None}.values())
    before_usage = {id(c): (c.usage.input_tokens, c.usage.output_tokens, c.usage.calls) for c in clients}
    plan_meta = load_artifact(plan_path)
    pack = pack or get_domain_pack(plan_meta.get("domain_id", "tcm"))
    if registry is None:
        registry = BookAdaptedRegistry(plan_meta.get("book_adaptations", {}))
    constrained = ConstrainedTemplateRegistry(
        target_template,
        constraint_block,
        prompts_root=registry.prompts_root,
        base_registry=registry,
    )
    cache_dir = work_dir / "extract_cache"
    verify_cache_dir = work_dir / "verify_cache"

    # 1) 真实抽取（Model A）
    records = await extract_records(
        book_id=book_id,
        blocks_path=blocks_path,
        plan_path=plan_path,
        cache_dir=cache_dir,
        client=extract_client,
        registry=constrained,
        pack=pack,
        vision_client=vision_client,
    )

    sections = _plan_sections(plan_path)
    scoped = _scope_records(records, sections, target_template)

    # 2) 确定性校验（廉价预筛；对范围内记录）
    blocks = load_blocks(blocks_path)
    validation = validate_records(list(scoped), blocks=blocks, pack=pack)

    # 3) B 核验（可采样以控成本）
    if verify_sample is not None and verify_sample < 1:
        raise ValueError("verify_sample 必须大于 0 或为 None")
    # Deterministic, section-stratified sample; a long first chapter cannot consume it all.
    ordered = sorted(scoped, key=lambda r: (r.source.section_path, r.record_id))
    groups: dict[tuple[str, ...], list[SourceRecord]] = {}
    for record in ordered:
        groups.setdefault(tuple(record.source.section_path), []).append(record)
    verify_targets: list[SourceRecord] = []
    limit = len(scoped) if verify_sample is None else min(verify_sample, len(scoped))
    while len(verify_targets) < limit:
        for group in groups.values():
            if group and len(verify_targets) < limit:
                verify_targets.append(group.pop(0))
    verification = await verify_records(
        records=list(verify_targets),
        blocks_path=blocks_path,
        client=verify_client,
        registry=constrained,
        cache_dir=verify_cache_dir,
        pack=pack,
        vision_client=vision_verify_client or vision_client,
    )

    # 4) 映射 → 适应度
    issues = verdict_to_issues(verify_targets, verification, validation, pack=pack)
    # Unassessed records are never counted as passed.
    fitness = compute_fitness(total_candidates=len(verify_targets), issues=issues)
    feedback = build_feedback_text(scoped, verification, validation)

    outcome = EvalOutcome(
        fitness=fitness,
        feedback=feedback,
        per_instance_scores={book_id: fitness.fitness},
        total_records=len(records),
        records_in_scope=len(scoped),
        evaluated_records=len(verify_targets),
        unassessed_records=len(scoped) - len(verify_targets),
        input_tokens=sum(c.usage.input_tokens-before_usage[id(c)][0] for c in clients),
        output_tokens=sum(c.usage.output_tokens-before_usage[id(c)][1] for c in clients),
        llm_calls=sum(c.usage.calls-before_usage[id(c)][2] for c in clients),
        issues=issues,
        quality_metrics={"assessed_records": len(verify_targets), "unassessed_records": len(scoped)-len(verify_targets),
                         "schema_errors": sum(i.severity.value == "ERROR" for i in validation.issues),
                         "located_records": sum(bool(r.source.block_ids) for r in scoped),
                         "reference_recall": None, "gold_status": "pending_human_confirmation",
                         "assessment": "model-assisted proxy; not gold accuracy"},
    )
    from tcm_extract_engine.core.artifacts import _atomic_write_json
    _atomic_write_json(work_dir / "evaluation.json", {
        "book_id": book_id, "template": target_template, "domain_id": pack.id,
        "fitness": fitness.fitness, "quality_metrics": outcome.quality_metrics,
        "issues": issues, "feedback": feedback, "record_ids": [r.record_id for r in scoped],
    })
    logger.info(
        "[GEPABridge] 评估 book=%s template=%s: fitness=%.4f 范围记录=%d/%d issues=%d calls=%d",
        book_id,
        target_template,
        fitness.fitness,
        len(scoped),
        len(records),
        len(issues),
        outcome.llm_calls,
    )
    return outcome


# ── P2: EvalOutcome → FitnessResult（折叠 feedback / 每实例分）───────────────


def fold_outcome_to_fitness(outcome: EvalOutcome) -> FitnessResult:
    """把 EvalOutcome 折叠成 GEPAEngine 注入点期望的 FitnessResult。

    P0 阶段 feedback / per_instance_scores 挂在 EvalOutcome 上；P1 给 FitnessResult
    扩了同名字段（反思变异读 feedback，Pareto 选择读 per_instance_scores）。
    这里用 dataclasses.replace 把两者补齐，保持 FitnessResult 不可变。
    """
    return replace(
        outcome.fitness,
        feedback=outcome.feedback,
        per_instance_scores=dict(outcome.per_instance_scores),
    )


def make_evaluate_fn(
    *,
    book_id: str,
    blocks_path: Path,
    plan_path: Path,
    target_template: str,
    extract_client: UnifiedLLMClient,
    verify_client: UnifiedLLMClient,
    work_dir_root: Path,
    registry: PromptTemplateRegistry | None = None,
    verify_sample: int | None = None,
    pack: Any = None,
    vision_client: UnifiedLLMClient | None = None,
    vision_verify_client: UnifiedLLMClient | None = None,
) -> EvaluateFn:
    """构造 GEPAEngine 的 evaluate_fn 注入：把「个体」映射为真实评估。

    约定：个体的 extraction_prompt 字段即「候选追加约束块」（正文模板不动，防漂移）。
    每个候选按 fingerprint 分独立 work_dir → 同约束块跨代复用缓存（强缓存），
    不同约束块互不污染。chunks 入参被忽略（真实评估吃 book 产物，不吃 SourceChunk）。
    """

    async def evaluate(
        individual: PromptIndividual, chunks: Sequence[Any] | None = None
    ) -> FitnessResult:
        _ = chunks  # 真实评估不依赖 SourceChunk 样本
        work_dir = Path(work_dir_root) / f"cand_{individual.fingerprint()}"
        outcome = await evaluate_candidate(
            book_id=book_id,
            blocks_path=blocks_path,
            plan_path=plan_path,
            target_template=target_template,
            constraint_block=individual.extraction_prompt,
            extract_client=extract_client,
            verify_client=verify_client,
            work_dir=work_dir,
            registry=registry,
            verify_sample=verify_sample,
            pack=pack,
            vision_client=vision_client,
            vision_verify_client=vision_verify_client,
        )
        return fold_outcome_to_fitness(outcome)

    return evaluate


# ── P2: 反思式变异（官方 GEPA 机制①）────────────────────────────────────────

_REFLECTION_SYSTEM_PROMPT = (
    "你是提示词进化器的反思模块。基于上一轮抽取的核验反馈，为抽取提示词"
    "追加一小段针对性约束，用以减少同类错误。严格遵守：只输出 JSON；"
    "禁止复述或改写原始模板正文；禁止编造原文不存在的医学规则；"
    "约束必须可执行、聚焦高频错误类型。"
)


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _sanitize_constraint_block(block: str, max_chars: int) -> str:
    """清洗反思产出的约束块：去代码围栏、剔除占位符（防破坏 render）、限长。"""
    text = (block or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.replace(CHUNK_PLACEHOLDER, "").strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _build_reflection_prompt(
    parent_block: str, fitness: FitnessResult, target_template: str
) -> str:
    stats = fitness.issue_stats
    return json.dumps(
        {
            "target_template": target_template,
            "current_constraint_block": parent_block,
            "fitness": round(fitness.fitness, 4),
            "top_issues": fitness.top_issues,
            "issue_counts": {
                "unsupported": stats.unsupported_count,
                "missing": stats.missing_count,
                "dosage_unit": stats.dosage_unit_count,
                "misclassified": stats.misclassified_count,
                "inconsistent": stats.inconsistent_count,
                "typo_or_ocr": stats.typo_or_ocr_count,
            },
            "feedback": (fitness.feedback or "")[:4000],
            "instruction": (
                "诊断上述反馈的根因，产出下一轮追加约束块。要求："
                "(1) 只写可执行的抽取约束，针对高频错误；"
                "(2) 简洁（建议 300 字以内），不要复述模板正文；"
                "(3) 不得引入原文之外的医学知识；"
                "(4) 若当前约束已足够，可返回精炼后的等价约束。"
            ),
            "output_schema": {
                "constraint_block": "string：追加到抽取模板末尾的约束正文",
                "rationale": "string：一句话说明为何这样改",
            },
        },
        ensure_ascii=False,
    )


def make_reflective_proposer(
    *,
    llm_client: UnifiedLLMClient,
    target_template: str,
    max_constraint_chars: int = 600,
    temperature: float = 0.4,
) -> ProposalFn:
    """构造 GEPAEngine 的 instruction_proposer 注入：反思式变异。

    读父代的富文本 feedback（verdict + 证据 + 修正理由）+ 问题分布，让 LLM 诊断
    根因并产出「新的追加约束块」。失败或无新意时回退内置静态 mutate_prompt，
    保证进化不因单次反思失败而停滞。
    """

    async def propose(parent: PromptIndividual, parent_fitness: FitnessResult) -> PromptIndividual:
        parent_block = parent.extraction_prompt.strip()
        user_prompt = _build_reflection_prompt(parent_block, parent_fitness, target_template)
        try:
            resp = await llm_client.complete_json(
                system_prompt=_REFLECTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        except Exception:  # noqa: BLE001 - 反思失败不应中断进化
            logger.warning("[GEPABridge] 反思调用失败，回退静态变异 parent=%s", parent.prompt_id)
            return mutate_prompt(parent, parent_fitness, 1.0)

        block = _sanitize_constraint_block(
            str(resp.get("constraint_block", "")), max_constraint_chars
        )
        rationale = str(resp.get("rationale", "")).strip()
        if not block or block == parent_block:
            # 反思无产出/无新意 → 回退静态变异，维持种群探索
            logger.info("[GEPABridge] 反思无新约束，回退静态变异 parent=%s", parent.prompt_id)
            return mutate_prompt(parent, parent_fitness, 1.0)

        return PromptIndividual(
            prompt_id=f"{parent.prompt_id}_r{parent.generation + 1}_{_short_hash(block)}",
            extraction_prompt=block,
            verification_prompt=parent.verification_prompt,
            generation=parent.generation + 1,
            parent_ids=(parent.prompt_id,),
            mutation_applied=f"reflect: {rationale[:60]}",
        )

    return propose


# ── P5: system-aware 合并（官方 GEPA 机制④）─────────────────────────────────

_MERGE_SYSTEM_PROMPT = (
    "你是提示词进化器的合并模块。给你两段各自在不同评估样本上取胜的抽取约束块，"
    "请合成一段新约束块，兼顾双方优势、消除冲突与重复。严格遵守：只输出 JSON；"
    "禁止复述或改写原始模板正文；禁止编造原文不存在的医学规则；"
    "合并结果必须可执行、简洁、无语义冲突。"
)


def _build_merge_prompt(
    block_a: str,
    block_b: str,
    fa: FitnessResult,
    fb: FitnessResult,
    target_template: str,
) -> str:
    return json.dumps(
        {
            "target_template": target_template,
            "parent_a": {
                "constraint_block": block_a,
                "fitness": round(fa.fitness, 4),
                "per_instance_scores": fa.per_instance_scores,
                "top_issues": fa.top_issues,
            },
            "parent_b": {
                "constraint_block": block_b,
                "fitness": round(fb.fitness, 4),
                "per_instance_scores": fb.per_instance_scores,
                "top_issues": fb.top_issues,
            },
            "instruction": (
                "合并上述两段约束块为一段。要求："
                "(1) 保留双方各自在样本上奏效的约束；"
                "(2) 去除重复项，化解相互矛盾的表述；"
                "(3) 简洁（建议 400 字以内），不要复述模板正文；"
                "(4) 不得引入原文之外的医学知识。"
            ),
            "output_schema": {
                "constraint_block": "string：合并后的追加约束正文",
                "rationale": "string：一句话说明如何取舍",
            },
        },
        ensure_ascii=False,
    )


def make_system_aware_merger(
    *,
    llm_client: UnifiedLLMClient,
    target_template: str,
    max_constraint_chars: int = 700,
    temperature: float = 0.3,
) -> MergeFn:
    """构造 GEPAEngine 的 merge_fn 注入：system-aware 合并（官方 GEPA 机制④）。

    给定两个父代约束块 + 各自适应度（含每实例分，揭示谁在哪本书上取胜），
    让 LLM 合成兼顾双方优势的新约束块。失败或无产出时抛异常，
    由引擎 _merge 回退内置 crossover，保证进化不中断。
    """

    async def merge(
        pa: PromptIndividual,
        pb: PromptIndividual,
        fa: FitnessResult,
        fb: FitnessResult,
    ) -> PromptIndividual:
        block_a = pa.extraction_prompt.strip()
        block_b = pb.extraction_prompt.strip()
        # 两父代约束一致 → 无需 LLM，直接沿用其一（省钱）
        if not block_a or not block_b or block_a == block_b:
            raise ValueError("empty or identical parents; fall back to crossover")

        user_prompt = _build_merge_prompt(block_a, block_b, fa, fb, target_template)
        resp = await llm_client.complete_json(
            system_prompt=_MERGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        block = _sanitize_constraint_block(
            str(resp.get("constraint_block", "")), max_constraint_chars
        )
        rationale = str(resp.get("rationale", "")).strip()
        if not block:
            raise ValueError("merge produced empty block; fall back to crossover")

        gen = max(pa.generation, pb.generation) + 1
        return PromptIndividual(
            prompt_id=f"merge_{gen}_{_short_hash(block)}",
            extraction_prompt=block,
            verification_prompt=pa.verification_prompt,
            generation=gen,
            parent_ids=(pa.prompt_id, pb.prompt_id),
            mutation_applied=f"merge: {rationale[:60]}",
        )

    return merge
