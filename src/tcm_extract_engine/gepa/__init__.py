"""GEPA 自反馈自优化：引擎内核 + 真实评估桥接 + 单书进化 + 跨书飞轮。"""

from tcm_extract_engine.gepa.bridge import (
    ConstrainedTemplateRegistry,
    EvalOutcome,
    evaluate_candidate,
    make_evaluate_fn,
    make_reflective_proposer,
    make_system_aware_merger,
)
from tcm_extract_engine.gepa.engine import (
    FitnessResult,
    GenerationReport,
    GEPAConfig,
    GEPAEngine,
    PromptIndividual,
    compute_fitness,
    crossover,
    mutate_prompt,
)
from tcm_extract_engine.gepa.flywheel import evolve_across_books, evolve_all_templates
from tcm_extract_engine.gepa.runner import (
    evolve_template,
    ledger_path_for,
    load_run_best_constraint,
    load_trend,
    promote_constraint,
)

__all__ = [
    "ConstrainedTemplateRegistry",
    "EvalOutcome",
    "GEPAConfig",
    "GEPAEngine",
    "FitnessResult",
    "GenerationReport",
    "PromptIndividual",
    "compute_fitness",
    "crossover",
    "evaluate_candidate",
    "evolve_across_books",
    "evolve_all_templates",
    "evolve_template",
    "ledger_path_for",
    "load_run_best_constraint",
    "load_trend",
    "make_evaluate_fn",
    "make_reflective_proposer",
    "make_system_aware_merger",
    "mutate_prompt",
    "promote_constraint",
]
