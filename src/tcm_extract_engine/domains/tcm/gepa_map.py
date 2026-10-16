"""TCM 域 GEPA 适应度信号映射（收编自 gepa/bridge.py）。

把 B 核验 verdict / 确定性校验 category 映射为 GEPA compute_fitness 的
issue 三元组（issue_type, severity, confidence）。骨架 gepa/bridge.py
通过领域包协议读取，信号加权策略（WARNING 降级/折扣）留在骨架。
"""

from __future__ import annotations

# B 核验 verdict → (issue_type, severity, confidence)
# 仅对 review_verdicts 发问题；SUPPORTED / PARTIAL 不计入（PARTIAL 多为兜底，噪声大）
VERDICT_ISSUE_MAP: dict[str, tuple[str, str, float]] = {
    "UNSUPPORTED": ("unsupported", "high", 0.9),
    "MISSING": ("missing", "medium", 0.85),
    "WRONG_DOSE": ("dosage_unit", "critical", 0.9),
    "WRONG_RELATION": ("misclassified", "critical", 0.9),
    "WRONG_NORMALIZATION": ("misclassified", "high", 0.85),
    "OCR_UNCERTAIN": ("typo_or_ocr", "low", 0.6),
}

# 确定性校验 category → (issue_type, severity, confidence)
CATEGORY_ISSUE_MAP: dict[str, tuple[str, str, float]] = {
    "SOURCE": ("unsupported", "high", 0.95),      # 无原文支撑 = 幻觉嫌疑
    "SCHEMA": ("misclassified", "medium", 0.9),
    "NUMERIC": ("dosage_unit", "high", 0.9),
    "ENTITY": ("missing", "medium", 0.85),        # 关系组成员/妊娠目标缺失
    "SEMANTIC": ("inconsistent", "high", 0.9),    # 措辞强度被违规升级
    "COVERAGE": ("missing", "medium", 0.8),
}
