"""TCM 域 B 核验判决词表（收编自 cab/verifier.py）。

Verdict 枚举值一经落盘（verification_results.json）即不可更改。
需要人工复核的判决子集由 manifest.review_verdicts 声明（骨架经协议读取）；
PASSTHROUGH 为流水线级判决（表格直通免 LLM 核验），与 TABLE_ROW 同理寄居于包枚举。
"""

from __future__ import annotations

import enum


class Verdict(str, enum.Enum):
    """B 核验判定。"""

    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    MISSING = "MISSING"
    WRONG_DOSE = "WRONG_DOSE"
    WRONG_RELATION = "WRONG_RELATION"
    WRONG_NORMALIZATION = "WRONG_NORMALIZATION"
    OCR_UNCERTAIN = "OCR_UNCERTAIN"
    PASSTHROUGH = "PASSTHROUGH"  # 表格直通记录：原表逐字透传，免 LLM 核验
    FIGURE_REVIEW = "FIGURE_REVIEW"  # 图片多模态记录：文本 B 无法比对图像，转人工复核
