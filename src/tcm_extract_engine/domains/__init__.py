"""领域包（Domain Pack）：骨架（域无关流水线）+ 可插拔域知识的边界。

包 = 域知识的单一事实源（schema / 分类 / 解析 / 校验 / 判决 / 导出 / 标签 / 文案）。
骨架通过 :func:`get_domain_pack` 惰性获取当前书的包；默认包 ``tcm``。

设计红线：
  - 包只在**编写期**由人闸（或 C0 起草 + 人闸）演进，运行期只读；
  - 通用不变量（溯源完备、不裁决、bounded output）留在骨架，不进包；
  - manifest.json 是声明面（goal/标签/风险文案/适配字段），行为代码在包模块。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

_DEFAULT_PACK = "tcm"
_registry: dict[str, Any] = {}


@dataclass(frozen=True)
class RiskRule:
    """域风险提示规则：类型计数达到阈值时写入 plan.risk_points。"""

    min_counts: dict[str, int] = field(default_factory=dict)
    text: str = ""


@dataclass(frozen=True)
class AdaptationField:
    """每书适配块的结构化字段声明（json key、渲染小标题、是否列表）。"""

    key: str
    title: str
    is_list: bool = False


class DomainPack(Protocol):
    """领域包协议：声明面（manifest）+ 行为入口（parse_record 等）。"""

    id: str
    version: str
    figure_template: str
    content_extractors: dict[str, str]
    extractor_templates: dict[str, str]
    chunk_strategies: dict[str, str]
    route_section: Callable[..., Any]
    infer_document_type: Callable[..., Any]
    normalize_correction: Callable[..., Any]
    validate_payload: Callable[..., Any]
    flatten_for_review: Callable[..., Any]
    subject_of: Callable[..., Any]
    related_subjects: Callable[..., Any]
    label: str
    goal: str
    profile_domain: str  # 写入 document_profile.domain（落盘后不可更改）
    dose_outlier_grams: float  # 剂量离群阈值（骨架 NUMERIC 校验）
    review_verdicts: frozenset[str]  # 需人工复核的 B 判决子集
    # GEPA 适应度信号映射（gepa/bridge.py 经协议读取）
    verdict_issue_map: dict[str, tuple[str, str, float]]
    category_issue_map: dict[str, tuple[str, str, float]]
    labels: dict[str, Any]
    schema_spec: dict[str, Any]  # schema_spec.json 声明面（一致性测试族对账）
    risk_rules: list[RiskRule]
    adaptation_fields: list[AdaptationField]

    # LLM 输出 → SourceRecord（骨架 shim 委托给包；Callable 避免 self 绑定歧义）
    parse_record: Callable[..., Any]
    # 域确定性校验（产出 ValidationIssue；通用不变量留在骨架）
    validate_record: Callable[..., Any]
    coverage_member_count: Callable[..., Any]
    coverage_table_rows: Callable[..., Any]
    dedup_key: Callable[..., Any]


def register_domain(domain_id: str, pack: DomainPack) -> None:
    if domain_id != pack.id:
        raise ValueError("领域包 ID 不匹配")
    _registry[domain_id] = pack


def registered_domains() -> tuple[str, ...]:
    return tuple(sorted({"tcm", "general", *_registry}))


def get_domain_pack(domain_id: str = _DEFAULT_PACK) -> DomainPack:
    """获取领域包（进程内缓存；未知包名抛 ValueError）。"""
    pack = _registry.get(domain_id)
    if pack is None:
        if domain_id == "tcm":
            from tcm_extract_engine.domains.tcm import TCMPack

            pack = TCMPack()
        elif domain_id == "general":
            from tcm_extract_engine.domains.general import GeneralPack

            pack = GeneralPack()
        else:
            raise ValueError(f"未知领域包: {domain_id}")
        _registry[domain_id] = pack
    return pack
