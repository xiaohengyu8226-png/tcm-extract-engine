"""Model C：Extraction Planner + Prompt Template Registry。

C 的真正职责不是"凭感觉写 Prompt"，而是：
  1. 根据 document_profile 为每个可抽取章节选择 extractor + 模板
  2. 配置模板参数（禁止生成全新 Prompt 正文 → 防 Prompt Drift）
  3. 产出 extraction_plan.json（供人工确认 → WAITING_PLAN_CONFIRMATION）

Prompt Template Registry：prompts/c|a|b/*.md，按内容 hash 生成稳定 version。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tcm_extract_engine.cab.profiler import (
    DocumentProfile,
    SectionProfile,
)
from tcm_extract_engine.common.paths import project_paths
from tcm_extract_engine.core.llm import UnifiedLLMClient
from tcm_extract_engine.domains import DomainPack, get_domain_pack

logger = logging.getLogger(__name__)

CHUNK_PLACEHOLDER = "{{chunk_json}}"

# 每书适配（Model C）：元提示词模板与占位符（不复用 render，避免 CHUNK_PLACEHOLDER 强校验）
ADAPTATION_TEMPLATE = "book_adaptation.md"
ADAPTATION_PLACEHOLDER = "{{adaptation_input}}"
_SYSTEM_PROMPT_C = "你是书籍抽取适配器。严格按指令输出 JSON。"

# 每书 goal 归纳（Model C）：按文档画像生成目标声明（纯展示元数据，不进任何抽取模板）
GOAL_TEMPLATE = "book_goal.md"
GOAL_PLACEHOLDER = "{{goal_input}}"
_SYSTEM_PROMPT_GOAL = "你是书籍抽取目标归纳器。严格按指令输出 JSON。"
# goal 长度上限（字符）：bounded output，防注入与冗长
_MAX_GOAL_CHARS = 60

# extractor 名 → A 模板文件名（领域包 classify 声明，shim 重导出）


class SectionPlan(BaseModel):
    """单个章节的抽取计划。"""

    section_id: str
    title: str
    heading_path: list[str] = Field(default_factory=list)
    extractor: str
    template: str  # 模板文件名
    prompt_version: str  # 模板内容 hash 前 12 位
    block_ids: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    skip_reason: str | None = None  # 跳过原因（不抽时填写）


class ExtractionPlan(BaseModel):
    """Model C 输出的抽取方案（03_plan/extraction_plan.json 主体）。"""

    domain_id: str = "tcm"
    goal: str = ""  # 领域包 goal（build_extraction_plan 注入）
    document_type: str
    sections: list[SectionPlan] = Field(default_factory=list)
    risk_points: list[str] = Field(default_factory=list)
    # 允许的字段推断开关（默认关闭：所有字段直接来自文本）
    allow_normalization: bool = False

    @property
    def planned_sections(self) -> list[SectionPlan]:
        """实际会执行抽取的章节。"""
        return [s for s in self.sections if not s.skip_reason]


class PromptTemplate:
    """单个 Prompt 模板（文件 + 版本）。"""

    def __init__(self, name: str, path: Path, content: str) -> None:
        self.name = name
        self.path = path
        self.content = content
        self.version = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

    def render(self, chunks_payload: str) -> str:
        """注入待抽取文本（保持其余部分逐字不变）。"""
        if CHUNK_PLACEHOLDER not in self.content:
            raise ValueError(f"模板 {self.name} 缺少 {CHUNK_PLACEHOLDER} 占位符")
        return self.content.replace(CHUNK_PLACEHOLDER, chunks_payload)


class PromptTemplateRegistry:
    """Prompt 模板注册表：按 prompts/a|b|c 目录加载，hash 即版本。"""

    def __init__(self, prompts_root: Path | None = None) -> None:
        self.prompts_root = prompts_root or project_paths.prompts
        self._cache: dict[str, PromptTemplate] = {}

    def load(self, role: str, name: str) -> PromptTemplate:
        """加载模板（role ∈ {a, b, c}）；带内存缓存。"""
        key = f"{role}/{name}"
        if key in self._cache:
            return self._cache[key]
        path = self.prompts_root / role / name
        if not path.is_file():
            raise FileNotFoundError(f"Prompt 模板不存在: {path}")
        content = path.read_text(encoding="utf-8")
        template = PromptTemplate(name=name, path=path, content=content)
        self._cache[key] = template
        return template

    def version_of(self, role: str, name: str) -> str:
        return self.load(role, name).version


def build_extraction_plan(
    profile: DocumentProfile,
    *,
    registry: PromptTemplateRegistry | None = None,
    goal: str | None = None,
    pack: DomainPack | None = None,
) -> ExtractionPlan:
    """从文档画像构建抽取计划（确定性规则，不调用 LLM）。

    规则：ContentType → extractor → 模板；REFERENCE_ONLY/IRRELEVANT 跳过。
    LLM 介入（可选增强）留给后续版本：仅在标题无法分类时调用 prompt 模型。

    Args:
        goal: 每书自适应目标声明（Model C 生成）；为空时回退领域包默认 goal。
            纯展示元数据，不影响任何抽取行为。
    """
    pack = pack or get_domain_pack(
        profile.domain if profile.domain in {"tcm", "general"} else "tcm"
    )
    registry = registry or PromptTemplateRegistry()
    sections: list[SectionPlan] = []

    for sec in profile.sections:
        extractor = pack.content_extractors.get(sec.content_type, "")
        if not extractor:
            sections.append(
                SectionPlan(
                    section_id=sec.section_id,
                    title=sec.title,
                    heading_path=sec.heading_path,
                    extractor="",
                    template="",
                    prompt_version="",
                    block_ids=sec.block_ids,
                    page_start=sec.page_start,
                    page_end=sec.page_end,
                    skip_reason=f"content_type={sec.content_type} 不抽取",
                )
            )
            continue

        template_name = pack.extractor_templates.get(extractor)
        if template_name is None:
            raise ValueError(f"extractor {extractor} 无对应模板")
        template = registry.load("a", template_name)
        sections.append(
            SectionPlan(
                section_id=sec.section_id,
                title=sec.title,
                heading_path=sec.heading_path,
                extractor=extractor,
                template=template_name,
                prompt_version=template.version,
                block_ids=sec.block_ids,
                page_start=sec.page_start,
                page_end=sec.page_end,
            )
        )

    plan = ExtractionPlan(
        domain_id=pack.id,        goal=(goal or "").strip() or pack.goal,
        document_type=profile.document_type.value,
        sections=sections,
        risk_points=_collect_risk_points(profile, pack=pack),
    )
    logger.info(
        "[Planner] 计划生成: %d 章节（%d 可抽取, %d 跳过）",
        len(sections),
        len(plan.planned_sections),
        len(sections) - len(plan.planned_sections),
    )
    return plan


def _collect_risk_points(profile: DocumentProfile, *, pack: DomainPack) -> list[str]:
    """基于画像归纳的风险提示（写入 plan 供人工确认时关注）。

    域风险由领域包声明（manifest.risk_rules，min_counts 触发阈值）；
    通用规模风险由骨架补充。
    """
    counts = profile.type_counts()
    risks = [
        rule.text
        for rule in pack.risk_rules
        if all(counts.get(k, 0) >= v for k, v in rule.min_counts.items())
    ]
    if profile.total_blocks > 3000:
        risks.append("文档较大：抽取分批执行，优先保证断点续传")
    return risks


# ── 每书抽取适配（Model C 用 LLM 生成，分层叠加在 base 模板之上）──────────────

# 适配块结构化字段 → 渲染小标题：由领域包声明（manifest.adaptation_fields），
# 固定顺序保证同输入同输出 → version 稳定


class BookAdaptation(BaseModel):
    """单本书 × 单个 A 模板的适配块。"""

    template: str
    block: str = ""  # 渲染后的 Markdown 块（追加到 base 模板末尾）
    version: str = ""  # sha256(block)[:12]；空块为空串
    structured: dict[str, Any] = Field(default_factory=dict)  # C-LLM 原始结构化输出
    degraded: bool = False  # 生成失败降级为 base-only


def render_adaptation_block(structured: dict[str, Any], *, pack: DomainPack | None = None) -> str:
    """把结构化适配字段确定性渲染为 Markdown 块（空结构返回空串）。"""
    pack = pack or get_domain_pack()
    lines: list[str] = []
    for field in pack.adaptation_fields:
        value = structured.get(field.key)
        if field.is_list:
            items = [str(v).strip() for v in (value or []) if str(v).strip()]
            if not items:
                continue
            lines.append(f"- {field.title}：")
            lines.extend(f"  - {it}" for it in items)
        else:
            text = str(value or "").strip()
            if not text:
                continue
            lines.append(f"- {field.title}：{text}")
    if not lines:
        return ""
    return "## 本书适配提示（Model C 生成，人工确认后生效）\n" + "\n".join(lines)


def adaptation_version(block: str) -> str:
    """适配块版本 = sha256(block)[:12]（与 PromptTemplate.version 同规则）。"""
    return hashlib.sha256(block.encode("utf-8")).hexdigest()[:12]


class BookAdaptedRegistry(PromptTemplateRegistry):
    """在指定 A 模板末尾追加「每书适配块」的模板注册表。

    - 只影响 role=="a" 且命中 adaptations 的模板；其余（含 B/C）原样返回。
    - base 正文逐字不变，适配块追加在末尾；合成正文 hash → version 变，
      天然驱动 extract 缓存失效与 manifest.prompt_version bump。
    - 与 gepa.bridge.ConstrainedTemplateRegistry 逻辑对齐（两处独立，互不影响）。
    """

    def __init__(
        self,
        adaptations: dict[str, str] | None = None,
        *,
        prompts_root: Path | None = None,
    ) -> None:
        super().__init__(prompts_root)
        self.adaptations = {k: v.strip() for k, v in (adaptations or {}).items() if v and v.strip()}

    def load(self, role: str, name: str) -> PromptTemplate:
        block = self.adaptations.get(name) if role == "a" else None
        if not block:
            return super().load(role, name)
        key = f"{role}/{name}|+adapt({adaptation_version(block)})"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        base = super().load(role, name)
        content = f"{base.content.rstrip()}\n\n{block}\n"
        template = PromptTemplate(name=base.name, path=base.path, content=content)
        self._cache[key] = template
        return template


def templates_for_profile(profile: DocumentProfile, *, pack: DomainPack | None = None) -> list[str]:
    """推出本书实际用到的 A 模板集合（去重、排序，保证调用顺序确定）。"""
    pack = pack or get_domain_pack(
        profile.domain if profile.domain in {"tcm", "general"} else "tcm"
    )
    templates: set[str] = set()
    for sec in profile.sections:
        if not sec.worth_extracting:
            continue
        extractor = pack.content_extractors.get(sec.content_type, "")
        template = pack.extractor_templates.get(extractor)
        if template:
            templates.add(template)
    return sorted(templates)


def _sections_for_template(
    profile: DocumentProfile, template: str, *, pack: DomainPack | None = None
) -> list[SectionProfile]:
    """本书中路由到指定模板的可抽取章节。"""
    pack = pack or get_domain_pack(
        profile.domain if profile.domain in {"tcm", "general"} else "tcm"
    )
    out: list[SectionProfile] = []
    for sec in profile.sections:
        if not sec.worth_extracting:
            continue
        extractor = pack.content_extractors.get(sec.content_type, "")
        if pack.extractor_templates.get(extractor) == template:
            out.append(sec)
    return out


def _adaptation_input(
    profile: DocumentProfile, template: str, sections: list[SectionProfile]
) -> str:
    """构造喂给 C-LLM 的画像输入（该模板覆盖章节的标题 + 采样）。"""
    payload = {
        "document_type": profile.document_type.value,
        "target_template": template,
        "covered_sections": [
            {
                "title": s.title,
                "heading_path": s.heading_path,
                "table_count": s.table_count,
                "sample_text": s.sample_text,
            }
            for s in sections
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _clean_structured(raw: Any, *, pack: DomainPack | None = None) -> dict[str, Any]:
    """把 C-LLM 输出裁剪为已知字段、string/list 值的有界结构（防注入）。"""
    if not isinstance(raw, dict):
        return {}
    pack = pack or get_domain_pack()
    cleaned: dict[str, Any] = {}
    for field in pack.adaptation_fields:
        value = raw.get(field.key)
        if field.is_list:
            if isinstance(value, list):
                items = [str(v).strip() for v in value if str(v).strip()]
                if items:
                    cleaned[field.key] = items
        elif value is not None and str(value).strip():
            cleaned[field.key] = str(value).strip()
    return cleaned


async def generate_book_adaptations(
    *,
    profile: DocumentProfile,
    templates: list[str],
    client: UnifiedLLMClient,
    registry: PromptTemplateRegistry | None = None,
) -> dict[str, BookAdaptation]:
    """为每个 A 模板调用一次 C-LLM，产出每书适配块（优雅降级）。

    - 每模板一次调用（≤ len(templates)，通常 ≤4）。
    - 单模板 LLM 失败/解析异常 → 该模板 degraded=True、block=""（回退 base-only），
      不整批失败（遵循 extract/verify 优雅处理 LLM 错误的约定）。
    """
    registry = registry or PromptTemplateRegistry()
    meta = registry.load("c", ADAPTATION_TEMPLATE).content
    pack = get_domain_pack(profile.domain_id)
    adaptations: dict[str, BookAdaptation] = {}
    for template in templates:
        sections = _sections_for_template(profile, template)
        if not sections:
            continue
        user_prompt = meta.replace(
            ADAPTATION_PLACEHOLDER,
            _adaptation_input(profile, template, sections),
        )
        try:
            result = await client.complete_json(
                system_prompt=_SYSTEM_PROMPT_C,
                user_prompt=user_prompt,
            )
        except Exception as exc:  # noqa: BLE001 - 单模板失败降级为 base-only
            logger.warning("[Planner] 适配块生成失败(%s)，降级为 base-only: %s", template, exc)
            adaptations[template] = BookAdaptation(template=template, degraded=True)
            continue
        structured = _clean_structured(result, pack=pack)
        block = render_adaptation_block(structured, pack=pack)
        adaptations[template] = BookAdaptation(
            template=template,
            block=block,
            version=adaptation_version(block) if block else "",
            structured=structured,
        )
        logger.info(
            "[Planner] 适配块生成(%s): %d 字段, version=%s",
            template,
            len(structured),
            adaptations[template].version or "-",
        )
    return adaptations


# ── 每书 goal 归纳（Model C 用 LLM 生成，纯展示元数据）─────────────────────────
#
# 与适配块不同：goal 不追加进任何模板、不参与 prompt_version、不影响抽取行为，
# 仅作为 extraction_plan.goal 落盘供人工确认与 web 展示（去领域中心化）。


def _goal_input(profile: DocumentProfile) -> str:
    """构造喂给 C-LLM 的 goal 归纳输入（全书画像摘要）。"""
    payload = {
        "document_type": profile.document_type.value,
        "type_counts": profile.type_counts(),
        "sections": [
            {
                "title": s.title,
                "heading_path": s.heading_path,
                "content_type": s.content_type,
                "table_count": s.table_count,
                "sample_text": s.sample_text,
            }
            for s in profile.sections
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


async def generate_book_goal(
    *,
    profile: DocumentProfile,
    client: UnifiedLLMClient,
    registry: PromptTemplateRegistry | None = None,
) -> str:
    """按文档画像归纳本书 goal（一次 C-LLM 调用，优雅降级）。

    Returns:
        goal 字符串；LLM 失败/解析异常/空结果时返回 ""（调用方回退领域包默认 goal）。
        失败不抛出，遵循 plan 阶段优雅降级约定。
    """
    registry = registry or PromptTemplateRegistry()
    meta = registry.load("c", GOAL_TEMPLATE).content
    user_prompt = meta.replace(GOAL_PLACEHOLDER, _goal_input(profile))
    try:
        result = await client.complete_json(
            system_prompt=_SYSTEM_PROMPT_GOAL,
            user_prompt=user_prompt,
        )
    except Exception as exc:  # noqa: BLE001 - 失败降级为领域包默认 goal
        logger.warning("[Planner] goal 归纳失败，回退领域包默认: %s", exc)
        return ""
    goal = str(result.get("goal") or "").strip()
    if not goal:
        return ""
    goal = goal[:_MAX_GOAL_CHARS]
    logger.info("[Planner] goal 归纳: %s", goal)
    return goal
