"""Model C：Extraction Planner + Prompt Template Registry。

C 的真正职责不是"凭感觉写 Prompt"，而是：
  1. 根据 document_profile 为每个可抽取章节选择 extractor + 模板
  2. 配置模板参数（禁止生成全新 Prompt 正文 → 防 Prompt Drift）
  3. 产出 extraction_plan.json（供人工确认 → WAITING_PLAN_CONFIRMATION）

Prompt Template Registry：prompts/c|a|b/*.md，按内容 hash 生成稳定 version。
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from tcm_extract_engine.cab.profiler import (
    CONTENT_TYPE_TO_EXTRACTOR,
    DocumentProfile,
)
from tcm_extract_engine.common.paths import project_paths

logger = logging.getLogger(__name__)

CHUNK_PLACEHOLDER = "{{chunk_json}}"

# extractor 名 → A 模板文件名
EXTRACTOR_TEMPLATES: dict[str, str] = {
    "GeneralRuleExtractor": "general_rule.md",
    "HerbTableExtractor": "herb_table.md",
    "InteractionExtractor": "interaction_group.md",
    "PregnancyRuleExtractor": "pregnancy_list.md",
}


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

    goal: str = "中药饮片安全审方规则库建设"
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
) -> ExtractionPlan:
    """从文档画像构建抽取计划（确定性规则，不调用 LLM）。

    规则：ContentType → extractor → 模板；REFERENCE_ONLY/IRRELEVANT 跳过。
    LLM 介入（可选增强）留给后续版本：仅在标题无法分类时调用 prompt 模型。
    """
    registry = registry or PromptTemplateRegistry()
    sections: list[SectionPlan] = []

    for sec in profile.sections:
        extractor = CONTENT_TYPE_TO_EXTRACTOR.get(sec.content_type, "")
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
                    skip_reason=f"content_type={sec.content_type.value} 不抽取",
                )
            )
            continue

        template_name = EXTRACTOR_TEMPLATES.get(extractor)
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
        document_type=profile.document_type.value,
        sections=sections,
        risk_points=_collect_risk_points(profile),
    )
    logger.info(
        "[Planner] 计划生成: %d 章节（%d 可抽取, %d 跳过）",
        len(sections),
        len(plan.planned_sections),
        len(sections) - len(plan.planned_sections),
    )
    return plan


def _collect_risk_points(profile: DocumentProfile) -> list[str]:
    """基于画像归纳的风险提示（写入 plan 供人工确认时关注）。"""
    risks: list[str] = []
    counts = profile.type_counts()
    if counts.get("RELATION_GROUP", 0) > 0:
        risks.append("含十八反/十九畏关系组：关系组禁止被 chunk 切断，成员覆盖必须完整")
    if counts.get("ENTITY_GROUP_LIST", 0) > 0:
        risks.append("含妊娠禁用/慎用清单：括号内炮制品成员需完整保留，措辞级别禁止互换")
    if counts.get("STRUCTURED_TABLE", 0) > 0:
        risks.append("含结构化表格：按行抽取并做行数覆盖率校验；跨页表格需合并表头")
    if profile.total_blocks > 3000:
        risks.append("文档较大：抽取分批执行，优先保证断点续传")
    return risks
