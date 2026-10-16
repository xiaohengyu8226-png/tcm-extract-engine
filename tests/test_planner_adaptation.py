"""Model C 每书适配块单元测试：渲染确定性 / 注入 registry / 生成与降级。"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from tcm_extract_engine.cab.planner import (
    BookAdaptedRegistry,
    PromptTemplateRegistry,
    adaptation_version,
    build_extraction_plan,
    generate_book_adaptations,
    generate_book_goal,
    render_adaptation_block,
    templates_for_profile,
)
from tcm_extract_engine.cab.profiler import DocumentProfile, SectionProfile
from tcm_extract_engine.core.llm import LLMError
from tcm_extract_engine.domains.tcm.classify import ContentType, DocumentType

# ── render_adaptation_block ──────────────────────────────────────────────────


def test_render_adaptation_block_deterministic_and_empty() -> None:
    assert render_adaptation_block({}) == ""
    # 未知字段忽略、空列表跳过
    assert render_adaptation_block({"unknown": "x", "watch_points": []}) == ""

    structured = {"terminology_aliases": ["仙灵脾即淫羊藿"], "table_format_notes": "跨页需合并表头"}
    b1 = render_adaptation_block(structured)
    b2 = render_adaptation_block(structured)
    assert b1 == b2  # 确定性
    assert "术语别名" in b1 and "表格版式" in b1
    assert adaptation_version(b1) == adaptation_version(b2)
    assert adaptation_version(b1) == hashlib.sha256(b1.encode()).hexdigest()[:12]


# ── BookAdaptedRegistry ──────────────────────────────────────────────────────


def test_book_adapted_registry_appends_only_role_a() -> None:
    base_reg = PromptTemplateRegistry()
    base = base_reg.load("a", "general_rule.md")
    block = "## 本书适配提示\n- 重点盯防：剂量"
    reg = BookAdaptedRegistry({"general_rule.md": block})

    composed = reg.load("a", "general_rule.md")
    assert composed.content == f"{base.content.rstrip()}\n\n{block}\n"
    assert composed.version == hashlib.sha256(composed.content.encode()).hexdigest()[:12]
    assert composed.version != base.version  # 合成版本 bump

    # 非目标模板 / 非 a 角色原样返回
    assert reg.load("a", "herb_table.md").content == base_reg.load("a", "herb_table.md").content
    assert (
        reg.load("b", "factual_verifier.md").content
        == base_reg.load("b", "factual_verifier.md").content
    )

    # 占位符仍完整，render 照常替换
    assert "{{chunk_json}}" in composed.content
    rendered = composed.render("PAYLOAD")
    assert "PAYLOAD" in rendered and block in rendered


def test_book_adapted_registry_ignores_blank_blocks() -> None:
    base_reg = PromptTemplateRegistry()
    reg = BookAdaptedRegistry({"general_rule.md": "   "})  # 空白块被过滤
    assert reg.load("a", "general_rule.md").content == base_reg.load("a", "general_rule.md").content


# ── generate_book_adaptations ────────────────────────────────────────────────


class _FakePromptClient:
    """按序出队的假 C 客户端；异常项抛出，空队列抛 LLMError。"""

    def __init__(self, responses: list[Any]) -> None:
        self.model = "c-fake"
        self._responses = list(responses)
        self.calls: list[str] = []

    async def complete_json(
        self, *, system_prompt: str, user_prompt: str, **_: Any
    ) -> dict[str, Any]:
        self.calls.append(user_prompt)
        item = self._responses.pop(0) if self._responses else None
        if isinstance(item, Exception):
            raise item
        if item is None:
            raise LLMError("api", "空队列")
        return item


def _profile_two_templates() -> DocumentProfile:
    s1 = SectionProfile(
        section_id="sec-0000",
        title="剂量规范",
        content_type=ContentType.NARRATIVE_RULE,
        sample_text="川乌极量不得超过二克",
    )
    s2 = SectionProfile(
        section_id="sec-0001",
        title="毒性中药表",
        content_type=ContentType.STRUCTURED_TABLE,
        sample_text="制川乌 3g",
    )
    return DocumentProfile(document_type=DocumentType.HANDBOOK, sections=[s1, s2], total_blocks=2)


def test_templates_for_profile_sorted_unique() -> None:
    templates = templates_for_profile(_profile_two_templates())
    assert templates == sorted(templates)
    assert set(templates) == {"general_rule.md", "herb_table.md"}


def test_generate_book_adaptations_renders_and_degrades() -> None:
    profile = _profile_two_templates()
    templates = templates_for_profile(profile)  # [general_rule.md, herb_table.md]
    client = _FakePromptClient(
        [
            {"watch_points": ["盯剂量单位"], "terminology_aliases": []},
            LLMError("api", "boom"),  # 第二个模板失败 → 降级
        ]
    )
    adaptations = asyncio.run(
        generate_book_adaptations(
            profile=profile,
            templates=templates,
            client=client,
            registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
        )
    )
    assert len(client.calls) == 2
    gr = adaptations["general_rule.md"]
    assert gr.block and not gr.degraded and gr.version
    assert "重点盯防" in gr.block
    ht = adaptations["herb_table.md"]
    assert ht.degraded and ht.block == "" and ht.version == ""


def test_generate_book_adaptations_cleans_unknown_fields() -> None:
    profile = _profile_two_templates()
    templates = ["general_rule.md"]
    client = _FakePromptClient(
        [{"unknown_field": "x", "watch_points": ["仅保留已知字段"], "table_format_notes": ""}]
    )
    # 只给 general_rule.md 一个模板；herb_table 无对应 response 也不调用
    adaptations = asyncio.run(
        generate_book_adaptations(
            profile=profile,
            templates=templates,
            client=client,
            registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
        )
    )
    gr = adaptations["general_rule.md"]
    assert "unknown_field" not in gr.structured
    assert gr.structured.get("watch_points") == ["仅保留已知字段"]
    assert "table_format_notes" not in gr.structured  # 空串被裁剪


# ── generate_book_goal / goal 覆盖 ──────────────────────────────────────


def test_generate_book_goal_returns_bounded_goal() -> None:
    profile = _profile_two_templates()
    client = _FakePromptClient([{"goal": "  医学知识抽取与规则库建设  "}])
    goal = asyncio.run(
        generate_book_goal(
            profile=profile,
            client=client,
            registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
        )
    )
    assert goal == "医学知识抽取与规则库建设"  # 去空白
    assert len(client.calls) == 1
    assert "{{goal_input}}" not in client.calls[0]  # 占位符已替换


def test_generate_book_goal_truncates_overlong() -> None:
    profile = _profile_two_templates()
    client = _FakePromptClient([{"goal": "长" * 200}])
    goal = asyncio.run(
        generate_book_goal(
            profile=profile,
            client=client,
            registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
        )
    )
    assert len(goal) == 60  # _MAX_GOAL_CHARS 上限


def test_generate_book_goal_degrades_to_empty() -> None:
    profile = _profile_two_templates()
    # LLM 失败 → ""（调用方回退领域包默认）
    client = _FakePromptClient([LLMError("api", "boom")])
    assert (
        asyncio.run(
            generate_book_goal(
                profile=profile,
                client=client,
                registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
            )
        )
        == ""
    )
    # 空结果 → ""
    client2 = _FakePromptClient([{"goal": "   "}])
    assert (
        asyncio.run(
            generate_book_goal(
                profile=profile,
                client=client2,
                registry=PromptTemplateRegistry(),  # type: ignore[arg-type]
            )
        )
        == ""
    )


def test_build_extraction_plan_goal_override_and_fallback() -> None:
    from tcm_extract_engine.domains import get_domain_pack

    profile = _profile_two_templates()
    # 非空 goal → 覆盖（去空白）
    plan = build_extraction_plan(profile, goal="  自定义目标  ")
    assert plan.goal == "自定义目标"
    # 空/空白 goal → 回退领域包默认
    plan2 = build_extraction_plan(profile, goal="   ")
    assert plan2.goal == get_domain_pack().goal
