"""Profiler / Planner 测试：章节分型路由 + 抽取计划生成 + 模板注册表。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcm_extract_engine.cab.planner import (
    PromptTemplateRegistry,
    build_extraction_plan,
)
from tcm_extract_engine.cab.profiler import (
    ContentType,
    DocumentType,
    build_profile,
    classify_section,
    infer_document_type,
)

# ── classify_section 规则路由 ────────────────────────────────────────────────


def test_classify_interaction_sections() -> None:
    assert classify_section("附录C 十八反、十九畏") == ContentType.RELATION_GROUP
    assert classify_section("十九畏歌诀") == ContentType.RELATION_GROUP
    assert classify_section("配伍禁忌") == ContentType.RELATION_GROUP


def test_classify_pregnancy_sections() -> None:
    assert classify_section("附录D 妊娠禁用药") == ContentType.ENTITY_GROUP_LIST
    assert classify_section("妊娠慎用中药清单") == ContentType.ENTITY_GROUP_LIST


def test_classify_toxic_tables() -> None:
    assert classify_section("附录A 有毒中药饮片用法与用量") == ContentType.STRUCTURED_TABLE
    assert classify_section("附录B 28种医疗用毒性中药") == ContentType.STRUCTURED_TABLE


def test_classify_irrelevant() -> None:
    assert classify_section("目录") == ContentType.IRRELEVANT
    assert classify_section("前言") == ContentType.IRRELEVANT
    assert classify_section("参考文献") == ContentType.IRRELEVANT


def test_classify_rule_by_body_text() -> None:
    """无匹配标题时按正文措辞密度分类。"""
    body = "本品禁用。孕妇慎用。儿童不宜使用。不得过量。"
    assert classify_section("第四章 其他要求", body_text=body) == ContentType.NARRATIVE_RULE
    # 正文无规则措辞 → REFERENCE_ONLY
    assert classify_section("第四章 其他要求", body_text="本章介绍背景知识。") == ContentType.REFERENCE_ONLY


def test_classify_table_fallback() -> None:
    assert classify_section("第四章 其他要求", table_count=3) == ContentType.STRUCTURED_TABLE


def test_table_priority_over_soft_keyword() -> None:
    """有表格时 STRUCTURED_TABLE 压过叙述性关键词（如「煎」）。"""
    assert classify_section(
        "需特殊煎煮中药目录", table_count=1
    ) == ContentType.STRUCTURED_TABLE


def test_high_semantic_types_survive_table_priority() -> None:
    """禁忌/妊娠目录本身就是表格，不得被降级成 HerbTable 路由。"""
    assert classify_section(
        "附录C 十八反、十九畏", table_count=2
    ) == ContentType.RELATION_GROUP
    assert classify_section(
        "妊娠禁用中药", table_count=1
    ) == ContentType.ENTITY_GROUP_LIST
    assert classify_section("参考文献", table_count=1) == ContentType.IRRELEVANT


def test_infer_document_type() -> None:
    assert infer_document_type("中药饮片临床应用规范") == DocumentType.STANDARD
    assert infer_document_type("中医临床用药禁忌手册") == DocumentType.HANDBOOK
    assert infer_document_type("某医案集") == DocumentType.CASE_COLLECTION
    assert infer_document_type("随便的书") == DocumentType.UNKNOWN


# ── build_profile 端到端（构造 blocks）───────────────────────────────────────


def _make_blocks_path(tmp_path: Path) -> Path:
    blocks = [
        {"type": "meta", "version": "1.0"},
        _block("第一章 总则", 1, "中药饮片应用应遵循规范。"),
        _block("第一章 总则", 1, "处方应审核剂量与用法。"),
        _block("4.4 剂量规范要求", 12, "有毒饮片剂量不得超过规定极量。"),
        _block("附录C 十八反、十九畏", 40, "甘草反甘遂、大戟、芫花、海藻。", heading_level=2),
        _block("附录D 妊娠禁用药", 50, "巴豆霜、牵牛子、商陆禁用。", heading_level=2),
        _block("目录", 2, "第一章 总则……第四章 剂量规范。", heading_level=1),
    ]
    path = tmp_path / "blocks.jsonl"
    path.write_text(
        "\n".join(json.dumps(b, ensure_ascii=False) for b in blocks) + "\n",
        encoding="utf-8",
    )
    return path


def _block(heading: str, page: int, content: str, heading_level: int = 1) -> dict:
    return {
        "type": "content",
        "blockid": f"blk-{heading[:4]}-{page}-{len(content)}",
        "format": "plain_text",
        "content": content,
        "heading": heading,
        "parent_headings": [],
        "level": heading_level,
        "positions": [{"type": "bbox", "anchor": str(page), "range": [0, 0, 100, 100]}],
    }


def test_build_profile_sections(tmp_path: Path) -> None:
    profile = build_profile(_make_blocks_path(tmp_path), book_name="中药饮片临床应用规范")

    assert profile.document_type == DocumentType.STANDARD
    assert profile.total_blocks == 6
    assert profile.total_pages == 50

    by_title = {s.title: s for s in profile.sections}
    # 十八反 → RELATION_GROUP；妊娠 → ENTITY_GROUP_LIST；目录 → IRRELEVANT
    assert by_title["附录C 十八反、十九畏"].content_type == ContentType.RELATION_GROUP
    assert by_title["附录D 妊娠禁用药"].content_type == ContentType.ENTITY_GROUP_LIST
    assert by_title["目录"].content_type == ContentType.IRRELEVANT

    # 章节聚合：第一章两个块合为一节
    assert by_title["第一章 总则"].block_count == 2

    # 页码范围
    assert by_title["第一章 总则"].page_start == 1
    assert by_title["第一章 总则"].page_end == 1


def test_build_profile_extractable(tmp_path: Path) -> None:
    profile = build_profile(_make_blocks_path(tmp_path), book_name="x手册")
    extractable = profile.extractable_sections
    titles = {s.title for s in extractable}
    assert "目录" not in titles
    assert "附录C 十八反、十九畏" in titles


# ── Planner ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def registry() -> PromptTemplateRegistry:
    return PromptTemplateRegistry()


def test_template_registry_loads_versions(registry: PromptTemplateRegistry) -> None:
    tpl = registry.load("a", "interaction_group.md")
    assert "{{chunk_json}}" in tpl.content
    assert len(tpl.version) == 12
    # 缓存命中同一实例
    assert registry.load("a", "interaction_group.md") is tpl


def test_template_render(registry: PromptTemplateRegistry) -> None:
    tpl = registry.load("a", "general_rule.md")
    rendered = tpl.render('{"blocks": ["甘草反甘遂"]}')
    assert "甘草反甘遂" in rendered
    assert "{{chunk_json}}" not in rendered


def test_template_missing_raises(registry: PromptTemplateRegistry) -> None:
    with pytest.raises(FileNotFoundError):
        registry.load("a", "nonexistent.md")


def test_build_extraction_plan(tmp_path: Path) -> None:
    profile = build_profile(_make_blocks_path(tmp_path), book_name="中药饮片临床应用规范")
    plan = build_extraction_plan(profile)

    assert plan.document_type == "STANDARD"
    planned = {s.title: s for s in plan.planned_sections}
    skipped = [s for s in plan.sections if s.skip_reason]

    # 关系组 → InteractionExtractor；妊娠 → PregnancyRuleExtractor
    assert planned["附录C 十八反、十九畏"].extractor == "InteractionExtractor"
    assert planned["附录D 妊娠禁用药"].extractor == "PregnancyRuleExtractor"
    # 目录被跳过
    assert all(s.title != "目录" for s in plan.planned_sections)
    # 目录（IRRELEVANT）+ 无规则措辞的正文（REFERENCE_ONLY）均被跳过
    assert len(skipped) == 2
    assert "IRRELEVANT" in (skipped[1].skip_reason or "")
    # 每个计划章节都有 prompt_version（模板 hash）
    for sec in plan.planned_sections:
        assert len(sec.prompt_version) == 12

    # 风险点包含关系组提示
    assert any("十八反" in r for r in plan.risk_points)
