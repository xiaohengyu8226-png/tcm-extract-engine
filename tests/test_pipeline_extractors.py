"""Model A 抽取器测试：组块策略 / LLM 输出解析 / 批级缓存 / 截断拆分。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tcm_extract_engine.cab.extractors import (
    MAX_SECTION_CHARS_PER_CALL,
    ExtractionError,
    SectionChunk,
    _load_batch_cache,
    _parse_record,
    _save_batch_cache,
    extract_records,
    split_section_batches,
)
from tcm_extract_engine.cab.planner import (
    PromptTemplateRegistry,
    SectionPlan,
    build_extraction_plan,
)
from tcm_extract_engine.cab.profiler import build_profile
from tcm_extract_engine.core.artifacts import ArtifactHeader, save_artifact
from tcm_extract_engine.core.llm import LLMError, LLMUsage
from tcm_extract_engine.core.records import RecordType

# ── 测试夹具 ─────────────────────────────────────────────────────────────────


def test_parse_record_attributes_fallback() -> None:
    """旧模板输出：类型化载荷在 attributes 中 → 解析器按 record_type 回退兼容。"""
    section = _section_plan()
    raw = {
        "record_type": "POPULATION",
        "raw_text": "肾功能不全者忌用雷公藤。",
        "attributes": {
            "population": "肾功能不全",
            "restriction_level": "忌用",
            "target_herbs": ["雷公藤"],
        },
    }
    record = _parse_record(raw, book_id="abc123def456", seq=1, section=section)
    assert record.population is not None
    assert record.population.population == "肾功能不全"
    assert record.population.restriction_level == "忌用"
    assert record.population.target_herbs == ["雷公藤"]

    raw_preg = {
        "record_type": "PREGNANCY",
        "raw_text": "孕妇禁用附子。",
        "attributes": {"restriction_level": "禁用", "target_herbs": ["附子"]},
    }
    record_preg = _parse_record(raw_preg, book_id="abc123def456", seq=2, section=section)
    assert record_preg.pregnancy is not None
    assert record_preg.pregnancy.restriction_level == "禁用"


def test_parse_record_toplevel_payload_wins_over_attributes() -> None:
    """顶层字段优先：attributes 中的同名载荷不覆盖正牌结构化字段。"""
    section = _section_plan()
    raw = {
        "record_type": "PREGNANCY",
        "raw_text": "孕妇禁用附子。",
        "pregnancy": {"restriction_level": "禁用", "target_herbs": ["附子"]},
        "attributes": {"restriction_level": "慎用", "target_herbs": ["错误"]},
    }
    record = _parse_record(raw, book_id="abc123def456", seq=1, section=section)
    assert record.pregnancy is not None
    assert record.pregnancy.restriction_level == "禁用"
    assert record.pregnancy.target_herbs == ["附子"]


def test_parse_record_attributes_fallback_ignores_wrong_type() -> None:
    """非对应 record_type 时不回退：attributes 里的 restriction_level 不产生 pregnancy 载荷。"""
    section = _section_plan()
    raw = {
        "record_type": "GENERAL_RULE",
        "raw_text": "附子先煎。",
        "attributes": {"restriction_level": "慎用", "target_herbs": ["附子"]},
    }
    record = _parse_record(raw, book_id="abc123def456", seq=1, section=section)
    assert record.pregnancy is None
    assert record.population is None


class FakeLLMClient:
    """顺序出队 canned 响应的假客户端；记录收到的 prompt。"""

    def __init__(self) -> None:
        self.usage = LLMUsage()
        self.queue: list[Any] = []  # dict=正常返回；Exception=抛出
        self.prompts: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    async def complete_json(
        self, *, system_prompt: str, user_prompt: str, semaphore: Any = None, **_: Any
    ) -> dict[str, Any]:
        self.prompts.append(user_prompt)
        if not self.queue:
            raise LLMError("api", "FakeLLMClient 队列为空")
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _block(blockid: str, heading: str, page: int, content: str) -> dict[str, Any]:
    return {
        "type": "content",
        "blockid": blockid,
        "format": "plain_text",
        "content": content,
        "heading": heading,
        "parent_headings": [],
        "level": 1,
        "positions": [{"type": "bbox", "anchor": str(page), "range": [0, 0, 1, 1]}],
    }


def _write_blocks(tmp_path: Path, blocks: list[dict[str, Any]]) -> Path:
    path = tmp_path / "blocks.jsonl"
    path.write_text(
        "\n".join(json.dumps(b, ensure_ascii=False) for b in blocks) + "\n",
        encoding="utf-8",
    )
    return path


def _section_plan(section_id: str = "sec-1", extractor: str = "GeneralRuleExtractor",
                   template: str = "general_rule.md", block_ids: list[str] | None = None,
                   title: str = "4.4 剂量规范要求") -> SectionPlan:
    registry = PromptTemplateRegistry()
    return SectionPlan(
        section_id=section_id,
        title=title,
        heading_path=[title],
        extractor=extractor,
        template=template,
        prompt_version=registry.version_of("a", template),
        block_ids=block_ids or [],
        page_start=1,
        page_end=10,
    )


# ── split_section_batches 组块策略 ───────────────────────────────────────────


def test_split_relation_group_never_split() -> None:
    """关系组章节（十八反/十九畏）整章一批，禁止切断。"""
    texts = [f"第{i}段" * 100 for i in range(10)]
    blocks_by_id = {
        f"b{i}": _block(f"b{i}", "附录C 十八反", 1, t)
        for i, t in enumerate(texts)
    }
    section = _section_plan(
        extractor="InteractionExtractor", template="interaction_group.md",
        block_ids=list(blocks_by_id), title="附录C 十八反",
    )
    batches = split_section_batches(section, blocks_by_id)
    assert len(batches) == 1
    assert len(batches[0].texts) == len(texts)


def test_split_table_block_is_own_batch() -> None:
    """表格块（<drawing>）自成批次，不与文本混批。"""
    blocks_by_id = {
        "t1": _block("t1", "附录A", 1, "<drawing id='tb-1'><table><tr><td>川乌</td></tr></table></drawing>"),
        "p1": _block("p1", "附录A", 1, "正文说明。"),
        "p2": _block("p2", "附录A", 2, "更多正文。"),
    }
    section = _section_plan(block_ids=["t1", "p1", "p2"], title="附录A")
    batches = split_section_batches(section, blocks_by_id)
    # 表格独立 + 文本聚合 = 2 批
    assert len(batches) == 2
    assert "<drawing" in batches[0].texts[0]
    assert all("<drawing" not in t for t in batches[1].texts)


def test_split_text_aggregation_char_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """文本块按字符上限聚合分批。"""
    monkeypatch.setattr(
        "tcm_extract_engine.cab.extractors.MAX_SECTION_CHARS_PER_CALL", 100
    )
    blocks_by_id = {
        f"b{i}": _block(f"b{i}", "4.4", 1, "字" * 60)
        for i in range(4)
    }
    section = _section_plan(block_ids=list(blocks_by_id))
    batches = split_section_batches(section, blocks_by_id)
    # 每批 ≥100 字即切：60+60=120 超限 → 每批 2 块 → 2 批
    assert len(batches) == 2
    assert all(len(b.texts) == 2 for b in batches)
    # 批次 chunk_key 唯一
    assert len({b.chunk_key for b in batches}) == 2


def test_split_missing_blocks_skipped() -> None:
    """block_ids 中不存在于 blocks_by_id 的块被跳过。"""
    section = _section_plan(block_ids=["gone"])
    assert split_section_batches(section, {}) == []


# ── 批级缓存 ─────────────────────────────────────────────────────────────────


def test_batch_cache_roundtrip_and_version_invalidation(tmp_path: Path) -> None:
    from tcm_extract_engine.core.records import SourceRecord

    record = SourceRecord(
        record_id="SR-test-000001", book_id="test",
        record_type=RecordType.GENERAL_RULE, raw_text="孕妇慎用红花。",
    )
    _save_batch_cache(tmp_path, "sec:abc", [record], prompt_version="v1")
    loaded = _load_batch_cache(tmp_path, "sec:abc", prompt_version="v1")
    assert loaded is not None and loaded[0].record_id == "SR-test-000001"

    # 模板版本变化 → 缓存失效
    assert _load_batch_cache(tmp_path, "sec:abc", prompt_version="v2") is None


# ── extract_records 端到端（FakeLLMClient）──────────────────────────────────


_DOSAGE_RESPONSE = {
    "records": [{
        "record_type": "DOSAGE",
        "raw_text": "制川乌 1.5～3g，先煎",
        "page_hint": 12,
        "herb_name": "制川乌",
        "dose": {"min_dose": 1.5, "max_dose": 3, "unit": "g", "special_note": "先煎"},
    }],
}
_INTERACTION_RESPONSE = {
    "records": [{
        "record_type": "INTERACTION",
        "raw_text": "甘草反甘遂、大戟、芫花、海藻",
        "page_hint": 40,
        "interaction": {
            "interaction_type": "十八反",
            "group_name": "十八反-甘草组",
            "anchor": {"herb_name": "甘草"},
            "members": [{"herb_name": "甘遂"}, {"herb_name": "大戟"},
                        {"herb_name": "芫花"}, {"herb_name": "海藻"}],
        },
        "action": "PROHIBIT",
    }],
}


def _setup_book(tmp_path: Path) -> tuple[Path, Path]:
    """构造 2 个可抽取章节的 blocks + plan artifact。"""
    blocks = [
        {"type": "meta", "version": "1.0"},
        _block("b1", "4.4 剂量规范要求", 12, "制川乌 1.5～3g，先煎。"),
        _block("b2", "附录C 十八反、十九畏", 40, "甘草反甘遂、大戟、芫花、海藻。"),
    ]
    blocks_path = _write_blocks(tmp_path, blocks)
    profile = build_profile(blocks_path, book_name="测试手册")
    plan = build_extraction_plan(profile)
    plan_path = tmp_path / "extraction_plan.json"
    save_artifact(
        plan_path,
        header=ArtifactHeader(schema_version="1", book_id="testbook01", task_id="t1"),
        data=plan.model_dump(mode="json"),
    )
    return blocks_path, plan_path


def test_extract_records_end_to_end(tmp_path: Path) -> None:
    blocks_path, plan_path = _setup_book(tmp_path)
    client = FakeLLMClient()
    client.queue = [dict(_DOSAGE_RESPONSE), dict(_INTERACTION_RESPONSE)]

    records = asyncio.run(extract_records(
        book_id="testbook01", blocks_path=blocks_path, plan_path=plan_path,
        cache_dir=tmp_path / "cache", client=client,
    ))

    assert client.calls == 2
    assert len(records) == 2
    # 全局序号重排
    assert records[0].record_id == "SR-testbook-000001"
    assert records[1].record_id == "SR-testbook-000002"
    # 记录类型与载荷
    assert records[0].record_type == RecordType.DOSAGE
    assert records[0].dose is not None and records[0].dose.max_dose == 3
    assert records[1].record_type == RecordType.INTERACTION
    assert records[1].interaction is not None
    assert len(records[1].interaction.members) == 4
    assert records[1].source.page_start == 40  # page_hint 生效


def test_extract_records_uses_cache_on_rerun(tmp_path: Path) -> None:
    """第二次运行命中批级缓存，不再调用 LLM。"""
    blocks_path, plan_path = _setup_book(tmp_path)
    cache_dir = tmp_path / "cache"

    first = FakeLLMClient()
    first.queue = [dict(_DOSAGE_RESPONSE), dict(_INTERACTION_RESPONSE)]
    records1 = asyncio.run(extract_records(
        book_id="testbook01", blocks_path=blocks_path, plan_path=plan_path,
        cache_dir=cache_dir, client=first,
    ))

    second = FakeLLMClient()  # 队列为空：若未命中缓存将抛错
    records2 = asyncio.run(extract_records(
        book_id="testbook01", blocks_path=blocks_path, plan_path=plan_path,
        cache_dir=cache_dir, client=second,
    ))
    assert second.calls == 0
    assert [r.record_id for r in records2] == [r.record_id for r in records1]


def test_extract_records_partial_failure_then_resume(tmp_path: Path) -> None:
    """单批失败：成功批次已缓存；重跑只补失败批次。"""
    blocks_path, plan_path = _setup_book(tmp_path)
    cache_dir = tmp_path / "cache"

    failing = FakeLLMClient()
    failing.queue = [dict(_DOSAGE_RESPONSE), LLMError("api", "模拟 API 故障")]
    with pytest.raises(ExtractionError, match="1/2"):
        asyncio.run(extract_records(
            book_id="testbook01", blocks_path=blocks_path, plan_path=plan_path,
            cache_dir=cache_dir, client=failing,
        ))

    # resume：第一批走缓存，第二批重新调用
    resumed = FakeLLMClient()
    resumed.queue = [dict(_INTERACTION_RESPONSE)]
    records = asyncio.run(extract_records(
        book_id="testbook01", blocks_path=blocks_path, plan_path=plan_path,
        cache_dir=cache_dir, client=resumed,
    ))
    assert resumed.calls == 1
    assert len(records) == 2
    assert records[1].record_type == RecordType.INTERACTION


def test_extract_records_truncation_split_retry(tmp_path: Path) -> None:
    """输出截断 → 对半拆分重试。"""
    blocks = [
        {"type": "meta", "version": "1.0"},
        _block("t1", "4.4 剂量规范要求", 12, "制川乌 1.5～3g，先煎。"),
        _block("t2", "4.4 剂量规范要求", 13, "生川乌仅为外用，内服禁用。"),
    ]
    blocks_path = _write_blocks(tmp_path, blocks)
    profile = build_profile(blocks_path, book_name="测试手册")
    plan = build_extraction_plan(profile)
    plan_path = tmp_path / "extraction_plan.json"
    save_artifact(
        plan_path,
        header=ArtifactHeader(schema_version="1", book_id="truncbook", task_id="t1"),
        data=plan.model_dump(mode="json"),
    )

    client = FakeLLMClient()
    truncation = LLMError("parse", "LLM 输出非法 JSON: Expecting value (char 95)")
    truncation.content = "x" * 100  # char 95 ≥ 80%
    client.queue = [
        truncation,
        {"records": [{"record_type": "DOSAGE", "raw_text": "制川乌", "herb_name": "制川乌"}]},
        {"records": [{"record_type": "USAGE", "raw_text": "生川乌外用", "herb_name": "生川乌"}]},
    ]

    records = asyncio.run(extract_records(
        book_id="truncbook", blocks_path=blocks_path, plan_path=plan_path,
        cache_dir=tmp_path / "cache", client=client,
    ))
    assert client.calls == 3  # 1 次截断 + 2 次拆分后
    assert len(records) == 2


def test_extract_records_progress_callback(tmp_path: Path) -> None:
    """进度回调按批次推进。"""
    blocks_path, plan_path = _setup_book(tmp_path)
    client = FakeLLMClient()
    client.queue = [dict(_DOSAGE_RESPONSE), dict(_INTERACTION_RESPONSE)]

    progress: list[tuple[int, int]] = []
    asyncio.run(extract_records(
        book_id="testbook01", blocks_path=blocks_path, plan_path=plan_path,
        cache_dir=tmp_path / "cache", client=client,
        on_progress=lambda p: progress.append((p.current, p.total)),
    ))
    assert progress == [(1, 2), (2, 2)]


def test_section_chunk_dataclass() -> None:
    section = _section_plan()
    chunk = SectionChunk(section=section, texts=["a"], pages=[1], chunk_key="k")
    assert chunk.section is section
    assert MAX_SECTION_CHARS_PER_CALL > 0
