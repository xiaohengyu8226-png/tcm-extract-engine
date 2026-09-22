"""Model B 核验器测试：verdict 解析 / needs_review / 修正日志 / 缓存。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tcm_extract_engine.cab.verifier import (
    Verdict,
    VerificationError,
    verify_records,
)
from tcm_extract_engine.core.llm import LLMError, LLMUsage
from tcm_extract_engine.core.records import (
    ActionLevel,
    HerbDose,
    InteractionGroup,
    InteractionMember,
    RecordType,
    SourceRecord,
)


class FakeLLMClient:
    """顺序出队 canned 响应的假客户端。"""

    def __init__(self) -> None:
        self.usage = LLMUsage()
        self.queue: list[Any] = []
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


def _write_blocks(tmp_path: Path) -> Path:
    blocks = [
        {"type": "meta", "version": "1.0"},
        {
            "type": "content", "blockid": "b1", "format": "plain_text",
            "content": "甘草反甘遂、大戟、芫花、海藻。", "heading": "附录C 十八反、十九畏",
            "parent_headings": [], "level": 2,
            "positions": [{"type": "bbox", "anchor": "40", "range": [0, 0, 1, 1]}],
        },
        {
            "type": "content", "blockid": "b2", "format": "plain_text",
            "content": "制川乌 3～15g，先煎。", "heading": "4.4 剂量规范要求",
            "parent_headings": [], "level": 2,
            "positions": [{"type": "bbox", "anchor": "12", "range": [0, 0, 1, 1]}],
        },
    ]
    path = tmp_path / "blocks.jsonl"
    path.write_text(
        "\n".join(json.dumps(b, ensure_ascii=False) for b in blocks) + "\n",
        encoding="utf-8",
    )
    return path


def make_interaction_record(**overrides: Any) -> SourceRecord:
    defaults: dict[str, Any] = {
        "record_id": "SR-testbook-000001",
        "book_id": "testbook01",
        "record_type": RecordType.INTERACTION,
        "raw_text": "甘草反甘遂、大戟、芫花、海藻",
        "interaction": InteractionGroup(
            interaction_type="十八反",
            anchor=InteractionMember(herb_name="甘草", role="anchor"),
            members=[
                InteractionMember(herb_name="甘遂"),
                InteractionMember(herb_name="大戟"),
                InteractionMember(herb_name="芫花"),
                InteractionMember(herb_name="海藻"),
            ],
        ),
        "action": ActionLevel.PROHIBIT,
        "source": {"page_start": 40, "section_path": ["附录C 十八反、十九畏"]},
    }
    defaults.update(overrides)
    return SourceRecord.model_validate(defaults)


def make_dosage_record(**overrides: Any) -> SourceRecord:
    defaults: dict[str, Any] = {
        "record_id": "SR-testbook-000002",
        "book_id": "testbook01",
        "record_type": RecordType.DOSAGE,
        "raw_text": "制川乌 3～15g，先煎",
        "herb_name": "制川乌",
        "dose": HerbDose(min_dose=150, max_dose=3, unit="g", special_note="先煎"),
        "source": {"page_start": 12, "section_path": ["4.4 剂量规范要求"]},
    }
    defaults.update(overrides)
    return SourceRecord.model_validate(defaults)


# ── verdict 解析与 needs_review ──────────────────────────────────────────────


def test_verify_supported(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path)
    record = make_interaction_record()
    client = FakeLLMClient()
    client.queue = [{
        "records": [{
            "record_id": record.record_id,
            "verdict": "SUPPORTED",
            "source_evidence": "甘草反甘遂、大戟、芫花、海藻",
            "needs_human_review": False,
        }],
    }]

    report = asyncio.run(verify_records(
        records=[record], blocks_path=blocks_path, client=client,
    ))

    assert client.calls == 1
    assert report.total == 1
    assert report.results[0].verdict == Verdict.SUPPORTED
    assert report.results[0].evidence_quote == "甘草反甘遂、大戟、芫花、海藻"
    assert record.needs_review is False  # SUPPORTED 不标记
    assert report.corrections == []
    assert report.needs_review_count == 0


def test_verify_unsupported_marks_review(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path)
    record = make_interaction_record()
    client = FakeLLMClient()
    client.queue = [{
        "records": [{
            "record_id": record.record_id,
            "verdict": "UNSUPPORTED",
            "issues": [{"field": "interaction.members",
                        "problem": "原文未提及海藻", "source_evidence": None}],
            "needs_human_review": True,
        }],
    }]

    report = asyncio.run(verify_records(
        records=[record], blocks_path=blocks_path, client=client,
    ))

    assert report.results[0].verdict == Verdict.UNSUPPORTED
    assert record.needs_review is True
    assert report.needs_review_count == 1
    assert report.results[0].explanation and "海藻" in report.results[0].explanation


def test_verify_lowercase_verdict_normalized(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path)
    record = make_interaction_record()
    client = FakeLLMClient()
    client.queue = [{
        "records": [{"record_id": record.record_id, "verdict": "supported"}],
    }]
    report = asyncio.run(verify_records(
        records=[record], blocks_path=blocks_path, client=client,
    ))
    assert report.results[0].verdict == Verdict.SUPPORTED


def test_verify_unknown_verdict_falls_back_partial(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path)
    record = make_interaction_record()
    client = FakeLLMClient()
    client.queue = [{
        "records": [{"record_id": record.record_id, "verdict": "也许吧"}],
    }]
    report = asyncio.run(verify_records(
        records=[record], blocks_path=blocks_path, client=client,
    ))
    assert report.results[0].verdict == Verdict.PARTIAL
    assert record.needs_review is True  # 无法解析 → 复核


# ── 修正日志（最小修改原则：不改记录，只记日志）──────────────────────────────


def test_verify_correction_log_not_applied_to_record(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path)
    record = make_dosage_record()  # min=150, max=3（错误值）
    original_dump = record.model_dump(mode="json")
    client = FakeLLMClient()
    client.queue = [{
        "records": [{
            "record_id": record.record_id,
            "verdict": "WRONG_DOSE",
            "issues": [{"field": "dose", "problem": "原文为 3～15g",
                        "source_evidence": "制川乌 3～15g，先煎"}],
            "corrected_record": {
                "record_type": "DOSAGE",
                "herb_name": "制川乌",
                "dose": {"min_dose": 3, "max_dose": 15, "unit": "g",
                         "special_note": "先煎"},
            },
            "needs_human_review": True,
        }],
    }]

    report = asyncio.run(verify_records(
        records=[record], blocks_path=blocks_path, client=client,
    ))

    # 记录的事实内容不被修改（修正待人工确认；needs_review 是允许更新的质量标记）
    original_facts = {
        k: v for k, v in original_dump.items()
        if k not in {"needs_review", "confidence", "ambiguities"}
    }
    current_facts = {
        k: v for k, v in record.model_dump(mode="json").items()
        if k not in {"needs_review", "confidence", "ambiguities"}
    }
    assert current_facts == original_facts
    assert record.needs_review is True

    # 修正进入 correction_log
    assert len(report.corrections) == 1
    entry = report.corrections[0]
    assert entry.field == "dose"
    assert entry.original["min_dose"] == 150
    assert entry.suggested["min_dose"] == 3
    assert entry.suggested["max_dose"] == 15
    assert entry.verdict == Verdict.WRONG_DOSE


def test_verify_correction_invalid_goes_to_manual(tmp_path: Path) -> None:
    """B 修正无法通过 SourceRecord 校验 → 转人工条目。"""
    blocks_path = _write_blocks(tmp_path)
    record = make_dosage_record()
    client = FakeLLMClient()
    client.queue = [{
        "records": [{
            "record_id": record.record_id,
            "verdict": "WRONG_DOSE",
            "corrected_record": {"dose": "不是对象"},
        }],
    }]
    report = asyncio.run(verify_records(
        records=[record], blocks_path=blocks_path, client=client,
    ))
    assert len(report.corrections) == 1
    assert report.corrections[0].field == "(record)"
    assert "转人工" in report.corrections[0].reason


# ── 原文定位 ────────────────────────────────────────────────────────────────


def test_verify_no_source_skips_llm(tmp_path: Path) -> None:
    """无法定位原文的记录：不调 LLM，直接 UNSUPPORTED。"""
    blocks_path = _write_blocks(tmp_path)
    record = make_interaction_record(source={"page_start": 40})
    client = FakeLLMClient()  # 队列为空

    report = asyncio.run(verify_records(
        records=[record], blocks_path=blocks_path, client=client,
    ))
    assert client.calls == 0
    assert report.results[0].verdict == Verdict.UNSUPPORTED
    assert record.needs_review is True


def test_verify_falls_back_to_block_ids(tmp_path: Path) -> None:
    """section 标题缺失时回退 block_ids 定位。"""
    blocks_path = _write_blocks(tmp_path)
    record = make_interaction_record(
        source={"page_start": 40, "block_ids": ["b1"], "section_path": []}
    )
    client = FakeLLMClient()
    client.queue = [{
        "records": [{"record_id": record.record_id, "verdict": "SUPPORTED"}],
    }]
    report = asyncio.run(verify_records(
        records=[record], blocks_path=blocks_path, client=client,
    ))
    assert client.calls == 1  # 定位成功 → 送 LLM
    assert report.results[0].verdict == Verdict.SUPPORTED


# ── 批次与兜底 ──────────────────────────────────────────────────────────────


def test_verify_missing_result_fallback(tmp_path: Path) -> None:
    """B 漏答的记录 → PARTIAL 兜底并标记复核。"""
    blocks_path = _write_blocks(tmp_path)
    record = make_interaction_record()
    client = FakeLLMClient()
    client.queue = [{"records": []}]  # B 返回空

    report = asyncio.run(verify_records(
        records=[record], blocks_path=blocks_path, client=client,
    ))
    assert report.total == 1
    assert report.results[0].verdict == Verdict.PARTIAL
    assert "漏答" in (report.results[0].explanation or "")
    assert record.needs_review is True


def test_verify_multiple_records_batched(tmp_path: Path) -> None:
    """batch_size=1 时逐条调用，结果齐全。"""
    blocks_path = _write_blocks(tmp_path)
    r1 = make_interaction_record()
    r2 = make_dosage_record()
    client = FakeLLMClient()
    client.queue = [
        {"records": [{"record_id": r1.record_id, "verdict": "SUPPORTED"}]},
        {"records": [{"record_id": r2.record_id, "verdict": "PARTIAL"}]},
    ]

    progress: list[tuple[int, int]] = []
    report = asyncio.run(verify_records(
        records=[r1, r2], blocks_path=blocks_path, client=client,
        batch_size=1, on_progress=lambda d, t: progress.append((d, t)),
    ))

    assert client.calls == 2
    assert progress == [(1, 2), (2, 2)]
    assert report.total == 2
    assert report.verdict_counts() == {"SUPPORTED": 1, "PARTIAL": 1}


def test_verify_batch_failure_raises(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path)
    record = make_interaction_record()
    client = FakeLLMClient()
    client.queue = [LLMError("api", "模拟核验失败")]

    with pytest.raises(VerificationError, match="1/1"):
        asyncio.run(verify_records(
            records=[record], blocks_path=blocks_path, client=client,
        ))


def test_verify_cache_reuse(tmp_path: Path) -> None:
    """第二次运行命中批级缓存，不调 LLM。"""
    blocks_path = _write_blocks(tmp_path)
    record = make_interaction_record()
    cache_dir = tmp_path / "vcache"

    first = FakeLLMClient()
    first.queue = [{
        "records": [{"record_id": record.record_id, "verdict": "SUPPORTED"}],
    }]
    report1 = asyncio.run(verify_records(
        records=[record], blocks_path=blocks_path, client=first, cache_dir=cache_dir,
    ))

    second = FakeLLMClient()  # 队列为空：未命中缓存会抛错
    report2 = asyncio.run(verify_records(
        records=[record], blocks_path=blocks_path, client=second, cache_dir=cache_dir,
    ))
    assert second.calls == 0
    assert report2.verdict_counts() == report1.verdict_counts()


def test_verify_records_content_change_invalidates_cache(tmp_path: Path) -> None:
    """记录内容变化 → 缓存失效，重新核验。"""
    blocks_path = _write_blocks(tmp_path)
    record = make_interaction_record()
    cache_dir = tmp_path / "vcache"

    first = FakeLLMClient()
    first.queue = [{"records": [{"record_id": record.record_id, "verdict": "SUPPORTED"}]}]
    asyncio.run(verify_records(
        records=[record], blocks_path=blocks_path, client=first, cache_dir=cache_dir,
    ))

    # 记录内容变化（成员数不同）
    changed = make_interaction_record(
        interaction=InteractionGroup(
            interaction_type="十八反",
            anchor=InteractionMember(herb_name="甘草", role="anchor"),
            members=[InteractionMember(herb_name="甘遂"), InteractionMember(herb_name="大戟")],
        )
    )
    second = FakeLLMClient()
    second.queue = [{"records": [{"record_id": changed.record_id, "verdict": "PARTIAL"}]}]
    report = asyncio.run(verify_records(
        records=[changed], blocks_path=blocks_path, client=second, cache_dir=cache_dir,
    ))
    assert second.calls == 1  # 缓存未命中
    assert report.verdict_counts() == {"PARTIAL": 1}


def test_verify_summary_and_empty_records(tmp_path: Path) -> None:
    """空记录列表：零调用、零结果。"""
    blocks_path = _write_blocks(tmp_path)
    client = FakeLLMClient()
    report = asyncio.run(verify_records(
        records=[], blocks_path=blocks_path, client=client,
    ))
    assert client.calls == 0
    assert report.total == 0
    assert "0 条记录" in report.summary()
