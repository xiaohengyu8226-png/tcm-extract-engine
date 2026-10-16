"""BookPipeline Runner 测试：端到端编排 / 断点恢复 / 重跑 / 人工确认门。

策略：用 Markdown 源文件驱动真实 parse（ensure_markdown_artifact）
+ 真实 profile/plan/validate（确定性），仅注入 Fake LLM 客户端
（extract = Model A，verify = Model B）。
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from tcm_extract_engine.cab.planner import PromptTemplateRegistry
from tcm_extract_engine.core.artifacts import load_artifact, load_artifact_records
from tcm_extract_engine.core.book import ADAPTATION_FILENAME, BookWorkspace, sha256_file
from tcm_extract_engine.core.llm import LLMError, LLMUsage
from tcm_extract_engine.core.manifest import BookStatus
from tcm_extract_engine.domains.tcm.schema import RecordType
from tcm_extract_engine.domains.tcm.schema import TCMRecord as SourceRecord
from tcm_extract_engine.runner import BookPipeline

# ── 测试夹具 ─────────────────────────────────────────────────────────────────


class FakeLLMClient:
    """Runner 测试用假客户端：model 属性 + 顺序出队响应。"""

    def __init__(self, model: str = "fake-model") -> None:
        self.model = model
        self.usage = LLMUsage()
        self.queue: list[Any] = []
        self.calls: list[str] = []

    async def complete_json(
        self, *, system_prompt: str, user_prompt: str, semaphore: Any = None, **_: Any
    ) -> dict[str, Any]:
        self.calls.append(user_prompt)
        if not self.queue:
            raise LLMError("api", "FakeLLMClient 队列为空")
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if "核验器" in system_prompt:
            import copy
            import re

            item = copy.deepcopy(item)
            ids = list(dict.fromkeys(re.findall(r'"record_id":\s*"(SR-[^"]+)"', user_prompt)))
            for row, actual_id in zip(item.get("records", []), ids):
                row["record_id"] = actual_id
        return item


SAMPLE_MD = """# 第一章 剂量规范

川乌制草乌生用内服极量不得超过二克，以免中毒。

# 附录C 十八反

甘草反甘遂、大戟、海藻、芫花；乌头反贝母、瓜蒌、半夏、白蔹、白及。
"""

GENERAL_RULE_RESPONSE = {
    "records": [
        {
            "record_type": "GENERAL_RULE",
            "raw_text": "川乌制草乌生用内服极量不得超过二克",
            "subject_text": "川乌",
            "action": "PROHIBIT",
        }
    ]
}

INTERACTION_RESPONSE = {
    "records": [
        {
            "record_type": "INTERACTION",
            "raw_text": "甘草反甘遂、大戟、海藻、芫花",
            "interaction": {
                "group_name": "十八反-甘草组",
                "interaction_type": "十八反",
                "anchor": {"herb_name": "甘草"},
                "members": [
                    {"herb_name": "甘遂"},
                    {"herb_name": "大戟"},
                    {"herb_name": "海藻"},
                    {"herb_name": "芫花"},
                ],
            },
        }
    ]
}


def _make_source(tmp_path: Path, name: str = "测试手册.md") -> Path:
    source = tmp_path / "sources" / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(SAMPLE_MD, encoding="utf-8")
    return source


def _make_pipeline(
    books_root: Path,
    extract: FakeLLMClient | None = None,
    verify: FakeLLMClient | None = None,
    prompt: FakeLLMClient | None = None,
    structure: FakeLLMClient | None = None,
) -> BookPipeline:
    return BookPipeline(
        books_root,
        extract_client=extract or FakeLLMClient(),  # type: ignore[arg-type]
        verify_client=verify or FakeLLMClient(),  # type: ignore[arg-type]
        # 默认空队列 fake：适配块生成即刻降级为 base-only，不触发真实网络
        prompt_client=prompt or FakeLLMClient(),  # type: ignore[arg-type]
        # 默认空队列 fake：大纲修复即刻降级为平铺标题，不触发真实网络
        structure_client=structure or FakeLLMClient(),  # type: ignore[arg-type]
    )


def _queue_success(extract: FakeLLMClient, verify: FakeLLMClient, book_id: str) -> None:
    """排入一轮完整成功所需的 A/B 响应。"""
    extract.queue = [GENERAL_RULE_RESPONSE, INTERACTION_RESPONSE]
    verify.queue = [
        {
            "records": [
                {
                    "record_id": f"SR-{book_id[:8]}-000001",
                    "verdict": "SUPPORTED",
                    "source_evidence": "川乌制草乌生用内服极量不得超过二克",
                },
                {
                    "record_id": f"SR-{book_id[:8]}-000002",
                    "verdict": "SUPPORTED",
                    "source_evidence": "甘草反甘遂、大戟、海藻、芫花",
                },
            ]
        }
    ]


def _book_id_of(source: Path) -> str:
    return sha256_file(source).removeprefix("sha256:")[:12]


# ── 端到端 ───────────────────────────────────────────────────────────────────


def test_run_end_to_end(tmp_path: Path) -> None:
    """Markdown 源 → 六阶段全部完成 → WAITING_REVIEW。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    extract, verify = FakeLLMClient("fake-a"), FakeLLMClient("fake-b")
    _queue_success(extract, verify, book_id)
    pipeline = _make_pipeline(tmp_path / "books", extract, verify)

    manifest = asyncio.run(pipeline.run(source, domain_id="tcm"))

    assert manifest.status is BookStatus.WAITING_REVIEW
    assert manifest.book_id == book_id
    assert manifest.prompt_version  # plan 阶段写入
    # 阶段记录：6 个自动阶段全部 DONE
    for stage in ("parse", "profile", "plan", "extract", "validate", "verify"):
        assert manifest.stages[stage].status == "DONE", stage
    assert manifest.stages["extract"].records == 2
    assert manifest.stages["parse"].records == 2  # block_count

    # LLM 调用：A=2 批（每章一批），B=1 批（2 条记录）
    assert len(extract.calls) == 2
    assert len(verify.calls) == 1

    # Artifact 链全部落盘
    ws = BookWorkspace(tmp_path / "books", book_id)
    for stage, filename in (
        ("parse", "parse_summary.json"),
        ("profile", "document_profile.json"),
        ("plan", "extraction_plan.json"),
        ("extract", "records.jsonl"),
        ("validate", "validation_report.json"),
        ("verify", "verified_records.jsonl"),
    ):
        assert ws.stage_artifact(stage).is_file(), f"{stage}/{filename}"
    assert (ws.stage_dir("parse") / "blocks.jsonl").is_file()

    # 记录内容：类型与动作正确
    records = load_artifact_records(ws.stage_artifact("extract"), SourceRecord)
    assert [r.record_type for r in records] == [
        RecordType.GENERAL_RULE,
        RecordType.INTERACTION,
    ]

    # 无 ERROR 校验问题 → 不产生 suspicious_records.jsonl
    assert not (ws.stage_dir("validate") / "suspicious_records.jsonl").is_file()

    # B 全部 SUPPORTED → needs_review 保持 False
    verified = load_artifact_records(ws.stage_artifact("verify"), SourceRecord)
    assert all(not r.needs_review for r in verified)


def test_run_is_idempotent(tmp_path: Path) -> None:
    """同源文件重复 run：复用工作目录，全部阶段跳过（0 次 LLM 调用）。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    extract, verify = FakeLLMClient(), FakeLLMClient()
    _queue_success(extract, verify, book_id)
    pipeline = _make_pipeline(tmp_path / "books", extract, verify)
    asyncio.run(pipeline.run(source, domain_id="tcm"))

    # 第二次 run：全新客户端（空队列）→ 若阶段被重跑会因队列为空而失败
    pipeline2 = _make_pipeline(tmp_path / "books")
    manifest2 = asyncio.run(pipeline2.run(source, domain_id="tcm"))

    assert manifest2.status is BookStatus.WAITING_REVIEW
    assert manifest2.book_id == book_id


def test_run_missing_source_raises(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path / "books")
    with pytest.raises(FileNotFoundError):
        asyncio.run(pipeline.run(tmp_path / "不存在.md", domain_id="tcm"))


# ── 人工确认门 ───────────────────────────────────────────────────────────────


def test_run_stops_at_plan_confirmation_gate(tmp_path: Path) -> None:
    """--no-confirm-plan：停在 WAITING_PLAN_CONFIRMATION，不调 LLM。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    extract, verify = FakeLLMClient(), FakeLLMClient()
    pipeline = _make_pipeline(tmp_path / "books", extract, verify)

    manifest = asyncio.run(pipeline.run(source, confirm_plan=False, domain_id="tcm"))

    assert manifest.status is BookStatus.WAITING_PLAN_CONFIRMATION
    assert manifest.stages["plan"].status == "DONE"
    assert "extract" not in manifest.stages
    assert not extract.calls and not verify.calls

    # 人工确认 → 续跑完成
    _queue_success(extract, verify, book_id)
    pipeline.confirm_plan(book_id)
    assert pipeline.status(book_id).status is BookStatus.PLAN_CONFIRMED

    manifest2 = asyncio.run(pipeline.resume(book_id))
    assert manifest2.status is BookStatus.WAITING_REVIEW
    assert len(extract.calls) == 2
    assert len(verify.calls) == 1


def test_confirm_plan_requires_waiting(tmp_path: Path) -> None:
    """未到确认门时 confirm_plan 应报非法迁移。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    books_root = tmp_path / "books"
    BookWorkspace.create(books_root, source)  # 建目录，状态停在 PENDING
    pipeline = _make_pipeline(books_root)
    with pytest.raises(ValueError, match="非法状态迁移"):
        pipeline.confirm_plan(book_id)


def test_run_stage_plan_skip_keeps_confirm_gate(tmp_path: Path) -> None:
    """plan 已 DONE 时 run_stage(confirm_plan=False) 不得静默越过确认门。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    extract, verify = FakeLLMClient(), FakeLLMClient()
    pipeline = _make_pipeline(tmp_path / "books", extract, verify)
    asyncio.run(pipeline.run(source, confirm_plan=False, domain_id="tcm"))
    assert pipeline.status(book_id).status is BookStatus.WAITING_PLAN_CONFIRMATION

    # 重复点「生成提示词」：跳过路径也必须停在确认门，不触发 extract
    manifest = asyncio.run(pipeline.run_stage(book_id, "plan", confirm_plan=False))
    assert manifest.status is BookStatus.WAITING_PLAN_CONFIRMATION
    assert not extract.calls


def test_rerun_from_plan_can_stop_at_gate(tmp_path: Path) -> None:
    """rerun --from plan 带 confirm_plan=False：重建 plan 后停在确认门。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    books_root = tmp_path / "books"
    extract, verify = FakeLLMClient(), FakeLLMClient()
    _queue_success(extract, verify, book_id)
    asyncio.run(_make_pipeline(books_root, extract, verify).run(source, domain_id="tcm"))

    extract2, verify2 = FakeLLMClient(), FakeLLMClient()
    manifest = asyncio.run(
        _make_pipeline(books_root, extract2, verify2).rerun(
            book_id, from_stage="plan", confirm_plan=False
        )
    )
    assert manifest.status is BookStatus.WAITING_PLAN_CONFIRMATION
    assert manifest.stages["plan"].status == "DONE"
    assert "extract" not in manifest.stages
    assert not extract2.calls


# ── 失败 / 续跑 / 重试 ───────────────────────────────────────────────────────


def test_resume_after_verify_failure(tmp_path: Path) -> None:
    """B 核验失败 → FAILED；resume 跳过已完成阶段只重跑 verify。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    extract = FakeLLMClient()
    _queue_success(extract, FakeLLMClient(), book_id)
    verify_fail = FakeLLMClient()
    verify_fail.queue = [LLMError("api", "模拟核验 API 故障")]
    pipeline = _make_pipeline(tmp_path / "books", extract, verify_fail)

    with pytest.raises(Exception, match="核验失败"):
        asyncio.run(pipeline.run(source, domain_id="tcm"))

    failed = pipeline.status(book_id)
    assert failed.status is BookStatus.FAILED
    assert failed.stages["verify"].status == "FAILED"
    assert failed.stages["extract"].status == "DONE"
    assert len(extract.calls) == 2

    # 换一个健康的 verify 客户端续跑：extract 不重跑（调用数不变）
    verify_ok = FakeLLMClient()
    _queue_success(FakeLLMClient(), verify_ok, book_id)
    pipeline2 = _make_pipeline(tmp_path / "books", extract, verify_ok)
    manifest = asyncio.run(pipeline2.resume(book_id))

    assert manifest.status is BookStatus.WAITING_REVIEW
    assert len(extract.calls) == 2  # 未重抽
    assert len(verify_ok.calls) == 1  # 只重跑核验


def test_retry_after_extract_failure(tmp_path: Path) -> None:
    """A 抽取失败 → FAILED；retry 复位后续跑成功。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    pipeline = _make_pipeline(tmp_path / "books")  # 空队列 → 抽取必失败

    with pytest.raises(Exception, match="批次抽取失败"):
        asyncio.run(pipeline.run(source, domain_id="tcm"))

    failed = pipeline.status(book_id)
    assert failed.status is BookStatus.FAILED
    assert failed.stages["parse"].status == "DONE"
    assert failed.stages["extract"].status == "FAILED"

    # 健康客户端 retry：parse/profile/plan 跳过，extract 起重跑
    extract, verify = FakeLLMClient(), FakeLLMClient()
    _queue_success(extract, verify, book_id)
    pipeline2 = _make_pipeline(tmp_path / "books", extract, verify)
    manifest = asyncio.run(pipeline2.retry(book_id))

    assert manifest.status is BookStatus.WAITING_REVIEW
    assert len(extract.calls) == 2
    assert len(verify.calls) == 1


def test_rerun_from_extract_reexecutes_downstream(tmp_path: Path) -> None:
    """rerun --from extract：下游阶段作废重跑（清缓存后必须重新调 LLM）。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    extract, verify = FakeLLMClient(), FakeLLMClient()
    _queue_success(extract, verify, book_id)
    books_root = tmp_path / "books"
    pipeline = _make_pipeline(books_root, extract, verify)
    asyncio.run(pipeline.run(source, domain_id="tcm"))

    # 作废缓存，强制重跑时真实调用 LLM
    ws = BookWorkspace(books_root, book_id)
    shutil.rmtree(ws.stage_dir("extract") / "cache")
    shutil.rmtree(ws.stage_dir("verify") / "cache")

    extract2, verify2 = FakeLLMClient(), FakeLLMClient()
    _queue_success(extract2, verify2, book_id)
    pipeline2 = _make_pipeline(books_root, extract2, verify2)
    manifest = asyncio.run(pipeline2.rerun(book_id, from_stage="extract"))

    assert manifest.status is BookStatus.WAITING_REVIEW
    assert len(extract2.calls) == 2  # extract 重跑
    assert len(verify2.calls) == 1  # verify 重跑
    for stage in ("parse", "profile", "plan"):
        assert manifest.stages[stage].status == "DONE"  # 上游保留
    # 抽取缓存重建
    assert (ws.stage_dir("extract") / "cache").is_dir()


def test_rerun_from_plan_clears_prompt_version(tmp_path: Path) -> None:
    """rerun --from plan：prompt_version 清空后重建，extract 重跑。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    books_root = tmp_path / "books"
    extract, verify = FakeLLMClient(), FakeLLMClient()
    _queue_success(extract, verify, book_id)
    asyncio.run(_make_pipeline(books_root, extract, verify).run(source, domain_id="tcm"))
    old_prompt_version = _make_pipeline(books_root).status(book_id).prompt_version
    assert old_prompt_version

    ws = BookWorkspace(books_root, book_id)
    shutil.rmtree(ws.stage_dir("extract") / "cache")
    shutil.rmtree(ws.stage_dir("verify") / "cache")

    extract2, verify2 = FakeLLMClient(), FakeLLMClient()
    _queue_success(extract2, verify2, book_id)
    manifest = asyncio.run(
        _make_pipeline(books_root, extract2, verify2).rerun(book_id, from_stage="plan")
    )

    assert manifest.status is BookStatus.WAITING_REVIEW
    assert manifest.prompt_version == old_prompt_version  # 模板未变 → 重建后一致
    assert len(extract2.calls) == 2  # plan 作废 → extract 重跑


def test_rerun_invalid_stage_rejected(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    pipeline = _make_pipeline(tmp_path / "books")
    book_id = _book_id_of(source)
    with pytest.raises(ValueError, match="未知阶段"):
        asyncio.run(pipeline.rerun(book_id, from_stage="nonexistent"))


def test_rewind_pops_downstream_but_keeps_artifacts(tmp_path: Path) -> None:
    """rewind：作废本阶段及下游记录、状态回退，但产物文件保留供回看。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    books_root = tmp_path / "books"
    extract, verify = FakeLLMClient(), FakeLLMClient()
    _queue_success(extract, verify, book_id)
    asyncio.run(_make_pipeline(books_root, extract, verify).run(source, domain_id="tcm"))
    ws = BookWorkspace(books_root, book_id)
    assert ws.stage_artifact("verify").is_file()

    manifest = _make_pipeline(books_root).rewind(book_id, to_stage="extract")

    assert manifest.status is BookStatus.PLAN_CONFIRMED
    assert set(manifest.stages) == {"parse", "profile", "plan"}
    assert ws.stage_artifact("verify").is_file()  # 产物保留
    assert ws.stage_artifact("extract").is_file()


def test_rewind_idempotent_when_status_already_at_pre_status(tmp_path: Path) -> None:
    """状态已处于目标前置态时 rewind 幂等（Web「返回上一步」后重跑下游的场景）。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    books_root = tmp_path / "books"
    extract, verify = FakeLLMClient(), FakeLLMClient()
    _queue_success(extract, verify, book_id)
    asyncio.run(_make_pipeline(books_root, extract, verify).run(source, domain_id="tcm"))
    ws = BookWorkspace(books_root, book_id)

    # 模拟 back：只回退状态到 EXTRACTED，阶段记录全部保留
    m = ws.load_manifest()
    m.rewind(BookStatus.EXTRACTED)
    m.save(ws.book_dir)

    # 再 rewind 到 validate：前置态 == 当前态，不得报错，记录照常作废
    manifest = _make_pipeline(books_root).rewind(book_id, to_stage="validate")
    assert manifest.status is BookStatus.EXTRACTED
    assert set(manifest.stages) == {"parse", "profile", "plan", "extract"}


# ── 状态查询 ─────────────────────────────────────────────────────────────────


def test_status_and_list_all(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    extract, verify = FakeLLMClient(), FakeLLMClient()
    _queue_success(extract, verify, book_id)
    pipeline = _make_pipeline(tmp_path / "books", extract, verify)
    asyncio.run(pipeline.run(source, domain_id="tcm"))

    manifest = pipeline.status(book_id)
    assert manifest.status is BookStatus.WAITING_REVIEW
    assert manifest.file_name == source.name

    books = pipeline.list_all()
    assert [m.book_id for m in books] == [book_id]

    with pytest.raises(FileNotFoundError):
        pipeline.status("0" * 12)


# ── 源文件定位 ───────────────────────────────────────────────────────────────


def test_locate_source_markdown_copy(tmp_path: Path) -> None:
    """Markdown 源：00_source 里的副本即内容本体。"""
    source = _make_source(tmp_path)
    books_root = tmp_path / "books"
    ws = BookWorkspace.create(books_root, source)
    manifest = ws.load_manifest()
    pipeline = _make_pipeline(books_root)

    located = pipeline._locate_source(ws, manifest)  # noqa: SLF001
    assert located == ws.source_dir() / source.name
    assert located.read_text(encoding="utf-8") == SAMPLE_MD


def test_locate_source_pdf_magic_and_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PDF 源：%PDF 魔数走副本；复制失败退化为文本指针并回溯原路径。"""
    books_root = tmp_path / "books"

    # 1) 真 PDF 副本（魔数开头）
    pdf = tmp_path / "sources" / "book.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.7 fake content")
    ws = BookWorkspace.create(books_root, pdf)
    pipeline = _make_pipeline(books_root)
    located = pipeline._locate_source(ws, ws.load_manifest())  # noqa: SLF001
    assert located == ws.source_dir() / pdf.name

    # 2) 复制失败 → 文本指针 → _locate_source 回溯原始路径
    pdf2 = tmp_path / "sources" / "book2.pdf"
    pdf2.write_bytes(b"%PDF-1.7 second")

    import tcm_extract_engine.core.book as book_module

    def _fail_copy(src: Path, dst: Path) -> None:  # noqa: ARG001
        raise OSError("模拟复制失败（跨设备）")

    monkeypatch.setattr(book_module.shutil, "copy2", _fail_copy)
    ws2 = BookWorkspace.create(books_root, pdf2)
    manifest2 = ws2.load_manifest()
    located2 = pipeline._locate_source(ws2, manifest2)  # noqa: SLF001
    assert located2 == pdf2.resolve()


# ── Artifact 血缘链 ─────────────────────────────────────────────────────────


def test_artifact_chain_hashes(tmp_path: Path) -> None:
    """extract/verify Artifact 头携带 model 与 prompt_version，形成可追溯链。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    extract, verify = FakeLLMClient("model-a-1"), FakeLLMClient("model-b-1")
    _queue_success(extract, verify, book_id)
    books_root = tmp_path / "books"
    asyncio.run(_make_pipeline(books_root, extract, verify).run(source, domain_id="tcm"))

    ws = BookWorkspace(books_root, book_id)

    def _jsonl_header(path: Path) -> dict[str, Any]:
        import json as _json

        first_line = next(ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
        return _json.loads(first_line)["header"]

    extract_header = _jsonl_header(ws.stage_artifact("extract"))
    assert extract_header["model"] == "model-a-1"
    assert extract_header["prompt_version"]
    assert extract_header["upstream_hash"]

    verify_header = _jsonl_header(ws.stage_artifact("verify"))
    assert verify_header["model"] == "model-b-1"
    assert verify_header["upstream_hash"]

    # 记录类型解析（真实 SourceRecord）
    records = load_artifact_records(ws.stage_artifact("verify"), SourceRecord)
    assert [r.record_type for r in records] == [
        RecordType.GENERAL_RULE,
        RecordType.INTERACTION,
    ]
    assert records[1].interaction is not None
    assert records[1].interaction.anchor.herb_name == "甘草"


# ── 每书适配块（Model C LLM 参与）─────────────────────────────────────────────


ADAPTATION_RESPONSE = {
    "terminology_aliases": ["川乌即制川乌"],
    "watch_points": ["剂量数字易 OCR 出错"],
}

GOAL_RESPONSE = {"goal": "中药临床应用知识抽取与规则库建设"}


def test_plan_generates_per_book_adaptation(tmp_path: Path) -> None:
    """开启适配：plan 生成每书适配块、prompt_version 相对 base bump、extract 用合成模板。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    extract, verify = FakeLLMClient("a"), FakeLLMClient("b")
    _queue_success(extract, verify, book_id)
    prompt = FakeLLMClient("c")
    # 适配块两个模板各一次，然后 goal 归纳一次（_run_plan 先适配后 goal）
    prompt.queue = [ADAPTATION_RESPONSE, ADAPTATION_RESPONSE, GOAL_RESPONSE]
    books_root = tmp_path / "books"
    pipeline = _make_pipeline(books_root, extract, verify, prompt)
    asyncio.run(pipeline.run(source, domain_id="tcm"))

    ws = BookWorkspace(books_root, book_id)
    # 1) 适配产物落盘
    adapt_path = ws.stage_dir("plan") / ADAPTATION_FILENAME
    assert adapt_path.is_file()
    adapt_data = load_artifact(adapt_path)
    assert adapt_data["profile_hash"]
    assert "general_rule.md" in adapt_data["adaptations"]
    assert adapt_data["adaptations"]["general_rule.md"]["block"]

    # 2) prompt_version 相对 base 模板 bump（合成 hash）；goal 为每书自适应值
    base_version = PromptTemplateRegistry().version_of("a", "general_rule.md")
    plan = load_artifact(ws.stage_artifact("plan"))
    gr = next(s for s in plan["sections"] if s["template"] == "general_rule.md")
    assert gr["prompt_version"] != base_version
    assert plan["goal"] == GOAL_RESPONSE["goal"]

    # 3) extract 实际收到「base + 适配块」合成提示词；prompt 共 3 次调用（2 适配 + 1 goal）
    assert len(prompt.calls) == 3
    assert any("本书适配提示" in c for c in extract.calls)


def test_plan_adaptation_cache_reuse_on_rerun(tmp_path: Path) -> None:
    """profile 未变时重跑 plan 命中适配/goal 缓存，不再调 prompt client（可复现）。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    extract, verify = FakeLLMClient("a"), FakeLLMClient("b")
    _queue_success(extract, verify, book_id)
    prompt = FakeLLMClient("c")
    prompt.queue = [ADAPTATION_RESPONSE, ADAPTATION_RESPONSE, GOAL_RESPONSE]
    books_root = tmp_path / "books"
    pipeline = _make_pipeline(books_root, extract, verify, prompt)
    asyncio.run(pipeline.run(source, domain_id="tcm"))
    assert len(prompt.calls) == 3

    # rewind 到 plan（适配/goal 产物保留），只重跑 plan：命中缓存，prompt 调用不增
    pipeline.rewind(book_id, to_stage="plan")
    asyncio.run(pipeline.run_stage(book_id, "plan", confirm_plan=False))
    assert len(prompt.calls) == 3


def test_plan_adaptation_degrades_on_llm_failure(tmp_path: Path) -> None:
    """prompt client 失败 → 适配块降级为 base-only，流水线照常完成。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    extract, verify = FakeLLMClient("a"), FakeLLMClient("b")
    _queue_success(extract, verify, book_id)
    prompt = FakeLLMClient("c")  # 空队列 → complete_json 抛 LLMError
    books_root = tmp_path / "books"
    pipeline = _make_pipeline(books_root, extract, verify, prompt)
    manifest = asyncio.run(pipeline.run(source, domain_id="tcm"))

    assert manifest.status is BookStatus.WAITING_REVIEW
    ws = BookWorkspace(books_root, book_id)
    adapt_data = load_artifact(ws.stage_dir("plan") / ADAPTATION_FILENAME)
    assert all(a["degraded"] for a in adapt_data["adaptations"].values())
    # 降级 → prompt_version 回落 base 版本
    base_version = PromptTemplateRegistry().version_of("a", "general_rule.md")
    plan = load_artifact(ws.stage_artifact("plan"))
    gr = next(s for s in plan["sections"] if s["template"] == "general_rule.md")
    assert gr["prompt_version"] == base_version
    assert not any("本书适配提示" in c for c in extract.calls)


def test_parse_applies_outline_repair(tmp_path: Path) -> None:
    """parse 阶段大纲修复：structure 客户端排队决策 → blocks 重建 + summary 记账。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    books_root = tmp_path / "books"
    BookWorkspace.create(books_root, source)

    structure = FakeLLMClient("structure")
    structure.queue = [
        {
            "entries": [
                {"level": 1, "keep": True, "title": "第一章 剂量规范"},
                {"level": 2, "keep": True, "title": "附录C 十八反"},
            ]
        }
    ]
    pipeline = _make_pipeline(books_root, structure=structure)
    asyncio.run(pipeline.run_stage(book_id, "parse"))
    assert len(structure.calls) == 1

    # parse summary 记录修复审计
    parse_data = load_artifact(books_root / book_id / "01_parse" / "parse_summary.json")
    assert parse_data["outline_repair"]["source"] == "llm"
    assert parse_data["outline_repair"]["kept"] == 2

    # blocks.jsonl 已重建：层级 1/2，meta 首行保留并盖审计标记
    from tcm_extract_engine.cab.profiler import load_blocks

    blocks_path = books_root / book_id / "01_parse" / "blocks.jsonl"
    meta = json.loads(blocks_path.read_text(encoding="utf-8").splitlines()[0])
    assert meta["type"] == "meta"
    assert meta["outline_repair"]["source"] == "llm"
    headings = {b["heading"]: b["level"] for b in load_blocks(blocks_path) if b["heading"]}
    assert headings["第一章 剂量规范"] == 1
    assert headings["附录C 十八反"] == 2


def test_parse_outline_repair_degrades_silently(tmp_path: Path) -> None:
    """structure 客户端失败 → 降级平铺标题，parse 照常完成。"""
    source = _make_source(tmp_path)
    book_id = _book_id_of(source)
    books_root = tmp_path / "books"
    BookWorkspace.create(books_root, source)

    pipeline = _make_pipeline(books_root)  # structure 空队列 → LLMError → 降级
    manifest = asyncio.run(pipeline.run_stage(book_id, "parse"))

    parse_data = load_artifact(books_root / book_id / "01_parse" / "parse_summary.json")
    assert parse_data["outline_repair"]["source"] == "degraded"
    assert manifest.status is BookStatus.PARSED
