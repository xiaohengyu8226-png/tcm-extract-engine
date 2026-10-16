"""统一 LLM 客户端工具函数测试：JSON 容错解析 + 截断检测 + 错误类别/思考关闭参数。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import APITimeoutError

from tcm_extract_engine.common.config import ModelConfig
from tcm_extract_engine.core.llm import (
    LLMError,
    UnifiedLLMClient,
    _is_local_endpoint,
    extract_json_object,
    is_truncation_error,
)

# ── extract_json_object ──────────────────────────────────────────────────────


def test_extract_plain_json() -> None:
    assert extract_json_object('{"records": []}') == {"records": []}


def test_extract_markdown_fenced_json() -> None:
    text = "```json\n{\"records\": [1]}\n```"
    assert extract_json_object(text) == {"records": [1]}


def test_extract_json_with_surrounding_text() -> None:
    text = "好的，以下是抽取结果：\n{\"records\": [1, 2]}\n以上。"
    assert extract_json_object(text) == {"records": [1, 2]}


def test_extract_invalid_json_raises() -> None:
    with pytest.raises(LLMError, match="parse"):
        extract_json_object("这不是 JSON")


def test_extract_non_object_raises() -> None:
    with pytest.raises(LLMError, match="JSON 对象"):
        extract_json_object("[1, 2, 3]")


def test_extract_truncated_json_raises_with_position() -> None:
    # 截断示例：不完整的 JSON（records 数组未闭合）
    truncated = json.dumps({"records": [{"a": 1}, {"b": 2}]}, ensure_ascii=False)
    truncated = truncated[: int(len(truncated) * 0.9)]
    with pytest.raises(LLMError) as exc_info:
        extract_json_object(truncated)
    # 错误消息应包含 char 位置（供截断检测使用）
    assert "char" in str(exc_info.value)


# ── is_truncation_error ─────────────────────────────────────────────────────


def _truncation_error(content: str, char_pos: int) -> LLMError:
    exc = LLMError("parse", f"LLM 输出非法 JSON: Expecting ',' (char {char_pos})")
    exc.content = content
    return exc


def test_truncation_detected_near_end() -> None:
    content = "x" * 100
    exc = _truncation_error(content, char_pos=95)
    assert is_truncation_error(exc)


def test_truncation_not_detected_early_position() -> None:
    content = "x" * 100
    exc = _truncation_error(content, char_pos=10)
    assert not is_truncation_error(exc)


def test_truncation_requires_content() -> None:
    exc = LLMError("parse", "LLM 输出非法 JSON: Expecting ',' (char 95)")
    assert exc.content is None
    assert not is_truncation_error(exc)


def test_truncation_ignores_other_kinds() -> None:
    exc = LLMError("api", "connection error (char 95)")
    assert not is_truncation_error(exc)


def test_truncation_ignores_non_llm_error() -> None:
    assert not is_truncation_error(ValueError("char 95"))


# ── UnifiedLLMClient：错误类别保真 + 思考关闭参数 ────────────────────
#
# 回归护栏：旧 bug 是 complete_text 重试耗尽后把 "empty" 一律改写成 "api"，
# 导致 verifier 的「empty→拆半重试」分支永远命中不到（exc.kind=="empty" 恒 False）。
# 以下测试走真实 UnifiedLLMClient 路径（仅 mock 最底层 AsyncOpenAI），避免
# “测试替身直接抛 empty”与生产契约不一致而掩盖回归。


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeUsage:
    prompt_tokens = 1
    completion_tokens = 1


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeCompletions:
    """记录每次 create() 的 kwargs，并按 content 队列出响应。"""

    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.captured: list[dict] = []

    async def create(self, **kwargs: object) -> _FakeResponse:
        self.captured.append(kwargs)
        content = self._contents.pop(0) if self._contents else ""
        return _FakeResponse(content)


class _FakeChat:
    def __init__(self, contents: list[str]) -> None:
        self.completions = _FakeCompletions(contents)


class _FakeAsyncOpenAI:
    def __init__(self, contents: list[str]) -> None:
        self.chat = _FakeChat(contents)

    async def close(self) -> None:
        pass


def _make_client(base_url: str, *, thinking_enabled: bool) -> UnifiedLLMClient:
    cfg = ModelConfig(
        base_url=base_url, api_key="EMPTY", model="m",
        timeout=5, max_tokens=64, thinking_enabled=thinking_enabled,
    )
    return UnifiedLLMClient(config=cfg)


async def _instant_sleep(_seconds: float) -> None:
    return None


def test_is_local_endpoint_detection() -> None:
    assert _is_local_endpoint("http://127.0.0.1:59001/v1")
    assert _is_local_endpoint("http://localhost:8000/v1")
    assert not _is_local_endpoint("https://tokenhub.tencentmaas.com/v1")


def test_complete_text_preserves_empty_kind_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空响应重试耗尽后必须仍以 kind=='empty' 抛出（否则 verifier 拆半兜底变死代码）。"""
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    client = _make_client("http://127.0.0.1:59001/v1", thinking_enabled=False)
    fake: Any = _FakeAsyncOpenAI(["", "", ""])  # 每次都空
    client._client = fake
    with pytest.raises(LLMError) as exc_info:
        asyncio.run(client.complete_text(
            system_prompt="s", user_prompt="u", max_retries=3,
        ))
    assert exc_info.value.kind == "empty"


def test_complete_text_local_disable_sends_chat_template_kwargs() -> None:
    """本地端点关思考：走 chat_template_kwargs.enable_thinking=false，不发 thinking。"""
    client = _make_client("http://127.0.0.1:59001/v1", thinking_enabled=False)
    fake: Any = _FakeAsyncOpenAI(['{"ok": 1}'])
    client._client = fake
    asyncio.run(client.complete_text(system_prompt="s", user_prompt="u", max_retries=1))
    extra_body = fake.chat.completions.captured[0]["extra_body"]
    assert extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_complete_text_cloud_disable_sends_thinking_param() -> None:
    """云端端点关思考：沿用 thinking={'type':'disabled'}，不注入本地专属字段。"""
    client = _make_client("https://tokenhub.tencentmaas.com/v1", thinking_enabled=False)
    fake: Any = _FakeAsyncOpenAI(['{"ok": 1}'])
    client._client = fake
    asyncio.run(client.complete_text(system_prompt="s", user_prompt="u", max_retries=1))
    extra_body = fake.chat.completions.captured[0]["extra_body"]
    assert extra_body == {"thinking": {"type": "disabled"}}


def test_complete_text_thinking_enabled_sends_no_extra_body() -> None:
    """思考开启时不附任何关思考参数（extra_body=None）。"""
    client = _make_client("http://127.0.0.1:59001/v1", thinking_enabled=True)
    fake: Any = _FakeAsyncOpenAI(['{"ok": 1}'])
    client._client = fake
    asyncio.run(client.complete_text(system_prompt="s", user_prompt="u", max_retries=1))
    assert fake.chat.completions.captured[0]["extra_body"] is None


@pytest.mark.parametrize("explicit_gate", [False, True])
def test_multimodal_requests_share_a_separate_capacity_limit(
    monkeypatch: pytest.MonkeyPatch, explicit_gate: bool,
) -> None:
    """Cloud text concurrency must not flood the local image service."""
    from types import SimpleNamespace

    from tcm_extract_engine.common import concurrency
    from tcm_extract_engine.core.llm import TextContent

    monkeypatch.setattr(
        concurrency, "get_settings",
        lambda: SimpleNamespace(llm_concurrency=200, vision_concurrency=2),
    )

    async def run() -> int:
        monkeypatch.setattr(concurrency, "_llm_semaphore", asyncio.Semaphore(200))
        active = peak = 0

        async def create(**_kwargs: Any) -> _FakeResponse:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return _FakeResponse('{"records": []}')

        clients = [_make_client("http://localhost:59001/v1", thinking_enabled=False)
                   for _ in range(3)]
        for client in clients:
            fake: Any = _FakeAsyncOpenAI([])
            fake.chat.completions.create = create
            client._client = fake
        gate = asyncio.Semaphore(200) if explicit_gate else None
        await asyncio.gather(*(
            clients[i % 3].complete_multimodal_json(
                system_prompt="extract", user_contents=[TextContent(text="image context")],
                semaphore=gate, max_retries=1,
            ) for i in range(12)
        ))
        return peak

    # A contended semaphore must also be safe across separate CLI event loops.
    assert asyncio.run(run()) == 2
    assert asyncio.run(run()) == 2


@pytest.mark.parametrize("multimodal", [False, True])
def test_sdk_timeout_keeps_timeout_error_kind(multimodal: bool) -> None:
    from tcm_extract_engine.core.llm import TextContent

    async def create(**_kwargs: Any) -> None:
        raise APITimeoutError(request=httpx.Request("POST", "http://localhost/v1/chat/completions"))

    client = _make_client("http://localhost:59001/v1", thinking_enabled=False)
    fake: Any = _FakeAsyncOpenAI([])
    fake.chat.completions.create = create
    client._client = fake
    call = (
        client.complete_multimodal_json(system_prompt="s", user_contents=[TextContent(text="u")], max_retries=1)
        if multimodal else client.complete_text(system_prompt="s", user_prompt="u", max_retries=1)
    )
    with pytest.raises(LLMError) as error:
        asyncio.run(call)
    assert error.value.kind == "timeout"


def test_vision_capacity_is_configurable_and_rejects_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tcm_extract_engine.common import concurrency, config
    from tcm_extract_engine.common.config import Settings

    monkeypatch.setattr(config, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.delenv("VISION_CONCURRENCY", raising=False)
    monkeypatch.setenv("VLLM_MAX_CONCURRENCY", "5")
    assert Settings.from_env().vision_concurrency == 5
    monkeypatch.setenv("VISION_CONCURRENCY", "3")
    assert Settings.from_env().vision_concurrency == 3
    monkeypatch.setattr(concurrency, "get_settings", lambda: SimpleNamespace(vision_concurrency=0))

    async def run() -> None:
        with pytest.raises(ValueError, match="VISION_CONCURRENCY"):
            concurrency.get_vision_semaphore()

    asyncio.run(run())


@pytest.mark.parametrize("guessed_path", [False, True])
def test_explicit_legacy_cache_migration_preserves_outputs_without_model_calls(
    tmp_path: Path, guessed_path: bool,
) -> None:
    from tcm_extract_engine.cab.extractors import (
        _load_batch_cache,
        _request_identity,
        _safe_key,
        extract_records,
        migrate_legacy_batch_cache,
        split_section_batches,
    )
    from tcm_extract_engine.cab.planner import PromptTemplateRegistry, SectionPlan
    from tcm_extract_engine.core.artifacts import ArtifactHeader, save_artifact
    from tcm_extract_engine.domains import get_domain_pack

    book_id = "legacy-book"
    image = tmp_path / "original.jpg"
    image.write_bytes(b"image-identity-only-cache-must-avoid-decoding")
    block = {"type": "content", "blockid": "image1", "format": "drawing",
             "content": "<drawing>舌色淡红</drawing>", "asset_path": str(image),
             "positions": [{"anchor": "1"}]}
    blocks_path = tmp_path / "blocks.jsonl"
    blocks_path.write_text(json.dumps(block, ensure_ascii=False) + "\n")
    registry = PromptTemplateRegistry()
    pack = get_domain_pack("tcm")
    version = registry.version_of("a", pack.figure_template)
    section = SectionPlan(section_id="sec-1", title="舌象", extractor="FigureExtractor",
                          template=pack.figure_template, prompt_version=version, block_ids=["image1"])
    batch = split_section_batches(section, {"image1": block}, pack=pack)[0]
    plan_path = tmp_path / "plan.json"
    save_artifact(plan_path, header=ArtifactHeader(schema_version="1", book_id=book_id, task_id="test"),
                  data={"sections": [section.model_dump(mode="json")]})
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / f"batch_{_safe_key(batch.chunk_key)}.json"
    cached_path = "/model-guessed/image.jpg" if guessed_path else str(image)
    original = json.dumps({"prompt_version": version, "records": [{
        "record_id": "SR-old-000000", "book_id": book_id, "record_type": "FIGURE",
        "raw_text": "舌色淡红", "image_ref": {"asset_path": cached_path, "figure_type": "舌象"},
    }]}).encode()
    path.write_bytes(original)
    client = _make_client("http://localhost:59001/v1", thinking_enabled=False)
    identity = _request_identity(batch, client)
    assert _load_batch_cache(cache, batch.chunk_key, prompt_version=version, request_identity=identity) is None
    assert migrate_legacy_batch_cache(cache, batch, book_id=book_id, registry=registry, client=client, pack=pack)
    assert path.with_name(path.name + ".legacy-v1.bak").read_bytes() == original
    records = asyncio.run(extract_records(book_id=book_id, blocks_path=blocks_path, plan_path=plan_path,
        cache_dir=cache, client=client, vision_client=client, registry=registry, pack=pack))
    assert len(records) == 1
    assert records[0].needs_review
    assert records[0].generation.model == "legacy:unattributed"
    assert records[0].generation.modality == "legacy_unknown"
    assert records[0].payload["image_ref"]["asset_path"] == cached_path
    if guessed_path:
        assert any("图片引用" in text for text in records[0].ambiguities)
    assert records[0].source.block_ids == ["image1"]
    assert client.usage.calls == 0
    assert _load_batch_cache(cache, batch.chunk_key, prompt_version=version, request_identity="different-config") is None
    assert not migrate_legacy_batch_cache(cache, batch, book_id=book_id, registry=registry, client=client, pack=pack)
