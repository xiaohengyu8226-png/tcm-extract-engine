"""统一 LLM 客户端：抽取/校验/规划共用。

收敛 ingestion 里 4 处重复的 LLM 调用（extractor/prompt_generator/proofreader/verifier），
提供：JSON mode 调用、容错解析、指数退避重试、截断检测、Token 用量统计。
并发复用 common.concurrency 的全局信号量。

多模态扩展（v2）：
  - TextContent / ImageContent 类型化消息内容
  - complete_multimodal() 支持图文混合输入
  - 兼容 OpenAI Vision API 格式（base64 data URL）
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from openai import APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, Field

from tcm_extract_engine.common.config import ModelConfig, get_settings

logger = logging.getLogger(__name__)


# ── 多模态消息内容类型 ────────────────────────────────────────────────────────


class TextContent(BaseModel):
    """文本内容（OpenAI Vision API 格式）。"""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """图片内容（OpenAI Vision API 格式）。

    image_url 格式：
      - base64: {"url": "data:image/jpeg;base64,..."}
      - HTTP:   {"url": "https://..."}
      - 本地:   {"url": "file:///abs/path"}  # 需先转 base64
    """

    type: Literal["image_url"] = "image_url"
    image_url: dict[str, str] = Field(default_factory=dict)
    detail: Literal["auto", "low", "high"] = "auto"


# 多模态内容联合类型
ContentType = TextContent | ImageContent


def text_content(text: str) -> TextContent:
    """快捷创建文本内容。"""
    return TextContent(text=text)


def image_content(url: str, *, detail: str = "auto") -> ImageContent:
    """快捷创建图片内容。"""
    return ImageContent(image_url={"url": url}, detail=detail)  # type: ignore[arg-type]


@dataclass(slots=True)
class LLMUsage:
    """单次运行累计的 Token 用量。"""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    failures: int = 0
    retries: int = 0
    latency_total: float = 0.0

    def add(self, response: Any, latency: float) -> None:
        self.calls += 1
        self.latency_total += latency
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)


@dataclass(slots=True)
class LLMResult:
    """单次调用结果。"""

    content: str
    latency: float
    usage: LLMUsage = field(default_factory=LLMUsage)


class LLMError(Exception):
    """LLM 调用错误（含分类信息）。"""

    def __init__(self, kind: str, message: str, content: str | None = None) -> None:
        self.kind = kind  # timeout | parse | empty | api | truncation
        self.content = content  # 原始响应文本（parse 失败时附上，供截断检测）
        super().__init__(f"[{kind}] {message}")


def extract_json_object(text: str) -> dict[str, Any]:
    """从 LLM 响应提取 JSON 对象（去 markdown 代码块 + 最外层 {} 兜底）。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                raise LLMError("parse", f"LLM 输出非法 JSON: {exc}") from exc
        else:
            raise LLMError("parse", f"LLM 输出中未找到 JSON 对象: {text[:200]}")
    if not isinstance(value, dict):
        raise LLMError("parse", "LLM 输出必须为 JSON 对象")
    return value


def is_truncation_error(exc: Exception) -> bool:
    """输出截断检测：JSON 解析错误位置接近输出末尾（≥80%）。

    依赖 LLMError.content（complete_json 在 parse 失败时附上原始响应）。
    """
    if not isinstance(exc, LLMError) or exc.kind != "parse" or not exc.content:
        return False
    match = re.search(r"char (\d+)", str(exc))
    if not match:
        return False
    error_pos = int(match.group(1))
    return error_pos >= len(exc.content) * 0.8


def _is_local_endpoint(base_url: str) -> bool:
    """本地推理服务（llama.cpp / vLLM）判定。

    关闭思考的参数因后端而异：本地 llama.cpp 只认 Jinja 模板变量
    ``chat_template_kwargs={"enable_thinking": false}``，云端 deepseek 网关认
    ``thinking={"type": "disabled"}``。按 host 区分，避免给云端调用注入本地专属字段。
    """
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"localhost", "0.0.0.0", "::1"} or host.startswith("127.")


class UnifiedLLMClient:
    """三模型统一客户端；按 ModelConfig 调用，自带重试与用量统计。"""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or get_settings().extract_llm
        self._client: AsyncOpenAI | None = None
        self.usage = LLMUsage()

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def cache_identity(self) -> str:
        identity = {key: getattr(self._config, key) for key in ("base_url", "model", "max_tokens", "thinking_enabled")}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    async def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            # max_retries=0：SDK 默认对超时静默重试 2 次，会与本类的指数退避重试
            # 叠乘（单轮实际耗时 3×timeout），超时批次在服务端反复重排队尾空转；
            # 重试策略只保留本类 complete_text 的一层。
            self._client = AsyncOpenAI(
                base_url=self._config.base_url,
                api_key=self._config.api_key,
                max_retries=0,
                http_client=httpx.AsyncClient(
                    trust_env=not _is_local_endpoint(self._config.base_url)
                ),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        semaphore: asyncio.Semaphore | None = None,
        max_retries: int = 3,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """调用 LLM 并解析 JSON（容错 + 指数退避）。

        Returns:
            解析后的 JSON 对象。
        Raises:
            LLMError: 重试耗尽或响应非法。
        """
        content = await self.complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            semaphore=semaphore,
            max_retries=max_retries,
            temperature=temperature,
        )
        last_exc: LLMError | None = None
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                # 截断重采样：思考型模型的思考链长度随采样波动，重跑常能落在输出预算内
                content = await self.complete_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    semaphore=semaphore,
                    max_retries=max_retries,
                    temperature=temperature,
                )
            try:
                return extract_json_object(content)
            except LLMError as exc:
                exc.content = content  # 附上原始响应供截断检测
                last_exc = exc
                if not is_truncation_error(exc):
                    raise
                if attempt < max_retries:
                    self.usage.retries += 1
                    logger.warning(
                        "[LLM] 输出截断 (%d/%d)，重采样",
                        attempt,
                        max_retries,
                    )
        assert last_exc is not None
        raise last_exc

    async def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        semaphore: asyncio.Semaphore | None = None,
        max_retries: int = 3,
        temperature: float = 0.1,
    ) -> str:
        """调用 LLM 返回文本（含指数退避重试）。"""
        from tcm_extract_engine.common.concurrency import get_llm_semaphore

        sem = semaphore or get_llm_semaphore()
        client = await self._ensure_client()
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            started = time.perf_counter()
            try:
                extra_body: dict[str, Any] = {}
                if not self._config.thinking_enabled:
                    # 关思考参数按后端区分（实测）：本地 llama.cpp（jiuwei-tcm 校验服务）
                    # 只认 chat_template_kwargs.enable_thinking=false，发 thinking={...}
                    # 会被忽略（reasoning_content 仍非空、思考链照吃输出预算）；云端沿用 thinking。
                    if _is_local_endpoint(self._config.base_url):
                        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                    else:
                        extra_body["thinking"] = {"type": "disabled"}
                async with sem:
                    response = await client.chat.completions.create(
                        model=self._config.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=self._config.max_tokens,
                        temperature=temperature,
                        timeout=self._config.timeout,
                        extra_body=extra_body or None,
                    )
                latency = time.perf_counter() - started
                self.usage.add(response, latency)

                content = response.choices[0].message.content
                if not content:
                    # 思考型模型的推理链可能吃满输出预算（服务端 n_ctx 扣除 prompt
                    # 后的余量更紧），content 为空；思考链长度随采样波动，重采样
                    # 常能落在预算内，故按可重试瞬态处理而非立即失败。
                    last_error = LLMError("empty", "LLM 返回空响应（推理链可能吃满输出预算）")
                else:
                    return content
            except LLMError:
                self.usage.failures += 1
                raise
            except (TimeoutError, httpx.TimeoutException, APITimeoutError) as exc:
                last_error = LLMError("timeout", str(exc))
            except Exception as exc:  # noqa: BLE001 - API 层错误统一分类重试
                last_error = LLMError("api", str(exc))

            self.usage.failures += 1
            if attempt < max_retries:
                self.usage.retries += 1
                wait = 2**attempt
                logger.warning(
                    "[LLM] 调用失败 (%d/%d): %s, %ds 后重试",
                    attempt,
                    max_retries,
                    last_error,
                    wait,
                )
                await asyncio.sleep(wait)

        # 重试耗尽：保留原始错误类别（empty/timeout）。一律改写成 "api" 会让 verifier 的
        # 「empty→拆半重试」分支永远命中不到（exc.kind == "empty" 恒 False），最重批次只能硬失败。
        if isinstance(last_error, LLMError):
            raise LLMError(
                last_error.kind,
                f"LLM 调用失败（重试 {max_retries} 次）: {last_error}",
                content=last_error.content,
            )
        raise LLMError("api", f"LLM 调用失败（重试 {max_retries} 次）: {last_error}")

    # ── 多模态调用（图文混合）────────────────────────────────────────────────

    async def complete_multimodal(
        self,
        *,
        system_prompt: str,
        user_contents: list[ContentType],
        semaphore: asyncio.Semaphore | None = None,
        max_retries: int = 3,
        temperature: float = 0.1,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """多模态 LLM 调用（支持图文混合输入）。

        Args:
            system_prompt: 系统提示词
            user_contents: 用户内容列表（TextContent / ImageContent 混合）
            semaphore: 并发信号量
            max_retries: 最大重试次数
            temperature: 采样温度
            response_format: 响应格式（默认 json_object）

        Returns:
            LLM 响应文本

        Raises:
            LLMError: 重试耗尽或响应非法
        """
        from tcm_extract_engine.common.concurrency import get_vision_semaphore

        vision_sem = get_vision_semaphore()
        # Caller limits (e.g. verification concurrency) may only tighten the
        # shared image-service limit, never replace it with cloud concurrency.
        caller_gate = semaphore if semaphore is not vision_sem else None
        client = await self._ensure_client()
        last_error: Exception | None = None

        # 构建多模态消息内容
        content_parts = [c.model_dump() for c in user_contents]

        for attempt in range(1, max_retries + 1):
            started = time.perf_counter()
            try:
                extra_body: dict[str, Any] = {}
                if not self._config.thinking_enabled:
                    if _is_local_endpoint(self._config.base_url):
                        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                    else:
                        extra_body["thinking"] = {"type": "disabled"}

                async with vision_sem, caller_gate or nullcontext():
                    # 多模态消息需要显式类型转换（OpenAI SDK 类型检查）
                    messages: list[dict[str, Any]] = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content_parts},
                    ]
                    response = await client.chat.completions.create(
                        model=self._config.model,
                        messages=messages,  # type: ignore[arg-type]
                        response_format=response_format or {"type": "json_object"},  # type: ignore[arg-type]
                        max_tokens=self._config.max_tokens,
                        temperature=temperature,
                        timeout=self._config.timeout,
                        extra_body=extra_body or None,
                    )
                latency = time.perf_counter() - started
                self.usage.add(response, latency)

                content = response.choices[0].message.content
                if not content:
                    last_error = LLMError("empty", "LLM 返回空响应（推理链可能吃满输出预算）")
                else:
                    return content
            except LLMError:
                self.usage.failures += 1
                raise
            except (TimeoutError, httpx.TimeoutException, APITimeoutError) as exc:
                last_error = LLMError("timeout", str(exc))
            except Exception as exc:  # noqa: BLE001
                last_error = LLMError("api", str(exc))

            self.usage.failures += 1
            if attempt < max_retries:
                self.usage.retries += 1
                wait = 2**attempt
                logger.warning(
                    "[LLM-Multimodal] 调用失败 (%d/%d): %s, %ds 后重试",
                    attempt,
                    max_retries,
                    last_error,
                    wait,
                )
                await asyncio.sleep(wait)

        if isinstance(last_error, LLMError):
            raise LLMError(
                last_error.kind,
                f"多模态 LLM 调用失败（重试 {max_retries} 次）: {last_error}",
                content=last_error.content,
            )
        raise LLMError("api", f"多模态 LLM 调用失败（重试 {max_retries} 次）: {last_error}")

    async def complete_multimodal_json(
        self,
        *,
        system_prompt: str,
        user_contents: list[ContentType],
        semaphore: asyncio.Semaphore | None = None,
        max_retries: int = 3,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """多模态 LLM 调用并解析 JSON（容错 + 指数退避）。

        Returns:
            解析后的 JSON 对象
        Raises:
            LLMError: 重试耗尽或响应非法
        """
        content = await self.complete_multimodal(
            system_prompt=system_prompt,
            user_contents=user_contents,
            semaphore=semaphore,
            max_retries=max_retries,
            temperature=temperature,
        )
        last_exc: LLMError | None = None
        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                content = await self.complete_multimodal(
                    system_prompt=system_prompt,
                    user_contents=user_contents,
                    semaphore=semaphore,
                    max_retries=max_retries,
                    temperature=temperature,
                )
            try:
                return extract_json_object(content)
            except LLMError as exc:
                exc.content = content
                last_exc = exc
                if not is_truncation_error(exc):
                    raise
                if attempt < max_retries:
                    self.usage.retries += 1
                    logger.warning(
                        "[LLM-Multimodal] 输出截断 (%d/%d)，重采样",
                        attempt,
                        max_retries,
                    )
        assert last_exc is not None
        raise last_exc
