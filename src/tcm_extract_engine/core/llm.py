"""统一 LLM 客户端：抽取/校验/规划共用。

收敛 ingestion 里 4 处重复的 LLM 调用（extractor/prompt_generator/proofreader/verifier），
提供：JSON mode 调用、容错解析、指数退避重试、截断检测、Token 用量统计。
并发复用 common.concurrency 的全局信号量。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from tcm_extract_engine.common.config import ModelConfig, get_settings

logger = logging.getLogger(__name__)


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


class UnifiedLLMClient:
    """三模型统一客户端；按 ModelConfig 调用，自带重试与用量统计。"""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or get_settings().extract_llm
        self._client: AsyncOpenAI | None = None
        self.usage = LLMUsage()

    @property
    def model(self) -> str:
        return self._config.model

    async def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self._config.base_url, api_key=self._config.api_key
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
        try:
            return extract_json_object(content)
        except LLMError as exc:
            exc.content = content  # 附上原始响应供截断检测
            raise

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
                    raise LLMError("empty", "LLM 返回空响应")
                return content
            except LLMError:
                self.usage.failures += 1
                raise
            except TimeoutError as exc:
                last_error = LLMError("timeout", str(exc))
            except Exception as exc:  # noqa: BLE001 - API 层错误统一分类重试
                last_error = LLMError("api", str(exc))

            self.usage.failures += 1
            if attempt < max_retries:
                self.usage.retries += 1
                wait = 2**attempt
                logger.warning(
                    "[LLM] 调用失败 (%d/%d): %s, %ds 后重试",
                    attempt, max_retries, last_error, wait,
                )
                await asyncio.sleep(wait)

        raise LLMError("api", f"LLM 调用失败（重试 {max_retries} 次）: {last_error}")
