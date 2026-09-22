"""模型网关与分块的最小类型集合（自 tcm-rag knowledge.workflow 裁剪而来）。

GEPA 引擎与评估桥接只依赖这里的协议/数据类，不依赖任何具体供应商实现，
便于更换模型、记录用量并在测试中完全隔离网络。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """一次模型调用的 Token 用量。缓存命中不计入当前运行用量。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    thinking_tokens: int = 0

    def __post_init__(self) -> None:
        values = (
            self.input_tokens,
            self.output_tokens,
            self.cached_input_tokens,
            self.thinking_tokens,
        )
        if any(value < 0 for value in values):
            raise ValueError("Token 用量不能为负数")

    def __add__(self, other: ModelUsage) -> ModelUsage:
        return ModelUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
        )


@dataclass(frozen=True, slots=True)
class StructuredModelResponse:
    """模型网关返回值；payload 可为已解析对象或原始 JSON 文本。"""

    payload: Mapping[str, Any] | str
    usage: ModelUsage = field(default_factory=ModelUsage)


@runtime_checkable
class AsyncModelGateway(Protocol):
    """外部结构化模型的最小异步接口。"""

    async def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> StructuredModelResponse:
        """请求模型生成结构化 JSON。"""
        ...


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """知识抽取所需的最小分块及可信旁路元数据。"""

    chunk_id: str
    content: str
    source_document_id: str
    document_version: str = ""
    source_pdf: str = ""
    chapter: str = ""
    page: str | int = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
