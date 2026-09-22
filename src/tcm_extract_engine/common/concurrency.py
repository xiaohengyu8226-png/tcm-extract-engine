"""全局并发控制：LLM / Parse 两层信号量保护 GPU/显存/API。

- parse: 限制 MinerU 并发，保护 GPU/CPU/显存
- llm: 限制 LLM API 并发，防止限流

注：原 tcm-rag 的 master 信号量依赖 max_concurrent_jobs 字段，引擎 Settings 无此字段
且自动化线为单本编排、未使用 master 层，故此处裁剪掉，仅保留 llm / parse 两层。
"""

from __future__ import annotations

import asyncio

from tcm_extract_engine.common.config import get_settings

_llm_semaphore: asyncio.Semaphore | None = None
_parse_semaphore: asyncio.Semaphore | None = None


def get_llm_semaphore() -> asyncio.Semaphore:
    """全局 LLM 并发控制，防 API 限流。"""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(get_settings().llm_concurrency)
    return _llm_semaphore


def get_parse_semaphore() -> asyncio.Semaphore:
    """全局 MinerU 解析并发控制，保护 GPU/显存/CPU。"""
    global _parse_semaphore
    if _parse_semaphore is None:
        _parse_semaphore = asyncio.Semaphore(get_settings().mineru_max_concurrent_parse)
    return _parse_semaphore
