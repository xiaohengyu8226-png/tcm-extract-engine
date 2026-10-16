"""全局并发控制：LLM / Verify / Parse 三层信号量保护 GPU/显存/API。

- parse: 限制 MinerU 并发，保护 GPU/CPU/显存
- llm: 限制 LLM API 并发，防止限流
- verify: 限制校验批次并发，贴着本地校验服务的真实槽位数（超发只会服务端
  排队，排队时间计入客户端超时，造成「超时→重排队尾→再超时」空转）

注：原 tcm-rag 的 master 信号量依赖 max_concurrent_jobs 字段，引擎 Settings 无此字段
且自动化线为单本编排、未使用 master 层，故此处裁剪掉，仅保留 llm / verify / parse 三层。
"""

from __future__ import annotations

import asyncio
from weakref import WeakKeyDictionary

from tcm_extract_engine.common.config import get_settings

_llm_semaphore: asyncio.Semaphore | None = None
_verify_semaphore: asyncio.Semaphore | None = None
_parse_semaphore: asyncio.Semaphore | None = None
_vision_semaphores: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = WeakKeyDictionary()


def get_vision_semaphore() -> asyncio.Semaphore:
    """共享视觉请求上限；不同 CLI event loop 使用独立的信号量。"""
    loop = asyncio.get_running_loop()
    semaphore = _vision_semaphores.get(loop)
    if semaphore is None:
        capacity = get_settings().vision_concurrency
        if capacity < 1:
            raise ValueError("VISION_CONCURRENCY 必须大于 0")
        semaphore = asyncio.Semaphore(capacity)
        _vision_semaphores[loop] = semaphore
    return semaphore


def get_llm_semaphore() -> asyncio.Semaphore:
    """全局 LLM 并发控制，防 API 限流。"""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(get_settings().llm_concurrency)
    return _llm_semaphore


def get_verify_semaphore() -> asyncio.Semaphore:
    """校验批次并发控制，贴校验服务端槽位数（VERIFY_CONCURRENCY）。"""
    global _verify_semaphore
    if _verify_semaphore is None:
        _verify_semaphore = asyncio.Semaphore(get_settings().verify_concurrency)
    return _verify_semaphore


def get_parse_semaphore() -> asyncio.Semaphore:
    """全局 MinerU 解析并发控制，保护 GPU/显存/CPU。"""
    global _parse_semaphore
    if _parse_semaphore is None:
        _parse_semaphore = asyncio.Semaphore(get_settings().mineru_max_concurrent_parse)
    return _parse_semaphore
