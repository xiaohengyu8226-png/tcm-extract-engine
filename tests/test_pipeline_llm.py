"""统一 LLM 客户端工具函数测试：JSON 容错解析 + 截断检测。"""

from __future__ import annotations

import json

import pytest

from tcm_extract_engine.core.llm import LLMError, extract_json_object, is_truncation_error

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
