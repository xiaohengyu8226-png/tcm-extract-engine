"""MinerU backend 实际值落盘与缓存身份测试。

背景：``MINERU_FALLBACK_TO_PIPELINE`` 打开时，hybrid 失败会静默降级成 pipeline，
而旧版只把**请求**的 backend 写进 manifest 与缓存变体名，于是 hybrid 目录里躺着
pipeline 产物、并声称自己是 hybrid，后续真 hybrid 又会被"缓存命中"跳过。
E0 实测两个 backend 的产物并不等价（45 vs 48 个标题，``【辨证禁忌】`` 有无），
所以这个误标有实际后果。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tcm_extract_engine.parse import mineru
from tcm_extract_engine.parse.mineru import (
    _backend_cache_usable,
    _load_manifest,
    _manifest_backend_used,
    ensure_mineru_artifact,
)

CONTENT_LIST: list[dict[str, Any]] = [
    {"type": "text", "text": "一、感 冒", "text_level": 2, "page_idx": 0,
     "bbox": [0, 0, 100, 20]},
    {"type": "text", "text": "【辨证禁忌】", "text_level": 2, "page_idx": 1,
     "bbox": [0, 0, 100, 20]},
    {"type": "text", "text": "忌将时行感冒误为普通感冒。", "page_idx": 1,
     "bbox": [0, 0, 100, 40]},
]


@pytest.fixture()
def fake_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake\n")
    return pdf


@pytest.fixture()
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """把 ``_parse_pdf_local`` 换成替身，记录每次实际尝试的 backend。"""
    recorded: list[str] = []

    async def _fake(
        source_pdf: Path, *, output_dir: Path, backend: str, lang: str | None = None,
        formula: bool = True, table: bool = True, image_analysis: bool = True,
        hybrid_effort: str = "medium", vlm_gpu_memory_utilization: float | None = None,
    ) -> list[dict[str, Any]]:
        recorded.append(backend)
        if backend == "hybrid-engine":
            raise RuntimeError("CUDA out of memory")
        return [dict(item) for item in CONTENT_LIST]

    monkeypatch.setattr(mineru, "_parse_pdf_local", _fake)
    return recorded


def _configure(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    monkeypatch.setenv("MINERU_BACKEND", backend)
    monkeypatch.setenv("MINERU_FALLBACK_TO_PIPELINE", "1")
    monkeypatch.setenv("MINERU_PARSE_RETRIES", "1")
    monkeypatch.setenv("MINERU_PARSE_RETRY_DELAY", "0")
    monkeypatch.delenv("MINERU_VLM_GPU_MEMORY_UTILIZATION", raising=False)


# ── 纯函数：manifest 读取与缓存可用性判定 ────────────────────────────────────


def test_manifest_backend_used_prefers_explicit_field() -> None:
    manifest = {
        "mineru_parse_options": {"backend": "hybrid-engine"},
        "mineru_backend_used": "pipeline",
    }
    assert _manifest_backend_used(manifest) == "pipeline"


def test_manifest_backend_used_falls_back_to_requested_for_legacy() -> None:
    """历史 manifest 没有实际字段时按"请求即实际"处理，避免一次性作废全部缓存。"""
    manifest = {"mineru_parse_options": {"backend": "pipeline"}}
    assert _manifest_backend_used(manifest) == "pipeline"


def test_manifest_backend_used_empty_when_unreadable() -> None:
    assert _manifest_backend_used({}) == ""
    assert _manifest_backend_used({"mineru_parse_options": "broken"}) == ""


def test_backend_cache_usable_rejects_degraded_artifact() -> None:
    manifest = {"mineru_backend_used": "pipeline"}
    assert _backend_cache_usable(manifest, requested_backend="hybrid-engine") is False
    assert _backend_cache_usable(manifest, requested_backend="pipeline") is True


def test_backend_cache_usable_allows_unknown_legacy() -> None:
    assert _backend_cache_usable({}, requested_backend="hybrid-engine") is True


# ── 端到端：降级记账 + 缓存身份 ──────────────────────────────────────────────


async def test_degraded_parse_records_actual_backend(
    fake_pdf: Path, calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """hybrid 失败降级 pipeline 后，manifest 必须记 pipeline，而非请求的 hybrid。"""
    _configure(monkeypatch, "hybrid-engine")

    result = await ensure_mineru_artifact(fake_pdf)

    assert calls == ["hybrid-engine", "pipeline"]
    assert result.mineru_backend_used == "pipeline"
    assert result.mineru_degraded is True

    manifest = _load_manifest(result.raw_dir)
    assert manifest is not None
    assert manifest["mineru_backend_used"] == "pipeline"
    assert manifest["mineru_degraded"] is True
    # 请求值仍留在 parse_options 里，两者可对照审计
    assert manifest["mineru_parse_options"]["backend"] == "hybrid-engine"


async def test_degraded_artifact_is_not_reused_as_requested_backend(
    fake_pdf: Path, calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """降级产物下次必须判为未命中重新解析，而不是被当成 hybrid 缓存永久锁定。"""
    _configure(monkeypatch, "hybrid-engine")
    first = await ensure_mineru_artifact(fake_pdf)
    assert first.cache_hit is False
    assert len(calls) == 2

    second = await ensure_mineru_artifact(fake_pdf)

    assert second.cache_hit is False, "降级产物不得以 hybrid 身份命中缓存"
    assert len(calls) == 4, "应重新尝试 hybrid，而非复用 pipeline 产物"


async def test_matching_backend_hits_cache(
    fake_pdf: Path, calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """实际 backend 与请求一致时缓存正常命中，不产生多余解析。"""
    _configure(monkeypatch, "pipeline")
    monkeypatch.setenv("MINERU_FALLBACK_TO_PIPELINE", "0")

    first = await ensure_mineru_artifact(fake_pdf)
    assert first.mineru_backend_used == "pipeline"
    assert first.mineru_degraded is False

    second = await ensure_mineru_artifact(fake_pdf)

    assert second.cache_hit is True
    assert second.mineru_backend_used == "pipeline"
    assert calls == ["pipeline"], "命中缓存后不应再次解析"


async def test_legacy_manifest_still_counts_as_cache_hit(
    fake_pdf: Path, calls: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺实际字段的旧 manifest 在同 backend 请求下仍应命中（向后兼容）。"""
    _configure(monkeypatch, "pipeline")
    monkeypatch.setenv("MINERU_FALLBACK_TO_PIPELINE", "0")
    first = await ensure_mineru_artifact(fake_pdf)

    manifest_path = first.raw_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("mineru_backend_used")
    manifest.pop("mineru_degraded")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    second = await ensure_mineru_artifact(fake_pdf)

    assert second.cache_hit is True
    assert second.mineru_backend_used == "pipeline"
    assert calls == ["pipeline"]
