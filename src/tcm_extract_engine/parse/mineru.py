"""MinerU PDF 解析：进程内直调、SHA256 缓存、blocks 构建。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MINERU_MODEL_SOURCE = "modelscope"
DEFAULT_MINERU_BACKEND = "pipeline"
DEFAULT_MINERU_FORMULA_ENABLE = True
DEFAULT_MINERU_TABLE_ENABLE = True
DEFAULT_MINERU_IMAGE_ANALYSIS = True
DEFAULT_MINERU_HYBRID_EFFORT = "medium"
# None 表示不干预，让 MinerU 自己取默认值（它按显存大小给 0.5 或 0.7）
DEFAULT_MINERU_VLM_GPU_MEMORY_UTILIZATION: float | None = None
BLOCKS_BUILDER_VERSION = "2026-09-08.preserve-layout-content-v2"


# ── 公开数据模型 ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ParseResult:
    """MinerU 解析结果。"""

    parsed_dir: Path
    cache_hit: bool
    block_count: int
    source_pdf: Path
    raw_dir: Path
    blocks_path: Path
    #: MinerU 原始 content_list 落盘路径；表格直通通路需要它才能拿到表体 HTML 与页坐标
    content_list_path: Path | None = None

    def load_content_list(self) -> list[dict[str, Any]]:
        """读取本产物对应的 content_list；Markdown 直转没有该文件时返回空。"""
        if self.content_list_path is None or not self.content_list_path.is_file():
            return []
        payload = json.loads(self.content_list_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []


# ── 内部工具函数 ─────────────────────────────────────────────────────────────


def _get_parse_semaphore() -> asyncio.Semaphore:
    """延迟导入全局 parse 信号量，保护 GPU/显存/CPU。"""
    from tcm_extract_engine.common.concurrency import get_parse_semaphore
    return get_parse_semaphore()


def _doc_id(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:32]
    return f"doc-{digest}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _artifact_roots(
    source_pdf: Path, *, cache_variant: str | None = None
) -> tuple[Path, Path]:
    """返回解析产物目录；指定 variant 时不触碰历史缓存目录。"""
    parsed_root = source_pdf.parent / "__parsed__"
    raw_root = parsed_root / f"{source_pdf.name}.mineru_raw"
    parsed_output_root = parsed_root / f"{source_pdf.name}.parsed"
    if cache_variant:
        raw_dir = raw_root / "variants" / cache_variant
        parsed_dir = parsed_output_root / "variants" / cache_variant
    else:
        raw_dir = raw_root
        parsed_dir = parsed_output_root
    return raw_dir, parsed_dir


def _manifest_path(raw_dir: Path) -> Path:
    return raw_dir / "_manifest.json"


def _blocks_path(parsed_dir: Path, source_pdf: Path) -> Path:
    return parsed_dir / f"{source_pdf.stem}.blocks.jsonl"


def _normalize_caption_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _content_value_to_text(value: Any) -> str:
    """将 MinerU 视觉块的 content 规范为可供下游 LLM 消费的文本。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).strip()
    return str(value).strip()


def _join_unique_texts(parts: list[str]) -> str:
    """按输入顺序拼接非空且不重复的视觉语义文本。"""
    unique_parts: list[str] = []
    for part in parts:
        normalized = part.strip()
        if normalized and normalized not in unique_parts:
            unique_parts.append(normalized)
    return "\n".join(unique_parts)


def _page_index(item: dict[str, Any]) -> int:
    try:
        return int(item.get("page_idx", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _positions_for_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    bbox = item.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            coords = [float(value) for value in bbox]
        except (TypeError, ValueError):
            coords = [0.0, 0.0, 0.0, 0.0]
        return [{"type": "bbox", "anchor": str(_page_index(item) + 1), "range": coords}]
    return []


def _looks_like_prescription_line(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?(?:克|g|mg|毫克|钱|两)", compact, re.IGNORECASE):
        return True
    dose_matches = re.findall(r"\d+(?:\.\d+)?\s*(?:克|g|mg|毫克|钱|两)", text, re.IGNORECASE)
    if len(dose_matches) >= 1 and re.search(r"[\u4e00-\u9fff]{1,8}\s*\d", text):
        return True
    if compact.startswith(("克", "钱", "两")) and re.search(r"\d", compact):
        return True
    return False


def _looks_like_heading_text(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if _looks_like_prescription_line(text):
        return False
    if len(text) <= 3:
        return True
    if re.match(r"^\d+[.．、]\s*[\u4e00-\u9fff].{0,20}(?:证|案|痛|炎|病|治|法|证候)?$", text):
        return True
    if re.match(r"^[一二三四五六七八九十]+[、.．]\s*[\u4e00-\u9fff].{0,24}$", text):
        return True
    if len(text) <= 24 and not re.search(r"[，,。；;：:]", text):
        return True
    return False


def _heading_level(item: dict[str, Any]) -> int:
    raw_level = item.get("text_level")
    if raw_level is not None:
        if not _looks_like_heading_text(str(item.get("text") or "")):
            return 0
        try:
            return max(0, int(raw_level))
        except (TypeError, ValueError):
            return 0
    if item.get("type") == "header":
        return 1
    return 0


def asset_id(prefix: str, source_pdf: Path, item: dict[str, Any], index: int) -> str:
    """多模态块（图/表/公式）的稳定身份号；表格直通通路用同一算法反查块归属。"""
    basis = "|".join([
        prefix,
        source_pdf.name,
        str(item.get("page_idx", "")),
        str(item.get("img_path", "")),
        str(item.get("table_img_path", "")),
        str(index),
    ])
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def _content_list_path(raw_dir: Path) -> Path | None:
    path = raw_dir / "content_list.json"
    return path if path.is_file() else None


def _content_text_for_item(
    item: dict[str, Any], *, source_pdf: Path, index: int
) -> tuple[str, str, bool]:
    item_type = str(item.get("type") or "text")
    text = str(item.get("text") or "").strip()

    if item_type == "page_number":
        return "", "", False

    if item_type in {"image", "chart"}:
        caption = " ".join(_normalize_caption_list(item.get("image_caption")))
        if item_type == "chart":
            caption = " ".join(_normalize_caption_list(item.get("chart_caption"))) or caption
        footnote = " ".join(_normalize_caption_list(item.get("image_footnote")))
        if item_type == "chart":
            footnote = " ".join(_normalize_caption_list(item.get("chart_footnote"))) or footnote
        payload = _join_unique_texts([
            caption,
            footnote,
            text,
            _content_value_to_text(item.get("content")),
        ])
        prefix = "im" if item_type == "image" else "ch"
        asset = asset_id(prefix, source_pdf, item, index)
        fallback = item.get("img_path") or item_type
        content = f'<drawing id="{prefix}-{asset}">{payload or fallback}</drawing>'
        if payload:
            label = "图片解析" if item_type == "image" else "图表解析"
            content = f"{content}\n[{label}] {payload}"
        return content, item_type, True

    if item_type == "table":
        table_body = item.get("table_data") or item.get("table_body") or text
        payload = str(table_body or "").strip()
        asset = asset_id("tb", source_pdf, item, index)
        content = f'<drawing id="tb-{asset}">{payload or "table"}</drawing>'
        return content, text or payload or "table", True

    if item_type == "equation":
        equation = str(item.get("latex") or item.get("equation") or text).strip()
        asset = asset_id("eq", source_pdf, item, index)
        content = f'<drawing id="eq-{asset}">{equation or "equation"}</drawing>'
        return content, text or equation or "equation", True

    if item_type == "list":
        # MinerU 3 将有序/无序列表放在 ``list_items``，而不是 ``text``。
        # 若只读取 text，会把“煎服禁忌”“生活禁忌”标题下的全部条目静默丢弃。
        raw_items = item.get("list_items")
        if isinstance(raw_items, list):
            list_text = "\n".join(
                _content_value_to_text(list_item)
                for list_item in raw_items
                if _content_value_to_text(list_item)
            )
        else:
            list_text = _content_value_to_text(raw_items)
        return list_text, "", False

    if not text:
        return "", "", False

    level = _heading_level(item)
    if level > 0:
        return f'{"#" * min(level, 6)} {text}', text, False
    return text, "", False


def _content_list_to_blocks(
    source_pdf: Path, content_list: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    heading_stack: list[str] = []
    current: dict[str, Any] | None = None
    # MinerU 有时会把章节标题、表格说明或注释标为 header/footer。
    # 不能按类型直接丢弃；只去除跨页重复的版心文字。
    seen_layout_texts: set[tuple[str, str]] = set()

    def start_block(
        *, content: str, heading: str, parent_headings: list[str],
        level: int, positions: list[dict[str, Any]],
    ) -> None:
        nonlocal current
        current = {
            "lines": [content], "heading": heading,
            "parent_headings": parent_headings, "level": level, "positions": positions,
        }

    def append_to_current(content: str, positions: list[dict[str, Any]]) -> None:
        if current is None:
            start_block(
                content=content,
                heading=heading_stack[-1] if heading_stack else "",
                parent_headings=heading_stack[:-1] if heading_stack else [],
                level=0, positions=positions,
            )
            return
        current["lines"].append(content)
        current["positions"].extend(positions)

    def current_has_body() -> bool:
        if current is None:
            return False
        return any(not str(line).lstrip().startswith("#") for line in current["lines"])

    def block_from_current() -> dict[str, Any] | None:
        if current is None:
            return None
        content = "\n".join(str(line).strip() for line in current["lines"] if str(line).strip())
        if not content:
            return None
        block_id_basis = "|".join([
            source_pdf.name, str(current.get("heading", "")),
            content, json.dumps(current.get("positions", []), ensure_ascii=False),
        ])
        block_id = hashlib.sha256(block_id_basis.encode("utf-8")).hexdigest()[:32]
        return {
            "type": "content", "blockid": block_id, "format": "plain_text",
            "content": content, "heading": current.get("heading", ""),
            "parent_headings": current.get("parent_headings", []),
            "level": current.get("level", 0), "session_type": "body",
            "table_slice": "none", "positions": current.get("positions", []),
        }

    def flush_current() -> None:
        nonlocal current
        block = block_from_current()
        if block is not None:
            blocks.append(block)
        current = None

    def append_multimodal_block(
        content: str, heading: str, level: int, positions: list[dict[str, Any]],
    ) -> None:
        block_id_basis = "|".join([
            source_pdf.name, heading, content, json.dumps(positions, ensure_ascii=False),
        ])
        block_id = hashlib.sha256(block_id_basis.encode("utf-8")).hexdigest()[:32]
        blocks.append({
            "type": "content", "blockid": block_id, "format": "drawing",
            "content": content, "heading": heading,
            "parent_headings": list(heading_stack), "level": level,
            "session_type": "body", "table_slice": "none", "positions": positions,
        })

    case_start = re.compile(r"^\s*(?:【病例\s*\d+】|病例\s*\d+[：:、.．]?)")

    for index, item in enumerate(content_list):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "text")
        if item_type == "page_number":
            continue
        content, heading, is_multimodal = _content_text_for_item(
            item, source_pdf=source_pdf, index=index
        )
        if not content:
            continue
        if item_type in {"header", "footer"}:
            layout_key = (item_type, re.sub(r"\s+", " ", content).strip())
            if layout_key in seen_layout_texts:
                continue
            seen_layout_texts.add(layout_key)
        positions = _positions_for_item(item)
        # header/footer 的布局标签不等同于文档标题层级；作为正文保留，
        # 避免运行页眉把后续正文错误切成大量章节。
        level = 0 if item_type in {"header", "footer"} else _heading_level(item)

        if item_type in {"image", "chart", "table", "equation"}:
            flush_current()
            append_multimodal_block(content, heading, level, positions)
            continue
        if level > 0:
            flush_current()
            while len(heading_stack) >= level:
                heading_stack.pop()
            start_block(
                content=content, heading=heading,
                parent_headings=list(heading_stack), level=level, positions=positions,
            )
            heading_stack.append(heading)
            continue
        if case_start.match(content) and current_has_body():
            flush_current()
            first_line = content.splitlines()[0].strip()
            start_block(
                content=content, heading=first_line,
                parent_headings=list(heading_stack), level=0, positions=positions,
            )
            continue
        append_to_current(content, positions)

    flush_current()
    return blocks


def _write_jsonl(path: Path, meta: dict[str, Any], blocks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for block in blocks:
            fh.write(json.dumps(block, ensure_ascii=False) + "\n")


def _count_blocks(blocks_path: Path) -> int:
    if not blocks_path.is_file():
        return 0
    count = 0
    with blocks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                item = json.loads(line)
                if item.get("type") == "content":
                    count += 1
    return count


def _blocks_builder_version(blocks_path: Path) -> str:
    if not blocks_path.is_file():
        return ""
    try:
        first_line = blocks_path.read_text(encoding="utf-8").splitlines()[0]
        meta = json.loads(first_line)
    except (IndexError, json.JSONDecodeError):
        return ""
    return str(meta.get("blocks_builder_version") or "")


def _load_manifest(raw_dir: Path) -> dict[str, Any] | None:
    manifest_path = _manifest_path(raw_dir)
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_manifest(raw_dir: Path, manifest: dict[str, Any]) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(raw_dir)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest_path


def _safe_clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _meta_for_content_list(
    *, source_pdf: Path, source_hash: str, doc_id: str,
    content_list: list[dict[str, Any]], blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "meta", "format": "lightrag", "version": "1.0",
        "blocks_builder_version": BLOCKS_BUILDER_VERSION,
        "document_name": source_pdf.name,
        "document_format": source_pdf.suffix.lstrip(".").lower(),
        "document_hash": source_hash,
        "table_file": any(
            isinstance(item, dict) and str(item.get("type")) == "table"
            for item in content_list
        ),
        "equation_file": any(
            isinstance(item, dict) and str(item.get("type")) == "equation"
            for item in content_list
        ),
        "drawing_file": any(
            isinstance(item, dict) and str(item.get("type")) in {"image", "chart"}
            for item in content_list
        ),
        "asset_dir": any(
            isinstance(item, dict) and str(item.get("type")) in {"image", "chart", "table", "equation"}
            for item in content_list
        ),
        "split_option": {}, "blocks": len(blocks), "doc_id": doc_id,
        "parse_engine": "mineru",
        "parse_time": datetime.now(timezone.utc).isoformat(),
        "doc_title": source_pdf.stem,
        "bbox_attributes": {"origin": "LEFTTOP", "max": 1000},
    }


def _write_blocks_from_content_list(
    *, source_pdf: Path, parsed_dir: Path, source_hash: str,
    doc_id: str, content_list: list[dict[str, Any]],
) -> tuple[Path, list[dict[str, Any]]]:
    blocks = _content_list_to_blocks(source_pdf, content_list)
    meta = _meta_for_content_list(
        source_pdf=source_pdf, source_hash=source_hash,
        doc_id=doc_id, content_list=content_list, blocks=blocks,
    )
    blocks_path = _blocks_path(parsed_dir, source_pdf)
    _write_jsonl(blocks_path, meta, blocks)
    return blocks_path, blocks


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _mineru_backend() -> str:
    return os.getenv("MINERU_BACKEND", DEFAULT_MINERU_BACKEND).strip() or DEFAULT_MINERU_BACKEND


def _mineru_hybrid_effort() -> str:
    effort = os.getenv("MINERU_HYBRID_EFFORT", DEFAULT_MINERU_HYBRID_EFFORT).strip().lower()
    if effort not in {"medium", "high"}:
        raise ValueError("MINERU_HYBRID_EFFORT 必须为 medium 或 high")
    return effort


def _mineru_vlm_gpu_memory_utilization() -> float | None:
    """返回 vLLM 引擎的显存预算比例；未配置时返回 ``None``，交由 MinerU 取默认。

    MinerU 的默认值是整卡的一半（`backend/vlm/utils.py`：0.5，小卡才升到 0.7），
    而它按**总显存**算比例、不看剩余，所以在共享卡上会把整卡吃满。1.2B 的文档模型
    权重才 2 GiB，多出来的预算全部变成用不上的 KV cache。

    这个值只决定引擎预留多少显存，不改变解析产物，因此**不进** ``_mineru_parse_options``
    ——否则调一次参数就要重新解析整本书。
    """
    raw_value = os.getenv("MINERU_VLM_GPU_MEMORY_UTILIZATION", "").strip()
    if not raw_value:
        return DEFAULT_MINERU_VLM_GPU_MEMORY_UTILIZATION
    try:
        utilization = float(raw_value)
    except ValueError as exc:
        raise ValueError("MINERU_VLM_GPU_MEMORY_UTILIZATION 必须是 0-1 之间的小数") from exc
    if not 0 < utilization <= 1:
        raise ValueError("MINERU_VLM_GPU_MEMORY_UTILIZATION 必须落在 (0, 1] 区间")
    return utilization


def _mineru_parse_options() -> dict[str, Any]:
    """返回会改变 MinerU 原始产物的全部解析配置，用于调用和缓存身份。"""
    return {
        "backend": _mineru_backend(),
        "lang": os.getenv("MINERU_LANG") or "ch",
        "formula_enable": _env_bool("MINERU_FORMULA_ENABLE", DEFAULT_MINERU_FORMULA_ENABLE),
        "table_enable": _env_bool("MINERU_TABLE_ENABLE", DEFAULT_MINERU_TABLE_ENABLE),
        "image_analysis": _env_bool("MINERU_IMAGE_ANALYSIS", DEFAULT_MINERU_IMAGE_ANALYSIS),
        "hybrid_effort": _mineru_hybrid_effort(),
    }


def _mineru_cache_variant(parse_options: dict[str, Any]) -> str:
    """根据解析配置生成稳定、可读的缓存变体名。"""
    canonical_options = json.dumps(
        parse_options, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical_options.encode("utf-8")).hexdigest()[:12]
    backend = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(parse_options["backend"])).strip("-")
    return f"{backend or 'mineru'}-{digest}"


# ── 公开入口 ─────────────────────────────────────────────────────────────────


def _ensure_mineru_env() -> None:
    """确保 MinerU 所需环境变量已设置（幂等）。"""
    os.environ.setdefault("MINERU_MODEL_SOURCE", DEFAULT_MINERU_MODEL_SOURCE)


async def _parse_pdf_local(
    source_pdf: Path,
    *,
    output_dir: Path,
    backend: str,
    lang: str | None,
    formula: bool,
    table: bool,
    image_analysis: bool,
    hybrid_effort: str,
    vlm_gpu_memory_utilization: float | None = None,
) -> list[dict[str, Any]]:
    """进程内调用 MinerU do_parse，返回 content_list。"""
    try:
        from mineru.cli.common import do_parse
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MinerU 依赖未安装。请执行 `uv sync --extra mineru` 安装。"
        ) from exc

    _ensure_mineru_env()
    pdf_bytes = source_pdf.read_bytes()
    # 显存预算只对起了 vLLM 引擎的后端有意义，pipeline 会直接忽略，别把它传下去误导日志
    engine_kwargs: dict[str, Any] = {}
    if vlm_gpu_memory_utilization is not None and not backend.startswith("pipeline"):
        engine_kwargs["gpu_memory_utilization"] = vlm_gpu_memory_utilization
    logger.info(
        "[MinerU] 调用 do_parse: %s (%.1f MB, backend=%s, lang=%s, image_analysis=%s, effort=%s%s)",
        source_pdf.name, len(pdf_bytes) / 1024 / 1024, backend, lang or "ch",
        image_analysis, hybrid_effort,
        "".join(f", {key}={value}" for key, value in sorted(engine_kwargs.items())),
    )
    t0 = time.perf_counter()

    try:
        await asyncio.to_thread(
            do_parse,
            str(output_dir),
            [source_pdf.stem],
            [pdf_bytes],
            [lang or "ch"],
            backend=backend,
            formula_enable=formula,
            table_enable=table,
            image_analysis=image_analysis,
            effort=hybrid_effort,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,
            **engine_kwargs,
        )
    except Exception:
        logger.exception(
            "[MinerU] do_parse 异常 (backend=%s, pdf=%s)", backend, source_pdf.name
        )
        raise

    # do_parse 输出: {output_dir}/{stem}/auto/{stem}_content_list.json
    # MinerU 会在 stem 下创建语言子目录（如 auto/、ch/、en/），需动态查找
    stem_dir = output_dir / source_pdf.stem
    content_list_file = None
    for candidate in sorted(stem_dir.rglob(f"{source_pdf.stem}_content_list.json")):
        content_list_file = candidate
        break
    if content_list_file is None or not content_list_file.is_file():
        raise RuntimeError(f"MinerU 未生成 content_list: {stem_dir}")
    logger.info("[MinerU] do_parse 完成: %s (%.1fs)", source_pdf.name, time.perf_counter() - t0)
    return json.loads(content_list_file.read_text(encoding="utf-8"))


async def ensure_mineru_artifact(source_pdf: Path) -> ParseResult:
    """解析 PDF 为结构化 blocks，进程内直调 MinerU。

    支持 SHA256 文件级缓存和 batch 级断点续传。
    并发解析通过信号量控制，避免 GPU OOM。
    """
    source_pdf = source_pdf.resolve()
    if not source_pdf.is_file():
        raise FileNotFoundError(f"PDF 文件不存在: {source_pdf}")

    parse_options = _mineru_parse_options()
    cache_variant = _mineru_cache_variant(parse_options)
    raw_dir, parsed_dir = _artifact_roots(source_pdf, cache_variant=cache_variant)
    blocks_path = _blocks_path(parsed_dir, source_pdf)
    source_hash = _sha256_file(source_pdf)
    doc_id = _doc_id(source_pdf)

    # 缓存命中：manifest + blocks 均存在且版本匹配
    manifest = _load_manifest(raw_dir)
    if (
        manifest
        and manifest.get("source_sha256") == source_hash
        and manifest.get("mineru_parse_options") == parse_options
        and blocks_path.is_file()
        and _blocks_builder_version(blocks_path) == BLOCKS_BUILDER_VERSION
    ):
        logger.info("[MinerU] 缓存命中: %s (%d blocks)", source_pdf.name, _count_blocks(blocks_path))
        return ParseResult(
            parsed_dir=parsed_dir, cache_hit=True,
            block_count=_count_blocks(blocks_path),
            source_pdf=source_pdf, raw_dir=raw_dir, blocks_path=blocks_path,
            content_list_path=_content_list_path(raw_dir),
        )

    # 半缓存：content_list 存在，仅需重建 blocks
    content_list_path = raw_dir / "content_list.json"
    if (
        manifest
        and manifest.get("source_sha256") == source_hash
        and manifest.get("mineru_parse_options") == parse_options
        and content_list_path.is_file()
    ):
        logger.info("[MinerU] 半缓存命中，重建 blocks: %s", source_pdf.name)
        content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
        blocks_path, blocks = _write_blocks_from_content_list(
            source_pdf=source_pdf, parsed_dir=parsed_dir,
            source_hash=source_hash, doc_id=doc_id, content_list=content_list,
        )
        manifest["blocks_builder_version"] = BLOCKS_BUILDER_VERSION
        manifest["content_blocks"] = len(blocks)
        manifest["blocks_path"] = str(blocks_path)
        _write_manifest(raw_dir, manifest)
        return ParseResult(
            parsed_dir=parsed_dir, cache_hit=True, block_count=len(blocks),
            source_pdf=source_pdf, raw_dir=raw_dir, blocks_path=blocks_path,
            content_list_path=content_list_path,
        )

    # 全量解析
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    max_retries = int(os.getenv("MINERU_PARSE_RETRIES", "3"))
    retry_delay = float(os.getenv("MINERU_PARSE_RETRY_DELAY", "10"))

    logger.info(
        "[MinerU] 开始全量解析: %s (backend=%s, retries=%d)",
        source_pdf.name, parse_options["backend"], max_retries,
    )
    parse_start = time.perf_counter()

    # 配置 backend；非 pipeline backend 失败时自动降级到 pipeline，提高稳定性
    primary_backend = str(parse_options["backend"])
    # 解析前就求值：配错了要立刻报错，别等 vLLM 起来才失败
    vlm_gpu_memory_utilization = _mineru_vlm_gpu_memory_utilization()
    fallback_backends: list[str] = []
    if os.getenv("MINERU_FALLBACK_TO_PIPELINE", "1") not in ("0", "false", "False"):
        if primary_backend != "pipeline":
            fallback_backends.append("pipeline")

    last_exc: Exception | None = None
    content_list: list[dict[str, Any]] = []

    async def _try_parse_with_backend(backend: str) -> list[dict[str, Any]] | None:
        """尝试用指定 backend 解析，失败返回 None。"""
        nonlocal last_exc
        for attempt in range(1, max_retries + 1):
            _safe_clear_dir(parsed_dir)
            _safe_clear_dir(raw_dir)
            async with _get_parse_semaphore():
                try:
                    return await _parse_pdf_local(
                        source_pdf,
                        output_dir=raw_dir,
                        backend=backend,
                        lang=str(parse_options["lang"]),
                        formula=bool(parse_options["formula_enable"]),
                        table=bool(parse_options["table_enable"]),
                        image_analysis=bool(parse_options["image_analysis"]),
                        hybrid_effort=str(parse_options["hybrid_effort"]),
                        vlm_gpu_memory_utilization=vlm_gpu_memory_utilization,
                    )
                except Exception as exc:
                    last_exc = exc
                    err_msg = str(exc)
                    if attempt < max_retries:
                        logger.warning(
                            "[MinerU] %s 解析失败 (%d/%d): %s",
                            backend, attempt, max_retries, err_msg[:200],
                        )
                        await asyncio.sleep(retry_delay)
        return None

    for backend in [primary_backend] + fallback_backends:
        logger.info("[MinerU] 尝试 backend: %s", backend)
        content_list = await _try_parse_with_backend(backend)
        if content_list:
            if backend != primary_backend:
                logger.info(
                    "[MinerU] %s 解析失败，已降级到 %s 成功",
                    primary_backend, backend,
                )
            break
        logger.warning("[MinerU] %s 解析失败，准备尝试 fallback backend", backend)
    else:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"MinerU 解析失败: {source_pdf.name}")

    blocks_path, blocks = _write_blocks_from_content_list(
        source_pdf=source_pdf, parsed_dir=parsed_dir,
        source_hash=source_hash, doc_id=doc_id, content_list=content_list,
    )

    elapsed = time.perf_counter() - parse_start
    logger.info(
        "[MinerU] 解析完成: %s → %d blocks, 耗时 %.1fs",
        source_pdf.name, len(blocks), elapsed,
    )

    raw_manifest: dict[str, Any] = {
        "source_pdf": str(source_pdf),
        "source_sha256": source_hash,
        "mineru_parse_options": parse_options,
        "cache_variant": cache_variant,
        "blocks_builder_version": BLOCKS_BUILDER_VERSION,
        "doc_id": doc_id,
        "parser": "mineru.do_parse",
        "cache_key": str(source_pdf),
        "parsed_dir": str(parsed_dir),
        "blocks_path": str(blocks_path),
        "content_blocks": len(blocks),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(raw_dir, raw_manifest)

    content_list_path = raw_dir / "content_list.json"
    content_list_path.write_text(
        json.dumps(content_list, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    return ParseResult(
        parsed_dir=parsed_dir, cache_hit=False, block_count=len(blocks),
        source_pdf=source_pdf, raw_dir=raw_dir, blocks_path=blocks_path,
        content_list_path=content_list_path,
    )


# ── Markdown 直接解析（跳过 MinerU）──────────────────────────────────────────


MARKDOWN_SUFFIXES = {".md", ".markdown"}


def _markdown_to_content_list(md_text: str) -> list[dict[str, Any]]:
    """将 Markdown 文本转为 MinerU content_list 格式。

    仅识别 ATX 标题（# ~ ######）与段落（空行分隔），
    其余内容（列表、代码块等）作为普通文本块，交由 blocks 构建器统一处理。
    """
    items: list[dict[str, Any]] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        text = "\n".join(paragraph).strip()
        if text:
            items.append({"type": "text", "text": text, "page_idx": 0})
        paragraph.clear()

    for line in md_text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        atx = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", stripped)
        if atx:
            flush_paragraph()
            items.append({
                "type": "text",
                "text": atx.group(2).strip(),
                "text_level": len(atx.group(1)),
                "page_idx": 0,
            })
            continue
        paragraph.append(line)

    flush_paragraph()
    return items


async def ensure_markdown_artifact(source: Path) -> ParseResult:
    """将 Markdown 文件直接转为 blocks，跳过 MinerU 解析。

    复用 content_list → blocks 转换管线，产出与 MinerU 一致的 .blocks.jsonl，
    使下游 semantic_chunk 无需感知来源差异。支持 SHA256 文件级缓存。
    """
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Markdown 文件不存在: {source}")

    raw_dir, parsed_dir = _artifact_roots(source)
    blocks_path = _blocks_path(parsed_dir, source)
    source_hash = _sha256_file(source)
    doc_id = _doc_id(source)

    # 缓存命中：manifest + blocks 均存在且版本匹配
    manifest = _load_manifest(raw_dir)
    if (
        manifest
        and manifest.get("source_sha256") == source_hash
        and blocks_path.is_file()
        and _blocks_builder_version(blocks_path) == BLOCKS_BUILDER_VERSION
    ):
        logger.info("[Markdown] 缓存命中: %s (%d blocks)", source.name, _count_blocks(blocks_path))
        return ParseResult(
            parsed_dir=parsed_dir, cache_hit=True,
            block_count=_count_blocks(blocks_path),
            source_pdf=source, raw_dir=raw_dir, blocks_path=blocks_path,
        )

    # 全量转换
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    _safe_clear_dir(parsed_dir)
    _safe_clear_dir(raw_dir)

    md_text = source.read_text(encoding="utf-8")
    content_list = _markdown_to_content_list(md_text)
    blocks_path, blocks = _write_blocks_from_content_list(
        source_pdf=source, parsed_dir=parsed_dir,
        source_hash=source_hash, doc_id=doc_id, content_list=content_list,
    )

    raw_manifest: dict[str, Any] = {
        "source_pdf": str(source),
        "source_sha256": source_hash,
        "blocks_builder_version": BLOCKS_BUILDER_VERSION,
        "doc_id": doc_id,
        "parser": "markdown-direct",
        "cache_key": str(source),
        "parsed_dir": str(parsed_dir),
        "blocks_path": str(blocks_path),
        "content_blocks": len(blocks),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(raw_dir, raw_manifest)

    logger.info("[Markdown] 转换完成: %s → %d blocks", source.name, len(blocks))
    return ParseResult(
        parsed_dir=parsed_dir, cache_hit=False, block_count=len(blocks),
        source_pdf=source, raw_dir=raw_dir, blocks_path=blocks_path,
    )


async def ensure_parse_artifact(source: Path) -> ParseResult:
    """根据文件格式分发解析：Markdown 直接转 blocks，其余走 MinerU。"""
    if source.suffix.lower() in MARKDOWN_SUFFIXES:
        return await ensure_markdown_artifact(source)
    return await ensure_mineru_artifact(source)
