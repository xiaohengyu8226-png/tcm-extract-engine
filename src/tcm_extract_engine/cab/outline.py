"""Model C：文档大纲修复（Outline Repair）。

背景：解析器（MinerU 各 backend）给出的标题层级是**平铺**的——text_level 只有
1/2 两档，章/节/条的真实层级在解析时即丢失，目录页条目、前言条目、短正文行
还会被误标成标题。此前靠解析层书域特化的正则启发式补救，换一本书就失效。

本模块改为**根本解**：在 parse 之后对 blocks 做一次 C-LLM 结构修复——
把标题候选流（文本 + 页码 + bbox 版式特征）交给 LLM，由它按语义输出
「真实层级 + 剔除 + 标题规范化」，再确定性重建 heading/parent_headings/level。

设计要点：
  - 零硬规则：层级与剔除完全由 LLM 语义判断，换书不失效；
  - 版式信号：每条候选附带行高比/缩进/居中偏移/宽比等客观 bbox 特征，
    另附全书版式上下文，全部为描述性数值、无任何阈值；
  - 分批调用：候选流按 token 预算切批（大书可超单次上下文），批间携带
    前批尾部决策保持层级连续；批级缓存使中途失败重试不重复计费；
  - 幂等：全书缓存键 = 候选流（含特征）+ 版式上下文 + 模板版本 + 模型，
    命中后确定性重建，不重复计费（对齐三层内容 hash 幂等约定）；
  - 优雅降级：LLM 失败或输出非法时原样返回 blocks（degraded），流水线不中断；
  - 健康闸门：层级已非退化（≥3 档）的书跳过修复，避免无谓 LLM 调用。
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tcm_extract_engine.cab.planner import PromptTemplateRegistry
from tcm_extract_engine.core.artifacts import artifact_hash
from tcm_extract_engine.core.llm import UnifiedLLMClient

logger = logging.getLogger(__name__)

OUTLINE_TEMPLATE = "outline_repair.md"
OUTLINE_PLACEHOLDER = "{{outline_input}}"
_SYSTEM_PROMPT = "你是文档结构修复引擎。严格按指令输出 JSON。"
CACHE_FILENAME = "outline_repair.json"

# 层级已非退化的下限：>=3 档层级视为解析器已给出真实层级（如 Markdown ATX 直转），跳过修复
MIN_DISTINCT_LEVELS_HEALTHY = 3
# 标题层级上限（prompt 约定 1-4，重建时防御性放宽到 6）
MAX_LEVEL = 6
# 分批调用：大书候选流可超单次上下文（实测 908 候选 ≈ 4 万 token）。
# 每批按保守 token 估算切分，批间携带前批尾部决策保持层级连续。
BATCH_TOKEN_BUDGET = 8000  # 单批候选 token 预算（模板+上下文+输出余量之外）
PRIOR_TAIL_SIZE = 5        # 批间上下文保留的最近决策条目数
# 书签优先（确定性、零 LLM）：PDF 内嵌大纲是出版方声明的层级，信号充分时
# 直接用它定级；信号不足/歧义时放弃该路径，落入 LLM 修复兜底。
BOOKMARK_PAGE_TOLERANCE = 1  # 书签页码与块页码容差（封面偏移/跨页）
BOOKMARK_MIN_MATCHES = 3     # 至少匹配的候选数，过少说明书签残缺
# 编号优先（确定性、零 LLM）：中文编号体系（第X章/第X节/一、/（一）/1./1.1…）
# 按本书实际出现的族压缩成连续层级，换书自适应；体系不完整时交 LLM。
NUMBERING_MIN_MATCHES = 3    # 至少编号的正文候选数，过少说明编号体系不可用


@dataclass(slots=True)
class OutlineEntry:
    """单条标题候选的修复决策。"""

    level: int  # 1..MAX_LEVEL；keep=False 时为 0
    keep: bool
    title: str


@dataclass(slots=True)
class OutlineRepairMeta:
    """修复审计信息（写入 parse_summary，供 Web/日志展示）。"""

    candidates: int = 0
    kept: int = 0
    dropped: int = 0
    levels: dict[str, int] = field(default_factory=dict)
    source: str = ""  # llm | cache | bookmarks | numbering | healthy-skip | no-candidates | degraded
    prompt_version: str = ""
    model: str = ""
    batches: int = 0  # LLM 路径的分批数（大书 >1，小书 =1）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── 候选提取 ────────────────────────────────────────────────────────────────


def extract_candidates(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """提取标题候选流：level>0 且 heading 非空的块，按文档顺序。

    每个候选附带版式特征（无任何阈值，纯客观描述，判断交给 LLM）：
      - height: 标题行 bbox 行高
      - height_ratio: 行高 / 全书正文行高中位数（目录条目≈1，正文章题常>1.5）
      - indent: 左边界相对全书正文左边界中位数的偏移（正=更靠左/凸出）
      - center_offset: 标题中心偏离页中线的比例（0=居中，0.5=贴边）
      - width_ratio: 标题宽度 / 页宽
    基准缺失时对应字段为 null。正文行来源：level>0 块的 positions[1:]
    （首行是标题）与 level=0 块的全部行；行高取全书中位数，抗个别页
    表格/多行段污染（MinerU text 项绝大多数为行级，中位数≈真实行高）。
    """
    body_bench = _body_benchmarks(blocks)
    page_widths = _page_widths(blocks)
    candidates: list[dict[str, Any]] = []
    for block in blocks:
        if int(block.get("level") or 0) <= 0:
            continue
        heading = str(block.get("heading") or "").strip()
        if not heading:
            continue
        positions = block.get("positions") or []
        candidate: dict[str, Any] = {
            "text": heading,
            "page": _page_of(positions),
            "height": _height_of(positions),
        }
        candidate.update(_layout_features(
            _box_of(positions), body_bench, page_widths.get(candidate["page"]),
        ))
        candidates.append(candidate)
    return candidates


def _page_of(positions: list[Any]) -> int | None:
    if not positions or not isinstance(positions[0], dict):
        return None
    try:
        return int(str(positions[0].get("anchor")))
    except (ValueError, TypeError):
        return None


def _box_of(positions: list[Any]) -> tuple[float, float, float, float] | None:
    """positions[0] 的 bbox（行级：每块首行自身的版式框）。"""
    if not positions or not isinstance(positions[0], dict):
        return None
    box = positions[0].get("range")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        return tuple(float(v) for v in box)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _height_of(positions: list[Any]) -> int | None:
    """标题行版式行高（bbox 高度取整），作为 LLM 的辅助版式信号。"""
    box = _box_of(positions)
    return None if box is None else max(0, int(round(box[3] - box[1])))


def _body_benchmarks(blocks: list[dict[str, Any]]) -> dict[str, float | None]:
    """全书正文行基准：行高/左边界中位数（中位数抗表格/多行段污染）。"""
    heights: list[float] = []
    lefts: list[float] = []
    for block in blocks:
        positions = block.get("positions") or []
        for idx, position in enumerate(positions):
            if not isinstance(position, dict):
                continue
            # 正文行：level=0 块全行；level>0 块跳过首行（标题行）
            if block.get("level") and idx == 0:
                continue
            box = _box_of([position])
            if box is None:
                continue
            heights.append(box[3] - box[1])
            lefts.append(box[0])
    return {
        "body_h": statistics.median(heights) if heights else None,
        "body_x0": statistics.median(lefts) if lefts else None,
    }


def _page_widths(blocks: list[dict[str, Any]]) -> dict[Any, float]:
    """每页版宽：该页所有行（含标题行）x1 的最大值。"""
    widths: dict[Any, float] = {}
    for block in blocks:
        for position in block.get("positions") or []:
            page = _page_of([position])
            box = _box_of([position])
            if page is None or box is None:
                continue
            widths[page] = max(widths.get(page, 0.0), box[2])
    return widths


def _layout_context(
    blocks: list[dict[str, Any]], candidates: list[dict[str, Any]],
) -> dict[str, float | None]:
    """全书版式上下文：给逐条特征提供解释基准（LLM 判断层级的全局参照）。

    例如：正文行高中位数 90 时，目录条目 height_ratio≈0.2 与正文标题
    height_ratio≈0.4 的相对差异才是有效信号，绝对值本身无意义。
    """
    bench = _body_benchmarks(blocks)
    widths = list(_page_widths(blocks).values())
    heights = [c["height"] for c in candidates if c.get("height") is not None]
    return {
        "body_height_median": bench["body_h"],
        "body_indent_median": bench["body_x0"],
        "page_width_median": statistics.median(widths) if widths else None,
        "heading_height_min": min(heights) if heights else None,
        "heading_height_median": statistics.median(heights) if heights else None,
        "heading_height_max": max(heights) if heights else None,
    }


def _layout_features(
    box: tuple[float, float, float, float] | None,
    bench: dict[str, float | None],
    page_width: float | None,
) -> dict[str, float | None]:
    """由 bbox 与基准计算客观版式特征；基准缺失时字段为 None。"""
    empty: dict[str, float | None] = {
        "height_ratio": None, "indent": None, "center_offset": None, "width_ratio": None,
    }
    if box is None:
        return empty
    height_ratio = None
    body_h = bench.get("body_h")
    if body_h:
        height_ratio = round((box[3] - box[1]) / body_h, 2)
    indent = None
    body_x0 = bench.get("body_x0")
    if body_x0 is not None:
        indent = round(box[0] - body_x0, 1)
    center_offset: float | None = None
    width_ratio: float | None = None
    if page_width:
        center_offset = round(abs((box[0] + box[2]) / 2 - page_width / 2) / page_width, 3)
        width_ratio = round((box[2] - box[0]) / page_width, 2)
    return {
        "height_ratio": height_ratio,
        "indent": indent,
        "center_offset": center_offset,
        "width_ratio": width_ratio,
    }


def hierarchy_needs_repair(blocks: list[dict[str, Any]]) -> bool:
    """层级是否退化：存在标题候选且 distinct levels 不足 3 档（解析器平铺的书）。"""
    levels = {int(b.get("level") or 0) for b in blocks if b.get("level")}
    return 0 < len(levels) < MIN_DISTINCT_LEVELS_HEALTHY


# ── LLM 输出校验 ─────────────────────────────────────────────────────────────


def _validate_entries(raw: Any, count: int) -> list[OutlineEntry] | None:
    """把 LLM 输出裁剪为与候选一一对应的有界决策；非法则返回 None（触发降级）。"""
    if not isinstance(raw, dict):
        return None
    items = raw.get("entries")
    if not isinstance(items, list) or len(items) != count:
        return None
    entries: list[OutlineEntry] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        keep = item.get("keep") is True
        try:
            level = int(item.get("level") or 0)
        except (TypeError, ValueError):
            return None
        title = str(item.get("title") or "").strip()
        if keep:
            if not title or level < 1:
                return None
            level = min(max(level, 1), MAX_LEVEL)
        else:
            level = 0
        entries.append(OutlineEntry(level=level, keep=keep, title=title))
    return entries


# ── 确定性重建 ───────────────────────────────────────────────────────────────


_HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s*")


def _strip_heading_prefix(content: str) -> str:
    """标题降级为正文时剥掉首行的 ``#`` 标记，避免标记残留正文。"""
    lines = content.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines[0] = _HEADING_PREFIX_RE.sub("", lines[0].lstrip())
    return "\n".join(lines)


def apply_outline(blocks: list[dict[str, Any]], entries: list[OutlineEntry]) -> list[dict[str, Any]]:
    """按修复决策确定性重建 heading/parent_headings/level。

    候选块与 entries 按文档顺序一一配对（标题文本可能重复，不能用文本做键）；
    剔除的标题块降为正文并入当前章节；正文块刷新父级链并同步栈顶标题。
    """
    stack: list[tuple[str, int]] = []  # (title, level)
    it = iter(entries)
    for block in blocks:
        is_candidate = (
            int(block.get("level") or 0) > 0 and bool(str(block.get("heading") or "").strip())
        )
        if is_candidate:
            entry = next(it, None)
            if entry is None:  # 防御：数量不匹配时该块保持原样
                continue
            if entry.keep and entry.title:
                while stack and stack[-1][1] >= entry.level:
                    stack.pop()
                block["heading"] = entry.title
                block["parent_headings"] = [t for t, _ in stack]
                block["level"] = entry.level
                stack.append((entry.title, entry.level))
            else:
                # 标题降级为正文：并入当前章节（heading=栈顶，父级=栈[:-1]，
                # 与 builder 正文块语义一致，保证 B 核验原文仍可定位）
                block["heading"] = stack[-1][0] if stack else ""
                block["parent_headings"] = [t for t, _ in stack[:-1]]
                block["level"] = 0
                block["content"] = _strip_heading_prefix(str(block.get("content") or ""))
        else:
            if block.get("heading"):
                block["heading"] = stack[-1][0] if stack else ""
            block["parent_headings"] = [t for t, _ in stack[:-1]]
    return blocks


def _entries_to_dicts(entries: list[OutlineEntry]) -> list[dict[str, Any]]:
    return [asdict(e) for e in entries]


def _entries_from_dicts(items: Any) -> list[OutlineEntry] | None:
    if not isinstance(items, list):
        return None
    try:
        return [
            OutlineEntry(
                level=int(e.get("level") or 0), keep=bool(e.get("keep")), title=str(e.get("title") or "")
            )
            for e in items
            if isinstance(e, dict)
        ]
    except (TypeError, ValueError):
        return None


# ── 缓存 ─────────────────────────────────────────────────────────────────────


def _cache_key(
    candidates: list[dict[str, Any]],
    layout_context: dict[str, float | None],
    prompt_version: str,
    model: str,
) -> str:
    """缓存键 = 候选流（含版式特征）+ 版式上下文 + 模板版本 + 模型。

    候选流与上下文均由 blocks 派生，且包含特征提取逻辑的输出——
    特征计算方式一旦变化，旧缓存自动失效重新走 LLM，不需要手动清缓存。
    """
    return artifact_hash({
        "candidates": candidates,
        "layout_context": layout_context,
        "prompt_version": prompt_version,
        "model": model,
    })


def _load_cache(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _save_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(cache_path)


# ── 分批调用 ─────────────────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """保守估算文本 token 数，防止批次超出模型上下文。

    GPT 系 tokenizer 中文约 1-2 token/字，取上限：中文 1 字 ≈ 1 token，
    ASCII 按 4 字符 ≈ 1 token；估高不估低，超预算宁可多切一批。
    """
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + max(1, (other + 3) // 4)


def _split_candidates(candidates: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """按 token 预算把候选流切成连续批次；单条超预算时独占一批。"""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for cand in candidates:
        tokens = _estimate_tokens(json.dumps(cand, ensure_ascii=False))
        if current and current_tokens + tokens > BATCH_TOKEN_BUDGET:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(cand)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def _prior_tail(entries: list[OutlineEntry]) -> list[dict[str, Any]]:
    """批间上下文：已完成决策的最近若干条（title + 层级 + keep）。"""
    return [
        {"text": e.title or "", "level": e.level, "keep": e.keep}
        for e in entries[-PRIOR_TAIL_SIZE:]
    ]


def _batch_cache_key(
    batch: list[dict[str, Any]],
    prior_tail: list[dict[str, Any]],
    prompt_version: str,
    model: str,
) -> str:
    """批级缓存键：本批候选 + 前批尾部决策 + 模板版本 + 模型。

    前批决策变化（模型/模板/输入变化）会经 prior_tail 传导为后批键变化，
    整链级联失效，保证决策链一致性。
    """
    return artifact_hash({
        "batch": batch,
        "prior_tail": prior_tail,
        "prompt_version": prompt_version,
        "model": model,
    })


# ── 书签优先（确定性、零 LLM）──────────────────────────────────────────────


_TITLE_NORM_RE = re.compile(r"[\s\u3000\u00a0·.．。:：,，;；…―—-—＿_【】\[\]（）()]+")


def _norm_title(text: str) -> str:
    """标题规范化键：去空白/点线/常见标点，用于书签与正文标题的容差匹配。"""
    return _TITLE_NORM_RE.sub("", text).lower()


def _page_int(page: Any) -> int | None:
    try:
        return int(page) if page is not None else None
    except (TypeError, ValueError):
        return None


def load_pdf_outline(pdf_path: Path) -> list[dict[str, Any]]:
    """读取 PDF 内嵌书签大纲，嵌套树展开为带层级+页码的平铺列表。

    任何异常（pypdf 缺失/文件损坏/无书签）都返回空列表——书签只是优先信号，
    拿不到就交给 LLM 修复兜底，绝不阻断流水线。
    """
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return []
    try:
        from pypdf import PdfReader  # pypdf 随 mineru extra 安装
    except ImportError:
        return []
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 - 打开失败静默降级
        logger.warning("[Outline] PDF 打开失败，跳过书签大纲: %s", exc)
        return []

    outline: list[dict[str, Any]] = []

    def _walk(items: Any, depth: int) -> None:
        for item in items:
            if isinstance(item, list):
                _walk(item, depth + 1)
                continue
            page = None
            try:
                page_obj = getattr(item, "page", None)
                page_no = reader.get_page_number(page_obj) if page_obj is not None else None
                if page_no is not None:
                    page = page_no + 1
            except Exception:  # noqa: BLE001 - 单条书签无页码则跳过
                page = None
            title = str(getattr(item, "title", "") or "").strip()
            if title and page:
                outline.append({"level": max(1, depth), "title": title, "page": page})

    try:
        _walk(reader.outline, 1)
    except Exception as exc:  # noqa: BLE001 - 大纲损坏时静默降级
        logger.warning("[Outline] PDF 书签大纲遍历失败: %s", exc)
        return []
    finally:
        try:
            reader.stream.close()
        except Exception:  # noqa: BLE001
            pass
    return outline


def _repair_from_bookmarks(
    candidates: list[dict[str, Any]],
    bookmarks: list[dict[str, Any]],
) -> list[OutlineEntry] | None:
    """用 PDF 书签大纲给候选确定性定级；信号不足/歧义时返回 None（交 LLM）。

    匹配：规范化标题全等 + 页码容差（±BOOKMARK_PAGE_TOLERANCE）；两轮贪心——
    先同页精确匹配（把书签留给正文标题，而非差一页的目录条目），再容差兜底；
    一书签至多匹配一候选。安全门（Docling 同思路：书签只加正确层级、绝不减）：
    (1) 匹配数 >= BOOKMARK_MIN_MATCHES；(2) 匹配层级 >= 2 档；
    (3) 未匹配候选必须全部位于首个匹配候选之前（前置目录/前言噪声），
    夹在正文中的未匹配标题说明书签不全，放弃整条路径。
    """
    if len(bookmarks) < BOOKMARK_MIN_MATCHES:
        return None

    used: set[int] = set()

    def _match(cand_page: int, text: str, *, exact: bool) -> int | None:
        norm = _norm_title(text)
        best_idx: int | None = None
        best_dist = 10**9
        for idx, bm in enumerate(bookmarks):
            if idx in used or _norm_title(str(bm.get("title") or "")) != norm:
                continue
            bm_page = _page_int(bm.get("page"))
            if bm_page is None:
                continue
            dist = abs(bm_page - cand_page)
            if dist <= BOOKMARK_PAGE_TOLERANCE and (not exact or dist == 0) and dist < best_dist:
                best_idx, best_dist = idx, dist
        if best_idx is not None:
            used.add(best_idx)
        return best_idx

    matched: dict[int, int] = {}
    # 第一轮：同页精确匹配（防目录条目抢走正文标题的书签）
    for ci, cand in enumerate(candidates):
        page = _page_int(cand.get("page"))
        if page is None:
            continue
        idx = _match(page, str(cand.get("text") or ""), exact=True)
        if idx is not None:
            matched[ci] = idx
    # 第二轮：±1 页容差兜底（封面偏移等）
    for ci, cand in enumerate(candidates):
        if ci in matched:
            continue
        page = _page_int(cand.get("page"))
        if page is None:
            continue
        idx = _match(page, str(cand.get("text") or ""), exact=False)
        if idx is not None:
            matched[ci] = idx

    # 安全门
    if len(matched) < BOOKMARK_MIN_MATCHES:
        return None
    levels = {int(bookmarks[idx]["level"]) for idx in matched.values()}
    if len(levels) < 2:
        return None
    first_matched = min(matched)
    if any(ci not in matched for ci in range(first_matched + 1, len(candidates))):
        return None

    entries: list[OutlineEntry] = []
    for ci, cand in enumerate(candidates):
        idx = matched.get(ci)
        if idx is None:  # 前置噪声（目录/前言）降为正文
            entries.append(OutlineEntry(level=0, keep=False, title=""))
        else:
            level = min(max(int(bookmarks[idx]["level"]), 1), MAX_LEVEL)
            entries.append(OutlineEntry(level=level, keep=True, title=str(cand["text"])))
    return entries


# ── 编号优先（确定性、零 LLM）─────────────────────────────────────────────


# 编号族按固定优先级排序（族顺序＝层级序：篇 > 章 > 节 > 条 > 一、 > （一） >
# 1./1.1 > 1. > (1) > 附录）；实际层级只取“本书出现的族”压缩成连续 1..N，
# 不做任何跨书绝对层级假设（如“第X章一定是一级”）——换书自适应。
_NUMBERING_SCHEMES: list[tuple[str, re.Pattern[str]]] = [
    ("part", re.compile(r"^第[一二三四五六七八九十百千0-9]+[篇部编卷]")),
    ("chapter", re.compile(r"^第[一二三四五六七八九十百千0-9]+章")),
    ("section", re.compile(r"^第[一二三四五六七八九十百千0-9]+节")),
    ("clause", re.compile(r"^第[一二三四五六七八九十百千0-9]+条")),
    ("han", re.compile(r"^[一二三四五六七八九十]+[、.．]")),
    ("han_paren", re.compile(r"^[（(][一二三四五六七八九十]+[）)]")),
    ("dot", re.compile(r"^[0-9]+(?:[.．][0-9]+)+[.．、\s]")),
    # 负向断言："1." 后面紧跟数字（如 1.5g 剂量）不是编号标记
    ("arabic", re.compile(r"^[0-9]+[.．、](?!\s*[0-9])")),
    ("arabic_paren", re.compile(r"^[（(]?[0-9]+[）)]")),
    ("appendix", re.compile(r"^附录[一二三四五六七八九十0-9A-Za-z]")),
]

# 目录条目标志：点线引导符（≥2 个，允许空格间隔）接页码结尾，正文标题不带页码后缀。
_TOC_SUFFIX_RE = re.compile(r"(?:[·.．…]\s*){2,}[0-9]+\s*$")


def _detect_numbering(text: str) -> tuple[str, int] | None:
    """识别标题开头的编号标记，返回 (族, 深度)；无编号返回 None。

    深度仅对点分十进制族有意义（1. → 1，1.1 → 2，1.1.1 → 3），其余族恒为 1。
    """
    for family, pattern in _NUMBERING_SCHEMES:
        match = pattern.match(text)
        if not match:
            continue
        if family == "dot":
            # 深度按点分十进制段数算，末位分隔符（.、空格等）不计入
            head = match.group(0).rstrip(" .．、\u3000")
            depth = head.count(".") + head.count("．") + 1
        else:
            depth = 1
        return family, depth
    return None


def _repair_from_numbering(candidates: list[dict[str, Any]]) -> list[OutlineEntry] | None:
    """用中文编号体系给候选确定性定级；体系不完整/歧义时返回 None（交 LLM）。

    与书签路径同哲学（只加正确层级、绝不减）：
    - 族层级只在本节内比较（本书出现的族按优先级压缩成连续 1..N），
      不引入任何跨书硬规则；
    - 点分十进制族按深度下推（1./1.1/1.1.1 → 连续层级），深度相对本书
      出现的最浅深度归零（只写 1.1.x 的书不会空出第一层）；
    - 目录条目（点线+页码后缀）任何位置都降为正文，正文同款标题保留。

    安全门（不满足即返回 None，落入 LLM 修复兜底）：
    (1) 正文编号候选数 >= 3；(2) 至少 2 个编号族（点分十进制单族
    以深度定层级除外）；
    (3) 首个正文编号候选之后不允许出现未编号候选——编号体系不完整或
    MinerU 把正文短行误标为标题时，交给 LLM 语义判断。
    """
    numbered: list[tuple[int, str, int]] = []  # (候选下标, 族, 深度)
    tocish: set[int] = set()
    for ci, cand in enumerate(candidates):
        text = str(cand.get("text") or "").strip()
        scheme = _detect_numbering(text)
        if scheme is None:
            continue
        numbered.append((ci, scheme[0], scheme[1]))
        if _TOC_SUFFIX_RE.search(text):
            tocish.add(ci)

    body_numbered = [(ci, family, depth) for ci, family, depth in numbered if ci not in tocish]
    if len(body_numbered) < NUMBERING_MIN_MATCHES:
        return None
    families = {family for _, family, _ in body_numbered}
    # 单族仅点分十进制可行（深度即层级）；其余单族无法定层级 → 交 LLM
    dot_depths = {depth for _, family, depth in body_numbered if family == "dot"}
    if len(families) < 2 and len(dot_depths) < 2:
        return None
    first_body = min(ci for ci, _, _ in body_numbered)
    # 安全门 (3)：首个正文编号之后的未编号候选 → 编号体系不完整，交 LLM
    for ci in range(first_body + 1, len(candidates)):
        if any(ni == ci for ni, _, _ in numbered):
            continue
        return None

    # 层级压缩：本书实际出现的族按优先级映射为连续 1..N
    present = [family for family, _ in _NUMBERING_SCHEMES if family in families]
    base = {family: idx + 1 for idx, family in enumerate(present)}
    min_dot_depth = min((depth for _, family, depth in body_numbered if family == "dot"), default=1)

    entries: list[OutlineEntry] = []
    for ci, cand in enumerate(candidates):
        if ci in tocish:
            entries.append(OutlineEntry(level=0, keep=False, title=""))
            continue
        info = next(((family, depth) for ni, family, depth in numbered if ni == ci), None)
        if info is None:  # 前置未编号噪声（书名页/前言/编者语）降为正文
            entries.append(OutlineEntry(level=0, keep=False, title=""))
            continue
        family, depth = info
        level = base[family] + max(0, depth - min_dot_depth) if family == "dot" else base[family]
        entries.append(OutlineEntry(level=min(level, MAX_LEVEL), keep=True, title=str(cand["text"])))
    return entries


# ── 主入口 ──────────────────────────────────────────────────────────────────


def rewrite_blocks_file(blocks_path: Path, blocks: list[dict[str, Any]], *, meta_note: dict[str, Any] | None = None) -> None:
    """重写 blocks.jsonl：保留首行 meta，并在 meta 上盖上 outline_repair 审计标记。

    ``load_blocks`` 只返回内容块，重写时必须把 meta 首行原样读回再写，
    否则下游（blocks_builder_version 记账、Web 元信息）会丢失。
    """
    if not blocks_path.is_file():
        raise FileNotFoundError(f"blocks 文件不存在: {blocks_path}")
    with blocks_path.open("r", encoding="utf-8") as fh:
        meta = json.loads(fh.readline())
    if meta_note is not None:
        meta["outline_repair"] = meta_note
    tmp = blocks_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for block in blocks:
            fh.write(json.dumps(block, ensure_ascii=False) + "\n")
    tmp.replace(blocks_path)


async def repair_blocks(
    blocks: list[dict[str, Any]],
    *,
    client: UnifiedLLMClient,
    registry: PromptTemplateRegistry,
    cache_dir: Path | None = None,
    model: str = "",
    bookmarks: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], OutlineRepairMeta]:
    """对 blocks 做一次大纲修复；任何失败路径都原样返回（优雅降级）。

    确定性优先（零 LLM）：PDF 内嵌大纲（load_pdf_outline）可安全匹配时直接
    定级；否则中文编号体系（第X章/第X节/一、/（一）/1./1.1 等）按本书实际
    出现的族压缩层级；两者信号不足时 C-LLM 修复——候选流按 token 预算
    分批调用（大书可超单次上下文），批间携带前批尾部决策（prior_tail）
    保持层级连续；批级缓存使中途失败重试不重复计费。
    返回 (blocks, meta)：blocks 为修复后的块列表；meta 记录来源与统计，
    供 runner 写入 parse_summary 审计。
    """
    candidates = extract_candidates(blocks)
    meta = OutlineRepairMeta(candidates=len(candidates), model=model or getattr(client, "model", ""))
    if not candidates:
        meta.source = "no-candidates"
        return blocks, meta
    if not hierarchy_needs_repair(blocks):
        meta.source = "healthy-skip"
        return blocks, meta

    # 书签优先：PDF 内嵌大纲是出版方声明的层级，信号充分时确定性定级，
    # 零 LLM 调用；信号不足/歧义时返回 None，落入 LLM 修复兜底。
    if bookmarks:
        bookmark_entries = _repair_from_bookmarks(candidates, bookmarks)
        if bookmark_entries is not None:
            repaired = apply_outline(blocks, bookmark_entries)
            _fill_stats(meta, bookmark_entries)
            meta.source = "bookmarks"
            logger.info(
                "[Outline] 书签大纲修复完成: %d 候选 → 保留 %d / 剔除 %d, 层级分布 %s",
                meta.candidates, meta.kept, meta.dropped, meta.levels,
            )
            return repaired, meta

    # 编号优先：中文编号体系同样可确定性定级、零 LLM 调用；
    # 体系不完整/歧义时返回 None，落入 LLM 修复兜底。
    numbering_entries = _repair_from_numbering(candidates)
    if numbering_entries is not None:
        repaired = apply_outline(blocks, numbering_entries)
        _fill_stats(meta, numbering_entries)
        meta.source = "numbering"
        logger.info(
            "[Outline] 编号大纲修复完成: %d 候选 → 保留 %d / 剔除 %d, 层级分布 %s",
            meta.candidates, meta.kept, meta.dropped, meta.levels,
        )
        return repaired, meta

    template = registry.load("c", OUTLINE_TEMPLATE)
    meta.prompt_version = template.version
    layout_context = _layout_context(blocks, candidates)
    key = _cache_key(candidates, layout_context, template.version, meta.model)
    cache_path = cache_dir / CACHE_FILENAME if cache_dir is not None else None

    # 缓存命中：确定性重建，零 LLM 调用（可复现）
    cached_payload: dict[str, Any] | None = None
    if cache_path is not None:
        cached_payload = _load_cache(cache_path)
        if cached_payload and cached_payload.get("key") == key:
            cached_entries = _entries_from_dicts(cached_payload.get("entries"))
            if cached_entries is not None:
                repaired = apply_outline(blocks, cached_entries)
                _fill_stats(meta, cached_entries)
                meta.source = "cache"
                return repaired, meta

    # 批级缓存：中途失败重试时已成功批不重复计费（前批尾部决策参与键）
    batch_cache: dict[str, list[dict[str, Any]]] = {}
    raw_batches = (cached_payload or {}).get("batches")
    if isinstance(raw_batches, dict):
        for batch_key, items in raw_batches.items():
            if isinstance(items, list):
                batch_cache[str(batch_key)] = items

    # LLM 修复：按 token 预算分批调用（大书候选流可能超单次上下文）；
    # 任一失败降级为原样（不阻断流水线），已成功批写入批级缓存供重试复用
    batches = _split_candidates(candidates)
    meta.batches = len(batches)
    entries: list[OutlineEntry] = []
    for batch_idx, batch in enumerate(batches, 1):
        tail = _prior_tail(entries)
        batch_key = _batch_cache_key(batch, tail, template.version, meta.model)
        batch_entries = _entries_from_dicts(batch_cache.get(batch_key))
        if batch_entries is None or len(batch_entries) != len(batch):
            outline_input = json.dumps(
                {"layout_context": layout_context, "prior_tail": tail, "candidates": batch},
                ensure_ascii=False,
            )
            user_prompt = template.content.replace(OUTLINE_PLACEHOLDER, outline_input)
            try:
                result = await client.complete_json(
                    system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt
                )
            except Exception as exc:  # noqa: BLE001 - 修复失败降级，不阻断 parse
                logger.warning(
                    "[Outline] LLM 修复失败（第 %d/%d 批），降级为平铺标题: %s",
                    batch_idx, len(batches), exc,
                )
                meta.source = "degraded"
                if cache_path is not None and batch_cache:
                    _save_cache(cache_path, {"batches": batch_cache})
                return blocks, meta
            batch_entries = _validate_entries(result, len(batch))
            if batch_entries is None:
                logger.warning(
                    "[Outline] LLM 输出非法（第 %d/%d 批条目数不匹配/字段越界），"
                    "降级为平铺标题",
                    batch_idx, len(batches),
                )
                meta.source = "degraded"
                if cache_path is not None and batch_cache:
                    _save_cache(cache_path, {"batches": batch_cache})
                return blocks, meta
            batch_cache[batch_key] = _entries_to_dicts(batch_entries)
        entries.extend(batch_entries)

    if cache_path is not None:
        _save_cache(cache_path, {
            "key": key,
            "prompt_version": template.version,
            "model": meta.model,
            "entries": _entries_to_dicts(entries),
            "batches": batch_cache,
        })

    repaired = apply_outline(blocks, entries)
    _fill_stats(meta, entries)
    meta.source = "llm"
    logger.info(
        "[Outline] 修复完成: %d 候选 / %d 批 → 保留 %d / 剔除 %d, 层级分布 %s",
        meta.candidates, meta.batches, meta.kept, meta.dropped, meta.levels,
    )
    return repaired, meta


def _fill_stats(meta: OutlineRepairMeta, entries: list[OutlineEntry]) -> None:
    meta.kept = sum(1 for e in entries if e.keep)
    meta.dropped = len(entries) - meta.kept
    counts: dict[str, int] = {}
    for e in entries:
        if e.keep:
            counts[str(e.level)] = counts.get(str(e.level), 0) + 1
    meta.levels = counts
