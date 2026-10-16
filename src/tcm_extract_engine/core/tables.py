"""确定性表格解析：把 drawing 块里的原表（HTML/Markdown）拆成表头+数据行。

用途（表格直通通路）：
  - extract 阶段：表格块不再送 LLM，逐行透传为 TABLE_ROW 记录（逐字保真）。
  - export 阶段：把原表版式原样写入「原始表格」sheet 供医生对照。

只解析、不推断：空单元格保留为空串，不做任何补全/规范化。
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

# <drawing id="..."> 包裹层（MinerU 多模态块标记）
_DRAWING_OPEN = re.compile(r"^\s*<drawing\b[^>]*>", re.IGNORECASE)
_DRAWING_CLOSE = re.compile(r"</drawing>\s*$", re.IGNORECASE)
_DRAWING_ID = re.compile(r'<drawing\b[^>]*\bid=["\']([^"\']+)["\']', re.IGNORECASE)

# HTML 表格行/单元格
_TR = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TD = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")

# Markdown 分隔行：| --- | :---: |
_MD_SEPARATOR = re.compile(r"^:?-{2,}:?$")


@dataclass(slots=True)
class TableParse:
    """一张原表的解析结果（表头 + 数据行，单元格已去标签/去实体）。"""

    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    header_rows: list[list[str]] = field(default_factory=list)
    cell_origins: list[list[tuple[int, int]]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.rows)


def strip_drawing(content: str) -> str:
    """去掉 <drawing …> 包裹层，返回内部表体。"""
    text = content.strip()
    text = _DRAWING_OPEN.sub("", text)
    text = _DRAWING_CLOSE.sub("", text)
    return text.strip()


def drawing_id(content: str) -> str:
    """提取 drawing 块稳定 ID（无则空串）。"""
    m = _DRAWING_ID.search(content)
    return m.group(1) if m else ""


def _clean_cell(raw: str) -> str:
    """单元格去标签/去实体/压空白（保留原文文字，不做规范化）。"""
    text = _TAG.sub(" ", raw)
    text = html.unescape(text)
    # 全角/半角空格与换行压成单空格
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _parse_html(body: str) -> TableParse:
    parser = _GridParser()
    parser.feed(body)
    if not parser.rows:
        return TableParse()
    grid: dict[tuple[int, int], tuple[str, tuple[int, int]]] = {}
    header_count = 0
    for row_index, cells in enumerate(parser.rows):
        column = 0
        if all(cell[3] for cell in cells) and row_index == header_count:
            header_count += 1
        for text, rowspan, colspan, _is_header in cells:
            while (row_index, column) in grid:
                column += 1
            for row_offset in range(rowspan):
                for col_offset in range(colspan):
                    grid[row_index + row_offset, column + col_offset] = (
                        text,
                        (row_index, column),
                    )
            if row_index == 0:
                header_count = max(header_count, rowspan)
            column += colspan
    width = max((column for _, column in grid), default=-1) + 1
    height = max((row for row, _ in grid), default=-1) + 1
    matrix = [
        [grid.get((row, col), ("", (row, col)))[0] for col in range(width)] for row in range(height)
    ]
    header_count = min(max(header_count, 1), len(matrix))
    headers = []
    for column in range(width):
        labels = list(dict.fromkeys(row[column] for row in matrix[:header_count] if row[column]))
        headers.append(" / ".join(labels))
    origins = [
        [grid.get((row, col), ("", (row, col)))[1] for col in range(width)]
        for row in range(header_count, height)
    ]
    return TableParse(
        headers=headers,
        rows=matrix[header_count:],
        header_rows=matrix[:header_count],
        cell_origins=origins,
    )


class _GridParser(HTMLParser):
    """Expand declared spans while retaining the original cell coordinates."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int, bool]]] = []
        self.current_row: list[tuple[str, int, int, bool]] = []
        self.current_cell: list[str] | None = None
        self.spans = (1, 1)
        self.header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            values = dict(attrs)
            try:
                # Bound malformed source HTML, rather than allocating unbounded grids.
                self.spans = tuple(
                    min(100, max(1, int(values.get(key) or "1"))) for key in ("rowspan", "colspan")
                )
            except ValueError:
                self.spans = (1, 1)
            self.header = tag == "th"
            self.current_cell = []
        elif tag == "br" and self.current_cell is not None:
            self.current_cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.current_cell is not None:
            self.current_row.append(
                (_clean_cell("".join(self.current_cell)), self.spans[0], self.spans[1], self.header)
            )
            self.current_cell = None
        elif tag == "tr" and self.current_row:
            self.rows.append(self.current_row)
            self.current_row = []


def _split_md_line(line: str) -> list[str]:
    parts = line.strip().strip("|").split("|")
    return [_clean_cell(p) for p in parts]


def _parse_markdown(body: str) -> TableParse:
    rows: list[list[str]] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = _split_md_line(stripped)
        if all(_MD_SEPARATOR.fullmatch(c) or c == "" for c in cells) and any(cells):
            continue  # 分隔行 |---|---|
        if any(cells):
            rows.append(cells)
    if not rows:
        return TableParse()
    return TableParse(headers=rows[0], rows=rows[1:])


def parse_table(content: str) -> TableParse:
    """解析 drawing 块内容为表头+数据行（自动识别 HTML / Markdown）。"""
    body = strip_drawing(content)
    if not body:
        return TableParse()
    if "<tr" in body.lower():
        parsed = _parse_html(body)
    elif any(ln.strip().startswith("|") for ln in body.splitlines()):
        parsed = _parse_markdown(body)
    else:
        return TableParse()
    _normalize_width(parsed)
    return parsed


def _normalize_width(parsed: TableParse) -> None:
    """把数据行宽度对齐到表头（不足补空串、超出截断），保证列对齐。"""
    width = max([len(parsed.headers), *(len(row) for row in parsed.rows)])
    if width == 0:
        return
    if width > len(parsed.headers):
        parsed.warnings.append("数据行宽于表头；保留全部列，未标注表头留空")
        parsed.headers.extend([""] * (width - len(parsed.headers)))
    parsed.rows = [(row + [""] * width)[:width] for row in parsed.rows]
