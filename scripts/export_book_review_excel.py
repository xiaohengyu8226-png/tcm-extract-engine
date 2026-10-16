"""从 P0 流水线工件（extract / validate / verify 缓存）导出医生审核版 Excel。

用法：
    python scripts/export_book_review_excel.py --book-id d302b2a18b01

输出（默认）：
    workspace/books/<book_id>/07_review/<源文件名>_医生审核版.xlsx

说明：
    - 只读工件，不修改 manifest；review 阶段正式落地前的过渡导出。
    - verify 结论从 06_verify/cache 读取，兼容 records/verifications/results
      三种响应键名（LLM 键名漂移）。
"""

from __future__ import annotations

import argparse
import enum
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.worksheet import Worksheet

from tcm_extract_engine.cab.profiler import load_blocks
from tcm_extract_engine.common.config import get_settings
from tcm_extract_engine.core.artifacts import load_artifact_records
from tcm_extract_engine.core.records import SourceRecord
from tcm_extract_engine.core.tables import drawing_id, parse_table
from tcm_extract_engine.domains import get_domain_pack

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspace" / "books"

REVIEW_SHEET = "审核记录表"
STATS_SHEET = "统计汇总"
GUIDE_SHEET = "填写说明"
RAW_TABLE_SHEET = "原始表格"


def _pack_labels(group: str, domain_id: str = "tcm") -> dict[str, str]:
    """领域包标签表（record_type/action/verdict 等，见 manifest.labels）。"""
    return get_domain_pack(domain_id).labels.get(group, {})


REVIEW_HEADERS = [
    "记录编号",
    "类型",
    "药物/主体",
    "相关药物",
    "动作强度",
    "结构化内容",
    "证据原文",
    "章节路径",
    "页码",
    "核验结论",
    "B建议修正",
    "核验依据",
    "校验问题",
    "疑点标记",
    "审核结论",
    "医生备注",
]

HEADER_FILL = PatternFill("solid", fgColor="002060")
HEADER_FONT = Font(color="FFFFFF", bold=True)
DOCTOR_FILL = PatternFill("solid", fgColor="FFF2CC")
WARN_FILL = PatternFill("solid", fgColor="FCE4EC")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "、".join(str(v) for v in value if v not in (None, ""))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _structured_content(record: SourceRecord) -> str:
    return get_domain_pack(record.domain_id).flatten_for_review(record)


def _subject(record: SourceRecord) -> str:
    return get_domain_pack(record.domain_id).subject_of(record)


def _related_herbs(record: SourceRecord) -> str:
    return get_domain_pack(record.domain_id).related_subjects(record)


def _correction_diff(record: SourceRecord, corrected: dict[str, Any]) -> str:
    """对比 B 修正稿与原记录，列出被修改的字段。"""
    diff: list[str] = []
    for field, new_value in corrected.items():
        if field in {"record_id", "book_id", "section", "record_type", "raw_text"}:
            continue
        old_raw = getattr(record, field, None)
        old_value = old_raw.value if isinstance(old_raw, enum.Enum) else old_raw
        old_text = _text(old_value)
        new_text = _text(new_value)
        if new_text and new_text != old_text:
            diff.append(f"{field}：{old_text or '（空）'} → {new_text}")
    if corrected.get("raw_text") and _text(corrected["raw_text"]) != record.raw_text:
        diff.append(f"证据原文：{record.raw_text[:60]}… → {_text(corrected['raw_text'])[:60]}…")
    return "；".join(diff)


def _load_verify_results(
    book_dir: Path, records_by_id: dict[str, SourceRecord] | None = None
) -> dict[str, dict[str, Any]]:
    """每条记录的核验结论：record_id → {verdict, evidence, correction}。

    优先读权威 per-run 工件 06_verify/verification_results.jsonl（由 verify 阶段
    从 report.results 落盘，含全部记录与 PASSTHROUGH），record_id 与当前
    records.jsonl 一一对应——从根上杜绝「跨运行按 record_id 合并缓存目录」时，
    因全局 record_id 重排（换一次抽取序号即位移）而被陈旧 verdict 撞车污染。

    仅当该工件缺失（本次改动前产出的旧书）时，才回退合并 06_verify/cache。
    """
    results = _load_verification_results(book_dir)
    if results is not None:
        return results
    return _load_verify_results_from_cache(book_dir, records_by_id or {})


def _load_verification_results(book_dir: Path) -> dict[str, dict[str, Any]] | None:
    """读权威 verification_results.jsonl；文件不存在返回 None 以触发回退。"""
    path = book_dir / "06_verify" / "verification_results.jsonl"
    if not path.is_file():
        return None
    corrections = _load_correction_log(book_dir)
    results: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if set(item.keys()) == {"header"}:
            continue
        rid = str(item.get("record_id") or "")
        if not rid:
            continue
        results[rid] = {
            "verdict": str(item.get("verdict") or ""),
            "evidence": _text(item.get("evidence_quote")),
            "correction": _format_corrections(corrections.get(rid, [])),
        }
    return results


def _load_correction_log(book_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """correction_log.jsonl → 按 record_id 归组的字段级修正建议（权威 per-run）。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    path = book_dir / "06_verify" / "correction_log.jsonl"
    if not path.is_file():
        return grouped
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith('{"header"'):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = str(entry.get("record_id") or "")
        if rid:
            grouped.setdefault(rid, []).append(entry)
    return grouped


def _format_corrections(items: list[dict[str, Any]]) -> str:
    """字段级修正建议 → 「field：原值 → 建议值（原因）」拼接串。"""
    segs: list[str] = []
    for c in items:
        field = str(c.get("field") or "")
        original = _text(c.get("original")) or "（空）"
        suggested = _text(c.get("suggested"))
        reason = _text(c.get("reason"))
        seg = f"{field}：{original} → {suggested}"
        if reason:
            seg += f"（{reason}）"
        segs.append(seg)
    return "；".join(segs)


def _load_verify_results_from_cache(
    book_dir: Path, records_by_id: dict[str, SourceRecord]
) -> dict[str, dict[str, Any]]:
    """[回退路径] 合并 06_verify/cache/*.json；旧书无权威工件时使用。

    注意：跨运行 record_id 会重排，此路径可能注入陈旧 verdict，仅在缺少
    verification_results.jsonl 时作为兜底。
    """
    results: dict[str, dict[str, Any]] = {}
    cache_dir = book_dir / "06_verify" / "cache"
    if not cache_dir.is_dir():
        return results
    for path in sorted(cache_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        response = data.get("response")
        if not isinstance(response, dict):
            continue
        items = (
            response.get("records")
            or response.get("verifications")
            or response.get("results")
            or []
        )
        for item in items:
            if not (isinstance(item, dict) and item.get("record_id")):
                continue
            rid = str(item["record_id"])
            correction = ""
            corrected = item.get("corrected_record")
            record = records_by_id.get(rid)
            if isinstance(corrected, dict) and record is not None:
                correction = _correction_diff(record, corrected)
            results[rid] = {
                "verdict": str(item.get("verdict") or ""),
                "evidence": _text(item.get("source_evidence")),
                "correction": correction,
            }
    return results


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _render_section_path(parts: list[str]) -> str:
    """渲染章节路径：折叠被 MinerU 误当作标题的整段表格 HTML（原表内容污染 heading）。"""
    clean: list[str] = []
    for raw in parts:
        seg = str(raw).strip()
        if not seg:
            continue
        lowered = seg.lower()
        if "<" in seg and ("table" in lowered or "<td" in lowered or "<tr" in lowered):
            seg = "（表格）"  # 整段表格 HTML：压成占位，避免在路径列刷屏
        elif "<" in seg:
            seg = _HTML_TAG_RE.sub("", seg).strip()
        if seg and (not clean or clean[-1] != seg):
            clean.append(seg)
    return " / ".join(clean)


def _load_validation_issues(book_dir: Path) -> dict[str, list[str]]:
    report_path = book_dir / "05_validate" / "validation_report.json"
    if not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    issues: dict[str, list[str]] = {}
    for issue in report.get("issues", []):
        record_id = str(issue.get("record_id") or "")
        if not record_id:
            continue
        severity = str(issue.get("severity") or "")
        category = str(issue.get("category") or "")
        message = str(issue.get("message") or "")
        issues.setdefault(record_id, []).append(f"[{severity}][{category}] {message}")
    return issues


def _build_rows(
    records: list[SourceRecord],
    verify_results: dict[str, dict[str, Any]],
    validation_issues: dict[str, list[str]],
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for record in records:
        verify = verify_results.get(record.record_id)
        if verify is not None:
            verdict_key = str(verify.get("verdict") or "")
            verdict_label = _pack_labels("verdict").get(verdict_key, verdict_key)
            evidence = _text(verify.get("evidence"))
            correction = _text(verify.get("correction"))
        elif record.passthrough or record.record_type == "TABLE_ROW":
            verdict_label = _pack_labels("verdict")["PASSTHROUGH"]  # 表格直通：免 B 核验
            evidence = ""
            correction = ""
        else:
            verdict_label = "未核验"
            evidence = ""
            correction = ""
        problems = validation_issues.get(record.record_id, [])
        action = record.payload.get("action")
        action_label = (
            _pack_labels("action", record.domain_id).get(action, action) if action else ""
        )
        rows.append(
            [
                record.record_id,
                _pack_labels("record_type", record.domain_id).get(
                    record.record_type, record.record_type
                ),
                _subject(record),
                _related_herbs(record),
                action_label,
                _structured_content(record),
                record.raw_text,
                _render_section_path(record.source.section_path),
                (
                    f"{record.source.page_start}-{record.source.page_end}"
                    if record.source.page_start
                    else ""
                ),
                verdict_label,
                correction,
                evidence,
                "\n".join(problems),
                "是" if record.needs_review else "",
                "",
                "",
            ]
        )
    return rows


def _style_review_sheet(ws: Worksheet) -> None:
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    widths = [18, 10, 14, 24, 10, 40, 60, 22, 8, 12, 30, 40, 32, 8, 10, 20]
    for idx, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = width

    doctor_cols = (15, 16)  # 审核结论 / 医生备注
    verdict_col = 10
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        for col in doctor_cols:
            row[col - 1].fill = DOCTOR_FILL
        verdict = str(row[verdict_col - 1].value or "")
        if verdict in {"未获原文支持", "名称规范化有误", "关系抽取有误", "核验缺失", "未核验"}:
            row[verdict_col - 1].fill = WARN_FILL

    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions

    dropdown = DataValidation(
        type="list",
        formula1='"保留,修改,删除,待定"',
        allow_blank=True,
        showDropDown=False,
    )
    ws.add_data_validation(dropdown)
    dropdown.add(f"O2:O{ws.max_row}")


def _append_stats_sheet(
    ws: Worksheet,
    records: list[SourceRecord],
    verify_results: dict[str, dict[str, Any]],
    book_dir: Path,
) -> None:
    settings = get_settings()
    type_counts = Counter(r.record_type for r in records)
    # 核验结论以权威 per-run 工件为准；passthrough/表格直通记为「直通免核验」，
    # 不再因缓存未覆盖而虚报「未核验」，也不再把陈旧 verdict 计入。
    verdict_keys: list[str] = []
    for r in records:
        v = verify_results.get(r.record_id)
        if v is not None and v.get("verdict"):
            verdict_keys.append(str(v["verdict"]))
        elif r.passthrough or r.record_type == "TABLE_ROW":
            verdict_keys.append("PASSTHROUGH")
    verdict_counts = Counter(_pack_labels("verdict").get(k, k) for k in verdict_keys)
    covered = len(verdict_keys)
    unverified = len(records) - covered
    validation_path = book_dir / "05_validate" / "validation_report.json"
    validation = (
        json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
    )
    severity_counts = Counter(i.get("severity") for i in validation.get("issues", []))
    manifest_path = book_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    cache_count = len(list((book_dir / "06_verify" / "cache").glob("*.json")))
    verify_authoritative = (book_dir / "06_verify" / "verification_results.jsonl").is_file()
    suspicious_path = book_dir / "05_validate" / "suspicious_records.jsonl"
    suspicious_count = 0
    if suspicious_path.is_file():
        suspicious_count = sum(
            1
            for line in suspicious_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith('{"header"')
        )

    rows = [
        ("统计项", "数值", "说明"),
        ("源文件", manifest.get("file_name") or "", "书档来源 PDF"),
        (
            "流水线状态",
            manifest.get("status") or "",
            "核验结论来源："
            + (
                "verification_results.jsonl（权威 per-run）"
                if verify_authoritative
                else "06_verify/cache 合并（旧书回退，可能含陈旧 verdict）"
            ),
        ),
        (
            "模型",
            f"extract={settings.extract_llm.model} / verify={settings.verify_llm.model}",
            "抽取与核验分别配置（可不同）",
        ),
        ("总记录数", len(records), "04_extract/records.jsonl"),
        *(
            (
                "类型 "
                + _pack_labels("record_type", records[0].domain_id if records else "tcm").get(k, k),
                v,
                "",
            )
            for k, v in sorted(type_counts.items())
        ),
        ("核验覆盖", f"{covered}/{len(records)}", f"未核验 {unverified} 条"),
        *(("核验 " + k, v, "") for k, v in sorted(verdict_counts.items(), key=lambda kv: -kv[1])),
        ("校验问题总数", sum(severity_counts.values()), "05_validate/validation_report.json"),
        *(("校验 " + k, v, "") for k, v in sorted(severity_counts.items())),
        (
            "疑点记录（validate）",
            suspicious_count,
            "05_validate/suspicious_records.jsonl（ERROR 或语义升级等）",
        ),
        ("verify 缓存批次", cache_count, "06_verify/cache"),
    ]
    for row in rows:
        ws.append(row)
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 44
    ws.column_dimensions["C"].width = 42
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def _append_guide_sheet(ws: Worksheet) -> None:
    rows = [
        ("填写规则", ""),
        ("1", "本表每行是一条抽取记录；黄色两列（审核结论、医生备注）由医生填写，其余列请勿修改。"),
        (
            "2",
            "审核结论下拉选项：保留=抽取正确；修改=部分正确（请在医生备注说明，或直接修改“结构化内容/药物主体”等列）；删除=原文不支持该记录；待定=无法判断。",
        ),
        (
            "3",
            "粉色底“核验结论”列是模型 B 逐条比对原文的结果：已证实=原文可支撑；未获原文支持/名称规范化有误/关系抽取有误=建议重点复核；直通免核验=表格逐行透传不经核验；未核验=本次运行核验未完成。",
        ),
        (
            "4",
            "“B建议修正”列是核验模型给出的修正建议（字段：原值 → 新值），仅供参考，采纳与否由医生决定。",
        ),
        (
            "5",
            "“校验问题”列是确定性校验器输出的 ERROR/WARNING（如缺少结构化载荷、动作超原文上限），其中 ERROR 建议优先处理。",
        ),
        ("6", "证据原文为抽取依据的原文片段；如与原书不符请标“修改”并在备注说明。"),
        ("", ""),
        ("列说明", ""),
        ("类型", "一般规则/配伍禁忌（十八反十九畏）/妊娠禁忌/特殊人群/剂量规范/特殊用法。"),
        ("药物/主体", "记录的核心对象：药名、配伍组锚点药、人群名或规则主语。"),
        ("动作强度", "禁用/忌用不宜/慎用/需监测/提示（与原文措辞对齐，禁止升级）。"),
        ("结构化内容", "按类型压平的载荷：配伍组成员、妊娠限制级别与原因、人群条件、剂量区间等。"),
    ]
    for row in rows:
        ws.append(row)
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 100
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def _append_raw_table_sheet(ws: Worksheet, blocks: list[dict]) -> None:
    """把原表版式原样写入合并 sheet（每表一段：标题行+表头+数据行）。"""
    ws.append(["章节", "页码", "表ID", "行", *[f"列{i + 1}" for i in range(8)]])
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    for block in blocks:
        content = str(block.get("content") or "")
        if "<drawing" not in content:
            continue
        parsed = parse_table(content)
        if not parsed.rows:
            continue
        heading = str(block.get("heading") or "")
        positions = block.get("positions") or []
        try:
            page = int(str(positions[0].get("anchor"))) if positions else None
        except (ValueError, TypeError):
            page = None
        tid = drawing_id(content)
        ws.append([heading, page, tid, "表头", *parsed.headers])
        for ridx, row in enumerate(parsed.rows, 1):
            ws.append(["", "", "", f"数据{ridx}", *row])
        ws.append([])
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 8
    ws.column_dimensions["C"].width = 22
    ws.column_dimensions["D"].width = 8


def export_review_workbook(book_dir: Path, output: Path | None = None) -> Path:
    records = load_artifact_records(book_dir / "04_extract" / "records.jsonl", SourceRecord)
    if not records:
        raise ValueError(f"未在 {book_dir} 找到抽取记录")
    verify_results = _load_verify_results(book_dir, {r.record_id: r for r in records})
    validation_issues = _load_validation_issues(book_dir)

    rows = _build_rows(records, verify_results, validation_issues)

    wb = Workbook()
    ws_review = wb.active
    ws_review.title = REVIEW_SHEET
    ws_review.append(REVIEW_HEADERS)
    for row in rows:
        ws_review.append(row)
    _style_review_sheet(ws_review)

    ws_stats = wb.create_sheet(STATS_SHEET)
    _append_stats_sheet(ws_stats, records, verify_results, book_dir)

    ws_guide = wb.create_sheet(GUIDE_SHEET)
    _append_guide_sheet(ws_guide)

    blocks_path = book_dir / "01_parse" / "blocks.jsonl"
    if blocks_path.is_file():
        ws_raw = wb.create_sheet(RAW_TABLE_SHEET)
        _append_raw_table_sheet(ws_raw, load_blocks(blocks_path))

    if output is None:
        manifest = json.loads((book_dir / "manifest.json").read_text(encoding="utf-8"))
        stem = Path(manifest.get("file_name") or book_dir.name).stem
        output = book_dir / "07_review" / f"{stem}_医生审核版.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 P0 流水线工件的医生审核版 Excel")
    parser.add_argument("--book-id", required=True, help="书籍工作区 ID（workspace/books/<id>）")
    parser.add_argument("--workspace", default=str(WORKSPACE), help="书籍工作区根目录")
    parser.add_argument(
        "--output", default=None, help="输出 xlsx 路径（默认 07_review/<源文件名>_医生审核版.xlsx）"
    )
    args = parser.parse_args()

    book_dir = Path(args.workspace) / args.book_id
    if not book_dir.is_dir():
        raise FileNotFoundError(f"书籍工作区不存在: {book_dir}")
    output = export_review_workbook(book_dir, Path(args.output) if args.output else None)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
