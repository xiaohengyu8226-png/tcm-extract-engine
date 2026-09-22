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
from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.worksheet import Worksheet

from tcm_extract_engine.core.artifacts import load_artifact_records
from tcm_extract_engine.core.records import SourceRecord

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspace" / "books"

REVIEW_SHEET = "审核记录表"
STATS_SHEET = "统计汇总"
GUIDE_SHEET = "填写说明"

TYPE_LABELS: dict[str, str] = {
    "GENERAL_RULE": "一般规则",
    "HERB_TABLE": "药物条目",
    "INTERACTION": "配伍禁忌",
    "PREGNANCY": "妊娠禁忌",
    "DOSAGE": "剂量规范",
    "USAGE": "特殊用法",
    "POPULATION": "特殊人群",
}

ACTION_LABELS: dict[str, str] = {
    "PROHIBIT": "禁用",
    "AVOID": "忌用/不宜",
    "CAUTION": "慎用",
    "MONITOR": "需监测",
    "INFO": "提示",
}

VERDICT_LABELS: dict[str, str] = {
    "SUPPORTED": "已证实",
    "PARTIAL": "部分证实",
    "UNSUPPORTED": "未获原文支持",
    "WRONG_NORMALIZATION": "名称规范化有误",
    "WRONG_RELATION": "关系抽取有误",
    "MISSING": "核验缺失",
}

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


def _dose_text(record: SourceRecord) -> str:
    dose = record.dose
    if dose is None:
        return ""
    parts: list[str] = []
    if dose.min_dose is not None and dose.max_dose is not None:
        parts.append(f"{dose.min_dose:g}~{dose.max_dose:g}{dose.unit}")
    elif dose.max_dose is not None:
        parts.append(f"≤{dose.max_dose:g}{dose.unit}")
    elif dose.min_dose is not None:
        parts.append(f"≥{dose.min_dose:g}{dose.unit}")
    if dose.per_use:
        parts.append("（单次剂量）")
    if dose.frequency:
        parts.append(f"频次：{dose.frequency}")
    if dose.special_note:
        parts.append(f"备注：{dose.special_note}")
    return "；".join(parts)


def _structured_content(record: SourceRecord) -> str:
    """把类型化载荷压平成医生可读的一列文本。"""
    parts: list[str] = []
    if record.interaction is not None:
        group = record.interaction
        head = group.interaction_type
        if group.group_name:
            head += f"（{group.group_name}）"
        anchor = group.anchor.herb_name + (f"（{group.anchor.processed_form}）" if group.anchor.processed_form else "")
        members = "、".join(
            m.herb_name + (f"（{m.processed_form}）" if m.processed_form else "") for m in group.members
        )
        relation = f"{anchor} 畏 {members}" if group.direction else f"{anchor} 反/畏 {members}"
        parts.append(f"{head}：{relation}")
    if record.pregnancy is not None:
        rule = record.pregnancy
        targets = "、".join([*rule.target_herbs, *rule.target_formulas])
        seg = f"限制级别：{rule.restriction_level}"
        if targets:
            seg += f"；对象：{targets}"
        if rule.reason:
            seg += f"；原因：{rule.reason}"
        parts.append(seg)
    if record.population is not None:
        rule = record.population
        seg = f"人群：{rule.population}；限制：{rule.restriction_level}"
        if rule.target_herbs:
            seg += f"；药物：{'、'.join(rule.target_herbs)}"
        if rule.condition_text:
            seg += f"；条件：{rule.condition_text}"
        parts.append(seg)
    dose = _dose_text(record)
    if dose:
        parts.append(dose)
    if record.toxicity_level:
        parts.append(f"毒性：{record.toxicity_level}")
    if record.record_type == "USAGE" and record.attributes.get("usage_note"):
        parts.append(_text(record.attributes["usage_note"]))
    elif record.attributes:
        extras = {k: v for k, v in record.attributes.items() if k != "usage_note"}
        if extras:
            note = "；".join(f"{k}={_text(v)}" for k, v in extras.items() if _text(v))
            if note:
                parts.append(note[:300])
    return "；".join(parts)


def _subject(record: SourceRecord) -> str:
    if record.herb_name:
        return record.herb_name
    if record.interaction is not None:
        return record.interaction.anchor.herb_name
    if record.pregnancy is not None:
        return "、".join([*record.pregnancy.target_herbs, *record.pregnancy.target_formulas])
    if record.population is not None:
        return record.population.population
    return _text(record.subject_text)


def _related_herbs(record: SourceRecord) -> str:
    if record.interaction is not None:
        return "、".join(m.herb_name for m in record.interaction.members)
    if record.pregnancy is not None:
        return "、".join(record.pregnancy.target_herbs)
    if record.population is not None:
        return "、".join(record.population.target_herbs)
    return ""


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


def _load_verify_results(book_dir: Path) -> dict[str, dict[str, Any]]:
    """合并 verify 缓存；兼容 records/verifications/results 键名。"""
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
        items = response.get("records") or response.get("verifications") or response.get("results") or []
        for item in items:
            if isinstance(item, dict) and item.get("record_id"):
                results[str(item["record_id"])] = item
    return results


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
        verdict_label = ""
        correction = ""
        evidence = ""
        if verify is not None:
            verdict_label = VERDICT_LABELS.get(str(verify.get("verdict") or ""), str(verify.get("verdict") or ""))
            evidence = _text(verify.get("source_evidence"))
            corrected = verify.get("corrected_record")
            if isinstance(corrected, dict):
                correction = _correction_diff(record, corrected)
        else:
            verdict_label = "未核验"
        problems = validation_issues.get(record.record_id, [])
        action_label = ACTION_LABELS.get(record.action.value, record.action.value) if record.action else ""
        rows.append(
            [
                record.record_id,
                TYPE_LABELS.get(record.record_type.value, record.record_type.value),
                _subject(record),
                _related_herbs(record),
                action_label,
                _structured_content(record),
                record.raw_text,
                " / ".join(record.source.section_path),
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


def _append_stats_sheet(ws: Worksheet, records: list[SourceRecord], verify_results: dict[str, dict[str, Any]], book_dir: Path) -> None:
    type_counts = Counter(r.record_type.value for r in records)
    verdict_counts = Counter(
        VERDICT_LABELS.get(str(v.get("verdict") or ""), str(v.get("verdict") or ""))
        for v in verify_results.values()
    )
    unverified = sum(1 for r in records if r.record_id not in verify_results)
    validation = json.loads((book_dir / "05_validate" / "validation_report.json").read_text(encoding="utf-8"))
    severity_counts = Counter(i.get("severity") for i in validation.get("issues", []))
    manifest = json.loads((book_dir / "manifest.json").read_text(encoding="utf-8"))
    cache_count = len(list((book_dir / "06_verify" / "cache").glob("*.json")))
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
        ("流水线状态", manifest.get("status") or "", "verify 因 API 配额中断（137/146 批次已缓存）"),
        ("模型", "deepseek-v4-flash-202605", "extract / verify 共用"),
        ("总记录数", len(records), "04_extract/records.jsonl"),
        *(("类型 " + TYPE_LABELS.get(k, k), v, "") for k, v in sorted(type_counts.items())),
        ("核验覆盖", f"{len(verify_results)}/{len(records)}", f"未核验 {unverified} 条（verify 中断）"),
        *(("核验 " + k, v, "") for k, v in sorted(verdict_counts.items(), key=lambda kv: -kv[1])),
        ("校验问题总数", sum(severity_counts.values()), "05_validate/validation_report.json"),
        *(("校验 " + k, v, "") for k, v in sorted(severity_counts.items())),
        ("疑点记录（validate）", suspicious_count, "05_validate/suspicious_records.jsonl（ERROR 或语义升级等）"),
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
        ("2", "审核结论下拉选项：保留=抽取正确；修改=部分正确（请在医生备注说明，或直接修改“结构化内容/药物主体”等列）；删除=原文不支持该记录；待定=无法判断。"),
        ("3", "粉色底“核验结论”列是模型 B 逐条比对原文的结果：已证实=原文可支撑；未获原文支持/名称规范化有误/关系抽取有误=建议重点复核；未核验=本次运行核验未完成。"),
        ("4", "“B建议修正”列是核验模型给出的修正建议（字段：原值 → 新值），仅供参考，采纳与否由医生决定。"),
        ("5", "“校验问题”列是确定性校验器输出的 ERROR/WARNING（如缺少结构化载荷、动作超原文上限），其中 ERROR 建议优先处理。"),
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


def export_review_workbook(book_dir: Path, output: Path | None = None) -> Path:
    records = load_artifact_records(book_dir / "04_extract" / "records.jsonl", SourceRecord)
    if not records:
        raise ValueError(f"未在 {book_dir} 找到抽取记录")
    verify_results = _load_verify_results(book_dir)
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
    parser.add_argument("--output", default=None, help="输出 xlsx 路径（默认 07_review/<源文件名>_医生审核版.xlsx）")
    args = parser.parse_args()

    book_dir = Path(args.workspace) / args.book_id
    if not book_dir.is_dir():
        raise FileNotFoundError(f"书籍工作区不存在: {book_dir}")
    output = export_review_workbook(book_dir, Path(args.output) if args.output else None)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
