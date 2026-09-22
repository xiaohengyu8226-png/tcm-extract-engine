"""逐步控制抽取 Web API：把 BookPipeline 的 8 阶段映射成交互式工作流。

设计要点：
  - manifest.json 是唯一权威状态；任务列表实时扫描 workspace/books 得出，
    无需额外持久化，服务重启后自动恢复。
  - 每个「步骤」= 调用 ``BookPipeline.run_stage`` 推进一个（或一组）引擎阶段，
    阶段产物落盘后前端读取展示。
  - 步骤执行放后台协程，前端轮询 ``/api/interactive/jobs`` 获取进度。

步骤 ↔ 引擎阶段映射：
  parse            → parse
  generate-prompt  → profile + plan（确定性，止于 WAITING_PLAN_CONFIRMATION）
  confirm(prompt)  → confirm_plan
  extract          → extract
  compare          → 只读视图（records vs blocks）
  verify           → validate + verify
  export           → 生成医生审核版 Excel + 人工审核门放行（REVIEWED → PUBLISHED）
  workflow/back    → 回退一个 UI 步骤（仅回退状态；阶段记录与产物全部保留，重新X 才作废）
  workflow/retry-* → 单步重跑（rewind 作废该步及下游记录后只跑该步，不自动续跑下游）
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from tcm_extract_engine.cab.planner import PromptTemplateRegistry
from tcm_extract_engine.cab.profiler import CONTENT_TYPE_TO_EXTRACTOR, load_blocks
from tcm_extract_engine.cab.validator import Severity
from tcm_extract_engine.cab.verifier import CorrectionEntry
from tcm_extract_engine.common.paths import project_paths
from tcm_extract_engine.core.artifacts import (
    ArtifactHeader,
    load_artifact,
    load_artifact_records,
    save_artifact,
)
from tcm_extract_engine.core.book import BookWorkspace, list_books
from tcm_extract_engine.core.manifest import BookManifest, BookStatus
from tcm_extract_engine.core.records import SourceRecord
from tcm_extract_engine.runner import BookPipeline

logger = logging.getLogger(__name__)

BOOKS_ROOT = project_paths.root / "workspace" / "books"
UPLOAD_TMP = project_paths.root / "workspace" / "uploads"
GEPA_ROOT = project_paths.gepa
STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

_pipeline: BookPipeline | None = None
# 运行期临时进度（非权威）：book_id → {message, step_started_at}
_runtime: dict[str, dict[str, Any]] = {}
# 后台任务强引用，防止被 GC
_background: set[asyncio.Task[None]] = set()

# 引擎阶段名（与 UI 步骤的映射见模块 docstring）
_RUNNING_STAGES = frozenset({"parsing", "generating_prompt", "extracting", "verifying"})

# manifest 状态 → UI 步骤（与 _view_stage 的 done 集合派生互为兼容）
_STATUS_TO_VIEW: dict[str, str] = {
    "PENDING": "uploaded",
    "PARSING": "parsing",
    "PARSED": "parsed",
    "PROFILING": "generating_prompt",
    "PROFILED": "generating_prompt",
    "PLAN_GENERATED": "generating_prompt",
    "WAITING_PLAN_CONFIRMATION": "prompt_ready",
    "PLAN_CONFIRMED": "prompt_ready",
    "EXTRACTING": "extracting",
    "EXTRACTED": "extracted",
    "VALIDATING": "verifying",
    "VALIDATED": "verifying",
    "VERIFYING": "verifying",
    "VERIFIED": "verified",
    "WAITING_REVIEW": "verified",
    "REVIEWED": "done",
    "PUBLISHED": "done",
}


def _pipe() -> BookPipeline:
    global _pipeline
    if _pipeline is None:
        BOOKS_ROOT.mkdir(parents=True, exist_ok=True)
        _pipeline = BookPipeline(BOOKS_ROOT)
    return _pipeline


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    if _pipeline is not None:
        await _pipeline.close()


app = FastAPI(title="中医智能审方系统 · 逐步控制抽取", version="0.1.0", lifespan=_lifespan)


# ── 状态派生 ────────────────────────────────────────────────────────────────


def _manifest(book_id: str) -> BookManifest:
    ws = BookWorkspace(BOOKS_ROOT, book_id)
    if not ws.manifest_path.is_file():
        raise HTTPException(404, f"任务不存在: {book_id}")
    return ws.load_manifest()


def _view_stage(m: BookManifest) -> str:
    """把 manifest 状态机映射成 UI 的 workflow_stage。

    以当前状态为准（「返回上一步」只回退状态、不作废阶段记录，
    若按 completed_stages 取最远步骤会把回退吃掉）；
    状态不在映射表时（如 FAILED）退回按已完成阶段派生。
    """
    done = set(m.completed_stages())
    s = m.status.value
    view = _STATUS_TO_VIEW.get(s)
    if view is not None:
        return view
    if "verify" in done:
        return "verified"
    if s == "VERIFYING":
        return "verifying"
    if "extract" in done:
        return "extracted"
    if s == "EXTRACTING":
        return "extracting"
    if "plan" in done:
        return "prompt_ready"
    if s in ("PROFILING", "PROFILED", "PLAN_GENERATED"):
        return "generating_prompt"
    if "parse" in done:
        return "parsed"
    if s == "PARSING":
        return "parsing"
    return "uploaded"


def _prompt_confirmed(m: BookManifest) -> bool:
    order = list(BookStatus)
    return order.index(m.status) >= order.index(BookStatus.PLAN_CONFIRMED)


def _job_view(ws_book_id: str, m: BookManifest) -> dict[str, Any]:
    rt = _runtime.get(ws_book_id, {})
    extract = m.stages.get("extract")
    parse = m.stages.get("parse")
    issues = 0
    validate_path = BookWorkspace(BOOKS_ROOT, ws_book_id).stage_artifact("validate")
    if validate_path.is_file():
        try:
            issues = len(load_artifact(validate_path).get("issues", []))
        except Exception:  # noqa: BLE001 - 坏报告不阻断任务列表
            issues = 0
    return {
        "job_id": m.book_id,
        "filename": m.file_name,
        "workflow_stage": _view_stage(m),
        "status": m.status.value,
        "message": rt.get("message") or (m.error or ""),
        "chunks_count": parse.records if parse and parse.records is not None else 0,
        "records_count": extract.records if extract and extract.records is not None else 0,
        "verification_issues": issues,
        "prompt_confirmed": _prompt_confirmed(m),
        "created_at": m.created_at,
        "updated_at": m.updated_at,
        "step_started_at": rt.get("step_started_at", 0),
        "error": m.error,
    }


# ── 后台步骤执行 ─────────────────────────────────────────────────────────────


def _spawn(book_id: str, coro: Any) -> None:
    async def _runner() -> None:
        try:
            await coro
        except Exception as exc:  # noqa: BLE001 - 失败已落 manifest，这里只记运行期消息
            _runtime.setdefault(book_id, {})["message"] = f"步骤失败: {exc}"
            logger.exception("[Web %s] 步骤异常", book_id)
        finally:
            _runtime.get(book_id, {}).pop("step_started_at", None)

    task = asyncio.create_task(_runner())
    _background.add(task)
    task.add_done_callback(_background.discard)


def _mark_running(book_id: str, message: str) -> None:
    rt = _runtime.setdefault(book_id, {})
    rt["message"] = message
    rt["step_started_at"] = time.time()


async def _bg_parse(book_id: str) -> None:
    await _pipe().run_stage(book_id, "parse")


async def _bg_plan(book_id: str) -> None:
    pipe = _pipe()
    await pipe.run_stage(book_id, "profile")
    await pipe.run_stage(book_id, "plan", confirm_plan=False)


async def _bg_extract(book_id: str) -> None:
    await _pipe().run_stage(book_id, "extract")


async def _bg_verify(book_id: str) -> None:
    pipe = _pipe()
    await pipe.run_stage(book_id, "validate")
    await pipe.run_stage(book_id, "verify")


def _guard_not_running(book_id: str) -> None:
    m = _manifest(book_id)
    if _view_stage(m) in _RUNNING_STAGES:
        raise HTTPException(409, "当前步骤正在执行，请完成后再操作")


# ── 路由：页面 & 任务列表 ─────────────────────────────────────────────────────


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/interactive/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    """扫描 workspace/books 列出全部任务（manifest 为权威状态）。"""
    views: list[dict[str, Any]] = []
    for ws in list_books(BOOKS_ROOT):
        try:
            m = ws.load_manifest()
        except Exception as exc:  # noqa: BLE001 - 坏 manifest 跳过但记日志
            logger.warning("[Web] %s manifest 读取失败: %s", ws.book_id, exc)
            continue
        views.append(_job_view(ws.book_id, m))
    views.sort(key=lambda v: v.get("updated_at", ""), reverse=True)
    return views


# ── 路由：上传 ────────────────────────────────────────────────────────────────


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    relative_path: str | None = Form(None),
    force: bool = Query(False),
) -> dict[str, Any]:
    """保存上传文件并建立书籍工作目录（幂等：同 hash 复用）。"""
    if not file.filename:
        raise HTTPException(400, "请上传文件")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".md", ".markdown"}:
        raise HTTPException(400, f"不支持的文件类型: {suffix or '未知'}")

    UPLOAD_TMP.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    dest = UPLOAD_TMP / safe_name
    if dest.exists() and not force:
        dest.unlink(missing_ok=True)
    with dest.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            fh.write(chunk)
            if fh.tell() > MAX_UPLOAD_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "文件超过大小限制（512MB）")

    try:
        ws = BookWorkspace.create(BOOKS_ROOT, dest)
    finally:
        # 源文件已被 create() 复制进 00_source，删除临时副本
        if dest.is_file():
            dest.unlink(missing_ok=True)

    m = ws.load_manifest()
    return {"job_id": m.book_id, "status": "uploaded", "filename": m.file_name}


# ── 路由：步骤 1 解析 ─────────────────────────────────────────────────────────


@app.post("/api/jobs/{job_id}/step/parse")
async def step_parse(job_id: str) -> dict[str, str]:
    _guard_not_running(job_id)
    _mark_running(job_id, "正在解析（PDF 走 MinerU，Markdown 直转）…")
    _spawn(job_id, _bg_parse(job_id))
    return {"status": "started", "stage": "parsing"}


@app.get("/api/jobs/{job_id}/parsed")
async def get_parsed(job_id: str, max_chars: int = Query(100_000, ge=1)) -> dict[str, Any]:
    m = _manifest(job_id)
    if "parse" not in m.completed_stages():
        raise HTTPException(400, "尚未解析")
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    blocks = load_blocks(ws.stage_dir("parse") / "blocks.jsonl")
    parts: list[str] = []
    total = 0
    for b in blocks:
        text = str(b.get("content") or "").strip()
        if total + len(text) > max_chars:
            parts.append("...（后续内容因长度限制省略）")
            break
        parts.append(f"[{b.get('blockid')}]\n{text}")
        total += len(text)
    return {
        "text": "\n\n".join(parts),
        "chunks": [
            {
                "chunk_id": str(b.get("blockid") or ""),
                "content": str(b.get("content") or ""),
                "content_length": len(str(b.get("content") or "")),
                "semantic_type": str(b.get("type") or "content"),
            }
            for b in blocks
        ],
        "chunks_count": len(blocks),
    }


# ── 路由：步骤 2 抽取方案（C：profile + plan）─────────────────────────────────


@app.post("/api/jobs/{job_id}/step/generate-prompt")
async def step_generate_prompt(job_id: str) -> dict[str, str]:
    m = _manifest(job_id)
    if "parse" not in m.completed_stages():
        raise HTTPException(400, "请先完成解析")
    _guard_not_running(job_id)
    _mark_running(job_id, "正在生成抽取方案（画像 + 章节规划）…")
    _spawn(job_id, _bg_plan(job_id))
    return {"status": "started", "stage": "generating_prompt"}


@app.get("/api/jobs/{job_id}/schema")
async def get_schema(job_id: str) -> dict[str, Any]:
    """把抽取计划映射成前端「Schema/Sheet」视图。"""
    _manifest(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    plan_path = ws.stage_artifact("plan")
    if not plan_path.is_file():
        raise HTTPException(400, "尚未生成抽取方案")
    plan = load_artifact(plan_path)
    sections = plan.get("sections", [])
    planned = [s for s in sections if not s.get("skip_reason")]
    registry = PromptTemplateRegistry()
    templates_dir = registry.prompts_root / "a"
    return {
        "schema": {"sections": sections},
        "domain": plan.get("goal", ""),
        "explanation": f"文档类型：{plan.get('document_type', '')}",
        "book_type": plan.get("document_type", ""),
        "confidence": 1.0,
        "structure": [s.get("title", "") for s in sections],
        "recommended_sheets": [
            {
                "name": s.get("template", ""),
                "section_id": s.get("section_id", ""),
                "title": s.get("title", ""),
                "extractor": s.get("extractor", ""),
                "prompt_version": s.get("prompt_version", ""),
                "skip_reason": s.get("skip_reason"),
            }
            for s in planned
        ],
        # 蓝图人工编辑：可选抽取器/模板 + 全章节可编辑视图
        "available_extractors": sorted({v for v in CONTENT_TYPE_TO_EXTRACTOR.values() if v}),
        "available_templates": sorted(p.name for p in templates_dir.glob("*.md")),
        "editable_sections": [
            {
                "section_id": s.get("section_id", ""),
                "title": s.get("title", ""),
                "extractor": s.get("extractor", ""),
                "template": s.get("template", ""),
                "skip": bool(s.get("skip_reason")),
            }
            for s in sections
        ],
        "qa_rules": plan.get("risk_points", []),
        "total_sections": len(sections),
        "planned_sections": len(planned),
    }


@app.put("/api/jobs/{job_id}/schema")
async def update_schema(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """人工编辑抽取蓝图：改章节的抽取器/模板/跳过标记，重写 plan 产物。

    已确认且抽过取的书：下游 extract 及之后记录作废（产物保留），需重新抽取。
    """
    m = _manifest(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    plan_path = ws.stage_artifact("plan")
    if not plan_path.is_file():
        raise HTTPException(400, "尚未生成抽取方案")
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    sections = raw.get("sections", [])
    edits = {e.get("section_id"): e for e in body.get("sections", []) if e.get("section_id")}
    registry = PromptTemplateRegistry()
    valid_ext = {v for v in CONTENT_TYPE_TO_EXTRACTOR.values() if v}
    valid_tpl = {p.name for p in (registry.prompts_root / "a").glob("*.md")}
    for s in sections:
        e = edits.get(s.get("section_id"))
        if not e:
            continue
        ext = str(e.get("extractor") or "")
        tpl = str(e.get("template") or "")
        if ext and ext not in valid_ext:
            raise HTTPException(400, f"未知抽取器: {ext}")
        if tpl and tpl not in valid_tpl:
            raise HTTPException(400, f"未知模板: {tpl}")
        if e.get("skip"):
            s["skip_reason"] = "人工标记不抽取"
            s["extractor"] = ""
            s["template"] = ""
            s["prompt_version"] = ""
        else:
            if not ext or not tpl:
                raise HTTPException(400, f"章节 {s.get('section_id')} 取消跳过后必须同时指定抽取器与模板")
            s["skip_reason"] = None
            s["extractor"] = ext
            s["template"] = tpl
            s["prompt_version"] = registry.version_of("a", tpl)
    raw["sections"] = sections
    raw.setdefault("header", {}).setdefault("extra", {})["plan_edited_at"] = (
        datetime.now(timezone.utc).isoformat()
    )
    plan_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    if _prompt_confirmed(m) and "extract" in m.completed_stages():
        _pipe().rewind(job_id, to_stage="extract")
    return await get_schema(job_id)


@app.get("/api/jobs/{job_id}/prompt")
async def get_prompt(job_id: str) -> dict[str, Any]:
    """返回本方案选用的 A 模板正文（引擎不允许自由生成 Prompt）。"""
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    plan_path = ws.stage_artifact("plan")
    if not plan_path.is_file():
        raise HTTPException(400, "尚未生成抽取方案")
    plan = load_artifact(plan_path)
    registry = PromptTemplateRegistry()
    names = sorted({s["template"] for s in plan.get("sections", []) if s.get("template")})
    blocks: list[str] = []
    for name in names:
        try:
            blocks.append(f"# ── {name} ──\n" + registry.load("a", name).content)
        except FileNotFoundError:
            blocks.append(f"# ── {name}（缺失）──")
    return {
        "prompt": "\n\n".join(blocks),
        "templates": names,
        "editable": False,
        "note": "引擎采用固定模板库（防 Prompt Drift）；如需调整请改 prompts/a/* 或走 GEPA 晋升。",
    }


@app.get("/api/jobs/{job_id}/meta-prompt")
async def get_meta_prompt(job_id: str) -> dict[str, Any]:
    registry = PromptTemplateRegistry()
    try:
        planner = registry.load("c", "planner.md").content
    except FileNotFoundError:
        planner = ""
    m = _manifest(job_id)
    return {
        "system_prompt": planner,
        "user_prompt": "",
        "generated_prompt": "",
        "prompt_version": m.prompt_version or "",
    }


@app.get("/api/jobs/{job_id}/prompt-versions")
async def get_prompt_versions(job_id: str) -> list[dict[str, Any]]:
    m = _manifest(job_id)
    if not m.prompt_version:
        return []
    return [
        {
            "version": m.prompt_version,
            "note": "当前抽取方案所用模板版本合成 hash",
            "created_at": m.updated_at,
        }
    ]


@app.put("/api/jobs/{job_id}/prompt")
async def confirm_prompt(job_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """确认抽取方案：WAITING_PLAN_CONFIRMATION → PLAN_CONFIRMED。"""
    m = _manifest(job_id)
    if m.status != BookStatus.WAITING_PLAN_CONFIRMATION:
        raise HTTPException(400, "当前没有待确认的抽取方案")
    _pipe().confirm_plan(job_id)
    return {
        "confirmed": True,
        "version": m.prompt_version or "",
        "prompt_length": 0,
        "schema_columns": [],
    }


# ── 路由：步骤 3 抽取（A）─────────────────────────────────────────────────────


@app.post("/api/jobs/{job_id}/step/extract")
async def step_extract(job_id: str) -> dict[str, str]:
    m = _manifest(job_id)
    if not _prompt_confirmed(m):
        raise HTTPException(400, "请先确认抽取方案")
    _guard_not_running(job_id)
    _mark_running(job_id, "正在调用抽取模型（Model A）…")
    _spawn(job_id, _bg_extract(job_id))
    return {"status": "started", "stage": "extracting"}


@app.get("/api/jobs/{job_id}/compare")
async def get_compare(job_id: str) -> dict[str, Any]:
    m = _manifest(job_id)
    if "extract" not in m.completed_stages():
        raise HTTPException(400, "请先完成抽取")
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    records = load_artifact_records(ws.stage_artifact("extract"), SourceRecord)
    blocks = load_blocks(ws.stage_dir("parse") / "blocks.jsonl")
    return {
        "chunks": [
            {
                "chunk_id": str(b.get("blockid") or ""),
                "content": str(b.get("content") or ""),
                "chars": len(str(b.get("content") or "")),
                "semantic_type": str(b.get("type") or "content"),
            }
            for b in blocks
        ],
        "records": [r.model_dump(mode="json") for r in records],
    }


# ── 路由：步骤 4 校验纠正（validate + verify）────────────────────────────────


@app.post("/api/jobs/{job_id}/step/verify")
async def step_verify(job_id: str) -> dict[str, str]:
    m = _manifest(job_id)
    if "extract" not in m.completed_stages():
        raise HTTPException(400, "请先完成抽取")
    _guard_not_running(job_id)
    _mark_running(job_id, "正在确定性校验 + 忠实性核验（Model B）…")
    _spawn(job_id, _bg_verify(job_id))
    return {"status": "started", "stage": "verifying"}


@app.get("/api/jobs/{job_id}/verification")
async def get_verification(job_id: str) -> dict[str, Any]:
    """QA 报告：校验问题 + 核验记录 + 原文分块（供对照）+ 纠错覆盖统计。"""
    _manifest(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    report_path = ws.stage_artifact("validate")
    if not report_path.is_file():
        raise HTTPException(400, "尚未执行校验")
    report = load_artifact(report_path)
    issues = report.get("issues", [])
    correction_path = ws.stage_dir("verify") / "correction_log.jsonl"
    corrections: list[CorrectionEntry] = []
    if correction_path.is_file():
        corrections = [
            c.model_dump(mode="json")
            for c in load_artifact_records(correction_path, CorrectionEntry)
        ]
    verify_path = ws.stage_artifact("verify")
    records = (
        load_artifact_records(verify_path, SourceRecord) if verify_path.is_file() else []
    )
    blocks_path = ws.stage_dir("parse") / "blocks.jsonl"
    chunks = (
        [
            {
                "chunk_id": str(b.get("blockid") or ""),
                "content": str(b.get("content") or ""),
                "chars": len(str(b.get("content") or "")),
            }
            for b in load_blocks(blocks_path)
        ]
        if blocks_path.is_file()
        else []
    )
    record_ids = {r.record_id for r in records}
    bad_ids = {i.get("record_id") for i in issues} | {
        c["record_id"] for c in corrections
    }
    total = len(records)
    applied = sum(1 for c in corrections if c.get("applied"))
    return {
        "total": total,
        "issues": issues,
        "error_count": sum(1 for i in issues if i.get("severity") == Severity.ERROR.value),
        "warning_count": sum(1 for i in issues if i.get("severity") == Severity.WARNING.value),
        "corrections": corrections,
        "records": [r.model_dump(mode="json") for r in records],
        "chunks": chunks,
        "stats": {
            "total": total,
            "pass_rate": round(1 - len(bad_ids & record_ids) / total, 4) if total else 1.0,
            "needs_review": sum(1 for r in records if r.needs_review),
            "correction_count": len(corrections),
            "applied_count": applied,
        },
        "corrected_records": [],
    }


def _first_jsonl_line(path: Path) -> str:
    """读 jsonl 首个非空行（header 信封）。"""
    return next(
        (ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()), "{}"
    )


@app.post("/api/jobs/{job_id}/step/apply-corrections")
async def apply_corrections(job_id: str) -> dict[str, Any]:
    """人工确认纠错：把 correction_log 未应用建议覆盖写回核验记录。

    字段级建议直接覆盖同名字段；``(record)`` 级建议合并整条记录；
    合并不过 SourceRecord 校验的建议保持未应用（转人工）。成功后需重新导出 Excel。
    """
    _manifest(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    verify_path = ws.stage_artifact("verify")
    correction_path = ws.stage_dir("verify") / "correction_log.jsonl"
    if not verify_path.is_file():
        raise HTTPException(400, "尚未执行校验")
    records = load_artifact_records(verify_path, SourceRecord)
    entries = (
        load_artifact_records(correction_path, CorrectionEntry)
        if correction_path.is_file()
        else []
    )
    by_id = {r.record_id: r for r in records}
    applied_now = failed = 0
    for c in entries:
        if c.applied:
            continue
        rec = by_id.get(c.record_id)
        if rec is None:
            continue
        dump = rec.model_dump(mode="json")
        if c.field == "(record)":
            merged = {**dump, **(c.suggested if isinstance(c.suggested, dict) else {})}
        else:
            merged = {**dump, c.field: c.suggested}
        try:
            by_id[c.record_id] = SourceRecord.model_validate(merged)
        except ValidationError:
            failed += 1
            continue
        c.applied = True
        applied_now += 1
    if applied_now:
        # 两个产物均为 jsonl：header 在首行信封里，不能整文件 json.loads
        raw_v = json.loads(_first_jsonl_line(verify_path)).get("header", {})
        ordered = [by_id[r.record_id] for r in records]
        save_artifact(
            verify_path,
            header=ArtifactHeader.model_validate(raw_v),
            data={
                "record_count": len(ordered),
                "needs_review_count": sum(1 for r in ordered if r.needs_review),
            },
            records=ordered,
        )
        raw_c = json.loads(_first_jsonl_line(correction_path)).get("header", {})
        save_artifact(
            correction_path,
            header=ArtifactHeader.model_validate(raw_c),
            data={"correction_count": len(entries)},
            records=entries,
        )
    return {
        "applied": applied_now,
        "failed": failed,
        "total": len(entries),
        "note": "纠错已覆盖写回核验记录；请重新导出 Excel 生效",
    }


# ── 路由：步骤 5 导出 ─────────────────────────────────────────────────────────


def _export_excel(job_id: str) -> Path:
    """调用 scripts/export_book_review_excel.py 生成审核版 Excel（需 export extra）。"""
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    script = project_paths.root / "scripts" / "export_book_review_excel.py"
    spec = importlib.util.spec_from_file_location("_tcm_export_excel", script)
    mod = importlib.util.module_from_spec(spec)
    try:
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # openpyxl 缺失在此抛 ImportError
    except ImportError as exc:
        raise HTTPException(
            500, "缺少 export 依赖（openpyxl）：请 uv sync --extra export 后重启服务"
        ) from exc
    try:
        return Path(mod.export_review_workbook(ws.book_dir))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/jobs/{job_id}/step/export")
async def step_export(job_id: str) -> dict[str, Any]:
    """导出 = 生成审核版 Excel + 放行人工审核门（WAITING_REVIEW → REVIEWED → PUBLISHED）。"""
    m = _manifest(job_id)
    if "verify" not in m.completed_stages():
        raise HTTPException(400, "请先完成校验")
    path = _export_excel(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    if m.status is BookStatus.WAITING_REVIEW:
        m.transition(BookStatus.REVIEWED)
        m.transition(BookStatus.PUBLISHED)
        m.save(ws.book_dir)
    elif m.status is BookStatus.REVIEWED:
        m.transition(BookStatus.PUBLISHED)
        m.save(ws.book_dir)
    return {
        "status": "done",
        "excel": path.name,
        "download": f"/api/download/{job_id}/excel",
    }


# ── 路由：下载 ────────────────────────────────────────────────────────────────


@app.get("/api/download/{job_id}")
async def download(job_id: str) -> FileResponse:
    m = _manifest(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    for stage in ("verify", "extract"):
        artifact = ws.stage_artifact(stage)
        if artifact.is_file():
            return FileResponse(
                artifact, filename=f"{m.book_id}_{stage}_records.jsonl",
                media_type="application/x-ndjson",
            )
    raise HTTPException(404, "尚无可导出的记录")


@app.get("/api/download/{job_id}/excel")
async def download_excel(job_id: str) -> FileResponse:
    _manifest(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    review_dir = ws.stage_dir("review")
    hits = sorted(review_dir.glob("*.xlsx")) if review_dir.is_dir() else []
    path = hits[0] if hits else _export_excel(job_id)
    return FileResponse(
        path, filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/api/download/{job_id}/json")
async def download_json(job_id: str) -> FileResponse:
    return await download(job_id)


@app.get("/api/download/{job_id}/pdf")
async def download_source(job_id: str) -> FileResponse:
    m = _manifest(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    src_dir = ws.source_dir()
    if src_dir.is_dir():
        for child in sorted(src_dir.iterdir()):
            if child.is_file() and child.name == m.file_name:
                return FileResponse(child, filename=m.file_name)
    raise HTTPException(404, "原文件不存在")


# ── 路由：重试 / 回退 ─────────────────────────────────────────────────────────


# UI 步骤 → 「返回上一步」的目标状态（仅回退状态，阶段记录与产物全部保留；
# 作废重跑走 retry-*，由 runner.rewind 完成）
_BACK_TO_STATUS: dict[str, BookStatus] = {
    "parsed": BookStatus.PARSED,
    "prompt_ready": BookStatus.WAITING_PLAN_CONFIRMATION,
    "extracted": BookStatus.EXTRACTED,
    "verified": BookStatus.EXTRACTED,  # 校验步回退到已抽取
    "done": BookStatus.WAITING_REVIEW,  # 导出可回退到已校验（Excel 产物保留）
}
# UI 步骤节点 → 「仅回退查看」目标状态（无损：只回退状态，阶段记录与产物全部保留）。
# 与 retry-* 区分：点更早的步骤节点=回退查看该步，点「重新X」才作废重跑；
# back 只能一跳，goto-<key> 可跳到任意更早步骤。
_GOTO_STATUS: dict[str, BookStatus] = {
    "parse": BookStatus.PARSED,
    "prompt": BookStatus.WAITING_PLAN_CONFIRMATION,
    "extract": BookStatus.EXTRACTED,
    "verify": BookStatus.VERIFIED,
}
# retry-<key> → 后台只跑这些阶段（rewind 在请求同步路径完成）
_RETRY_STEPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "retry-parse": ("parse", ("parse",)),
    "retry-prompt": ("profile", ("profile", "plan")),
    "retry-extract": ("extract", ("extract",)),
    "retry-verify": ("validate", ("validate", "verify")),
}
_RETRY_MESSAGES = {
    "retry-parse": "正在重新解析…",
    "retry-prompt": "正在重新生成抽取方案…",
    "retry-extract": "正在重新抽取（Model A）…",
    "retry-verify": "正在重新校验 + 核验（Model B）…",
}


async def _bg_retry(job_id: str, stages: tuple[str, ...]) -> None:
    pipe = _pipe()
    for stage in stages:
        await pipe.run_stage(job_id, stage, confirm_plan=False)


@app.post("/api/jobs/{job_id}/workflow/{operation}")
async def workflow_op(job_id: str, operation: str) -> dict[str, Any]:
    m = _manifest(job_id)  # 404 校验
    if operation == "back":
        _guard_not_running(job_id)
        view = _view_stage(m)
        target = _BACK_TO_STATUS.get(view)
        if target is None:
            raise HTTPException(400, "当前步骤无法回退")
        # 只回退状态：阶段记录/产物全部保留，重新执行（重新X）才作废重跑
        ws = BookWorkspace(BOOKS_ROOT, job_id)
        order = list(BookStatus)
        if order.index(target) < order.index(m.status):
            m.rewind(target)
            m.save(ws.book_dir)
        return {"status": "back", "stage": _view_stage(m)}
    if operation.startswith("goto-"):
        key = operation[len("goto-"):]
        _guard_not_running(job_id)
        target = _GOTO_STATUS.get(key)
        if target is None:
            raise HTTPException(400, "该步骤不支持跳转查看")
        # 仅回退状态查看该步：阶段记录/产物全部保留（无损，跳到任意更早步骤）
        ws = BookWorkspace(BOOKS_ROOT, job_id)
        order = list(BookStatus)
        if order.index(target) < order.index(m.status):
            m.rewind(target)
            m.save(ws.book_dir)
        return {"status": "goto", "stage": _view_stage(m)}
    if operation == "retry-export":
        path = _export_excel(job_id)
        return {"status": "exported", "excel": path.name}
    if operation not in _RETRY_STEPS:
        raise HTTPException(400, "无效的工作流操作")
    _guard_not_running(job_id)
    rewind_stage, run_stages = _RETRY_STEPS[operation]
    if "extract" in run_stages and not _prompt_confirmed(m):
        raise HTTPException(400, "请先确认抽取方案")
    # rewind 放同步路径：请求返回前状态已回退，前端轮询立即看到重跑中的步骤
    _pipe().rewind(job_id, to_stage=rewind_stage)
    _mark_running(job_id, _RETRY_MESSAGES[operation])
    _spawn(job_id, _bg_retry(job_id, run_stages))
    return {"status": "started", "stage": _view_stage(_manifest(job_id))}


# ── 兼容前端轮询的轻量聚合端点 ────────────────────────────────────────────────


@app.get("/api/jobs")
async def jobs_page(page: int = 1, page_size: int = 50, q: str = "") -> dict[str, Any]:
    items = await list_jobs()
    if q:
        items = [it for it in items if q.lower() in it["filename"].lower()]
    start = (page - 1) * page_size
    return {"total": len(items), "items": items[start : start + page_size]}


@app.get("/api/queue/status")
async def queue_status() -> dict[str, Any]:
    running = [b for b, _ in _runtime.items() if "step_started_at" in _runtime.get(b, {})]
    return {"running": len(running), "total": len(list_books(BOOKS_ROOT))}


# ── GEPA 自优化飞轮（只读概览 + 人工晋升门，与抽取流程解耦）────────────────────


def _resolve_run_dir(raw: str) -> Path:
    """把前端传来的 run_dir（相对 GEPA 根的路径）解析并限制在根目录内，防目录穿越。"""
    root = GEPA_ROOT.resolve()
    p = Path(raw)
    cand = (root / p).resolve() if not p.is_absolute() else p.resolve()
    if cand != root and root not in cand.parents:
        raise HTTPException(400, "非法的进化产物路径")
    if not (cand / "run.json").is_file():
        raise HTTPException(404, "进化产物不存在（缺 run.json）")
    return cand


def _token_total(tokens: dict[str, Any]) -> int:
    return int(
        sum(
            v
            for k, v in (tokens or {}).items()
            if k.endswith("_input") or k.endswith("_output")
        )
    )


def _scan_gepa_runs() -> list[dict[str, Any]]:
    """扫描 workspace/gepa/*/*/*/run.json（单书 + _multi 跨书同构）。"""
    root = GEPA_ROOT
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for run_json in root.glob("*/*/*/run.json"):
        try:
            data = load_artifact(run_json)
        except Exception:  # noqa: BLE001 - 坏产物跳过，不阻断列表
            continue
        header = data.get("header") or {}
        book_ids = data.get("book_ids") or [
            b for b in str(header.get("book_id", "")).split(",") if b
        ]
        runs.append(
            {
                "run_dir": str(run_json.parent.relative_to(root)),
                "book_id": header.get("book_id", ""),
                "book_ids": book_ids,
                "template": data.get("template", ""),
                "run_id": data.get("run_id") or run_json.parent.name,
                "mode": data.get("mode", "single"),
                "baseline_fitness": data.get("baseline_fitness"),
                "best_fitness": data.get("best_fitness"),
                "improved": bool(data.get("improved")),
                "total_generations": data.get("total_generations"),
                "eval_calls": data.get("eval_calls"),
                "elapsed_seconds": data.get("elapsed_seconds"),
                "total_tokens": _token_total(data.get("tokens")),
                "best_constraint": str(data.get("best_constraint", "")).strip(),
            }
        )
    # run_id 前缀是 UTC 时间戳 → 字典序即时间序，倒序 = 最新在前
    runs.sort(key=lambda r: r["run_id"], reverse=True)
    return runs


@app.get("/api/gepa/overview")
async def gepa_overview() -> dict[str, Any]:
    """飞轮只读概览：质量趋势台账 + 已落盘进化产物 + 各 run 是否已晋升。"""
    from tcm_extract_engine.gepa.runner import ledger_path_for, load_trend

    root = GEPA_ROOT
    runs = _scan_gepa_runs()
    trend = [
        {
            "ts": r.ts,
            "kind": r.kind,
            "template": r.template,
            "book_ids": r.book_ids,
            "baseline_fitness": r.baseline_fitness,
            "best_fitness": r.best_fitness,
            "improved": r.improved,
            "eval_calls": r.eval_calls,
            "total_tokens": r.total_tokens,
            "run_id": r.run_id,
            "promoted": r.promoted,
            "old_version": r.old_version,
            "new_version": r.new_version,
            "reviewer": r.reviewer,
        }
        for r in load_trend(ledger_path_for(root))
    ]
    promoted_run_ids = {
        t["run_id"] for t in trend if t["kind"] == "promote" and t["promoted"]
    }
    for r in runs:
        r["promoted"] = r["run_id"] in promoted_run_ids
    summary = {
        "runs_total": len(runs),
        "improved_runs": sum(1 for r in runs if r["improved"]),
        "promote_total": len(promoted_run_ids),
        "latest_best": runs[0]["best_fitness"] if runs else None,
    }
    return {
        "enabled": root.is_dir(),
        "gepa_root": str(root),
        "summary": summary,
        "runs": runs,
        "trend": trend,
    }


@app.post("/api/gepa/promote")
async def gepa_promote(body: dict[str, Any]) -> dict[str, Any]:
    """人工晋升门：把胜出约束块写入 prompts/a/<template> 受管块（触发 version bump）。

    dry_run=true 仅预览版本变化不落盘；晋升后需对相关书 rerun --from extract 才生效。
    空约束块 = 移除受管块（回滚基线）。
    """
    from dataclasses import asdict

    from tcm_extract_engine.gepa.runner import (
        ledger_path_for,
        load_run_best_constraint,
        promote_constraint,
    )

    raw = str(body.get("run_dir", "")).strip()
    if not raw:
        raise HTTPException(400, "缺少 run_dir")
    run_dir = _resolve_run_dir(raw)
    reviewer = (str(body.get("reviewer", "")).strip() or "web-user")[:64]
    template = str(body.get("template", "")).strip() or None
    dry_run = bool(body.get("dry_run"))
    try:
        block, run_template = load_run_best_constraint(run_dir)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"读取胜出约束块失败: {exc}") from exc
    try:
        res = promote_constraint(
            template=template or run_template,
            constraint_block=block,
            reviewer=reviewer,
            run_id=run_dir.name,
            dry_run=dry_run,
            ledger_path=None if dry_run else ledger_path_for(GEPA_ROOT),
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"晋升失败: {exc}") from exc
    out = asdict(res)
    out["prompts_path"] = str(res.prompts_path)
    if res.dry_run:
        out["note"] = "预览（未落盘）"
    elif res.version_bumped:
        out["note"] = "模板已更新并 bump 版本；对相关书 rerun --from extract 后作用于线上抽取"
    else:
        out["note"] = "模板内容未变化"
    return out
