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
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from tcm_extract_engine.cab.planner import (
    ADAPTATION_TEMPLATE,
    BookAdaptedRegistry,
    PromptTemplateRegistry,
    adaptation_version,
)
from tcm_extract_engine.cab.profiler import (
    load_blocks,
)
from tcm_extract_engine.cab.validator import Severity
from tcm_extract_engine.cab.verifier import CorrectionEntry, apply_correction
from tcm_extract_engine.common.paths import project_paths
from tcm_extract_engine.core.artifacts import (
    ArtifactHeader,
    artifact_hash,
    load_artifact,
    load_artifact_records,
    save_artifact,
)
from tcm_extract_engine.core.book import ADAPTATION_FILENAME, BookWorkspace, list_books
from tcm_extract_engine.core.manifest import BookManifest, BookStatus
from tcm_extract_engine.core.records import SourceRecord
from tcm_extract_engine.domains import get_domain_pack
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


app = FastAPI(title="书籍知识抽取工作台", version="0.1.0", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    ws = BookWorkspace(BOOKS_ROOT, ws_book_id)
    failed_stage = next(
        (name for name, stage in reversed(list(m.stages.items())) if stage.status == "FAILED"), None
    ) if m.status == BookStatus.FAILED else None
    progress: dict[str, Any] = {}
    ledger_path = ws.stage_dir("extract") / "batch_ledger.json"
    try:
        started = datetime.fromisoformat(extract.started_at).timestamp() if extract and extract.started_at else 0
        if ledger_path.is_file() and ledger_path.stat().st_mtime >= started:
            batches = json.loads(ledger_path.read_text(encoding="utf-8"))
            progress = {
                "total": len(batches),
                "completed": sum(b.get("status") == "DONE" for b in batches),
                "failed": sum(b.get("status") == "FAILED" for b in batches),
            }
        elif m.status in {BookStatus.EXTRACTING, BookStatus.FAILED}:
            progress = {"cached": len(list((ws.stage_dir("extract") / "cache").glob("batch_*.json")))}
    except (OSError, ValueError, TypeError):
        progress = {}
    quality = None
    quality_path = ws.stage_dir("publish") / "knowledge" / "quality_report.json"
    verified_path = ws.stage_artifact("verify")
    if "verify" in m.completed_stages() and not m.derived_artifacts_stale:
        try:
            if quality_path.is_file() and verified_path.is_file() and quality_path.stat().st_mtime >= verified_path.stat().st_mtime:
                report = json.loads(quality_path.read_text(encoding="utf-8"))
                quality = {k: report.get(k) for k in ("records", "usable_records", "quarantined_records", "evidence_count")}
        except (OSError, ValueError):
            pass
    message = rt.get("message") or ""
    if m.status == BookStatus.FAILED:
        message = m.error or message
    elif message.startswith("步骤失败"):
        message = ""
    return {
        "job_id": m.book_id,
        "filename": m.file_name,
        "workflow_stage": _view_stage(m),
        "status": m.status.value,
        "message": message,
        "domain_id": m.domain_id,
        "domain_label": get_domain_pack(m.domain_id).label,
        "failed_stage": failed_stage,
        "batch_progress": progress,
        "quality": quality,
        "chunks_count": parse.records if parse and parse.records is not None else 0,
        "records_count": extract.records if extract and extract.records is not None else 0,
        "verification_issues": issues,
        "prompt_confirmed": _prompt_confirmed(m),
        "has_extract": "extract" in m.completed_stages(),
        "created_at": m.created_at,
        "updated_at": m.updated_at,
        "step_started_at": rt.get("step_started_at", 0),
        "error": m.error if m.status == BookStatus.FAILED else None,
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
    domain: str = Form("general"),
    auto_run: bool = Form(True),
    force: bool = Query(False),
) -> dict[str, Any]:
    """保存上传文件并建立书籍工作目录（幂等：同 hash 复用）。"""
    try:
        get_domain_pack(domain)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
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
        ws = BookWorkspace.create(BOOKS_ROOT, dest, domain_id=domain)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    finally:
        # 源文件已被 create() 复制进 00_source，删除临时副本
        if dest.is_file():
            dest.unlink(missing_ok=True)

    m = ws.load_manifest()
    if auto_run:
        _guard_not_running(m.book_id)
        _spawn(m.book_id, _pipe().resume(m.book_id))
    return {"job_id": m.book_id, "status": "running" if auto_run else "uploaded", "filename": m.file_name}


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


# ── 蓝图编辑：展示标签、确定性派生与同源注册表 ───────────────────────────────


def _pack_labels(domain_id: str = "tcm") -> dict[str, Any]:
    """领域包声明面标签：extractor/template/content_type 中文名 + 人工跳过标记。"""
    return get_domain_pack(domain_id).labels


def _derive_routing(content_type: str, domain_id: str = "tcm") -> tuple[str, str]:
    pack = get_domain_pack(domain_id)
    extractor = pack.content_extractors.get(content_type, "")
    return extractor, pack.extractor_templates.get(extractor, "")


def _profile_content_types(ws: BookWorkspace) -> dict[str, str]:
    """读章节画像：section_id → content_type（缺画像/坏文件返回空表）。"""
    path = ws.stage_artifact("profile")
    if not path.is_file():
        return {}
    try:
        profile = load_artifact(path)
    except Exception:  # noqa: BLE001 - 坏画像不阻断蓝图展示
        return {}
    return {
        str(s.get("section_id")): str(s.get("content_type") or "")
        for s in profile.get("sections", [])
        if s.get("section_id")
    }


def _edited_registry(ws: BookWorkspace) -> PromptTemplateRegistry | BookAdaptedRegistry:
    """与抽取阶段同源的模板注册表：有适配块则合成 hash(base+块)，否则 base。"""
    adaptations = _load_adaptations(ws)
    blocks = {t: a["block"] for t, a in adaptations.items() if a.get("block")}
    if not blocks:
        return PromptTemplateRegistry()
    return BookAdaptedRegistry(blocks, prompts_root=PromptTemplateRegistry().prompts_root)


@app.get("/api/jobs/{job_id}/schema")
async def get_schema(job_id: str) -> dict[str, Any]:
    """把抽取计划映射成前端「蓝图复核」视图：路由结果 + 规划器原因 + 人工覆盖状态。"""
    _manifest(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    plan_path = ws.stage_artifact("plan")
    if not plan_path.is_file():
        raise HTTPException(400, "尚未生成抽取方案")
    plan = load_artifact(plan_path)
    sections = plan.get("sections", [])
    planned = [s for s in sections if not s.get("skip_reason")]
    profile_ct = _profile_content_types(ws)
    registry = PromptTemplateRegistry()
    templates_dir = registry.prompts_root / "a"
    labels = _pack_labels(_manifest(job_id).domain_id)
    extractor_labels = labels["extractor"]
    template_labels = labels["template"]
    content_type_labels = labels["content_type"]
    manual_skip = labels["manual_skip"]
    editable: list[dict[str, Any]] = []
    for s in sections:
        sid = s.get("section_id", "")
        ct = profile_ct.get(sid, "")
        ext = s.get("extractor", "")
        tpl = s.get("template", "")
        skip_reason = s.get("skip_reason")
        derived = _derive_routing(ct, _manifest(job_id).domain_id) if ct else ("", "")
        d_ext, d_tpl = derived
        if skip_reason == manual_skip:
            mode = "skip"  # 人工强制跳过
        elif not ct:
            mode = "auto"  # 无画像：维持现状，禁用恢复
        elif not d_ext:
            mode = "auto"  # 规划器自动跳过（仅参考/无关）
        elif (ext, tpl) == derived and not skip_reason:
            mode = "auto"  # 与规划器一致
        else:
            mode = "extract"  # 人工覆盖路由
        editable.append(
            {
                "section_id": sid,
                "title": s.get("title", ""),
                "heading_path": s.get("heading_path") or [],
                "page_start": s.get("page_start"),
                "page_end": s.get("page_end"),
                "block_count": len(s.get("block_ids") or []),
                "content_type": ct,
                "content_type_label": content_type_labels.get(ct, ct),
                "extractor": ext,
                "extractor_label": extractor_labels.get(ext, ext),
                "template": tpl,
                "template_label": template_labels.get(tpl, tpl),
                "skip": bool(skip_reason),
                "skip_reason": skip_reason,
                "skip_kind": "manual"
                if skip_reason == manual_skip
                else ("auto" if skip_reason else None),
                "mode": mode,
                "derived": {"extractor": d_ext, "template": d_tpl, "skip": not d_ext}
                if ct
                else None,
            }
        )
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
        # 蓝图人工调整：抽取器可选集 + 中文标签 + 富化后的章节视图
        "available_extractors": sorted(
            {
                v
                for v in get_domain_pack(_manifest(job_id).domain_id).content_extractors.values()
                if v
            }
        ),
        "available_templates": sorted(p.name for p in templates_dir.glob("*.md")),
        "extractor_labels": extractor_labels,
        "template_labels": template_labels,
        "editable_sections": editable,
        "qa_rules": plan.get("risk_points", []),
        "total_sections": len(sections),
        "planned_sections": len(planned),
    }


@app.put("/api/jobs/{job_id}/schema")
async def update_schema(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """人工调整抽取蓝图：冻结门 + 三态编辑契约。

    每章节一个 {section_id, mode, extractor?}：
      - auto    恢复规划器默认路由（由章节画像的 content_type 确定性派生）
      - skip    人工强制跳过
      - extract 人工强制抽取（extractor 必填；模板由 EXTRACTOR_TEMPLATES 唯一确定，杜绝错配）
    已确认方案（PLAN_CONFIRMED 及以后）蓝图冻结，需先「解锁编辑」
    （workflow/goto-prompt）回退到待确认状态；已抽取的书保存后作废 extract
    及下游阶段记录（产物保留），仅变更章节需重新调用 Model A。
    """
    m = _manifest(job_id)
    if m.status != BookStatus.WAITING_PLAN_CONFIRMATION:
        raise HTTPException(
            400, "蓝图已冻结（方案已确认）。请先「解锁编辑」回退到待确认状态，再调整路由。"
        )
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    plan_path = ws.stage_artifact("plan")
    if not plan_path.is_file():
        raise HTTPException(400, "尚未生成抽取方案")
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    sections = raw.get("sections", [])
    edits = {e.get("section_id"): e for e in body.get("sections", []) if e.get("section_id")}
    profile_ct = _profile_content_types(ws)
    # 与抽取阶段同源：有适配块则 prompt_version = hash(base+块)，保证缓存键一致
    registry = _edited_registry(ws)
    valid_ext = {
        v for v in get_domain_pack(_manifest(job_id).domain_id).content_extractors.values() if v
    }
    manual_skip = _pack_labels(m.domain_id)["manual_skip"]
    for s in sections:
        sid = s.get("section_id")
        e = edits.get(sid)
        if not e:
            continue
        mode = str(e.get("mode") or "")
        if mode == "auto":
            ct = profile_ct.get(sid, "")
            if not ct:
                raise HTTPException(
                    400,
                    f"章节 {s.get('title') or sid} 缺少章节画像，无法恢复自动路由；请重新生成蓝图",
                )
            ext, tpl = _derive_routing(ct, m.domain_id)
            if ext:
                s["extractor"], s["template"] = ext, tpl
                s["skip_reason"] = None
                s["prompt_version"] = registry.load("a", tpl).version
            else:
                s["extractor"] = s["template"] = s["prompt_version"] = ""
                s["skip_reason"] = f"content_type={ct} 不抽取"
        elif mode == "skip":
            s["skip_reason"] = manual_skip
            s["extractor"] = s["template"] = s["prompt_version"] = ""
        elif mode == "extract":
            ext = str(e.get("extractor") or "")
            if ext not in valid_ext:
                raise HTTPException(400, f"未知抽取器: {ext}")
            tpl = get_domain_pack(m.domain_id).extractor_templates[ext]
            s["extractor"], s["template"] = ext, tpl
            s["skip_reason"] = None
            s["prompt_version"] = registry.load("a", tpl).version
        else:
            raise HTTPException(400, f"未知编辑模式: {mode}")
    raw["sections"] = sections
    raw.setdefault("header", {}).setdefault("extra", {})["plan_edited_at"] = datetime.now(
        timezone.utc
    ).isoformat()
    plan_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    # 与 update_adaptations 同一公式重算合成版本；先落盘再作废下游
    # （invalidate_from 从磁盘重读 manifest，避免新版本被旧状态覆盖）
    versions = sorted({s.get("prompt_version") for s in sections if s.get("prompt_version")})
    m.prompt_version = artifact_hash({"prompt_versions": versions})[:20] if versions else None
    m.save(ws.book_dir)
    if "extract" in m.completed_stages():
        _pipe().invalidate_from(job_id, "extract")
    return await get_schema(job_id)


def _load_adaptations(ws: BookWorkspace) -> dict[str, dict[str, Any]]:
    """读 03_plan/prompt_adaptations.json → template → 适配块字典（坏文件退回空）。"""
    path = ws.stage_dir("plan") / ADAPTATION_FILENAME
    if not path.is_file():
        return {}
    try:
        data = load_artifact(path)
    except Exception:  # noqa: BLE001 - 坏产物不阻断展示
        return {}
    raw = data.get("adaptations") or {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _profile_summary(ws: BookWorkspace) -> str:
    """把 document_profile 压缩成供审核 C 输入的摘要文本。"""
    path = ws.stage_artifact("profile")
    if not path.is_file():
        return ""
    try:
        profile = load_artifact(path)
    except Exception:  # noqa: BLE001
        return ""
    sections = profile.get("sections", [])
    lines = [
        f"document_type: {profile.get('document_type', '')}",
        f"total_blocks: {profile.get('total_blocks', 0)}",
        f"sections: {len(sections)}",
    ]
    for s in sections[:40]:
        lines.append(f"- [{s.get('content_type', '')}] {s.get('title', '')}")
    return "\n".join(lines)


@app.get("/api/jobs/{job_id}/prompt")
async def get_prompt(job_id: str) -> dict[str, Any]:
    """返回本方案选用的 A 模板正文 + 每书适配块（base 逐字不变，适配块可复核）。"""
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    plan_path = ws.stage_artifact("plan")
    if not plan_path.is_file():
        raise HTTPException(400, "尚未生成抽取方案")
    plan = load_artifact(plan_path)
    registry = PromptTemplateRegistry()
    names = sorted({s["template"] for s in plan.get("sections", []) if s.get("template")})
    adaptations = _load_adaptations(ws)
    blocks: list[str] = []
    adaptation_view: list[dict[str, Any]] = []
    for name in names:
        try:
            base = registry.load("a", name).content
        except FileNotFoundError:
            blocks.append(f"# ── {name}（缺失）──")
            continue
        adapt = adaptations.get(name) or {}
        block = str(adapt.get("block") or "")
        composed = f"{base.rstrip()}\n\n{block}\n" if block else base
        blocks.append(f"# ── {name} ──\n" + composed)
        adaptation_view.append(
            {
                "template": name,
                "block": block,
                "version": adapt.get("version", ""),
                "degraded": bool(adapt.get("degraded", False)),
                "human_edited": bool(adapt.get("human_edited", False)),
            }
        )
    return {
        "prompt": "\n\n".join(blocks),
        "templates": names,
        "adaptations": adaptation_view,
        "editable": False,
        "note": "base 模板逐字只读（防 Prompt Drift，全局调整走磁盘/GEPA 晋升）；每书适配块可人工编辑/重生成，保存后重算版本并作废下游抽取。",
    }


@app.get("/api/jobs/{job_id}/meta-prompt")
async def get_meta_prompt(job_id: str) -> dict[str, Any]:
    registry = PromptTemplateRegistry()
    try:
        planner = registry.load("c", ADAPTATION_TEMPLATE).content
    except FileNotFoundError:
        try:
            planner = registry.load("c", "planner.md").content
        except FileNotFoundError:
            planner = ""
    m = _manifest(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    adaptations = _load_adaptations(ws)
    generated = "\n\n".join(
        f"# ── {name} ──\n{a.get('block', '')}".rstrip()
        for name, a in sorted(adaptations.items())
        if a.get("block")
    )
    return {
        "system_prompt": planner,
        "user_prompt": _profile_summary(ws),
        "generated_prompt": generated,
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


@app.post("/api/jobs/{job_id}/adaptations/regenerate")
async def regenerate_adaptations(job_id: str) -> dict[str, str]:
    """重新生成每书适配块：删缓存 → 回退 plan → 后台重跑，回到确认门。"""
    m = _manifest(job_id)
    if m.status != BookStatus.WAITING_PLAN_CONFIRMATION:
        raise HTTPException(400, "仅在待确认抽取方案时可重新生成适配块")
    _guard_not_running(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    (ws.stage_dir("plan") / ADAPTATION_FILENAME).unlink(missing_ok=True)
    _pipe().rewind(job_id, to_stage="plan")
    _mark_running(job_id, "正在重新生成每书适配块（Model C）…")
    _spawn(job_id, _bg_plan(job_id))
    return {"status": "started", "stage": "generating_prompt"}


@app.put("/api/jobs/{job_id}/adaptations")
async def update_adaptations(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """人工编辑每书适配块（base 模板仍只读）。

    保存后：重算 block version → 重写 plan 的 prompt_version（base+块合成 hash）
    → 重算 manifest.prompt_version → 若已抽取则作废 extract 及下游记录。
    """
    m = _manifest(job_id)
    _guard_not_running(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    adapt_path = ws.stage_dir("plan") / ADAPTATION_FILENAME
    if not adapt_path.is_file():
        raise HTTPException(400, "尚无适配块产物，请先生成抽取方案")
    edits = {
        str(e.get("template")): str(e.get("block") or "").strip()
        for e in body.get("adaptations", [])
        if e.get("template")
    }
    if not edits:
        raise HTTPException(400, "未提供待保存的适配块")

    data = load_artifact(adapt_path)
    adaptations = data.get("adaptations") or {}
    for tpl, block in edits.items():
        if tpl not in adaptations:
            raise HTTPException(400, f"未知模板的适配块: {tpl}")
        adaptations[tpl]["block"] = block
        adaptations[tpl]["version"] = adaptation_version(block) if block else ""
        adaptations[tpl]["degraded"] = False
        adaptations[tpl]["human_edited"] = True
    header = ArtifactHeader.model_validate(data["header"])
    header.extra["adapt_edited_at"] = datetime.now(timezone.utc).isoformat()
    body = {k: v for k, v in data.items() if k != "header"}
    body["adaptations"] = adaptations
    save_artifact(adapt_path, header=header, data=body)

    # 重算 plan 的 prompt_version（base + 适配块 合成 hash）
    registry = BookAdaptedRegistry(
        {t: a["block"] for t, a in adaptations.items() if a.get("block")},
        prompts_root=PromptTemplateRegistry().prompts_root,
    )
    plan_path = ws.stage_artifact("plan")
    plan_raw = load_artifact(plan_path)
    plan_header = ArtifactHeader.model_validate(plan_raw["header"])
    plan_body = {k: v for k, v in plan_raw.items() if k != "header"}
    for sec in plan_body.get("sections", []):
        tpl = sec.get("template")
        if tpl and tpl in adaptations:
            sec["prompt_version"] = registry.load("a", tpl).version
    save_artifact(plan_path, header=plan_header, data=plan_body)

    versions = sorted(
        {s["prompt_version"] for s in plan_body.get("sections", []) if s.get("prompt_version")}
    )
    m.prompt_version = artifact_hash({"prompt_versions": versions})[:20] if versions else None
    had_extract = "extract" in m.completed_stages()
    # 先落盘再作废下游：invalidate_from 从磁盘重读 manifest，避免合成版本被旧状态覆盖
    m.save(ws.book_dir)
    if had_extract:
        _pipe().invalidate_from(job_id, "extract")
        m = _manifest(job_id)
    return {
        "saved": sorted(edits),
        "prompt_version": m.prompt_version or "",
        "invalidated_downstream": had_extract,
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
                "page": (b.get("positions") or [{}])[0].get("anchor"),
                "section_path": [*(b.get("parent_headings") or []), b.get("heading") or ""],
                "asset_url": f"/api/jobs/{job_id}/assets/{b['blockid']}" if b.get("asset_path") else None,
            }
            for b in blocks
        ],
        "records": [r.model_dump(mode="json") for r in records],
    }


@app.get("/api/jobs/{job_id}/assets/{block_id}")
async def image_asset(job_id: str, block_id: str) -> FileResponse:
    """Serve only image assets referenced by this book's parsed blocks."""
    _manifest(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    blocks = load_blocks(ws.stage_dir("parse") / "blocks.jsonl")
    block = next((b for b in blocks if b.get("blockid") == block_id), None)
    if not block or not block.get("asset_path"):
        raise HTTPException(404, "该内容块没有可查看的图片")
    path = Path(block["asset_path"]).resolve()
    if not path.is_relative_to(ws.book_dir.resolve()) or not path.is_file() or path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff",
    }:
        raise HTTPException(404, "图片不存在或不属于当前文档")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})


_WORKBENCH_ARTIFACTS = {
    "extracted": ("extract", "04_extract/records.jsonl", "抽取记录", "全部初步抽取结果，包含待核验记录"),
    "verified": ("verify", "06_verify/verified_records.jsonl", "核验记录", "包含核验与待复核标记的完整记录"),
    "knowledge": ("knowledge", "08_publish/knowledge/records.jsonl", "结构化知识", "具有来源证据的结构化记录"),
    "usable": ("knowledge", "08_publish/knowledge/usable_records.jsonl", "可用知识", "通过当前证据与核验门槛的记录"),
    "evidence": ("knowledge", "08_publish/knowledge/evidence.jsonl", "原文证据", "原文片段、物理页码及定位信息"),
    "visual": ("knowledge", "08_publish/knowledge/visual_facts.jsonl", "图片知识", "图中观察、步骤和关系及其核验状态"),
    "assets": ("knowledge", "08_publish/knowledge/assets.jsonl", "图片索引", "原始图片的标识和相对路径"),
    "quality": ("knowledge", "08_publish/knowledge/quality_report.json", "质量报告", "可用、隔离、覆盖和待审核情况"),
    "quarantine": ("knowledge", "08_publish/knowledge/quarantine.jsonl", "隔离记录", "缺少证据或未通过确定性校验的记录"),
}


def _available_artifacts(job_id: str) -> list[dict[str, Any]]:
    manifest = _manifest(job_id)
    ws = BookWorkspace(BOOKS_ROOT, job_id)
    done = manifest.completed_stages()
    verified = ws.stage_artifact("verify")
    files = []
    for name, (stage, relative, label, description) in _WORKBENCH_ARTIFACTS.items():
        path = ws.book_dir / relative
        if not path.is_file():
            continue
        if stage == "knowledge":
            if "verify" not in done or manifest.derived_artifacts_stale or not verified.is_file():
                continue
            if path.stat().st_mtime < verified.stat().st_mtime:
                continue
        elif stage not in done:
            continue
        files.append({"name": name, "filename": path.name, "label": label, "description": description,
                      "bytes": path.stat().st_size, "url": f"/api/jobs/{job_id}/artifacts/{name}"})
    return files


@app.get("/api/jobs/{job_id}/artifacts")
async def workbench_artifacts(job_id: str) -> dict[str, Any]:
    return {"files": _available_artifacts(job_id)}


@app.get("/api/jobs/{job_id}/artifacts/{name}")
async def download_artifact(job_id: str, name: str) -> FileResponse:
    if name not in _WORKBENCH_ARTIFACTS or name not in {a["name"] for a in _available_artifacts(job_id)}:
        raise HTTPException(404, "文件尚未生成或已过期，请先完成相应处理阶段")
    _, relative, _, _ = _WORKBENCH_ARTIFACTS[name]
    path = BookWorkspace(BOOKS_ROOT, job_id).book_dir / relative
    return FileResponse(path, filename=path.name,
                        media_type="application/x-ndjson" if path.suffix == ".jsonl" else "application/json")


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
    corrections: list[dict[str, Any]] = []
    if correction_path.is_file():
        corrections = [
            c.model_dump(mode="json")
            for c in load_artifact_records(correction_path, CorrectionEntry)
        ]
    verify_path = ws.stage_artifact("verify")
    records = load_artifact_records(verify_path, SourceRecord) if verify_path.is_file() else []
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
    bad_ids = {i.get("record_id") for i in issues} | {c["record_id"] for c in corrections}
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
    return next((ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()), "{}")


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
        load_artifact_records(correction_path, CorrectionEntry) if correction_path.is_file() else []
    )
    by_id = {r.record_id: r for r in records}
    applied_now = failed = 0
    for c in entries:
        if c.applied:
            continue
        rec = by_id.get(c.record_id)
        if rec is None:
            continue
        try:
            by_id[c.record_id] = apply_correction(rec, c)
        except (ValidationError, ValueError):
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
    assert spec is not None and spec.loader is not None
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
    """生成审核版 Excel；保持人工审核和发布状态。"""
    m = _manifest(job_id)
    if "verify" not in m.completed_stages():
        raise HTTPException(400, "请先完成校验")
    path = _export_excel(job_id)
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
    stages = ("verify", "extract") if "verify" in m.completed_stages() and not m.derived_artifacts_stale else ("extract",)
    for stage in stages:
        if stage not in m.completed_stages():
            continue
        artifact = ws.stage_artifact(stage)
        if artifact.is_file():
            return FileResponse(
                artifact,
                filename=f"{m.book_id}_{stage}_records.jsonl",
                media_type="application/x-ndjson",
            )
    raise HTTPException(404, "尚无可导出的记录")


@app.get("/api/download/{job_id}/excel")
async def download_excel(job_id: str) -> FileResponse:
    if _manifest(job_id).derived_artifacts_stale:
        raise HTTPException(409, "旧审核表已过期，请重跑校验并重新导出")
    _manifest(job_id)
    path = _export_excel(job_id)  # Regenerate from current records; old files may predate a rerun.
    return FileResponse(
        path,
        filename=path.name,
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
        key = operation[len("goto-") :]
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
        sum(v for k, v in (tokens or {}).items() if k.endswith("_input") or k.endswith("_output"))
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
                "total_tokens": _token_total(data.get("tokens") or {}),
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
    promoted_run_ids = {t["run_id"] for t in trend if t["kind"] == "promote" and t["promoted"]}
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
