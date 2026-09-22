"""BookPipeline Runner：阶段编排 + 断点恢复 + 重跑。

职责：
  - 按 STAGE_ORDER 编排 parse → profile → plan → extract → validate → verify
  - 每阶段：状态迁移 → 执行 → Artifact 落盘 → manifest 更新（原子）
  - resume：跳过 manifest 中已完成阶段（幂等）
  - rerun --from <stage>：回退状态 + 作废该阶段及下游的阶段记录后重跑
  - review / publish 为人工阶段，runner 止步于 WAITING_REVIEW

命令语义（对应 CLI ``tcm-engine book ...``）：
  run / run-book   新建或续跑单本书直到 WAITING_REVIEW
  resume           继续未完成的书
  rerun            --from <stage> 从指定阶段强制重跑
  retry            FAILED → 重置后 resume
  confirm          WAITING_PLAN_CONFIRMATION → PLAN_CONFIRMED（人工确认计划）
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from tcm_extract_engine.cab.extractors import extract_records
from tcm_extract_engine.cab.planner import PromptTemplateRegistry, build_extraction_plan
from tcm_extract_engine.cab.profiler import DocumentProfile, build_profile, load_blocks
from tcm_extract_engine.cab.validator import validate_records
from tcm_extract_engine.cab.verifier import verify_records
from tcm_extract_engine.common.paths import project_paths
from tcm_extract_engine.core.artifacts import (
    ArtifactHeader,
    artifact_hash,
    load_artifact,
    load_artifact_records,
    save_artifact,
)
from tcm_extract_engine.core.book import BookWorkspace, list_books
from tcm_extract_engine.core.llm import UnifiedLLMClient
from tcm_extract_engine.core.manifest import (
    STAGE_ORDER,
    BookManifest,
    BookStatus,
)
from tcm_extract_engine.core.records import SourceRecord

logger = logging.getLogger(__name__)

# runner 自动执行的阶段（review/publish 为人工阶段）
AUTO_STAGES: tuple[str, ...] = ("parse", "profile", "plan", "extract", "validate", "verify")

# 阶段完成后的状态（用于 resume 快进与 rewind 定位）
_POST_STATUS: dict[str, BookStatus] = {
    "parse": BookStatus.PARSED,
    "profile": BookStatus.PROFILED,
    "plan": BookStatus.PLAN_CONFIRMED,
    "extract": BookStatus.EXTRACTED,
    "validate": BookStatus.VALIDATED,
    "verify": BookStatus.WAITING_REVIEW,
    "review": BookStatus.REVIEWED,
    "publish": BookStatus.PUBLISHED,
}

# 阶段开始前的状态（rerun --from 的回退目标）
_PRE_STATUS: dict[str, BookStatus] = {
    "parse": BookStatus.PENDING,
    "profile": BookStatus.PARSED,
    "plan": BookStatus.PROFILED,
    "extract": BookStatus.PLAN_CONFIRMED,
    "validate": BookStatus.EXTRACTED,
    "verify": BookStatus.VALIDATED,
    "review": BookStatus.WAITING_REVIEW,
    "publish": BookStatus.REVIEWED,
}

_STATUS_CHAIN: list[BookStatus] = list(BookStatus)[:-1]  # 去掉 FAILED，其余按序


class BookPipeline:
    """单本书/批量的流水线编排器。"""

    def __init__(
        self,
        books_root: Path | None = None,
        *,
        registry: PromptTemplateRegistry | None = None,
        extract_client: UnifiedLLMClient | None = None,
        verify_client: UnifiedLLMClient | None = None,
    ) -> None:
        self.books_root = books_root or project_paths.root / "workspace" / "books"
        self.registry = registry or PromptTemplateRegistry()
        self._extract_client = extract_client
        self._verify_client = verify_client
        self._owned_clients: list[UnifiedLLMClient] = []

    # ── 客户端（延迟创建，便于测试注入）────────────────────────────────────

    @property
    def extract_client(self) -> UnifiedLLMClient:
        if self._extract_client is None:
            from tcm_extract_engine.common.config import get_settings

            client = UnifiedLLMClient(get_settings().extract_llm)
            self._extract_client = client
            self._owned_clients.append(client)
        return self._extract_client

    @property
    def verify_client(self) -> UnifiedLLMClient:
        if self._verify_client is None:
            from tcm_extract_engine.common.config import get_settings

            client = UnifiedLLMClient(get_settings().verify_llm)
            self._verify_client = client
            self._owned_clients.append(client)
        return self._verify_client

    async def close(self) -> None:
        for client in self._owned_clients:
            await client.close()
        self._owned_clients.clear()

    # ── 对外入口 ──────────────────────────────────────────────────────────

    async def run(
        self, source: Path, *, confirm_plan: bool = True
    ) -> BookManifest:
        """新建（或复用）书籍工作目录并跑到 verify 完成。"""
        ws = BookWorkspace.create(self.books_root, source)
        manifest = ws.load_manifest()
        return await self._execute(ws, manifest, confirm_plan=confirm_plan)

    async def resume(self, book_id: str) -> BookManifest:
        """继续未完成的书（跳过已完成阶段）。"""
        ws = self._workspace(book_id)
        manifest = ws.load_manifest()
        return await self._execute(ws, manifest)

    async def retry(self, book_id: str) -> BookManifest:
        """FAILED → 重置后继续（等价 resume + 状态复位）。"""
        ws = self._workspace(book_id)
        manifest = ws.load_manifest()
        if manifest.status == BookStatus.FAILED:
            manifest.transition(BookStatus.PENDING)
            manifest.save(ws.book_dir)
        return await self._execute(ws, manifest)

    async def rerun(
        self, book_id: str, *, from_stage: str, confirm_plan: bool = True
    ) -> BookManifest:
        """从指定阶段强制重跑（下游阶段记录作废）。"""
        self.rewind(book_id, to_stage=from_stage)
        ws = self._workspace(book_id)
        return await self._execute(ws, ws.load_manifest(), confirm_plan=confirm_plan)

    def rewind(self, book_id: str, *, to_stage: str) -> BookManifest:
        """作废 to_stage 及下游阶段记录，状态回退到该阶段前置状态。

        产物文件保留在磁盘（回看视图仍可读），重跑时覆盖；
        供 Web「返回上一步 / 单步重跑」与 ``rerun`` 共用。
        """
        if to_stage not in STAGE_ORDER:
            raise ValueError(f"未知阶段: {to_stage}")
        ws = self._workspace(book_id)
        manifest = ws.load_manifest()
        target = _PRE_STATUS[to_stage]
        if manifest.status is BookStatus.FAILED:
            manifest.status = target
            manifest.error = None
        elif manifest.status is not target:
            # 已处于目标前置态时幂等（如 EXTRACTED 下重跑 validate），rewind 拒绝原地回退
            manifest.rewind(target)
        idx = STAGE_ORDER.index(to_stage)
        for stage in STAGE_ORDER[idx:]:
            manifest.stages.pop(stage, None)
        if to_stage == "plan":
            manifest.prompt_version = None
        manifest.save(ws.book_dir)
        return manifest

    async def run_stage(
        self, book_id: str, stage: str, *, confirm_plan: bool = True
    ) -> BookManifest:
        """只执行单个自动阶段（供 Web 逐步控制调用；DONE 阶段幂等跳过）。

        与 ``rerun`` 不同：不作废下游阶段，只推进 ``stage`` 这一步。
        """
        if stage not in AUTO_STAGES:
            raise ValueError(f"未知自动阶段: {stage}")
        ws = self._workspace(book_id)
        manifest = ws.load_manifest()
        await self._run_one_stage(ws, manifest, stage, confirm_plan=confirm_plan)
        return manifest

    def confirm_plan(self, book_id: str) -> BookManifest:
        """人工确认抽取计划：WAITING_PLAN_CONFIRMATION → PLAN_CONFIRMED。"""
        ws = self._workspace(book_id)
        manifest = ws.load_manifest()
        manifest.transition(BookStatus.PLAN_CONFIRMED)
        manifest.save(ws.book_dir)
        return manifest

    def status(self, book_id: str) -> BookManifest:
        return self._workspace(book_id).load_manifest()

    def list_all(self) -> list[BookManifest]:
        manifests: list[BookManifest] = []
        for ws in list_books(self.books_root):
            try:
                manifests.append(ws.load_manifest())
            except Exception as exc:  # noqa: BLE001 - 状态损坏的书籍也要能列出
                logger.warning("[Books] %s manifest 读取失败: %s", ws.book_id, exc)
        return manifests

    # ── 内部编排 ──────────────────────────────────────────────────────────

    def _workspace(self, book_id: str) -> BookWorkspace:
        ws = BookWorkspace(self.books_root, book_id)
        if not ws.manifest_path.is_file():
            raise FileNotFoundError(f"书籍不存在: {ws.book_dir}")
        return ws

    async def _execute(
        self,
        ws: BookWorkspace,
        manifest: BookManifest,
        *,
        confirm_plan: bool = True,
    ) -> BookManifest:
        """顺序执行未完成的阶段；单阶段失败 → FAILED 并抛出。"""
        for stage in AUTO_STAGES:
            stop = await self._run_one_stage(
                ws, manifest, stage, confirm_plan=confirm_plan
            )
            if stop:  # 停在计划确认门
                return manifest
        logger.info("[Book %s] 自动阶段全部完成（WAITING_REVIEW）", ws.book_id)
        return manifest

    async def _run_one_stage(
        self,
        ws: BookWorkspace,
        manifest: BookManifest,
        stage: str,
        *,
        confirm_plan: bool = True,
    ) -> bool:
        """执行单个阶段（已完成则跳过）；返回是否应中断流水线（计划确认门）。

        供 ``_execute`` 顺序循环与 Web 逐步控制 ``run_stage`` 共用，保证
        状态迁移、Artifact 落盘、失败落 FAILED 的语义完全一致。
        """
        record = manifest.stage(stage)
        artifact_path = ws.stage_artifact(stage)
        if record.status == "DONE" and artifact_path.is_file():
            post = _POST_STATUS[stage]
            if stage == "plan" and not confirm_plan:
                post = BookStatus.WAITING_PLAN_CONFIRMATION
            _advance_status(manifest, post)
            manifest.save(ws.book_dir)
            logger.info("[Book %s] %s 已完成，跳过", ws.book_id, stage)
            # 确认门同样适用于跳过路径：已停在待确认时不得静默放行 extract
            return manifest.status is BookStatus.WAITING_PLAN_CONFIRMATION

        handler = getattr(self, f"_run_{stage}")
        _normalize_status(manifest, _PRE_STATUS[stage])
        manifest.transition(_IN_PROGRESS_STATUS[stage])
        record.mark_running()
        manifest.save(ws.book_dir)
        try:
            if stage == "plan":
                await handler(ws, manifest, confirm_plan=confirm_plan)
            else:
                await handler(ws, manifest)
            record.mark_done(
                artifact=str(artifact_path.relative_to(ws.book_dir)),
                artifact_hash=_artifact_digest(artifact_path),
                records=_record_count(artifact_path),
            )
            post = _POST_STATUS[stage]
            if stage == "plan" and not confirm_plan:
                post = BookStatus.WAITING_PLAN_CONFIRMATION
            _advance_status(manifest, post)
            manifest.save(ws.book_dir)
            logger.info("[Book %s] %s 完成", ws.book_id, stage)
            if manifest.status == BookStatus.WAITING_PLAN_CONFIRMATION:
                logger.info(
                    "[Book %s] 计划待人工确认（tcm-engine book confirm %s）",
                    ws.book_id, ws.book_id,
                )
                return True
            return False
        except Exception as exc:  # noqa: BLE001 - 统一落 FAILED
            record.mark_failed(str(exc))
            manifest.fail(str(exc))
            manifest.save(ws.book_dir)
            logger.error("[Book %s] %s 失败: %s", ws.book_id, stage, exc)
            raise

    # ── 阶段实现 ──────────────────────────────────────────────────────────

    async def _run_parse(self, ws: BookWorkspace, manifest: BookManifest) -> None:
        """解析：MinerU/Markdown → blocks.jsonl。"""
        from tcm_extract_engine.parse.mineru import ensure_parse_artifact

        source = self._locate_source(ws, manifest)
        result = await ensure_parse_artifact(source)
        blocks_dest = ws.stage_dir("parse") / "blocks.jsonl"
        if result.blocks_path.resolve() != blocks_dest.resolve():
            shutil.copy2(result.blocks_path, blocks_dest)

        summary = {
            "block_count": result.block_count,
            "cache_hit": result.cache_hit,
            "source_file": str(source),
            "file_hash": manifest.file_hash,
        }
        save_artifact(
            ws.stage_artifact("parse"),
            header=self._header(manifest, "parse"),
            data=summary,
        )

    async def _run_profile(self, ws: BookWorkspace, manifest: BookManifest) -> None:
        """画像：blocks → document_profile.json。"""
        blocks_path = ws.stage_dir("parse") / "blocks.jsonl"
        profile = build_profile(blocks_path, book_name=manifest.file_name)
        manifest.source_type = profile.document_type.value
        save_artifact(
            ws.stage_artifact("profile"),
            header=self._header(manifest, "profile",
                                upstream=_artifact_digest(ws.stage_artifact("parse"))),
            data=profile.model_dump(mode="json"),
        )

    async def _run_plan(
        self, ws: BookWorkspace, manifest: BookManifest, *, confirm_plan: bool = True
    ) -> None:
        """计划：profile → extraction_plan.json（Model C，确定性规则）。"""
        profile_data = load_artifact(ws.stage_artifact("profile"))
        profile = DocumentProfile.model_validate(profile_data)
        plan = build_extraction_plan(profile, registry=self.registry)
        save_artifact(
            ws.stage_artifact("plan"),
            header=self._header(manifest, "plan",
                                upstream=_artifact_digest(ws.stage_artifact("profile"))),
            data=plan.model_dump(mode="json"),
        )
        # 幂等键的 prompt 版本：全部计划章节 prompt_version 的合成 hash
        versions = sorted({s.prompt_version for s in plan.sections if s.prompt_version})
        manifest.prompt_version = (
            artifact_hash({"prompt_versions": versions})[:20] if versions else None
        )

    async def _run_extract(self, ws: BookWorkspace, manifest: BookManifest) -> None:
        """抽取（Model A）：plan + blocks → records.jsonl。"""
        records = await extract_records(
            book_id=ws.book_id,
            blocks_path=ws.stage_dir("parse") / "blocks.jsonl",
            plan_path=ws.stage_artifact("plan"),
            cache_dir=ws.stage_dir("extract") / "cache",
            client=self.extract_client,
            registry=self.registry,
        )
        save_artifact(
            ws.stage_artifact("extract"),
            header=self._header(manifest, "extract",
                                model=self.extract_client.model,
                                prompt_version=manifest.prompt_version,
                                upstream=_artifact_digest(ws.stage_artifact("plan"))),
            data={"record_count": len(records)},
            records=records,
        )

    async def _run_validate(self, ws: BookWorkspace, manifest: BookManifest) -> None:
        """确定性校验：records → validation_report.json + suspicious_records.jsonl。"""
        records = load_artifact_records(ws.stage_artifact("extract"), SourceRecord)
        blocks = load_blocks(ws.stage_dir("parse") / "blocks.jsonl")
        report = validate_records(records, blocks=blocks)

        # ERROR 记录就地标记复核（记录不删除，人工决定）
        suspicious_ids = {i.record_id for i in report.issues if i.severity.value == "ERROR"}
        suspicious = [r for r in records if r.record_id in suspicious_ids]
        for record in suspicious:
            record.needs_review = True
        if suspicious:
            save_artifact(
                ws.stage_dir("validate") / "suspicious_records.jsonl",
                header=self._header(manifest, "validate"),
                data={"suspicious_count": len(suspicious)},
                records=suspicious,
            )
        save_artifact(
            ws.stage_artifact("validate"),
            header=self._header(manifest, "validate",
                                upstream=_artifact_digest(ws.stage_artifact("extract"))),
            data=report.model_dump(mode="json"),
        )

    async def _run_verify(self, ws: BookWorkspace, manifest: BookManifest) -> None:
        """核验（Model B）：records + blocks → verified_records + correction_log。"""
        records = load_artifact_records(ws.stage_artifact("extract"), SourceRecord)
        report = await verify_records(
            records=records,
            blocks_path=ws.stage_dir("parse") / "blocks.jsonl",
            client=self.verify_client,
            registry=self.registry,
            cache_dir=ws.stage_dir("verify") / "cache",
        )
        save_artifact(
            ws.stage_artifact("verify"),
            header=self._header(manifest, "verify",
                                model=self.verify_client.model,
                                prompt_version=self.registry.version_of("b", "factual_verifier.md"),
                                upstream=_artifact_digest(ws.stage_artifact("validate"))),
            data={"record_count": len(records),
                  "needs_review_count": sum(1 for r in records if r.needs_review)},
            records=records,
        )
        if report.corrections:
            save_artifact(
                ws.stage_dir("verify") / "correction_log.jsonl",
                header=self._header(manifest, "verify"),
                data={"correction_count": len(report.corrections)},
                records=report.corrections,
            )

    # ── 工具 ──────────────────────────────────────────────────────────────

    def _locate_source(self, ws: BookWorkspace, manifest: BookManifest) -> Path:
        """定位源文件：优先 00_source 副本；复制失败的退化路径（文本指针）回退。"""
        source_dir = ws.source_dir()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"源目录缺失: {source_dir}")
        for child in sorted(source_dir.iterdir()):
            if not child.is_file() or child.name != manifest.file_name:
                continue
            if child.suffix.lower() == ".pdf":
                # 真副本以 %PDF 魔数开头；否则是指向原始路径的文本指针
                if child.read_bytes()[:4] == b"%PDF":
                    return child
                candidate = Path(child.read_text(encoding="utf-8").strip())
                if candidate.is_file():
                    return candidate
            else:
                # Markdown 等文本源：本身就是内容
                return child
        raise FileNotFoundError(f"源文件缺失: {manifest.file_name}")

    def _header(
        self,
        manifest: BookManifest,
        stage: str,
        *,
        model: str | None = None,
        prompt_version: str | None = None,
        upstream: str | None = None,
    ) -> ArtifactHeader:
        return ArtifactHeader(
            schema_version="1",
            book_id=manifest.book_id,
            task_id=f"{manifest.book_id}:{stage}",
            model=model,
            prompt_version=prompt_version,
            upstream_hash=upstream,
        )


# ── 状态推进工具 ────────────────────────────────────────────────────────────


_IN_PROGRESS_STATUS: dict[str, BookStatus] = {
    "parse": BookStatus.PARSING,
    "profile": BookStatus.PROFILING,
    "plan": BookStatus.PLAN_GENERATED,
    "extract": BookStatus.EXTRACTING,
    "validate": BookStatus.VALIDATING,
    "verify": BookStatus.VERIFYING,
    "review": BookStatus.REVIEWED,
    "publish": BookStatus.PUBLISHED,
}


def _normalize_status(manifest: BookManifest, pre: BookStatus) -> None:
    """将状态对齐到阶段前置状态（阶段重跑/崩溃恢复后状态可能越位）。"""
    if manifest.status == BookStatus.FAILED:
        manifest.transition(BookStatus.PENDING)
    if manifest.status not in _STATUS_CHAIN or pre not in _STATUS_CHAIN:
        return
    cur = _STATUS_CHAIN.index(manifest.status)
    tgt = _STATUS_CHAIN.index(pre)
    if cur > tgt:
        manifest.rewind(pre)
    elif cur < tgt:
        for status in _STATUS_CHAIN[cur + 1 : tgt + 1]:
            manifest.transition(status)


def _advance_status(manifest: BookManifest, target: BookStatus) -> None:
    """沿状态链逐步推进到目标（跳过已完成阶段时的快进；禁止越位后向）。"""
    if manifest.status == BookStatus.FAILED:
        manifest.transition(BookStatus.PENDING)
    if manifest.status not in _STATUS_CHAIN or target not in _STATUS_CHAIN:
        return
    cur = _STATUS_CHAIN.index(manifest.status)
    tgt = _STATUS_CHAIN.index(target)
    if cur >= tgt:
        return  # 已在目标或更后（后向回退仅由 rewind 显式处理）
    for status in _STATUS_CHAIN[cur + 1 : tgt + 1]:
        manifest.transition(status)


def _artifact_digest(path: Path) -> str:
    """计算已落盘 Artifact 的内容 hash（去掉 header 的业务部分）。

    jsonl Artifact（首行 header + 记录行）只 hash 记录行，
    与 json Artifact 跳过 header 的语义一致（重跑重生成时 created_at 变化不影响 digest）。
    """
    if path.suffix == ".jsonl":
        lines = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        body = lines[1:]  # 首行为 header
        return artifact_hash(body)
    data = load_artifact(path)
    body = {k: v for k, v in data.items() if k != "header"}
    return artifact_hash(body)


def _record_count(artifact_path: Path) -> int | None:
    """从 Artifact 提取记录数（供 StageRecord.records）。"""
    if artifact_path.suffix == ".jsonl":
        if artifact_path.name == "verified_records.jsonl":
            return None  # 与 extract 相同数量，不重复统计
        return max(sum(1 for line in artifact_path.read_text(encoding="utf-8").splitlines() if line.strip()) - 1, 0)
    data = load_artifact(artifact_path)
    for key in ("record_count", "total_records", "total", "block_count", "total_blocks"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    return None
