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
from typing import Any

from tcm_extract_engine.cab.extractors import extract_records
from tcm_extract_engine.cab.planner import (
    BookAdaptation,
    BookAdaptedRegistry,
    PromptTemplateRegistry,
    build_extraction_plan,
    generate_book_adaptations,
    generate_book_goal,
    templates_for_profile,
)
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
from tcm_extract_engine.core.book import (
    ADAPTATION_FILENAME,
    GOAL_FILENAME,
    BookWorkspace,
    list_books,
)
from tcm_extract_engine.core.llm import UnifiedLLMClient
from tcm_extract_engine.core.manifest import (
    STAGE_ORDER,
    BookManifest,
    BookStatus,
)
from tcm_extract_engine.core.records import SourceRecord
from tcm_extract_engine.domains import get_domain_pack

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
        prompt_client: UnifiedLLMClient | None = None,
        structure_client: UnifiedLLMClient | None = None,
        vision_client: UnifiedLLMClient | None = None,
    ) -> None:
        self.books_root = books_root or project_paths.root / "workspace" / "books"
        self.registry = registry or PromptTemplateRegistry()
        self._extract_client = extract_client
        self._verify_client = verify_client
        self._prompt_client = prompt_client
        self._structure_client = structure_client
        self._vision_client = vision_client
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

    @property
    def prompt_client(self) -> UnifiedLLMClient:
        if self._prompt_client is None:
            from tcm_extract_engine.common.config import get_settings

            client = UnifiedLLMClient(get_settings().prompt_llm)
            self._prompt_client = client
            self._owned_clients.append(client)
        return self._prompt_client

    @property
    def structure_client(self) -> UnifiedLLMClient:
        """大纲修复专用 C-LLM 客户端（独立实例，与 prompt_client 各计各的调用数）。"""
        if self._structure_client is None:
            from tcm_extract_engine.common.config import get_settings

            client = UnifiedLLMClient(get_settings().prompt_llm)
            self._structure_client = client
            self._owned_clients.append(client)
        return self._structure_client

    @property
    def vision_client(self) -> UnifiedLLMClient:
        """多模态视觉客户端（图片批次专用，回退链见 settings.vision_llm）。"""
        if self._vision_client is None:
            from tcm_extract_engine.common.config import get_settings

            client = UnifiedLLMClient(get_settings().vision_llm)
            self._vision_client = client
            self._owned_clients.append(client)
        return self._vision_client

    async def close(self) -> None:
        for client in self._owned_clients:
            await client.close()
        self._owned_clients.clear()

    # ── 对外入口 ──────────────────────────────────────────────────────────

    async def run(
        self, source: Path, *, confirm_plan: bool = True, domain_id: str = "general"
    ) -> BookManifest:
        """新建（或复用）书籍工作目录并跑到 verify 完成。"""
        ws = BookWorkspace.create(self.books_root, source, domain_id=domain_id)
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

    def invalidate_from(self, book_id: str, stage: str) -> BookManifest:
        """作废 stage 及下游阶段记录；不回退状态（供蓝图/适配块编辑作废下游用）。

        与 ``rewind`` 的区别：不触碰 manifest.status —— 编辑发生在
        WAITING_PLAN_CONFIRMATION 确认门内（已解锁），作废下游后仍需人工
        重新确认才能再次抽取。
        """
        if stage not in STAGE_ORDER:
            raise ValueError(f"未知阶段: {stage}")
        ws = self._workspace(book_id)
        manifest = ws.load_manifest()
        idx = STAGE_ORDER.index(stage)
        for s in STAGE_ORDER[idx:]:
            manifest.stages.pop(s, None)
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
            stop = await self._run_one_stage(ws, manifest, stage, confirm_plan=confirm_plan)
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
            if stage == "verify":
                manifest.derived_artifacts_stale = False
            manifest.save(ws.book_dir)
            if stage == "verify":
                from tcm_extract_engine.core.export import export_knowledge
                export_knowledge(ws.book_dir)
            logger.info("[Book %s] %s 完成", ws.book_id, stage)
            if manifest.status == BookStatus.WAITING_PLAN_CONFIRMATION:
                logger.info(
                    "[Book %s] 计划待人工确认（tcm-engine book confirm %s）",
                    ws.book_id,
                    ws.book_id,
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
        """解析：MinerU/Markdown → blocks.jsonl（+ C-LLM 大纲修复）。"""
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
            # 记账用：MINERU_FALLBACK_TO_PIPELINE 会静默降级，而不同 backend 的标题检出
            # 并不等价；只记请求值会让人以为拿到的是 hybrid 产物。Markdown 直转为 None。
            "mineru_backend_used": result.mineru_backend_used,
            "mineru_degraded": result.mineru_degraded,
        }

        # 大纲修复：解析器标题层级平铺（text_level 仅 1/2）。书签优先——PDF 内嵌
        # 大纲可用时确定性定级（零 LLM）；否则由 C-LLM 按语义重建章/节/条层级并
        # 剔除目录/前言等假标题；失败或禁用时 blocks 保持原样，流水线不中断。
        # 修复合法的路径重写 blocks.jsonl（保留 meta 首行）。
        from tcm_extract_engine.cab.outline import (
            load_pdf_outline,
            repair_blocks,
            rewrite_blocks_file,
        )
        from tcm_extract_engine.cab.profiler import load_blocks
        from tcm_extract_engine.common.config import get_settings

        outline_note: dict | None = None
        if get_settings().structure_repair_enabled:
            settings = get_settings()
            bookmarks: list[dict[str, Any]] | None = None
            if settings.structure_bookmarks_enabled and source.suffix.lower() == ".pdf":
                bookmarks = load_pdf_outline(source)
            blocks = load_blocks(blocks_dest)
            blocks, repair_meta = await repair_blocks(
                blocks,
                client=self.structure_client,
                registry=self.registry,
                cache_dir=ws.stage_dir("parse") / "cache",
                bookmarks=bookmarks,
            )
            outline_note = repair_meta.to_dict()
            summary["outline_repair"] = outline_note
            if repair_meta.source in {"llm", "cache", "bookmarks", "numbering"}:
                rewrite_blocks_file(blocks_dest, blocks, meta_note=outline_note)
                summary["block_count"] = len(blocks)

        save_artifact(
            ws.stage_artifact("parse"),
            header=self._header(manifest, "parse"),
            data=summary,
        )

    async def _run_profile(self, ws: BookWorkspace, manifest: BookManifest) -> None:
        """画像：blocks → document_profile.json。"""
        blocks_path = ws.stage_dir("parse") / "blocks.jsonl"
        profile = build_profile(
            blocks_path, book_name=manifest.file_name, pack=get_domain_pack(manifest.domain_id)
        )
        manifest.source_type = profile.document_type.value
        save_artifact(
            ws.stage_artifact("profile"),
            header=self._header(
                manifest, "profile", upstream=_artifact_digest(ws.stage_artifact("parse"))
            ),
            data=profile.model_dump(mode="json"),
        )

    async def _run_plan(
        self, ws: BookWorkspace, manifest: BookManifest, *, confirm_plan: bool = True
    ) -> None:
        """计划：profile → extraction_plan.json（Model C：确定性规则 + 每书 LLM 适配块）。"""
        profile_data = load_artifact(ws.stage_artifact("profile"))
        profile = DocumentProfile.model_validate(profile_data)

        # 每书适配块（缓存优先，按 profile hash 命中；关闭或全降级则空）
        adaptations = await self._resolve_adaptations(ws, manifest, profile)
        registry = BookAdaptedRegistry(
            {t: a.block for t, a in adaptations.items() if a.block},
            prompts_root=self.registry.prompts_root,
        )

        # 每书 goal（Model C 归纳，缓存优先；失败/关闭回退领域包默认）
        goal = await self._resolve_goal(ws, manifest, profile)

        plan = build_extraction_plan(
            profile, registry=registry, goal=goal, pack=get_domain_pack(manifest.domain_id)
        )
        save_artifact(
            ws.stage_artifact("plan"),
            header=self._header(
                manifest, "plan", upstream=_artifact_digest(ws.stage_artifact("profile"))
            ),
            data={**plan.model_dump(mode="json"), "domain_id": manifest.domain_id},
        )
        # 幂等键的 prompt 版本：全部计划章节 prompt_version 的合成 hash
        versions = sorted({s.prompt_version for s in plan.sections if s.prompt_version})
        manifest.prompt_version = (
            artifact_hash({"prompt_versions": versions})[:20] if versions else None
        )

    async def _resolve_adaptations(
        self, ws: BookWorkspace, manifest: BookManifest, profile: DocumentProfile
    ) -> dict[str, BookAdaptation]:
        """载入或生成本书适配块。

        - 缓存命中（profile_hash 一致）→ 直接复用，LLM 零调用（可复现）。
        - 开关关闭 → 空适配（base-only），并清理遗留产物保证与 extract 一致。
        - 否则调 C-LLM 生成并落盘；整体失败优雅降级为 base-only（不阻断流水线）。
        """
        from tcm_extract_engine.common.config import get_settings

        adapt_path = self._adaptation_path(ws)
        profile_hash = artifact_hash(profile.model_dump(mode="json"))

        # 1) 缓存命中：profile 未变 → 复用
        if adapt_path.is_file():
            try:
                cached = load_artifact(adapt_path)
            except Exception:  # noqa: BLE001 - 坏缓存视为无
                cached = None
            if cached and cached.get("profile_hash") == profile_hash:
                return _parse_adaptations(cached)

        # 2) 开关关闭 → 空适配，清理遗留文件（防 extract 误注入旧适配块）
        if not get_settings().plan_adaptation_enabled:
            adapt_path.unlink(missing_ok=True)
            return {}

        # 3) 生成
        templates = templates_for_profile(profile)
        if not templates:
            adapt_path.unlink(missing_ok=True)
            return {}
        try:
            adaptations = await generate_book_adaptations(
                profile=profile,
                templates=templates,
                client=self.prompt_client,
                registry=self.registry,
            )
        except Exception as exc:  # noqa: BLE001 - 整体失败降级为 base-only
            logger.warning("[Book %s] 适配块生成失败，降级为 base-only: %s", ws.book_id, exc)
            adapt_path.unlink(missing_ok=True)
            return {}

        save_artifact(
            adapt_path,
            header=self._header(manifest, "plan", model=self.prompt_client.model),
            data={
                "profile_hash": profile_hash,
                "adaptations": {t: a.model_dump(mode="json") for t, a in adaptations.items()},
            },
        )
        return adaptations

    def _adaptation_path(self, ws: BookWorkspace) -> Path:
        return ws.stage_dir("plan") / ADAPTATION_FILENAME

    async def _resolve_goal(
        self, ws: BookWorkspace, manifest: BookManifest, profile: DocumentProfile
    ) -> str:
        """载入或生成本书 goal（每书自适应；缓存按 profile_hash）。

        - 缓存命中（profile_hash 一致且 goal 非空）→ 复用，LLM 零调用。
        - 开关关闭 → 回退领域包默认（返回 ""），并清理遗留产物。
        - 否则调 C-LLM 归纳并落盘；失败优雅降级为 ""（build_extraction_plan 回退默认）。
        """
        from tcm_extract_engine.common.config import get_settings

        goal_path = self._goal_path(ws)
        profile_hash = artifact_hash(profile.model_dump(mode="json"))

        # 1) 缓存命中：profile 未变 → 复用
        if goal_path.is_file():
            try:
                cached = load_artifact(goal_path)
            except Exception:  # noqa: BLE001 - 坏缓存视为无
                cached = None
            if cached and cached.get("profile_hash") == profile_hash:
                goal = str(cached.get("goal") or "").strip()
                if goal:
                    return goal

        # 2) 开关关闭 → 回退领域包默认，清理遗留文件
        if not get_settings().plan_goal_enabled:
            goal_path.unlink(missing_ok=True)
            return ""

        # 3) 生成（generate_book_goal 内部已优雅降级；空结果不落盘）
        goal = await generate_book_goal(
            profile=profile, client=self.prompt_client, registry=self.registry
        )
        if goal:
            save_artifact(
                goal_path,
                header=self._header(manifest, "plan", model=self.prompt_client.model),
                data={"profile_hash": profile_hash, "goal": goal},
            )
        return goal

    def _goal_path(self, ws: BookWorkspace) -> Path:
        return ws.stage_dir("plan") / GOAL_FILENAME

    def _extract_registry(self, ws: BookWorkspace) -> PromptTemplateRegistry:
        """由已落盘适配块重建注入 registry；无适配块则用 base registry。

        与 _run_plan 的适配状态保持一致（plan 写、extract 读同一产物），
        保证 resume/独立进程重跑 extract 时合成 prompt_version 与缓存键自洽。
        """
        adapt_path = self._adaptation_path(ws)
        if not adapt_path.is_file():
            return self.registry
        try:
            data = load_artifact(adapt_path)
        except Exception:  # noqa: BLE001 - 坏文件退回 base
            return self.registry
        blocks = {t: a.block for t, a in _parse_adaptations(data).items() if a.block}
        if not blocks:
            return self.registry
        return BookAdaptedRegistry(blocks, prompts_root=self.registry.prompts_root)

    async def _run_extract(self, ws: BookWorkspace, manifest: BookManifest) -> None:
        """抽取（Model A）：plan + blocks → records.jsonl（叠加每书适配块）。"""
        records = await extract_records(
            book_id=ws.book_id,
            blocks_path=ws.stage_dir("parse") / "blocks.jsonl",
            plan_path=ws.stage_artifact("plan"),
            cache_dir=ws.stage_dir("extract") / "cache",
            client=self.extract_client,
            vision_client=self.vision_client,
            registry=self._extract_registry(ws),
            pack=get_domain_pack(manifest.domain_id),
        )
        save_artifact(
            ws.stage_artifact("extract"),
            header=self._header(
                manifest,
                "extract",
                model=self.extract_client.model,
                prompt_version=manifest.prompt_version,
                upstream=_artifact_digest(ws.stage_artifact("plan")),
            ),
            data={"record_count": len(records)},
            records=records,
        )

    async def _run_validate(self, ws: BookWorkspace, manifest: BookManifest) -> None:
        """确定性校验：records → validation_report.json + suspicious_records.jsonl。"""
        records = load_artifact_records(ws.stage_artifact("extract"), SourceRecord)
        blocks = load_blocks(ws.stage_dir("parse") / "blocks.jsonl")
        report = validate_records(records, blocks=blocks, pack=get_domain_pack(manifest.domain_id))

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
            header=self._header(
                manifest, "validate", upstream=_artifact_digest(ws.stage_artifact("extract"))
            ),
            data=report.model_dump(mode="json"),
        )

    async def _run_verify(self, ws: BookWorkspace, manifest: BookManifest) -> None:
        """核验（Model B）：records + blocks → verified_records + correction_log。"""
        records = load_artifact_records(ws.stage_artifact("extract"), SourceRecord)
        validation_path = ws.stage_artifact("validate")
        if validation_path.is_file():
            validation = load_artifact(validation_path)
            flagged = {i["record_id"] for i in validation.get("issues", []) if i.get("severity") in {"ERROR", "WARNING"}}
            for record in records:
                if record.record_id in flagged:
                    record.needs_review = True
        report = await verify_records(
            records=records,
            blocks_path=ws.stage_dir("parse") / "blocks.jsonl",
            client=self.verify_client,
            vision_client=self.vision_client,
            pack=get_domain_pack(manifest.domain_id),
            registry=self.registry,
            cache_dir=ws.stage_dir("verify") / "cache",
        )
        save_artifact(
            ws.stage_artifact("verify"),
            header=self._header(
                manifest,
                "verify",
                model=self.verify_client.model,
                prompt_version=self.registry.version_of("b", "factual_verifier.md"),
                upstream=_artifact_digest(ws.stage_artifact("validate")),
            ),
            data={
                "record_count": len(records),
                "needs_review_count": sum(1 for r in records if r.needs_review),
            },
            records=records,
        )
        save_artifact(
            ws.stage_dir("verify") / "correction_log.jsonl",
            header=self._header(manifest, "verify"),
            data={"correction_count": len(report.corrections)},
            records=report.corrections,
        )
        # 权威 per-run verdict 工件：逐条核验结论（含全部 passthrough）落盘，供 review
        # 导出直接按 record_id 读取。避免下游合并原始缓存目录时，因全局 record_id 重排
        # 而被上一运行的陈旧 verdict 撞车污染（历史 WRONG_NORMALIZATION 1→48 即此因）。
        save_artifact(
            ws.stage_dir("verify") / "verification_results.jsonl",
            header=self._header(
                manifest,
                "verify",
                model=self.verify_client.model,
                prompt_version=self.registry.version_of("b", "factual_verifier.md"),
            ),
            data={"total": report.total, "verdict_counts": report.verdict_counts()},
            records=report.results,
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
            extra={
                "domain_id": manifest.domain_id,
                "pack_version": get_domain_pack(manifest.domain_id).version,
                "record_schema_version": "2",
            },
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


def _parse_adaptations(data: dict[str, Any]) -> dict[str, BookAdaptation]:
    """从适配产物（load_artifact 结果）解析 template → BookAdaptation。"""
    raw = data.get("adaptations") or {}
    out: dict[str, BookAdaptation] = {}
    if isinstance(raw, dict):
        for template, payload in raw.items():
            if isinstance(payload, dict):
                out[str(template)] = BookAdaptation.model_validate(payload)
    return out


def _record_count(artifact_path: Path) -> int | None:
    """从 Artifact 提取记录数（供 StageRecord.records）。"""
    if artifact_path.suffix == ".jsonl":
        if artifact_path.name == "verified_records.jsonl":
            return None  # 与 extract 相同数量，不重复统计
        return max(
            sum(
                1 for line in artifact_path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
            - 1,
            0,
        )
    data = load_artifact(artifact_path)
    for key in ("record_count", "total_records", "total", "block_count", "total_blocks"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    return None
