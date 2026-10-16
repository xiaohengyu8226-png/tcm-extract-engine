"""Book 工作目录管理：分层 Artifact 目录 + manifest 生命周期。

一本文档的工作目录结构：

    workspace/books/<book_id>/
      manifest.json          # 唯一权威状态
      00_source/             # 源 PDF 副本/引用
      01_parse/              # MinerU 解析产物（content_list/middle/blocks）
      02_profile/            # document_profile.json + section_map.json
      03_plan/               # extraction_plan.json + prompt_a.md + prompt_b.md
      04_extract/            # records.jsonl + extraction_summary.json
      05_validate/           # validation_report.json + suspicious_records.jsonl
      06_verify/             # verified_records.jsonl + correction_log.jsonl
      07_review/             # review.xlsx + review_result.json
      08_publish/            # final_records.jsonl + safety_rules.jsonl + export.xlsx
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
from pathlib import Path

from .manifest import MANIFEST_FILENAME, BookManifest

logger = logging.getLogger(__name__)

# 阶段目录名（顺序即 STAGE_ORDER）
STAGE_DIRS: dict[str, str] = {
    "parse": "01_parse",
    "profile": "02_profile",
    "plan": "03_plan",
    "extract": "04_extract",
    "validate": "05_validate",
    "verify": "06_verify",
    "review": "07_review",
    "publish": "08_publish",
}

# 各阶段的主要 Artifact 文件名
STAGE_ARTIFACTS: dict[str, str] = {
    "parse": "parse_summary.json",
    "profile": "document_profile.json",
    "plan": "extraction_plan.json",
    "extract": "records.jsonl",
    "validate": "validation_report.json",
    "verify": "verified_records.jsonl",
    "review": "review_result.json",
    "publish": "safety_rules.jsonl",
}

_BOOK_ID_RE = re.compile(r"^[a-z0-9]{12}$")

# 每书适配块产物（落 03_plan/ 同级，非 stage artifact，不进 STAGE_ARTIFACTS）
ADAPTATION_FILENAME = "prompt_adaptations.json"

# 每书 goal 归纳产物（同落 03_plan/，按 profile_hash 缓存；非 stage artifact）
GOAL_FILENAME = "book_goal.json"


class BookWorkspace:
    """单本书的工作目录 + manifest 操作封装。"""

    def __init__(self, books_root: Path, book_id: str) -> None:
        if not _BOOK_ID_RE.match(book_id):
            raise ValueError(f"book_id 必须是 12 位小写十六进制: {book_id}")
        self.books_root = books_root
        self.book_id = book_id
        self.book_dir = books_root / book_id

    # ── 目录 ────────────────────────────────────────────────────────────────

    @property
    def manifest_path(self) -> Path:
        return self.book_dir / MANIFEST_FILENAME

    def stage_dir(self, stage: str) -> Path:
        dir_name = STAGE_DIRS.get(stage)
        if dir_name is None:
            raise ValueError(f"未知阶段: {stage}")
        return self.book_dir / dir_name

    def stage_artifact(self, stage: str) -> Path:
        return self.stage_dir(stage) / STAGE_ARTIFACTS[stage]

    def source_dir(self) -> Path:
        return self.book_dir / "00_source"

    # ── manifest 生命周期 ───────────────────────────────────────────────────

    @classmethod
    def create(
        cls, books_root: Path, pdf_path: Path, *, domain_id: str = "general"
    ) -> BookWorkspace:
        """为新 PDF 建立工作目录 + 初始 manifest（幂等：同 hash 复用）。"""
        from tcm_extract_engine.domains import get_domain_pack

        get_domain_pack(domain_id)
        file_hash = sha256_file(pdf_path)
        book_id = file_hash.removeprefix("sha256:")[:12]

        ws = cls(books_root, book_id)
        existing = ws.read_manifest()
        if existing is not None:
            if existing.file_hash == file_hash:
                if existing.domain_id != domain_id:
                    raise ValueError(
                        f"书籍已有领域 {existing.domain_id}，请求 {domain_id}；请显式切包后从 profile 重跑"
                    )
                logger.info("[Book %s] 已存在（hash 相同），复用", book_id)
                return ws
            raise ValueError(f"book_id 冲突: {book_id} hash 不一致")

        ws.book_dir.mkdir(parents=True, exist_ok=True)
        for dir_name in ["00_source", *STAGE_DIRS.values()]:
            (ws.book_dir / dir_name).mkdir(parents=True, exist_ok=True)

        # 源文件引用（复制小型 PDF；复制失败时退化为记录原始路径）
        manifest = BookManifest(
            book_id=book_id,
            domain_id=domain_id,
            file_name=pdf_path.name,
            file_hash=file_hash,
            file_size=pdf_path.stat().st_size,
        )
        source_link = ws.source_dir() / pdf_path.name
        if not source_link.exists():
            try:
                shutil.copy2(pdf_path, source_link)
            except OSError as exc:
                logger.warning("[Book %s] 源文件复制失败，记录引用: %s", book_id, exc)
                source_link.write_text(str(pdf_path), encoding="utf-8")
        manifest.save(ws.book_dir)
        logger.info("[Book %s] 创建工作目录: %s", book_id, ws.book_dir)
        return ws

    def read_manifest(self) -> BookManifest | None:
        """读取 manifest；不存在返回 None。"""
        if not self.manifest_path.is_file():
            return None
        return BookManifest.load(self.book_dir)

    def load_manifest(self) -> BookManifest:
        """读取 manifest；不存在时抛 FileNotFoundError。"""
        return BookManifest.load(self.book_dir)

    def save_manifest(self, manifest: BookManifest) -> None:
        manifest.save(self.book_dir)

    # ── 幂等判断 ────────────────────────────────────────────────────────────

    def is_processed(
        self,
        manifest: BookManifest | None = None,
        *,
        prompt_version: str | None = None,
    ) -> bool:
        """幂等检查：同 file_hash + pipeline_version + prompt_version 已成功则跳过。"""
        m = manifest or self.read_manifest()
        if m is None:
            return False
        if m.status.value == "PUBLISHED":
            if prompt_version and m.prompt_version and m.prompt_version != prompt_version:
                return False
            return True
        return False


def sha256_file(path: Path) -> str:
    """计算文件 SHA256（流式读取）。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def list_books(books_root: Path) -> list[BookWorkspace]:
    """列出 books_root 下全部书籍工作目录。"""
    if not books_root.is_dir():
        return []
    result: list[BookWorkspace] = []
    for child in sorted(books_root.iterdir()):
        if child.is_dir() and (child / MANIFEST_FILENAME).is_file():
            try:
                result.append(BookWorkspace(books_root, child.name))
            except ValueError:
                logger.warning("[Books] 跳过非法 book_id 目录: %s", child.name)
    return result
