"""Artifact 统一封装：元数据信封 + 原子读写 + hash 链。

每个阶段产物必须携带：
  - schema_version / book_id / task_id / created_at
  - model 信息 / prompt_version（涉及时）
  - upstream Artifact hash（构成可校验的血缘链）
Artifact 可以单独重新生成，不依赖上一步的内存变量。
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ArtifactHeader(BaseModel):
    """Artifact 元数据信封（所有阶段产物共享）。"""

    schema_version: str
    book_id: str
    task_id: str  # 生成本 Artifact 的运行标识
    created_at: str = Field(default_factory=lambda: _now_iso())
    model: str | None = None  # 生成模型（未调用 LLM 时为 None）
    prompt_version: str | None = None
    upstream_hash: str | None = None  # 直接上游 Artifact 的内容 hash
    extra: dict[str, Any] = Field(default_factory=dict)


def artifact_hash(payload: Any) -> str:
    """计算 Artifact 内容 hash（与存储无关的规范化 JSON）。"""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_artifact(
    path: Path,
    *,
    header: ArtifactHeader,
    data: BaseModel | dict[str, Any],
    records: list[BaseModel] | None = None,
) -> str:
    """保存带元数据信封的 Artifact，返回内容 hash。

    Args:
        path: 目标文件路径（.json 或 .jsonl 由扩展名决定）
        header: 元数据信封
        data: 主体数据（Pydantic 模型或 dict）
        records: 可选的记录数组（jsonl 模式下逐行写入）

    Returns:
        Artifact 内容 hash（不含 header 的业务部分）
    """
    body: dict[str, Any]
    if isinstance(data, BaseModel):
        body = data.model_dump(mode="json")
    else:
        body = data

    digest = artifact_hash(body)
    payload: dict[str, Any] = {"header": header.model_dump(mode="json"), **body}

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl" and records is not None:
        _atomic_write_jsonl(path, payload["header"], records)
    else:
        _atomic_write_json(path, payload)
    logger.debug("[Artifact] 写入 %s (hash=%s)", path.name, digest[:16])
    return digest


def load_artifact(path: Path) -> dict[str, Any]:
    """读取 JSON Artifact（含 header）；文件不存在抛 FileNotFoundError。"""
    if not path.is_file():
        raise FileNotFoundError(f"Artifact 不存在: {path}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "header" not in data:
        raise ValueError(f"Artifact 缺少 header 信封: {path}")
    return data


def load_artifact_model(path: Path, model_cls: type[T], *, key: str) -> T:
    """读取 Artifact 中的指定业务字段并解析为 Pydantic 模型。"""
    data = load_artifact(path)
    body = data.get(key)
    if body is None:
        raise ValueError(f"Artifact 缺少业务字段 '{key}': {path}")
    return model_cls.model_validate(body)


def load_artifact_records(path: Path, model_cls: type[T]) -> list[T]:
    """读取 jsonl Artifact 中逐行记录并解析为 Pydantic 模型列表。

    首行为 header 信封（``{"header": {...}}``），跳过；
    兼容行包裹 ``{"record": {...}}`` 与裸记录两种格式。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Artifact 不存在: {path}")
    records: list[T] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if set(item.keys()) == {"header"}:
            continue  # 首行 header 信封
        body = item.get("record", item)
        records.append(model_cls.model_validate(body))
    return records


# ── 原子写入 ─────────────────────────────────────────────────────────────────


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def _atomic_write_jsonl(path: Path, header: dict[str, Any], records: list[BaseModel]) -> None:
    """jsonl 首行为 header，其后每行一条记录。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    lines = [json.dumps({"header": header}, ensure_ascii=False)]
    for record in records:
        row = record.model_dump(mode="json")
        lines.append(json.dumps(row, ensure_ascii=False))
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
