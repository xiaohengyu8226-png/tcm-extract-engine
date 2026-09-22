"""项目路径约定；业务代码不应自行拼接仓库相对路径。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """项目关键目录路径集合。"""

    root: Path

    @property
    def prompts(self) -> Path:
        return self.root / "prompts"

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def books(self) -> Path:
        return self.workspace / "books"

    @property
    def gepa(self) -> Path:
        return self.workspace / "gepa"


def _find_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (
            parent / "src" / "tcm_extract_engine"
        ).is_dir():
            return parent
    return Path.cwd()


project_paths = ProjectPaths(_find_root())
