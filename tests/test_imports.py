"""包内导入完整性冒烟测试。

防止两类回归：
1. 顶层 import 断裂——walk_packages 逐模块导入即可暴露；
2. 函数体内延迟 import 指向不存在的 vendored 模块——AST 扫描所有
   tcm_extract_engine.* 引用（含嵌套在函数/方法内的），用 find_spec 校验目标存在。

历史教训：拆分时 common/concurrency.py 漏拷，而 core/llm.py 与 parse/mineru.py
仅在函数体内延迟导入它，全部单测又 stub 掉真实客户端，导致 176 测试全绿，
却在真实 LLM/解析调用时 ModuleNotFoundError。本测试即为堵此盲区。
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pkgutil
from pathlib import Path

import tcm_extract_engine

_PACKAGE_ROOT = Path(tcm_extract_engine.__file__).resolve().parent
_PREFIX = "tcm_extract_engine"


def test_walk_packages_import_clean() -> None:
    """每个子模块都能被顶层导入，无语法错误 / 顶层 import 断裂。"""
    for mod in pkgutil.walk_packages(tcm_extract_engine.__path__, prefix=f"{_PREFIX}."):
        importlib.import_module(mod.name)


def test_all_internal_imports_resolvable() -> None:
    """AST 扫描所有内部 import（含函数体内延迟导入），目标模块必须真实存在。"""
    missing: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module and node.module.startswith(_PREFIX):
                    targets.append(node.module)
            elif isinstance(node, ast.Import):
                targets.extend(a.name for a in node.names if a.name.startswith(_PREFIX))
            for name in targets:
                if importlib.util.find_spec(name) is None:
                    rel = path.relative_to(_PACKAGE_ROOT)
                    missing.append(f"{rel}: {name}")
    assert not missing, "以下内部 import 指向不存在的模块：\n" + "\n".join(missing)
