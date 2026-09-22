"""集中式日志配置：统一格式、控制台 + 文件双输出、自动轮转。

使用方式：
  - 应用入口（CLI / Web）调用 ``setup_logging()`` 一次即可。
  - 业务模块保持 ``logger = logging.getLogger(__name__)`` 不变。

环境变量：
  - LOG_LEVEL: 全局日志级别（DEBUG / INFO / WARNING / ERROR），默认 INFO
  - LOG_DIR:   日志文件目录，默认 <project_root>/logs
  - LOG_TO_FILE: 是否启用文件日志（1/0），默认 1
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from tcm_extract_engine.common.paths import project_paths

# ── 常量 ──────────────────────────────────────────────────────────────────────

_DEFAULT_LEVEL = "INFO"
_LOG_DIR_ENV = "LOG_DIR"
_LOG_LEVEL_ENV = "LOG_LEVEL"
_LOG_TO_FILE_ENV = "LOG_TO_FILE"

# 单文件最大 20 MB，保留最近 5 个轮转文件（共 ~120 MB 上限）
_MAX_BYTES = 20 * 1024 * 1024
_BACKUP_COUNT = 5

# 格式：时间 | 级别 | 模块 | 消息
_CONSOLE_FORMAT = "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s"
_FILE_FORMAT = (
    "%(asctime)s │ %(levelname)-7s │ %(name)s:%(lineno)d │ %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ── 公开接口 ──────────────────────────────────────────────────────────────────


def setup_logging(*, level: str | None = None, log_dir: Path | None = None) -> None:
    """初始化全局日志配置（幂等，重复调用安全）。

    Args:
        level: 日志级别，优先级：参数 > 环境变量 > 默认 INFO
        log_dir: 日志文件目录，优先级：参数 > 环境变量 > <root>/logs
    """
    root = logging.getLogger()

    # 幂等保护：避免 uvicorn reload 等场景重复添加 handler
    if getattr(root, "_tcm_engine_configured", False):
        return
    root._tcm_engine_configured = True  # type: ignore[attr-defined]

    # 解析级别
    effective_level = (
        level or os.getenv(_LOG_LEVEL_ENV, _DEFAULT_LEVEL)
    ).upper()
    root.setLevel(getattr(logging, effective_level, logging.INFO))

    # ── 控制台 Handler ────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console_handler)

    # ── 文件 Handler（可选）────────────────────────────────────────────────
    to_file = os.getenv(_LOG_TO_FILE_ENV, "1") not in ("0", "false", "False", "")
    if to_file:
        env_dir = os.getenv(_LOG_DIR_ENV, "").strip()
        effective_dir = log_dir or (Path(env_dir) if env_dir else None) or project_paths.root / "logs"
        effective_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            effective_dir / "tcm-engine.log",
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(file_handler)

    # ── 降低第三方库噪音 ──────────────────────────────────────────────────
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.getLogger(__name__).debug(
        "[Logging] 初始化完成: level=%s, file=%s", effective_level, to_file
    )
