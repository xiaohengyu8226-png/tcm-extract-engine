"""统一配置：从 .env 加载，暴露类型化 Settings。

三模型独立配置：
  - 提示词生成模型 (prompt)：低频、高创造力
  - 抽取模型 (extract)：高频、结构化输出主力
  - 校验模型 (verify)：中频、需要深度推理

每个模型独立配置 base_url / api_key / model / timeout / max_tokens / thinking，
未显式设置时回退到全局 OPENAI_* 默认值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from tcm_extract_engine.common.paths import project_paths


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """单个模型的完整配置。"""

    base_url: str
    api_key: str
    model: str
    timeout: int
    max_tokens: int
    thinking_enabled: bool


@dataclass(frozen=True, slots=True)
class Settings:
    """全局配置，字段均有合理默认值。"""

    # ── 全局默认（三模型未显式配置时的回退值） ─────────────────────
    openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    openai_api_key: str = ""
    openai_model: str = "deepseek-v4-flash-202605"
    llm_timeout: int = 600
    llm_max_tokens: int = 65536  # 最大输出 token（非上下文窗口）
    llm_concurrency: int = 8  # LLM 并发请求数

    # ── 提示词生成模型（低频、高创造力） ──
    prompt_model: str = ""  # 空 = 使用 openai_model
    prompt_base_url: str = ""  # 空 = 使用 openai_base_url
    prompt_api_key: str = ""  # 空 = 使用 openai_api_key
    prompt_timeout: int = 0  # 0 = 使用 llm_timeout
    prompt_max_tokens: int = 0  # 0 = 使用 llm_max_tokens
    prompt_thinking_enabled: bool = False

    # ── 抽取模型（高频主力） ───
    extract_model: str = ""  # 空 = 使用 openai_model
    extract_base_url: str = ""  # 空 = 使用 openai_base_url
    extract_api_key: str = ""  # 空 = 使用 openai_api_key
    extract_timeout: int = 0  # 0 = 使用 llm_timeout
    extract_max_tokens: int = 0  # 0 = 使用 llm_max_tokens
    extract_thinking_enabled: bool = False

    # ── 校验模型（需要深度推理） ──
    verify_model: str = ""  # 空 = 使用 openai_model
    verify_base_url: str = ""  # 空 = 使用 openai_base_url
    verify_api_key: str = ""  # 空 = 使用 openai_api_key
    verify_timeout: int = 0  # 0 = 使用 llm_timeout
    verify_max_tokens: int = 0  # 0 = 使用 llm_max_tokens
    verify_thinking_enabled: bool = True  # 校验默认开启思考模式
    verify_concurrency: int = 8  # 校验批次并发数

    # Pipeline
    llm_batch_size: int = 5  # 每次 LLM 调用包含的 chunk 数
    parse_retries: int = 2

    # MinerU
    mineru_max_concurrent_parse: int = 1  # 进程内 Pipeline backend 默认串行，保护 GPU/显存

    # ── 便捷方法：获取各阶段的解析后 ModelConfig ────────────────────

    @property
    def prompt_llm(self) -> ModelConfig:
        """提示词生成阶段的模型配置。"""
        return ModelConfig(
            base_url=self.prompt_base_url or self.openai_base_url,
            api_key=self.prompt_api_key or self.openai_api_key,
            model=self.prompt_model or self.openai_model,
            timeout=self.prompt_timeout or self.llm_timeout,
            max_tokens=self.prompt_max_tokens or self.llm_max_tokens,
            thinking_enabled=self.prompt_thinking_enabled,
        )

    @property
    def extract_llm(self) -> ModelConfig:
        """抽取阶段的模型配置。"""
        return ModelConfig(
            base_url=self.extract_base_url or self.openai_base_url,
            api_key=self.extract_api_key or self.openai_api_key,
            model=self.extract_model or self.openai_model,
            timeout=self.extract_timeout or self.llm_timeout,
            max_tokens=self.extract_max_tokens or self.llm_max_tokens,
            thinking_enabled=self.extract_thinking_enabled,
        )

    @property
    def verify_llm(self) -> ModelConfig:
        """校验阶段的模型配置。"""
        return ModelConfig(
            base_url=self.verify_base_url or self.openai_base_url,
            api_key=self.verify_api_key or self.openai_api_key,
            model=self.verify_model or self.openai_model,
            timeout=self.verify_timeout or self.llm_timeout,
            max_tokens=self.verify_max_tokens or self.llm_max_tokens,
            thinking_enabled=self.verify_thinking_enabled,
        )

    @classmethod
    def from_env(cls) -> Settings:
        """从 .env 文件和环境变量加载配置。"""
        load_dotenv(project_paths.root / ".env", override=False)
        defaults = cls()

        def _bool(key: str, default: bool) -> bool:
            return os.getenv(key, "1" if default else "0") not in ("0", "false", "False", "")

        def _int(key: str, default: int) -> int:
            return int(os.getenv(key, str(default)))

        return cls(
            # 全局默认
            openai_base_url=os.getenv("OPENAI_BASE_URL", defaults.openai_base_url),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", defaults.openai_model),
            llm_timeout=_int("LLM_TIMEOUT", defaults.llm_timeout),
            llm_max_tokens=_int("LLM_MAX_TOKENS", defaults.llm_max_tokens),
            llm_concurrency=_int("LLM_CONCURRENCY", defaults.llm_concurrency),
            # 提示词生成模型
            prompt_model=os.getenv("PROMPT_MODEL", ""),
            prompt_base_url=os.getenv("PROMPT_BASE_URL", ""),
            prompt_api_key=os.getenv("PROMPT_API_KEY", ""),
            prompt_timeout=_int("PROMPT_TIMEOUT", 0),
            prompt_max_tokens=_int("PROMPT_MAX_TOKENS", 0),
            prompt_thinking_enabled=_bool("PROMPT_THINKING_ENABLED", False),
            # 抽取模型
            extract_model=os.getenv("EXTRACT_MODEL", ""),
            extract_base_url=os.getenv("EXTRACT_BASE_URL", ""),
            extract_api_key=os.getenv("EXTRACT_API_KEY", ""),
            extract_timeout=_int("EXTRACT_TIMEOUT", 0),
            extract_max_tokens=_int("EXTRACT_MAX_TOKENS", 0),
            extract_thinking_enabled=_bool("EXTRACT_THINKING_ENABLED", False),
            # 校验模型
            verify_model=os.getenv("VERIFY_MODEL", ""),
            verify_base_url=os.getenv("VERIFY_BASE_URL", ""),
            verify_api_key=os.getenv("VERIFY_API_KEY", ""),
            verify_timeout=_int("VERIFY_TIMEOUT", 0),
            verify_max_tokens=_int("VERIFY_MAX_TOKENS", 0),
            verify_thinking_enabled=_bool("VERIFY_THINKING_ENABLED", True),
            verify_concurrency=_int("VERIFY_CONCURRENCY", defaults.verify_concurrency),
            # Pipeline
            llm_batch_size=_int("LLM_BATCH_SIZE", defaults.llm_batch_size),
            parse_retries=_int("PARSE_RETRIES", defaults.parse_retries),
            mineru_max_concurrent_parse=_int(
                "MINERU_MAX_CONCURRENT_PARSE", defaults.mineru_max_concurrent_parse
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局配置单例。"""
    return Settings.from_env()
