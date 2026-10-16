"""图片加载与预处理：支持本地文件、URL、base64 转换。

为多模态 LLM 调用提供图片加载能力：
  - 本地文件 → base64 data URL
  - 尺寸压缩（保持宽高比）
  - 格式转换（统一 JPEG）
  - 缓存友好（按路径+尺寸 hash）

依赖：Pillow（已在 mineru extra 中）
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 默认图片最大尺寸（像素）
DEFAULT_MAX_SIZE = 1024
# 默认 JPEG 质量
DEFAULT_JPEG_QUALITY = 85


class ImageLoadError(Exception):
    """图片加载错误。"""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"图片加载失败 [{path}]: {reason}")


class ImageLoader:
    """图片加载器：支持本地文件、URL、base64。"""

    @staticmethod
    def load_as_base64(
        path: str | Path,
        *,
        max_size: int = DEFAULT_MAX_SIZE,
        quality: int = DEFAULT_JPEG_QUALITY,
    ) -> str:
        """加载图片并转为 base64（带尺寸压缩）。

        Args:
            path: 图片文件路径
            max_size: 最大边长（像素），超过则等比缩放
            quality: JPEG 压缩质量（1-100）

        Returns:
            base64 编码的图片数据

        Raises:
            ImageLoadError: 文件不存在或无法解析
        """
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImageLoadError(
                str(path), "Pillow 未安装，请运行 pip install pillow"
            ) from exc

        path = Path(path)
        if not path.is_file():
            raise ImageLoadError(str(path), "文件不存在")

        try:
            img = Image.open(path)
            img.load()  # 确保图片数据已加载
        except Exception as exc:
            raise ImageLoadError(str(path), f"无法解析图片: {exc}") from exc

        # 转换为 RGB（处理 RGBA、P 等模式）
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # 压缩到 max_size（保持宽高比）
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(
                "[ImageLoader] 压缩图片 %s: %s → %s",
                path.name, img.size, new_size,
            )

        # 转为 base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def to_data_url(
        path: str | Path,
        *,
        max_size: int = DEFAULT_MAX_SIZE,
        quality: int = DEFAULT_JPEG_QUALITY,
    ) -> str:
        """转为 OpenAI Vision API 格式的 data URL。

        Returns:
            data:image/jpeg;base64,... 格式的 URL
        """
        b64 = ImageLoader.load_as_base64(path, max_size=max_size, quality=quality)
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def to_image_url_dict(
        path: str | Path,
        *,
        max_size: int = DEFAULT_MAX_SIZE,
        quality: int = DEFAULT_JPEG_QUALITY,
    ) -> dict[str, str]:
        """转为 OpenAI Vision API 格式的 image_url 字典。

        Returns:
            {"url": "data:image/jpeg;base64,..."}
        """
        return {"url": ImageLoader.to_data_url(path, max_size=max_size, quality=quality)}

    @staticmethod
    def image_hash(path: str | Path) -> str:
        """计算图片内容 hash（用于缓存键）。

        Returns:
            sha256 hash 前 16 位
        """
        path = Path(path)
        if not path.is_file():
            return ""
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return digest

    @staticmethod
    def get_image_info(path: str | Path) -> dict[str, Any]:
        """获取图片元信息。

        Returns:
            {"width": int, "height": int, "mode": str, "format": str, "size_bytes": int}
        """
        try:
            from PIL import Image
        except ImportError:
            return {}

        path = Path(path)
        if not path.is_file():
            return {}

        try:
            img = Image.open(path)
            return {
                "width": img.size[0],
                "height": img.size[1],
                "mode": img.mode,
                "format": img.format,
                "size_bytes": path.stat().st_size,
            }
        except Exception:
            return {}

    @staticmethod
    def is_valid_image(path: str | Path) -> bool:
        """检查文件是否为有效图片。"""
        try:
            from PIL import Image
        except ImportError:
            return False

        path = Path(path)
        if not path.is_file():
            return False

        try:
            img = Image.open(path)
            img.verify()  # 验证图片完整性
            return True
        except Exception:
            return False


def load_drawing_block_image(
    block: dict[str, Any],
    *,
    max_size: int = DEFAULT_MAX_SIZE,
) -> str | None:
    """从 drawing block 加载图片并转为 data URL。

    Args:
        block: MinerU 产生的 drawing block（需有 asset_path 字段）
        max_size: 最大边长

    Returns:
        data URL 字符串，加载失败返回 None
    """
    asset_path = block.get("asset_path", "")
    if not asset_path:
        logger.debug("[ImageLoader] block 无 asset_path: %s", block.get("blockid", "?"))
        return None

    path = Path(asset_path)
    if not path.is_file():
        logger.warning("[ImageLoader] 图片文件不存在: %s", asset_path)
        return None

    try:
        return ImageLoader.to_data_url(path, max_size=max_size)
    except ImageLoadError as exc:
        logger.warning("[ImageLoader] 加载失败: %s", exc)
        return None
