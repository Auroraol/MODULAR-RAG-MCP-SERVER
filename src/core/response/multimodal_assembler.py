"""用于组合文本与图像内容的多模态组装器。

本模块通过以下方式组装多模态 MCP 响应：
- 检测 Chunk 元数据中的图像引用
- 读取图像文件并编码为 base64
- 构建符合 MCP 规范的 ImageContent 块

设计原则：
- 惰性加载：仅在显式请求时才加载图像
- 错误容错：图像缺失不会破坏文本响应
- 格式检测：根据文件内容自动推断 MIME 类型
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mcp import types

from src.core.types import RetrievalResult


logger = logging.getLogger(__name__)


# 支持的图像格式及其 MIME 类型
MIME_TYPE_MAP: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}

# 用于格式检测的魔数字节（扩展名不可靠时的回退）
MAGIC_BYTES: Dict[bytes, str] = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",  # WebP 以 RIFF 开头
    b"BM": "image/bmp",
}

# 文本中的图像占位符模式：[IMAGE: image_id]
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


@dataclass
class ImageReference:
    """Chunk 内对图像的引用。

    属性:
        image_id: 图像的唯一标识
        file_path: 图像文件的文件系统路径
        page: 来源文档中的可选页码
        text_offset: 占位符在 Chunk 文本中的字符偏移
        text_length: 占位符字符串的长度
        caption: 由 ImageCaptioner 生成的可选说明文字
    """
    image_id: str
    file_path: Optional[str] = None
    page: Optional[int] = None
    text_offset: Optional[int] = None
    text_length: Optional[int] = None
    caption: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "image_id": self.image_id,
            "file_path": self.file_path,
            "page": self.page,
            "text_offset": self.text_offset,
            "text_length": self.text_length,
            "caption": self.caption,
        }


@dataclass
class ImageContent:
    """已加载、可用于 MCP 响应的图像内容。

    属性:
        image_id: 图像的唯一标识
        data: Base64 编码的图像数据
        mime_type: MIME 类型（例如 "image/png"）
        caption: 图像的可选说明文字
    """
    image_id: str
    data: str  # base64 编码
    mime_type: str
    caption: Optional[str] = None

    def to_mcp_content(self) -> types.ImageContent:
        """转换为 MCP ImageContent 块。

        返回:
            用于协议响应的 MCP ImageContent 对象。
        """
        return types.ImageContent(
            type="image",
            data=self.data,
            mimeType=self.mime_type,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "image_id": self.image_id,
            "data": self.data,
            "mime_type": self.mime_type,
            "caption": self.caption,
        }


class MultimodalAssembler:
    """从检索结果组装多模态内容。

    本类从 Chunk 元数据中提取图像引用、加载图像文件，
    并构建组合文本与图像的符合 MCP 规范的内容块。

    示例:
        >>> assembler = MultimodalAssembler()
        >>> results = [RetrievalResult(chunk_id="doc1_001", ...)]
        >>> content_blocks = assembler.assemble(results)
        >>> # 返回 TextContent 与 ImageContent 块列表

    参数:
        image_storage: 可选的 ImageStorage 实例，用于路径查找。
            为 None 时直接使用元数据中的文件路径。
        max_images_per_result: 每个结果包含的最大图像数量（默认：5）。
        include_captions: 是否将图像说明文字作为文本包含（默认：True）。
    """

    def __init__(
        self,
        image_storage: Optional[Any] = None,
        max_images_per_result: int = 5,
        include_captions: bool = True,
    ) -> None:
        """初始化 MultimodalAssembler。

        参数:
            image_storage: 可选，用于解析图像路径的 ImageStorage。
            max_images_per_result: 每个检索结果包含的最大图像数量。
            include_captions: 是否添加说明文字文本块。
        """
        self._image_storage = image_storage
        self.max_images_per_result = max_images_per_result
        self.include_captions = include_captions

    def extract_image_refs(
        self,
        result: RetrievalResult,
    ) -> List[ImageReference]:
        """从检索结果中提取图像引用。

        从以下位置查找图像引用：
        1. metadata.images 列表（结构化的图像信息）
        2. 文本中的 [IMAGE: id] 占位符（回退）

        参数:
            result: 包含 Chunk 数据的 RetrievalResult。

        返回:
            在结果中找到的 ImageReference 对象列表。
        """
        refs: List[ImageReference] = []
        metadata = result.metadata or {}

        # 主要来源：metadata 中的结构化 images 列表
        images_list = metadata.get("images", [])
        if isinstance(images_list, list):
            for img_info in images_list[:self.max_images_per_result]:
                if isinstance(img_info, dict) and "id" in img_info:
                    ref = ImageReference(
                        image_id=img_info["id"],
                        file_path=img_info.get("path"),
                        page=img_info.get("page"),
                        text_offset=img_info.get("text_offset"),
                        text_length=img_info.get("text_length"),
                    )
                    refs.append(ref)

        # 如有可用说明文字则添加
        captions = metadata.get("image_captions", {})
        if isinstance(captions, dict):
            for ref in refs:
                if ref.image_id in captions:
                    ref.caption = captions[ref.image_id]

        # 回退：无结构化引用时从文本中解析占位符
        if not refs and result.text:
            placeholders = IMAGE_PLACEHOLDER_PATTERN.findall(result.text)
            for image_id in placeholders[:self.max_images_per_result]:
                image_id = image_id.strip()
                ref = ImageReference(
                    image_id=image_id,
                    caption=captions.get(image_id) if isinstance(captions, dict) else None,
                )
                refs.append(ref)

        return refs

    def resolve_image_path(
        self,
        ref: ImageReference,
        collection: Optional[str] = None,
    ) -> Optional[str]:
        """解析图像引用的文件系统路径。

        参数:
            ref: 待解析的 ImageReference。
            collection: 可选的 collection 名称，用于构建路径。

        返回:
            找到时返回绝对文件路径，否则返回 None。
        """
        # 如有显式路径则使用
        if ref.file_path:
            path = Path(ref.file_path)
            if path.exists():
                return str(path.resolve())

        # 尝试通过 ImageStorage 查找
        if self._image_storage is not None:
            try:
                path = self._image_storage.get_image_path(ref.image_id)
                if path and Path(path).exists():
                    return path
            except Exception as e:
                logger.warning(f"ImageStorage lookup failed for {ref.image_id}: {e}")

        # 基于约定的路径：data/images/{collection}/{image_id}.png
        if collection:
            from src.core.settings import resolve_path
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                candidate = resolve_path(f"data/images/{collection}/{ref.image_id}{ext}")
                if candidate.exists():
                    return str(candidate.resolve())

        return None

    def load_image(
        self,
        file_path: str,
    ) -> Optional[ImageContent]:
        """加载并编码图像文件。

        参数:
            file_path: 图像文件路径。

        返回:
            包含 base64 数据与 MIME 类型的 ImageContent，失败时返回 None。
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Image file not found: {file_path}")
                return None

            # 读取文件内容
            data = path.read_bytes()
            if not data:
                logger.warning(f"Empty image file: {file_path}")
                return None

            # 检测 MIME 类型
            mime_type = self._detect_mime_type(path, data)

            # 编码为 base64
            base64_data = base64.b64encode(data).decode("utf-8")

            return ImageContent(
                image_id=path.stem,
                data=base64_data,
                mime_type=mime_type,
            )

        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {e}")
            return None

    def _detect_mime_type(
        self,
        path: Path,
        data: bytes,
    ) -> str:
        """根据文件扩展名或魔数字节检测 MIME 类型。

        参数:
            path: 用于扩展名检测的文件路径。
            data: 用于魔数字节检测的文件内容。

        返回:
            MIME 类型字符串（未知时默认 "image/png"）。
        """
        # 优先尝试扩展名
        suffix = path.suffix.lower()
        if suffix in MIME_TYPE_MAP:
            return MIME_TYPE_MAP[suffix]

        # 回退到魔数字节
        for magic, mime_type in MAGIC_BYTES.items():
            if data.startswith(magic):
                return mime_type

        # 默认为 PNG
        logger.debug(f"Unknown image format for {path}, defaulting to image/png")
        return "image/png"

    def assemble_for_result(
        self,
        result: RetrievalResult,
        collection: Optional[str] = None,
    ) -> List[Union[types.TextContent, types.ImageContent]]:
        """为单个结果组装多模态内容块。

        参数:
            result: 待处理的 RetrievalResult。
            collection: 可选的 collection 名称，用于路径解析。

        返回:
            MCP 内容块列表（TextContent 与 ImageContent）。
        """
        blocks: List[Union[types.TextContent, types.ImageContent]] = []

        # 提取图像引用
        refs = self.extract_image_refs(result)

        # 加载并添加图像
        for ref in refs:
            # 解析路径
            file_path = self.resolve_image_path(ref, collection)
            if not file_path:
                logger.debug(f"Could not resolve path for image: {ref.image_id}")
                continue

            # 加载图像
            image_content = self.load_image(file_path)
            if image_content is None:
                continue

            # 更新引用信息
            image_content.image_id = ref.image_id
            image_content.caption = ref.caption

            # 添加图像块
            blocks.append(image_content.to_mcp_content())

            # 如启用则将说明文字添加为文本
            if self.include_captions and ref.caption:
                caption_text = f"**Image Caption ({ref.image_id}):** {ref.caption}"
                blocks.append(types.TextContent(type="text", text=caption_text))

        return blocks

    def assemble(
        self,
        results: List[RetrievalResult],
        collection: Optional[str] = None,
    ) -> List[Union[types.TextContent, types.ImageContent]]:
        """为多个结果组装多模态内容块。

        参数:
            results: 待处理的 RetrievalResult 列表。
            collection: 可选的 collection 名称，用于路径解析。

        返回:
            来自所有结果的 MCP 内容块列表。
        """
        all_blocks: List[Union[types.TextContent, types.ImageContent]] = []
        seen_image_ids: set = set()

        for result in results:
            blocks = self.assemble_for_result(result, collection)

            # 跨结果去重图像
            for block in blocks:
                if isinstance(block, types.ImageContent):
                    # 检查该图像是否已出现过
                    # ImageContent 没有 image_id，因此对数据做哈希
                    data_hash = hash(block.data[:100])  # 使用前缀以提高效率
                    if data_hash in seen_image_ids:
                        continue
                    seen_image_ids.add(data_hash)

                all_blocks.append(block)

        return all_blocks

    def has_images(self, result: RetrievalResult) -> bool:
        """检查结果是否包含图像引用。

        参数:
            result: 待检查的 RetrievalResult。

        返回:
            结果包含图像引用时返回 True，否则返回 False。
        """
        refs = self.extract_image_refs(result)
        return len(refs) > 0

    def count_images(self, results: List[RetrievalResult]) -> int:
        """统计所有结果中的图像总数。

        参数:
            results: 待统计的 RetrievalResult 列表。

        返回:
            找到的图像引用总数。
        """
        total = 0
        for result in results:
            refs = self.extract_image_refs(result)
            total += len(refs)
        return total
