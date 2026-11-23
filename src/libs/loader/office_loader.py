"""Office / 富文本文档 Loader（docx / pptx / xlsx / html / epub）。

基于 MarkItDown 转换为 Markdown；Office 格式需要对应的可选依赖：
    pip install "markitdown[docx,xlsx,pptx]"

docx / pptx 的内嵌图片会被提取到 image_storage_dir，并在正文中替换为
[IMAGE: {image_id}] 占位符，元数据记录在 Document.metadata.images。
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)

# MarkItDown 在 keep_data_uris=True 时把内嵌图片写成：![alt](data:image/png;base64,...)
_DATA_URI_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\(data:image/([A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=]+)\)"
)
# 提取失败或禁用时的兜底：把残留的图片 Markdown 一并删掉
_IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\(data:image/[^)]*\)")
# pptx 文本里的幻灯片分隔标记，用来给图片标页码
_SLIDE_NUMBER_RE = re.compile(r"<!--\s*Slide number:\s*(\d+)\s*-->")


class OfficeLoader(BaseLoader):
    """使用 MarkItDown 将 Office / 富文本格式转换为 Markdown 文本。"""

    SUPPORTED_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".html", ".htm", ".epub"}
    # MarkItDown 能内联图片的格式（xlsx/html/epub 不处理）
    IMAGE_EXTENSIONS = {".docx", ".pptx"}

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images",
    ) -> None:
        if not MARKITDOWN_AVAILABLE:
            raise ImportError(
                "MarkItDown is required for OfficeLoader. "
                "Install with: pip install 'markitdown[docx,xlsx,pptx]'"
            )
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        self._markitdown = MarkItDown()

    def load(self, file_path: str | Path) -> Document:
        """加载并解析 Office / 富文本文件。

        异常:
            FileNotFoundError: 文件不存在。
            ValueError: 扩展名不支持或内容为空。
            RuntimeError: 解析失败（常见原因：缺少该格式的可选依赖）。
        """
        path = self._validate_file(file_path)
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported office file type: {path.suffix}")

        doc_hash = self._compute_file_hash(path)
        has_embedded_images = suffix in self.IMAGE_EXTENSIONS

        try:
            result = self._markitdown.convert(
                str(path),
                keep_data_uris=has_embedded_images and self.extract_images,
            )
            text = result.text_content if hasattr(result, "text_content") else str(result)
        except Exception as e:
            raise RuntimeError(
                f"Document parsing failed for {path.name}: {e}. "
                f"If the format is docx/xlsx/pptx, install markitdown extras."
            ) from e

        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": suffix.lstrip("."),
            "doc_hash": doc_hash,
        }

        if has_embedded_images:
            if self.extract_images:
                try:
                    text, images_metadata = self._extract_embedded_images(text, doc_hash)
                    if images_metadata:
                        metadata["images"] = images_metadata
                except Exception as e:
                    logger.warning(
                        f"Image extraction failed for {path}, continuing with text-only: {e}"
                    )
                    text = _IMAGE_MARKDOWN_RE.sub("", text)
            else:
                # 没开图片提取时别把 base64 / 截断的 data URI 留在正文里
                text = _IMAGE_MARKDOWN_RE.sub("", text)

        if not text.strip():
            raise ValueError(f"No extractable text in {path.name}")

        title = self._extract_title(text)
        if title:
            metadata["title"] = title

        return Document(id=f"doc_{doc_hash[:16]}", text=text, metadata=metadata)

    def _extract_embedded_images(
        self, text: str, doc_hash: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """把内联的 data URI 图片落盘，并替换为 [IMAGE: id] 占位符。

        返回:
            (替换后的文本, 图片元数据列表)
        """
        slide_positions = [
            (m.start(), int(m.group(1))) for m in _SLIDE_NUMBER_RE.finditer(text)
        ]
        image_dir = self.image_storage_dir / doc_hash
        image_dir.mkdir(parents=True, exist_ok=True)
        images: List[Dict[str, Any]] = []

        def page_at(position: int) -> int:
            # 只有 pptx 有幻灯片标记；docx 记为 0
            page = 0
            for start, number in slide_positions:
                if start > position:
                    break
                page = number
            return page

        def replace(match: re.Match) -> str:
            page = page_at(match.start())
            subtype = match.group(1).split("+")[0]  # svg+xml -> svg
            image_id = f"{doc_hash[:8]}_{page}_{len(images) + 1}"
            image_path = image_dir / f"{image_id}.{subtype}"
            image_path.write_bytes(base64.b64decode(match.group(2)))
            images.append(
                {
                    "id": image_id,
                    "path": self._display_path(image_path),
                    "page": page,
                }
            )
            return f"[IMAGE: {image_id}]"

        new_text = _DATA_URI_IMAGE_RE.sub(replace, text)
        if images:
            logger.info(f"Extracted {len(images)} embedded images from doc {doc_hash[:8]}")
        return new_text, images

    @staticmethod
    def _display_path(path: Path) -> str:
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return str(path.absolute())

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        for line in text.split("\n")[:20]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        for line in text.split("\n")[:10]:
            line = line.strip()
            if line:
                return line[:200]
        return None