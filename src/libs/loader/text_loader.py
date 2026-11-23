"""纯文本类文档 Loader（txt / md / csv / json）。

不依赖 MarkItDown，直接按多种常见编码读取文本。
Markdown 中的本地图片会被提取到 image_storage_dir，并替换为
[IMAGE: {image_id}] 占位符；远程 URL 与找不到的本地文件保持原样。
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)

# Markdown 图片语法：![alt](src "title")，允许尖括号包裹
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
# Markdown 内嵌 HTML：<img src="...">
_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)
# data URI：data:image/png;base64,....
_DATA_URI_RE = re.compile(r"^data:image/([A-Za-z0-9.+-]+);base64,(.+)$", re.DOTALL)


class TextLoader(BaseLoader):
    """读取纯文本类文档，输出统一 Markdown 文本。"""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json"}
    IMAGE_EXTENSIONS = {".md", ".markdown"}
    _ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images",
    ) -> None:
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)

    def load(self, file_path: str | Path) -> Document:
        """加载并解析纯文本文件。

        异常:
            FileNotFoundError: 文件不存在。
            ValueError: 扩展名不支持、无法解码或内容为空。
        """
        path = self._validate_file(file_path)
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported text file type: {path.suffix}")

        raw = path.read_bytes()
        text = self._decode(raw, path)

        if not text.strip():
            raise ValueError(f"No text content in {path.name}")

        doc_hash = hashlib.sha256(raw).hexdigest()
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": suffix.lstrip("."),
            "doc_hash": doc_hash,
        }

        if self.extract_images and suffix in self.IMAGE_EXTENSIONS:
            text, images_metadata = self._extract_local_images(text, path.parent, doc_hash)
            if images_metadata:
                metadata["images"] = images_metadata

        title = self._extract_title(text)
        if title:
            metadata["title"] = title

        return Document(id=f"doc_{doc_hash[:16]}", text=text, metadata=metadata)

    def _extract_local_images(
        self, text: str, base_dir: Path, doc_hash: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """把 Markdown 引用的本地图片复制到存储目录，并替换为占位符。

        远程 URL、找不到的本地文件保持原样（不产生占位符）。
        """
        image_dir = self.image_storage_dir / doc_hash
        images: List[Dict[str, Any]] = []

        def replace(match: re.Match) -> str:
            src = match.group(1).strip()
            saved = self._save_image(src, base_dir, doc_hash, image_dir, images)
            return saved or match.group(0)

        text = _MD_IMAGE_RE.sub(replace, text)
        text = _HTML_IMAGE_RE.sub(replace, text)

        if images:
            logger.info(f"Extracted {len(images)} local images from doc {doc_hash[:8]}")
        return text, images

    def _save_image(
        self,
        src: str,
        base_dir: Path,
        doc_hash: str,
        image_dir: Path,
        images: List[Dict[str, Any]],
    ) -> Optional[str]:
        """保存单张图片，返回占位符；不支持/不存在时返回 None。"""
        if src.startswith(("http://", "https://", "//")):
            return None  # 远程图片不下载

        data_uri = _DATA_URI_RE.match(src)
        if data_uri:
            blob = base64.b64decode(data_uri.group(2))
            ext = data_uri.group(1).split("+")[0]
        else:
            # 去掉锚点/查询串，解码 %20 之类的转义
            local = Path(unquote(src.split("#")[0].split("?")[0]))
            if not local.is_absolute():
                local = base_dir / local
            if not local.is_file():
                return None
            blob = local.read_bytes()
            ext = local.suffix.lstrip(".").lower() or "png"  # ponytail: 无后缀按 png 存

        image_dir.mkdir(parents=True, exist_ok=True)
        image_id = f"{doc_hash[:8]}_0_{len(images) + 1}"
        image_path = image_dir / f"{image_id}.{ext}"
        image_path.write_bytes(blob)
        images.append(
            {
                "id": image_id,
                "path": self._display_path(image_path),
                "page": 0,
            }
        )
        return f"[IMAGE: {image_id}]"

    @staticmethod
    def _display_path(path: Path) -> str:
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return str(path.absolute())

    @classmethod
    def _decode(cls, raw: bytes, path: Path) -> str:
        for encoding in cls._ENCODINGS:
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(
            f"Cannot decode {path.name} with any of {', '.join(cls._ENCODINGS)}"
        )

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