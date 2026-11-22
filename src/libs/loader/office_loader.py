"""Office / 富文本文档 Loader（docx / pptx / xlsx / html / epub）。

基于 MarkItDown 转换为 Markdown；Office 格式需要对应的可选依赖：
    pip install "markitdown[docx,xlsx,pptx]"
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class OfficeLoader(BaseLoader):
    """使用 MarkItDown 将 Office / 富文本格式转换为 Markdown 文本。"""

    SUPPORTED_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".html", ".htm", ".epub"}

    def __init__(self) -> None:
        if not MARKITDOWN_AVAILABLE:
            raise ImportError(
                "MarkItDown is required for OfficeLoader. "
                "Install with: pip install 'markitdown[docx,xlsx,pptx]'"
            )
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

        try:
            result = self._markitdown.convert(str(path))
            text = result.text_content if hasattr(result, "text_content") else str(result)
        except Exception as e:
            raise RuntimeError(
                f"Document parsing failed for {path.name}: {e}. "
                f"If the format is docx/xlsx/pptx, install markitdown extras."
            ) from e

        if not text.strip():
            raise ValueError(f"No extractable text in {path.name}")

        doc_hash = self._compute_file_hash(path)
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": suffix.lstrip("."),
            "doc_hash": doc_hash,
        }
        title = self._extract_title(text)
        if title:
            metadata["title"] = title

        return Document(id=f"doc_{doc_hash[:16]}", text=text, metadata=metadata)

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