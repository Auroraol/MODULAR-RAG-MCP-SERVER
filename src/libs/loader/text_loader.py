"""纯文本类文档 Loader（txt / md / csv / json）。

不依赖 MarkItDown，直接按多种常见编码读取文本。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader


class TextLoader(BaseLoader):
    """读取纯文本类文档，输出统一 Markdown 文本。"""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json"}
    _ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")

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

        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": suffix.lstrip("."),
            "doc_hash": hashlib.sha256(raw).hexdigest(),
        }
        title = self._extract_title(text)
        if title:
            metadata["title"] = title

        return Document(id=f"doc_{metadata['doc_hash'][:16]}", text=text, metadata=metadata)

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