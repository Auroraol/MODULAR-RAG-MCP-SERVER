"""固定长度切分器实现。

不依赖任何外部包：按 ``chunk_size`` 硬切，相邻 Chunk 保留
``chunk_overlap`` 字符重叠；切点附近有空白时优先在空白处断开，
避免把英文单词切成两半。
"""

from __future__ import annotations

from typing import Any, List, Optional

from src.libs.splitter.base_splitter import BaseSplitter


class FixedLengthSplitter(BaseSplitter):
    """按固定字符数切分文本。

    属性:
        chunk_size: 每个 Chunk 的最大字符数。
        chunk_overlap: 相邻 Chunk 的重叠字符数。

    示例:
        >>> splitter = FixedLengthSplitter(settings)
        >>> chunks = splitter.split_text("a" * 25)
        >>> all(len(c) <= 10 for c in chunks)
        True
    """

    def __init__(
        self,
        settings: Any,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings

        try:
            ingestion_config = settings.ingestion
            self.chunk_size = chunk_size if chunk_size is not None else ingestion_config.chunk_size
            self.chunk_overlap = chunk_overlap if chunk_overlap is not None else ingestion_config.chunk_overlap
        except AttributeError as e:
            raise ValueError(
                "Missing ingestion configuration in settings. "
                "Expected settings.ingestion.chunk_size and settings.ingestion.chunk_overlap"
            ) from e

        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got: {self.chunk_size}")
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be a non-negative integer, got: {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """按固定长度切分文本。

        参数:
            text: 待切分的输入文本。
            trace: 可选 TraceContext（当前未使用）。
            **kwargs: 额外参数（当前未使用）。

        返回:
            文本 Chunk 列表，按原始顺序排列。
        """
        self.validate_text(text)

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]

            # 切点落在词中间时回退到最近的空白处（仅搜索尾部 20%，
            # ponytail: 超出 20% 宁可硬切，否则块会远小于 chunk_size）
            if end < len(text):
                boundary = max(chunk.rfind(" "), chunk.rfind("\n"))
                if boundary >= int(len(chunk) * 0.8):
                    end = start + boundary + 1
                    chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            if end >= len(text):
                break
            start = max(start + 1, end - self.chunk_overlap)

        if not chunks:
            chunks = [text]

        self.validate_chunks(chunks)
        return chunks