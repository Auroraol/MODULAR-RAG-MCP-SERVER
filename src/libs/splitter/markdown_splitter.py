"""基于 LangChain 的 Markdown 切分器实现。

相比 RecursiveSplitter，本切分器把 Markdown 结构（标题、代码块、分隔线）
作为高优先级分隔符，保证代码块和章节不会被拦腰截断。
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from langchain_text_splitters import MarkdownTextSplitter
except ImportError:
    MarkdownTextSplitter = None  # type: ignore[misc, assignment]

from src.libs.splitter.base_splitter import BaseSplitter


class MarkdownSplitter(BaseSplitter):
    """按 Markdown 结构切分的文本切分器。

    属性:
        chunk_size: 每个 Chunk 的最大字符数。
        chunk_overlap: 相邻 Chunk 的重叠字符数。

    异常:
        ImportError: 当未安装 langchain-text-splitters 时。
    """

    def __init__(
        self,
        settings: Any,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if MarkdownTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters is not installed. "
                "Install it with: pip install langchain-text-splitters"
            )

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

        self._splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            **kwargs,
        )

    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """按 Markdown 标题 / 代码块 / 段落切分文本。

        参数:
            text: 待切分的输入文本。
            trace: 可选 TraceContext（当前未使用）。
            **kwargs: 额外参数（当前未使用）。

        返回:
            文本 Chunk 列表，按原始顺序排列。
        """
        self.validate_text(text)

        chunks = self._splitter.split_text(text)
        if not chunks:
            chunks = [text]

        self.validate_chunks(chunks)
        return chunks