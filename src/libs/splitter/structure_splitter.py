"""结构化切分器实现。

Loader 输出的 Document.text 统一为 Markdown，因此可以按 ATX 标题
（``#`` / ``##`` …）切出章节，保证每个 Chunk 落在同一标题层级下：

- 相邻的小章节会合并，避免产出大量碎片
- 超大章节降级用 RecursiveSplitter 切分，并给每段补上所属标题路径，
  使每段脱离原文也能知道自己在哪一节
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from src.libs.splitter.base_splitter import BaseSplitter

# ATX 标题：行首 1-6 个 # 后跟空白
_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")


def _split_sections(text: str) -> List[tuple[str, str]]:
    """把 Markdown 文本切成 ``[(标题路径, 正文)]``。

    标题路径为所有祖先标题 + 自身标题的拼接；正文不含标题行。
    没有标题的文本整体作为一个 section（标题路径为空）。
    """
    sections: List[tuple[str, str]] = []
    stack: List[tuple[int, str]] = []  # [(level, 标题行)]
    body_lines: List[str] = []

    def flush() -> None:
        body = "".join(body_lines)
        if body.strip():
            sections.append(("".join(heading for _, heading in stack), body))
        body_lines.clear()

    for line in text.splitlines(keepends=True):
        match = _HEADING_RE.match(line.rstrip("\n"))
        if match:
            flush()
            level = len(match.group(1))
            stack = [(l, h) for l, h in stack if l < level]
            stack.append((level, line))
        else:
            body_lines.append(line)
    flush()
    return sections


class StructureSplitter(BaseSplitter):
    """按 Markdown 标题结构切分的文本切分器。

    属性:
        chunk_size: 每个 Chunk 的最大字符数。
        chunk_overlap: 超大章节二次切分时的重叠字符数。

    示例:
        >>> splitter = StructureSplitter(settings)
        >>> chunks = splitter.split_text("# A\\n\\ncontent")
        >>> chunks[0].startswith("# A")
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
        """按标题层级切分文本。

        参数:
            text: 待切分的 Markdown 文本。
            trace: 可选 TraceContext（当前未使用）。
            **kwargs: 额外参数（当前未使用）。

        返回:
            文本 Chunk 列表，每个 Chunk 以所属标题开头。
        """
        self.validate_text(text)

        chunks: List[str] = []
        current = ""
        for prefix, body in _split_sections(text):
            piece = prefix + body
            if len(piece) <= self.chunk_size:
                if current and len(current) + len(piece) > self.chunk_size:
                    chunks.append(current)
                    current = ""
                current += piece
            else:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_oversized(prefix, body))

        if current:
            chunks.append(current)
        if not chunks:
            chunks = [text]

        self.validate_chunks(chunks)
        return chunks

    def _split_oversized(self, prefix: str, body: str) -> List[str]:
        """章节超过 chunk_size 时二次切分，并为每段补上标题路径。"""
        # ponytail: 标题路径过长时 budget 兜底为 chunk_size 一半，
        # 此时部分 Chunk 会超过 chunk_size，正常文档不会触发
        budget = max(self.chunk_size // 2, self.chunk_size - len(prefix))
        overlap = min(self.chunk_overlap, budget - 1)

        try:
            from src.libs.splitter.recursive_splitter import RecursiveSplitter
            parts = RecursiveSplitter(
                self.settings, chunk_size=budget, chunk_overlap=overlap
            ).split_text(body)
            return [prefix + part for part in parts]
        except ImportError:
            from src.libs.splitter.fixed_length_splitter import FixedLengthSplitter
            return FixedLengthSplitter(
                self.settings, chunk_size=budget, chunk_overlap=overlap
            ).split_text(prefix + body)