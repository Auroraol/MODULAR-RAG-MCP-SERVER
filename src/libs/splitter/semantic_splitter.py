"""语义切分器实现。

先按句末标点切句，再用 Embedding 计算相邻句子的余弦距离，
在语义距离的高分位处断开，使每个 Chunk 尽量对应一个主题。
只对句数足够多的文本生效；句子太少或 Embedding 调用失败时降级为
RecursiveSplitter。

代价：切分阶段就要调用一次 Embedding API（全部句子），比
recursive / fixed_length 慢且贵，只在大文档、对语义边界敏感时使用。
"""

from __future__ import annotations

import math
import re
from typing import Any, List, Optional

from src.observability.logger import get_logger
from src.libs.splitter.base_splitter import BaseSplitter

logger = get_logger(__name__)

# 中英文句末标点后切句（零宽切分，不丢失标点）
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])\s*|(?<=\.)\s+")

# 句数少于此值时没有足够的统计意义，直接整段返回
_MIN_SENTENCES = 3


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _percentile(values: List[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[index]


def _join(sentences: List[str]) -> str:
    """还原句子连接符：中文标点后不补空格，英文句号后补空格。"""
    out = sentences[0]
    for sentence in sentences[1:]:
        out += ("" if out[-1] in "。！？；，、" else " ") + sentence
    return out


class SemanticSplitter(BaseSplitter):
    """按语义边界切分的文本切分器。

    属性:
        chunk_size: 单个 Chunk 的字符上限（超长的连续语义块会在句子边界强制断开）。
        breakpoint_percentile: 距离阈值分位数，默认 95（只在大约 top 5% 的
            语义跳变处断开）。越小切得越碎。
        embedding: 可选的外部 Embedding 实例；不传则从 settings 创建。

    示例:
        >>> splitter = SemanticSplitter(settings)
        >>> chunks = splitter.split_text("A. B. C. D.")
        >>> len(chunks) >= 1
        True
    """

    def __init__(
        self,
        settings: Any,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding: Optional[Any] = None,
        breakpoint_percentile: float = 95.0,
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
        if not 0 < breakpoint_percentile <= 100:
            raise ValueError(
                f"breakpoint_percentile must be in (0, 100], got: {breakpoint_percentile}"
            )

        self._embedding = embedding
        self.breakpoint_percentile = breakpoint_percentile

    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """按相邻句子的语义距离切分文本。

        参数:
            text: 待切分的输入文本。
            trace: 可选 TraceContext，透传给 Embedding 调用。
            **kwargs: 额外参数（当前未使用）。

        返回:
            文本 Chunk 列表，按原始顺序排列。
        """
        self.validate_text(text)

        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]

        if len(sentences) < _MIN_SENTENCES:
            # 句子太少时没有语义边界可言，但仍要遵守 chunk_size 上限
            chunks = [text] if len(text) <= self.chunk_size else self._fallback(text)
        else:
            try:
                vectors = self._get_embedding().embed(sentences, trace=trace)
                chunks = self._merge(sentences, vectors)
            except Exception as e:
                logger.warning("SemanticSplitter 失败，降级为 recursive: %s", e)
                chunks = self._fallback(text)

        if not chunks:
            chunks = [text]

        self.validate_chunks(chunks)
        return chunks

    def _get_embedding(self) -> Any:
        if self._embedding is None:
            from src.libs.embedding.embedding_factory import EmbeddingFactory
            self._embedding = EmbeddingFactory.create(self.settings)
        return self._embedding

    def _merge(self, sentences: List[str], vectors: List[List[float]]) -> List[str]:
        if len(vectors) != len(sentences):
            raise RuntimeError(
                f"Embedding count mismatch: {len(vectors)} vectors for {len(sentences)} sentences"
            )

        distances = [
            1.0 - _cosine(vectors[i], vectors[i + 1]) for i in range(len(sentences) - 1)
        ]
        # 所有距离相同说明没有语义跳变，不做语义断开，只按 chunk_size 强制切
        if max(distances) - min(distances) < 1e-9:
            threshold = math.inf
        else:
            threshold = _percentile(distances, self.breakpoint_percentile)

        chunks: List[str] = []
        current: List[str] = [sentences[0]]
        for i in range(1, len(sentences)):
            break_before = distances[i - 1] >= threshold
            over_limit = len(_join(current + [sentences[i]])) > self.chunk_size
            if break_before or over_limit:
                chunks.append(_join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        chunks.append(_join(current))
        return chunks

    def _fallback(self, text: str) -> List[str]:
        from src.libs.splitter.recursive_splitter import RecursiveSplitter

        fallback = RecursiveSplitter(
            self.settings,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return fallback.split_text(text)