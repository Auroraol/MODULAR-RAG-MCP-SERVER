"""命题分块（Propositional Chunking）实现。

用 LLM 把文本改写为一组「原子命题」——每条都是可脱离上下文独立
理解的陈述句——再把命题打包成不超过 ``chunk_size`` 的 Chunk。
检索时每条命题都自带主语和条件，不依赖邻近文本。

代价：每个窗口一次 LLM 调用，摄入明显变慢变贵，只适合高价值语料
（合同条款、政策条文、FAQ）。LLM 调用失败时保留原文窗口，不阻塞摄入。
"""

from __future__ import annotations

from typing import Any, List, Optional

from src.observability.logger import get_logger
from src.libs.llm.base_llm import Message
from src.libs.splitter.base_splitter import BaseSplitter

logger = get_logger(__name__)

_PROMPT = """请把下面的文本拆分成若干条「原子命题」，每条都能脱离上下文独立理解。
要求：
1. 一条一句话，保留原文的专有名词、数字、条件和否定，不要编造或改写事实；
2. 每条以 `- ` 开头，逐行输出，不要输出标题、解释或代码块；
3. 使用与原文相同的语言。

文本：
{text}"""


class PropositionalSplitter(BaseSplitter):
    """基于 LLM 的命题切分器。

    属性:
        chunk_size: 单个 Chunk 的字符上限。
        window_size: 每次送给 LLM 的原文窗口大小，默认等于 chunk_size。
        llm: 可选的外部 LLM 实例；不传则从 settings 创建。

    异常:
        ValueError: 配置无效时。
    """

    def __init__(
        self,
        settings: Any,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        llm: Optional[Any] = None,
        window_size: Optional[int] = None,
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

        self._llm = llm
        self.window_size = window_size if window_size is not None else self.chunk_size

    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """把文本切分为命题块。

        参数:
            text: 待切分的输入文本。
            trace: 可选 TraceContext，透传给 LLM 调用。
            **kwargs: 额外参数（当前未使用）。

        返回:
            由若干条命题组成的 Chunk 列表，按原始顺序排列。
        """
        self.validate_text(text)

        chunks: List[str] = []
        for window in self._windows(text):
            chunks.extend(self._pack(self._propositions(window, trace)))

        if not chunks:
            chunks = [text]

        self.validate_chunks(chunks)
        return chunks

    def _get_llm(self) -> Any:
        if self._llm is None:
            from src.libs.llm.llm_factory import LLMFactory
            self._llm = LLMFactory.create(self.settings)
        return self._llm

    def _windows(self, text: str) -> List[str]:
        """按 window_size 切出送给 LLM 的原文窗口。"""
        if len(text) <= self.window_size:
            return [text]

        try:
            from src.libs.splitter.recursive_splitter import RecursiveSplitter
            splitter = RecursiveSplitter(
                self.settings, chunk_size=self.window_size, chunk_overlap=0
            )
            return splitter.split_text(text)
        except ImportError:
            return [text[i:i + self.window_size] for i in range(0, len(text), self.window_size)]

    def _propositions(self, window: str, trace: Optional[Any]) -> List[str]:
        """调用 LLM 抽命题；失败时把原文窗口当作唯一命题保留。"""
        try:
            response = self._get_llm().chat(
                [Message(role="user", content=_PROMPT.format(text=window))],
                trace=trace,
            )
            items = []
            for line in response.content.splitlines():
                item = line.strip().lstrip("-*• ").strip()
                if item and not item.startswith("```"):
                    items.append(item)
            return items or [window]
        except Exception as e:
            logger.warning("PropositionalSplitter LLM 调用失败，保留原文窗口: %s", e)
            return [window]

    def _pack(self, propositions: List[str]) -> List[str]:
        """把命题按顺序打包成不超过 chunk_size 的 Chunk。"""
        chunks: List[str] = []
        current = ""
        for item in propositions:
            if len(item) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(
                    item[i:i + self.chunk_size] for i in range(0, len(item), self.chunk_size)
                )
                continue
            if current and len(current) + len(item) + 1 > self.chunk_size:
                chunks.append(current)
                current = ""
            current = f"{current}\n{item}" if current else item

        if current:
            chunks.append(current)
        return chunks