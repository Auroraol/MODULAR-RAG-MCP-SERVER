"""基于 LangChain 的递归切分器实现。

本模块提供基于递归字符的文本切分策略，
它遵循文档结构（标题、代码块），并以层级方式切分文本，
从而保持语义连贯性。
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None  # type: ignore[misc, assignment]

from src.libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """基于递归字符的文本切分器。

    此切分器使用 LangChain 的 RecursiveCharacterTextSplitter，按顺序尝试
    不同分隔符（段落、句子、单词）来切分文本，同时遵循 Markdown
    结构元素（如标题和代码块）。

    应用的设计原则：
    - 可插拔：实现 BaseSplitter 接口，供工厂实例化。
    - 配置驱动：从 settings 读取 chunk_size 与 chunk_overlap。
    - 快速失败：未安装 langchain-text-splitters 时抛出 ImportError。
    - 优雅降级：校验输入并提供清晰的错误消息。

    属性:
        chunk_size: 每个 Chunk 的最大字符数。
        chunk_overlap: Chunk 之间的重叠字符数。
        separators: 按顺序尝试的分隔符列表（默认支持 Markdown）。

    异常:
        ImportError: 当未安装 langchain-text-splitters 包时。
    """

    DEFAULT_SEPARATORS = [
        "\n\n",  # 双换行（段落）
        "\n",    # 单个换行
        ". ",    # 句子结尾
        "! ",
        "? ",
        "; ",
        ", ",
        " ",     # 空格
        "",      # 字符
    ]
    
    def __init__(
        self,
        settings: Any,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 RecursiveSplitter。

        参数:
            settings: 包含 ingestion 配置的应用设置。
            chunk_size: 可选的 Chunk 大小覆盖（默认为 settings.ingestion.chunk_size）。
            chunk_overlap: 可选的重叠大小覆盖（默认为 settings.ingestion.chunk_overlap）。
            separators: 可选的分隔符字符串列表（默认为 Markdown 感知的分隔符）。
            **kwargs: 传递给 LangChain 切分器的额外参数。

        异常:
            ImportError: 当未安装 langchain-text-splitters 时。
            ValueError: 当 chunk_size 或 chunk_overlap 无效时。
        """
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters is not installed. "
                "Install it with: pip install langchain-text-splitters"
            )
        
        self.settings = settings
        
        # 从 settings 提取配置（支持覆盖）
        try:
            ingestion_config = settings.ingestion
            self.chunk_size = chunk_size if chunk_size is not None else ingestion_config.chunk_size
            self.chunk_overlap = chunk_overlap if chunk_overlap is not None else ingestion_config.chunk_overlap
        except AttributeError as e:
            raise ValueError(
                "Missing ingestion configuration in settings. "
                "Expected settings.ingestion.chunk_size and settings.ingestion.chunk_overlap"
            ) from e
        
        # 校验配置
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got: {self.chunk_size}")
        
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be a non-negative integer, got: {self.chunk_overlap}")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS
        
        # 初始化 LangChain 切分器
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
            **kwargs,
        )
    
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """递归地将文本切分为 Chunk。

        此方法按层级顺序尝试不同分隔符来切分文本，
        以保留 Markdown 标题、代码块等文档结构。

        参数:
            text: 待切分的输入文本。必须为非空字符串。
            trace: 可选的 TraceContext，用于可观测性（预留给阶段 F）。
            **kwargs: 额外参数（当前未使用，预留给未来扩展）。

        返回:
            文本 Chunk 列表。每个 Chunk 均遵循配置的 chunk_size
            与 chunk_overlap。顺序与原始文本序列一致。

        异常:
            ValueError: 当输入文本无效时（为空、类型错误）。
            RuntimeError: 当切分意外失败时。

        示例:
            >>> splitter = RecursiveSplitter(settings)
            >>> chunks = splitter.split_text("# Header\\n\\nParagraph 1.\\n\\nParagraph 2.")
            >>> len(chunks)
            1  # 若文本未超过 chunk_size
        """
        # 校验输入
        self.validate_text(text)

        try:
            # 执行切分
            chunks = self._splitter.split_text(text)

            # 处理边界情况：LangChain 对极短文本可能返回空列表
            if not chunks:
                chunks = [text]

            # 校验输出
            self.validate_chunks(chunks)

            return chunks

        except Exception as e:
            # 捕获所有 LangChain 错误并提供上下文
            raise RuntimeError(
                f"RecursiveSplitter failed to split text: {e}. "
                f"Text length: {len(text)}, chunk_size: {self.chunk_size}, "
                f"chunk_overlap: {self.chunk_overlap}"
            ) from e
