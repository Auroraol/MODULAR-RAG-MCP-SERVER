"""文本切分器的抽象基类。

本模块定义了文本切分器 provider 的可插拔接口，通过配置驱动的实例化方式，
可在不同的切分策略（Recursive、Semantic、FixedLength 等）之间无缝切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseSplitter(ABC):
    """文本切分器的抽象基类。

    所有切分器实现都必须继承此类并实现 split_text() 方法。
    这可确保不同策略之间拥有一致的接口。

    应用的设计原则：
    - 可插拔：子类可在不修改上游代码的情况下替换。
    - 可观测：接受可选的 TraceContext 以集成可观测性。
    - 配置驱动：实例通过工厂基于配置创建。
    """
    
    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """将输入文本切分为 Chunk 列表。

        参数:
            text: 待切分的输入文本。必须为非空字符串。
            trace: 可选的 TraceContext，用于可观测性（预留给阶段 F）。
            **kwargs: 策略特定参数（chunk_size、overlap 等）。

        返回:
            文本 Chunk 列表。顺序必须与原始文本序列一致。

        异常:
            ValueError: 当输入文本无效时。
            RuntimeError: 当切分器意外失败时。
        """
        pass
    
    def validate_text(self, text: str) -> None:
        """校验输入文本。

        参数:
            text: 待校验的输入文本。

        异常:
            ValueError: 当 text 不是非空字符串时。
        """
        if not isinstance(text, str):
            raise ValueError(f"Input text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only")
    
    def validate_chunks(self, chunks: List[str]) -> None:
        """校验输出的 Chunk。

        参数:
            chunks: 待校验的 Chunk 字符串列表。

        异常:
            ValueError: 当 Chunk 列表为空或包含无效项时。
        """
        if not isinstance(chunks, list):
            raise ValueError("Chunks must be a list of strings")
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, str):
                raise ValueError(
                    f"Chunk at index {i} is not a string (type: {type(chunk).__name__})"
                )
            if not chunk.strip():
                raise ValueError(
                    f"Chunk at index {i} is empty or whitespace-only"
                )