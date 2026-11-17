"""Embedding Provider 的抽象基类。

本模块定义 Embedding 服务提供方的可插拔接口，
通过配置驱动的实例化，实现不同后端（OpenAI、Local/BGE 等）之间的无缝切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseEmbedding(ABC):
    """Embedding Provider 的抽象基类。

    所有 Embedding 实现都必须继承本类并实现 embed() 方法。
    这确保了不同 Provider（OpenAI、Local/BGE、Ollama 等）之间接口一致。

    应用的设计原则：
    - 可插拔：子类可在不修改上游代码的情况下替换。
    - 可观测：接受可选的 TraceContext，以集成可观测能力。
    - 配置驱动：实例由工厂根据 settings 创建。
    - 批优先：为批量处理而设计，以最大化效率。
    """
    
    @abstractmethod
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """为一组文本批量生成 Embedding。

        参数:
            texts: 待 Embedding 的文本字符串列表，不能为空。
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F）。
            **kwargs: Provider 特定参数（batch_size、dimensions 等）。

        返回:
            Embedding 向量列表，其中每个向量是一个浮点数列表。
            外层列表长度与 len(texts) 一致。
            每个内层列表的长度（向量维度）由具体 Provider 决定。

        异常:
            ValueError: 当 texts 列表为空或包含无效条目时抛出。
            RuntimeError: 当 Embedding Provider 调用失败时抛出。

        示例:
            >>> embeddings = embedding.embed(["hello", "world"])
            >>> len(embeddings)  # 2 vectors
            >>> len(embeddings[0])  # dimension (e.g., 1536 for OpenAI)
        """
        pass
    
    def validate_texts(self, texts: List[str]) -> None:
        """校验输入文本列表。

        参数:
            texts: 待校验的文本列表。

        异常:
            ValueError: 当 texts 列表为空或包含无效条目时抛出。
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"Text at index {i} is not a string (type: {type(text).__name__})"
                )
            if not text.strip():
                raise ValueError(
                    f"Text at index {i} is empty or whitespace-only. "
                    "Embedding providers typically reject empty strings."
                )
    
    def get_dimension(self) -> int:
        """获取该 Provider 产出的 Embedding 维度。

        返回:
            向量维度（例如 OpenAI text-embedding-3-small 为 1536）。

        异常:
            NotImplementedError: 当子类未重写该方法时抛出。

        注意:
            子类应重写此方法以返回各自的维度。
            这在校验和存储配置时很有用。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_dimension() method"
        )
