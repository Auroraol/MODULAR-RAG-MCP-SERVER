"""Evaluator Provider 的抽象基类。

本模块定义评估提供方的可插拔接口，
通过配置驱动的实例化，实现不同评估后端之间的无缝切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseEvaluator(ABC):
    """评估提供方的抽象基类。

    所有评估器实现都必须继承本类并实现 evaluate() 方法。
    这确保了不同评估后端之间接口一致。

    应用的设计原则：
    - 可插拔：子类可在不修改上游代码的情况下替换。
    - 可观测：接受可选的 TraceContext，以集成可观测能力。
    - 配置驱动：实例由工厂根据 settings 创建。
    """

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """评估检索与生成质量。

        参数:
            query: 用户查询字符串。
            retrieved_chunks: 待评估的检索 Chunk 或记录。
            generated_answer: 可选的生成答案文本。
            ground_truth: 可选的 ground truth 数据（id 或答案）。
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F）。
            **kwargs: Provider 特定参数。

        返回:
            指标名称到浮点值的字典。

        异常:
            ValueError: 当输入无效时抛出。
            RuntimeError: 当评估意外失败时抛出。
        """
        pass

    def validate_query(self, query: str) -> None:
        """校验查询字符串。

        参数:
            query: 待校验的查询字符串。

        异常:
            ValueError: 当查询无效时抛出。
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

    def validate_retrieved_chunks(self, retrieved_chunks: List[Any]) -> None:
        """校验检索 Chunk 的结构。

        参数:
            retrieved_chunks: 待校验的检索 Chunk 列表。

        异常:
            ValueError: 当 retrieved_chunks 无效时抛出。
        """
        if not isinstance(retrieved_chunks, list):
            raise ValueError("retrieved_chunks must be a list")
        if not retrieved_chunks:
            raise ValueError("retrieved_chunks cannot be empty")


class NoneEvaluator(BaseEvaluator):
    """返回空指标的空操作评估器。

    当评估被禁用时使用此实现。
    """

    def __init__(self, settings: Any = None, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)
        return {}