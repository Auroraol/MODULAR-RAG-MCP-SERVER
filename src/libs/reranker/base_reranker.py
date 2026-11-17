"""Reranker provider 的抽象基类。

本模块定义了 Reranker provider 的可插拔接口，通过配置驱动的实例化方式，
可在不同的重排策略（None、Cross-Encoder、LLM-based）之间无缝切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseReranker(ABC):
    """Reranker provider 的抽象基类。

    所有 Reranker 实现都必须继承此类并实现 rerank() 方法。
    这可确保不同重排策略之间拥有一致的接口。

    应用的设计原则：
    - 可插拔：子类可在不修改上游代码的情况下替换。
    - 可观测：接受可选的 TraceContext 以集成可观测性。
    - 配置驱动：实例通过工厂基于配置创建。
    - 可降级：实现应支持安全降级为原始顺序。
    """
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """针对给定查询对候选 Chunk 进行重排。

        参数:
            query: 用户查询字符串。
            candidates: 待重排的候选记录列表。每一项是一个 dict，至少包含
                标识符以及 Reranker 实现所需的字段（如 text、score、metadata）。
            trace: 可选的 TraceContext，用于可观测性（预留给阶段 F）。
            **kwargs: provider 特定的参数（top_k、timeout 等）。

        返回:
            按重排后顺序排列的候选列表。除非有明确文档说明，
            实现应保留候选对象本身，仅改变其顺序。

        异常:
            ValueError: 当 query 或 candidates 无效时。
            RuntimeError: 当 Reranker 意外失败时。
        """
        pass
    
    def validate_query(self, query: str) -> None:
        """校验查询字符串。

        参数:
            query: 待校验的查询字符串。

        异常:
            ValueError: 当 query 不是非空字符串时。
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
    
    def validate_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        """校验候选列表的结构。

        参数:
            candidates: 待校验的候选记录列表。

        异常:
            ValueError: 当候选列表为空或格式错误时。
        """
        if not isinstance(candidates, list):
            raise ValueError("Candidates must be a list of dicts")
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        for i, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise ValueError(
                    f"Candidate at index {i} is not a dict (type: {type(candidate).__name__})"
                )


class NoneReranker(BaseReranker):
    """保持原始顺序的空操作 Reranker。

    当重排被禁用或 provider 被设置为 'none' 时使用此实现。
    它会校验输入并原样返回候选。
    """
    
    def __init__(self, settings: Any = None, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """按原始顺序返回候选。

        参数:
            query: 查询字符串。
            candidates: 待返回的候选列表。
            trace: 可选的 TraceContext（未使用）。
            **kwargs: 忽略。

        返回:
            保持原有顺序的候选浅拷贝。
        """
        self.validate_query(query)
        self.validate_candidates(candidates)
        return list(candidates)
