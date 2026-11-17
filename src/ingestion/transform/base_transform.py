"""Chunk 变换操作的基类。"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """Chunk 变换操作的抽象基类。

    变换操作对 Chunk 进行处理，以提升其质量、添加元数据，
    或为下游处理（Embedding、索引）做好准备。

    设计原则:
        - 单一职责：每个变换只做一种增强
        - 原子操作：单个 Chunk 失败不会影响其他 Chunk
        - 可观测：在 TraceContext 中记录处理信息
        - 优雅降级：遇到不可恢复的错误时返回原始 Chunk
    """
    
    @abstractmethod
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """变换一组 Chunk。

        参数:
            chunks: 要变换的 Chunk 列表
            trace: 可选的可观测性 trace 上下文

        返回:
            变换后的 Chunk 列表（长度与输入一致）

        异常:
            ValueError: 当输入校验失败时
        """
        pass
