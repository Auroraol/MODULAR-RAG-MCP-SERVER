"""
Core Layer - 核心业务逻辑。

本包包含核心业务逻辑，包括：
- 配置管理（settings.py）
- 核心数据类型（types.py）—— 所有 Pipeline 阶段共享的契约
- 查询引擎
- 响应构建
- Trace 收集
"""

from src.core.types import Document, Chunk, ChunkRecord, Metadata, Vector, SparseVector

__all__ = [
    "Document",
    "Chunk", 
    "ChunkRecord",
    "Metadata",
    "Vector",
    "SparseVector"
]
