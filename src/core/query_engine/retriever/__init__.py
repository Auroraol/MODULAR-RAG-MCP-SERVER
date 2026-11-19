"""
Retriever 模块。

本包包含混合检索的两条召回路径：
- DenseRetriever：基于 Embedding 的语义检索
- SparseRetriever：基于 BM25 的关键词检索
"""

from src.core.query_engine.retriever.dense_retriever import (
    DenseRetriever,
    create_dense_retriever,
)
from src.core.query_engine.retriever.sparse_retriever import (
    SparseRetriever,
    create_sparse_retriever,
)

__all__ = [
    "DenseRetriever",
    "create_dense_retriever",
    "SparseRetriever",
    "create_sparse_retriever",
]
