"""Chunking 模块 - 文档切分适配层。

本模块提供文本切分的业务适配器，将 Document 对象转换为带有正确元数据
和可追溯性的 Chunk 对象。
"""

from src.ingestion.chunking.document_chunker import DocumentChunker

__all__ = ["DocumentChunker"]
