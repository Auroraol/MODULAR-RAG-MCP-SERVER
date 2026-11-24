"""用于将 Chunk 写入向量数据库的 Vector 写入器。

该模块实现 VectorUpserter 组件，负责：
- 根据内容生成确定性的 chunk ID
- 将 Chunk 和向量转换为存储记录
- 调用 VectorStore 进行幂等写入
- 支持顺序一致的批量操作

设计原则：
- 幂等：相同内容产生相同 ID，重复写入安全
- 可观测：接受 TraceContext 以便后续集成
- 配置驱动：使用来自 settings 的 VectorStoreFactory
- 确定性：基于哈希的稳定 ID 生成
- 类型安全：完整的类型提示与校验
"""

import hashlib
from typing import List, Dict, Any, Optional

from src.core.types import Chunk
from src.core.settings import Settings
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class VectorUpserter:
    """以幂等保证将 Chunk 和向量写入向量数据库。

    该写入器从 DenseEncoder 接收 Chunk 及其稠密向量，
    生成稳定的 chunk ID，并写入配置的向量存储。

    Chunk ID 格式:
        {source_path_hash}_{chunk_index:04d}_{content_hash}

        其中:
        - source_path_hash = SHA256(source_path) 的前 8 个字符
        - chunk_index = 补零为 4 位的索引
        - content_hash = SHA256(chunk.text) 的前 8 个字符

    这确保:
        - 相同内容 → 相同 ID（幂等）
        - 内容变化 → 不同 ID（版本化）
        - 人类可读且可溯源到来源

    示例:
        >>> upserter = VectorUpserter(settings)
        >>>
        >>> chunks = [
        ...     Chunk(id="temp1", text="Hello world", metadata={"source_path": "doc.pdf", "chunk_index": 0}),
        ...     Chunk(id="temp2", text="Python rocks", metadata={"source_path": "doc.pdf", "chunk_index": 1})
        ... ]
        >>> vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        >>>
        >>> upserter.upsert(chunks, vectors)
        >>> # Chunk 以稳定 ID 写入，例如: "a1b2c3d4_0000_e5f6g7h8"
    """
    
    def __init__(self, settings: Settings, collection_name: Optional[str] = None):
        """使用配置的向量存储初始化 VectorUpserter。

        参数:
            settings: 包含 vector_store 配置的应用设置。
            collection_name: 可选的集合名称，用于覆盖 settings 中的默认值。

        异常:
            ValueError: 当设置无效或无法创建向量存储时。
        """
        self.settings = settings
        kwargs = {}
        if collection_name:
            kwargs['collection_name'] = collection_name
        self.vector_store = VectorStoreFactory.create(settings, **kwargs)
    
    def upsert(
        self,
        chunks: List[Chunk],
        vectors: List[List[float]],
        trace: Optional[Any] = None,
    ) -> List[str]:
        """将 Chunk 及其向量写入向量存储。

        参数:
            chunks: 要存储的 Chunk 对象列表。
            vectors: Embedding 向量列表（顺序和长度与 chunks 一致）。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。

        返回:
            生成的 chunk ID 列表（顺序与输入的 chunks 一致）。

        异常:
            ValueError: 当 chunks 和 vectors 长度不匹配，或缺少必需的
                       元数据字段时。
            RuntimeError: 当向量存储写入操作失败时。

        示例:
            >>> chunks = [Chunk(...), Chunk(...)]
            >>> vectors = [[0.1, 0.2], [0.3, 0.4]]
            >>> chunk_ids = upserter.upsert(chunks, vectors)
            >>> len(chunk_ids) == len(chunks)  # True
        """
        # 校验输入长度是否匹配
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match vector count ({len(vectors)})"
            )

        if not chunks:
            raise ValueError("Cannot upsert empty chunks list")

        # 生成稳定的 chunk ID 并构建记录
        records = []
        chunk_ids = []

        # 生成稳定的 chunk ID 并构建记录
        for chunk, vector in zip(chunks, vectors):
            # 生成确定性的 chunk ID
            chunk_id = self._generate_chunk_id(chunk)
            chunk_ids.append(chunk_id)

            # 构建存储记录
            record = {
                "id": chunk_id,
                "vector": vector,
                "metadata": {
                    **chunk.metadata,  # 保留所有原始元数据
                    "text": chunk.text,  # 存储文本以便检索
                    "chunk_id": chunk_id,  # 冗余字段，但便于查询
                },
            }
            records.append(record)

        # 执行幂等写入
        try:
            self.vector_store.upsert(records, trace=trace)
        except Exception as e:
            raise RuntimeError(
                f"Vector store upsert failed: {str(e)}"
            ) from e

        return chunk_ids

    def _generate_chunk_id(self, chunk: Chunk) -> str:
        """根据内容生成确定性的 chunk ID。

        参数:
            chunk: 要为其生成 ID 的 Chunk 对象。

        返回:
            格式: {source_hash}_{index:04d}_{content_hash}
            稳定的 chunk ID 字符串。

        异常:
            ValueError: 当缺少必需的元数据字段时。
        """
        # 校验必需的元数据
        if "source_path" not in chunk.metadata:
            raise ValueError("Chunk metadata must contain 'source_path'")
        if "chunk_index" not in chunk.metadata:
            raise ValueError("Chunk metadata must contain 'chunk_index'")

        source_path = chunk.metadata["source_path"]
        chunk_index = chunk.metadata["chunk_index"]

        # 计算稳定哈希
        source_hash = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:8]
        content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:8]

        # 格式: {source_hash}_{index:04d}_{content_hash}
        chunk_id = f"{source_hash}_{chunk_index:04d}_{content_hash}"
        
        return chunk_id
    
    def upsert_batch(
        self,
        batches: List[tuple[List[Chunk], List[List[float]]]],
        trace: Optional[Any] = None,
    ) -> List[str]:
        """写入多批 Chunk 和向量。

        这是一个便捷方法，用于处理来自 BatchProcessor 的输出。
        所有批次会被展平并在单次写入操作中处理，
        以保持顺序并减少向量存储的往返次数。

        参数:
            batches: 来自批处理的 (chunks, vectors) 元组列表。
            trace: 可选的 TraceContext，用于可观测性。

        返回:
            按顺序生成的所有 chunk ID 列表。

        示例:
            >>> batch1 = ([chunk1, chunk2], [[0.1, 0.2], [0.3, 0.4]])
            >>> batch2 = ([chunk3], [[0.5, 0.6]])
            >>> chunk_ids = upserter.upsert_batch([batch1, batch2])
            >>> len(chunk_ids)  # 3
        """
        # 展平所有批次
        all_chunks = []
        all_vectors = []

        for chunks, vectors in batches:
            all_chunks.extend(chunks)
            all_vectors.extend(vectors)

        # 单次写入操作
        return self.upsert(all_chunks, all_vectors, trace=trace)
