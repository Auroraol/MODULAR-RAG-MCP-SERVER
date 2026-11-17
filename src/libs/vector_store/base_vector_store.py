"""VectorStore provider 的抽象基类。

本模块定义了 VectorStore provider 的可插拔接口，通过配置驱动的实例化方式，
可在不同的后端（Chroma、Qdrant、Milvus 等）之间无缝切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseVectorStore(ABC):
    """VectorStore provider 的抽象基类。

    所有 VectorStore 实现都必须继承此类并实现 upsert() 与 query() 方法。
    这可确保不同 provider（Chroma、Qdrant、Milvus 等）之间拥有一致的接口。

    应用的设计原则：
    - 可插拔：子类可在不修改上游代码的情况下替换。
    - 可观测：接受可选的 TraceContext 以集成可观测性。
    - 配置驱动：实例通过工厂基于配置创建。
    - 幂等：upsert() 操作应可安全重复执行。
    """
    
    @abstractmethod
    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """在 Vector Store 中插入或更新记录。

        参数:
            records: 待 upsert 的记录列表。每条记录是一个 dict，包含：
                - 'id'：唯一标识符（str）
                - 'vector'：Embedding 向量（List[float]）
                - 'metadata'：可选的元数据 dict（source、chunk_index 等）
            trace: 可选的 TraceContext，用于可观测性（预留给阶段 F）。
            **kwargs: provider 特定参数。

        异常:
            ValueError: 当记录列表为空或包含无效项时。
            RuntimeError: 当 Vector Store 操作失败时。

        示例:
            >>> records = [
            ...     {
            ...         'id': 'doc1_chunk0',
            ...         'vector': [0.1, 0.2, ..., 0.5],
            ...         'metadata': {'source': 'doc1.pdf', 'page': 1}
            ...     }
            ... ]
            >>> vector_store.upsert(records)

        注意:
            - 此操作应当幂等：多次 upsert 同一条记录
              应产生相同的最终状态。
            - 实现应高效地处理批量操作。
        """
        pass
    
    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """在 Vector Store 中查询相似的向量。

        参数:
            vector: 待检索的查询向量（Embedding）。
            top_k: 返回结果的最大数量。
            filters: 可选的元数据过滤条件（如 {'source': 'doc1.pdf'}）。
            trace: 可选的 TraceContext，用于可观测性（预留给阶段 F）。
            **kwargs: provider 特定参数。

        返回:
            匹配的记录列表，按相似度降序排列。
            每条记录是一个 dict，包含：
                - 'id'：记录标识符
                - 'score'：相似度分数（越高越相似）
                - 'metadata'：关联的元数据
                - 'vector'：可选，存储的向量（取决于 provider）

        异常:
            ValueError: 当向量为空或 top_k 无效时。
            RuntimeError: 当 Vector Store 查询失败时。

        示例:
            >>> query_vector = [0.1, 0.2, ..., 0.5]
            >>> results = vector_store.query(query_vector, top_k=5)
            >>> for result in results:
            ...     print(f"ID: {result['id']}, Score: {result['score']}")
        """
        pass
    
    def validate_records(self, records: List[Dict[str, Any]]) -> None:
        """在 upsert 前校验记录。

        参数:
            records: 待校验的记录列表。

        异常:
            ValueError: 当记录列表为空或包含无效项时。
        """
        if not records:
            raise ValueError("Records list cannot be empty")

        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Record at index {i} is not a dict (type: {type(record).__name__})"
                )

            # 校验必需字段
            if 'id' not in record:
                raise ValueError(f"Record at index {i} is missing required field: 'id'")
            if 'vector' not in record:
                raise ValueError(f"Record at index {i} is missing required field: 'vector'")

            # 校验向量格式
            vector = record['vector']
            if not isinstance(vector, (list, tuple)):
                raise ValueError(
                    f"Record at index {i} has invalid vector type: {type(vector).__name__}. "
                    "Expected list or tuple of floats."
                )
            
            if not vector:
                raise ValueError(f"Record at index {i} has empty vector")
    
    def validate_query_vector(self, vector: List[float], top_k: int) -> None:
        """校验查询参数。

        参数:
            vector: 待校验的查询向量。
            top_k: 待校验的结果数量。

        异常:
            ValueError: 当参数无效时。
        """
        if not isinstance(vector, (list, tuple)):
            raise ValueError(
                f"Query vector must be a list or tuple, got {type(vector).__name__}"
            )
        
        if not vector:
            raise ValueError("Query vector cannot be empty")
        
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
    
    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """按 ID 从 Vector Store 中删除记录。

        参数:
            ids: 待删除的记录 ID 列表。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: provider 特定参数。

        异常:
            ValueError: 当 ids 列表为空时。
            RuntimeError: 当删除操作失败时。
            NotImplementedError: 当 provider 不支持删除时。

        注意:
            这是可选操作。不支持删除的 provider
            应抛出带有明确消息的 NotImplementedError。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement delete() method. "
            "This operation is optional and provider-dependent."
        )
    
    def clear(
        self,
        collection_name: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """清空 Vector Store 中的所有记录，或清空指定 collection。

        参数:
            collection_name: 可选的待清空 collection 名称。若为 None，则清空默认 collection。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: provider 特定参数。

        异常:
            RuntimeError: 当清空操作失败时。
            NotImplementedError: 当 provider 不支持清空时。

        注意:
            这主要用于测试与开发。生产环境请谨慎使用。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement clear() method. "
            "This operation is optional and primarily for testing."
        )
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """按 ID 检索记录。

        此方法由 SparseRetriever 使用，用于获取 BM25 检索匹配到的 Chunk
        的文本与元数据（BM25 只返回 ID 与分数）。

        参数:
            ids: 待检索的记录 ID 列表。
            trace: 可选的 TraceContext，用于可观测性（预留给阶段 F）。
            **kwargs: provider 特定参数。

        返回:
            与输入 ids 顺序一致的记录列表。
            每条记录是一个 dict，包含：
                - 'id'：记录标识符
                - 'text'：存储的文本内容
                - 'metadata'：关联的元数据
            若某个 ID 未找到，则对应位置返回空 dict。

        异常:
            ValueError: 当 ids 列表为空时。
            RuntimeError: 当检索操作失败时。
            NotImplementedError: 当 provider 不支持此操作时。

        示例:
            >>> ids = ["chunk_001", "chunk_002", "chunk_003"]
            >>> records = vector_store.get_by_ids(ids)
            >>> for record in records:
            ...     print(f"ID: {record['id']}, Text: {record['text'][:50]}...")

        注意:
            此操作对混合检索至关重要：BM25 返回的 Chunk ID
            需要从 Vector Store 中补充文本与元数据。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_by_ids() method. "
            "This operation is required for SparseRetriever support."
        )
