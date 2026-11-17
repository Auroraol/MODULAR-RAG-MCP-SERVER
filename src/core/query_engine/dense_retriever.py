"""基于向量 Embedding 的语义检索 Dense Retriever。

本模块实现 DenseRetriever 组件，通过将查询向量化并从 Vector Store 中
检索相似的 Chunk 来执行语义检索。它构成混合检索引擎中的稠密检索路径。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.libs.embedding.base_embedding import BaseEmbedding
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """基于 Embedding 的语义检索 Dense Retriever。

    本类通过以下步骤执行语义检索：
    1. 使用配置的 Embedding 客户端对查询进行向量化
    2. 在 Vector Store 中查询相似向量
    3. 返回规范化的 RetrievalResult 对象

    遵循的设计原则：
    - 可插拔：通过依赖注入接收 embedding_client 与 vector_store。
    - 配置驱动：默认 top_k 读取自 settings.retrieval.dense_top_k。
    - 可观测：接收可选的 TraceContext 以集成可观测性。
    - 快速失败：尽早校验输入并给出清晰错误信息。
    - 类型安全：返回标准化的 RetrievalResult 对象。

    属性:
        embedding_client: 用于查询向量化的 Embedding 提供方。
        vector_store: 用于相似度检索的 Vector Store。
        default_top_k: 默认返回的结果数量。

    示例:
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        >>> from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        >>>
        >>> settings = Settings.load('config/settings.yaml')
        >>> embedding_client = EmbeddingFactory.create(settings)
        >>> vector_store = VectorStoreFactory.create(settings)
        >>>
        >>> retriever = DenseRetriever(
        ...     settings=settings,
        ...     embedding_client=embedding_client,
        ...     vector_store=vector_store
        ... )
        >>> results = retriever.retrieve("What is RAG?", top_k=5)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_client: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
        default_top_k: int = 10,
    ) -> None:
        """使用依赖初始化 DenseRetriever。

        参数:
            settings: 应用配置。未提供 default_top_k 时用于提取该值。
            embedding_client: 用于查询向量化的 Embedding 提供方。
                              执行实际检索操作时必需。
            vector_store: 用于相似度检索的 Vector Store。
                          执行实际检索操作时必需。
            default_top_k: 默认返回的结果数量（默认：10）。
                           可被 settings.retrieval.dense_top_k 覆盖。

        异常:
            ValueError: 当需要 embedding_client 或 vector_store 但其为 None 时。

        注意:
            依赖可被注入用于测试（配合 mock）或生产使用
            （使用工厂创建的真实实现）。
        """
        self.embedding_client = embedding_client
        self.vector_store = vector_store

        # 若配置可用，则从中提取 default_top_k
        self.default_top_k = default_top_k
        if settings is not None:
            retrieval_config = getattr(settings, 'retrieval', None)
            if retrieval_config is not None:
                self.default_top_k = getattr(
                    retrieval_config, 'dense_top_k', default_top_k
                )

        logger.info(
            f"DenseRetriever initialized with default_top_k={self.default_top_k}"
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """为查询检索语义相似的 Chunk。

        参数:
            query: 搜索查询字符串，不能为空。
            top_k: 返回的最大结果数量。为 None 时使用 default_top_k。
            filters: 可选的元数据过滤条件（例如 {"collection": "api-docs"}）。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。

        返回:
            RetrievalResult 对象列表，按相似度降序排序。
            每个结果包含 chunk_id、score、text 与 metadata。

        异常:
            ValueError: 查询为空或无效时。
            RuntimeError: 未配置 embedding_client 或 vector_store，
                          或检索操作失败时。

        示例:
            >>> results = retriever.retrieve("How to configure Azure OpenAI?")
            >>> for result in results:
            ...     print(f"[{result.score:.2f}] {result.chunk_id}: {result.text[:50]}...")
        """
        # 校验输入
        self._validate_query(query)
        self._validate_dependencies()

        # 未指定时使用默认 top_k
        effective_top_k = top_k if top_k is not None else self.default_top_k

        logger.debug(f"Retrieving for query='{query[:50]}...', top_k={effective_top_k}")

        # 步骤 1：对查询进行向量化
        try:
            query_vectors = self.embedding_client.embed([query], trace=trace)
            query_vector = query_vectors[0]
        except Exception as e:
            raise RuntimeError(
                f"Failed to embed query: {e}. "
                "Check embedding client configuration and connectivity."
            ) from e

        # 步骤 2：查询 Vector Store
        try:
            raw_results = self.vector_store.query(
                vector=query_vector,
                top_k=effective_top_k,
                filters=filters,
                trace=trace,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query vector store: {e}. "
                "Check vector store configuration and data availability."
            ) from e

        # 步骤 3：转换为 RetrievalResult 对象
        results = self._transform_results(raw_results)

        logger.debug(f"Retrieved {len(results)} results for query")
        return results

    def _validate_query(self, query: str) -> None:
        """校验查询字符串。

        参数:
            query: 待校验的查询字符串。

        异常:
            ValueError: 查询为空或不是字符串时。
        """
        if not isinstance(query, str):
            raise ValueError(
                f"Query must be a string, got {type(query).__name__}"
            )
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

    def _validate_dependencies(self) -> None:
        """校验所需依赖已配置。

        异常:
            RuntimeError: embedding_client 或 vector_store 为 None 时。
        """
        if self.embedding_client is None:
            raise RuntimeError(
                "DenseRetriever requires an embedding_client. "
                "Provide one during initialization or via setter."
            )
        if self.vector_store is None:
            raise RuntimeError(
                "DenseRetriever requires a vector_store. "
                "Provide one during initialization or via setter."
            )

    def _transform_results(
        self,
        raw_results: List[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """将 Vector Store 的原始结果转换为 RetrievalResult 对象。

        参数:
            raw_results: Vector Store 查询返回的原始结果。
                         每个结果应包含：id、score、text、metadata。

        返回:
            RetrievalResult 对象列表。
        """
        results = []
        for raw in raw_results:
            try:
                result = RetrievalResult(
                    chunk_id=str(raw.get('id', '')),
                    score=float(raw.get('score', 0.0)),
                    text=str(raw.get('text', '')),
                    metadata=raw.get('metadata', {}),
                )
                results.append(result)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Failed to transform result {raw.get('id', 'unknown')}: {e}. "
                    "Skipping this result."
                )
                continue
        
        return results


def create_dense_retriever(
    settings: Settings,
    embedding_client: Optional[BaseEmbedding] = None,
    vector_store: Optional[BaseVectorStore] = None,
) -> DenseRetriever:
    """创建 DenseRetriever 的工厂函数，支持可选的依赖注入。

    该函数简化了 DenseRetriever 的创建：未提供依赖时，
    会自动从工厂创建依赖。

    参数:
        settings: 应用配置。
        embedding_client: 可选，预先配置好的 Embedding 客户端。
                          为 None 时从 EmbeddingFactory 创建。
        vector_store: 可选，预先配置好的 Vector Store。
                      为 None 时从 VectorStoreFactory 创建。

    返回:
        配置好的 DenseRetriever 实例。

    示例:
        >>> settings = Settings.load('config/settings.yaml')
        >>> retriever = create_dense_retriever(settings)
    """
    # 延迟导入以避免循环依赖
    if embedding_client is None:
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        embedding_client = EmbeddingFactory.create(settings)
    
    if vector_store is None:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        vector_store = VectorStoreFactory.create(settings)
    
    return DenseRetriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
