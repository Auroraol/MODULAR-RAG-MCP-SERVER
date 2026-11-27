"""使用 BM25 的关键词检索 Sparse Retriever 稠密检索。

本模块实现 SparseRetriever 组件，使用 BM25 倒排索引执行基于关键词的检索。
它构成混合检索引擎中的稀疏检索路径，与 DenseRetriever 的语义检索互相补充。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class SparseRetriever:
    """基于 BM25 关键词检索的 Sparse Retriever。

    本类通过以下步骤执行基于关键词的检索：
    1. 使用关键词查询 BM25 索引，获得匹配的 Chunk ID 与分数
    2. 使用 get_by_ids() 从 Vector Store 获取文本与元数据
    3. 返回规范化的 RetrievalResult 对象

    遵循的设计原则：
    - 可插拔：通过依赖注入接收 bm25_indexer 与 vector_store。
    - 配置驱动：默认 top_k 与 collection 读取自 settings。
    - 可观测：接收可选的 TraceContext 以集成可观测性。
    - 快速失败：尽早校验输入并给出清晰错误信息。
    - 类型安全：返回与 DenseRetriever 相同的标准化 RetrievalResult 对象。

    属性:
        bm25_indexer: 用于关键词检索的 BM25 索引器。
        vector_store: 用于获取文本与元数据的 Vector Store。
        default_top_k: 默认返回的结果数量。
        default_collection: 默认查询的 BM25 索引 collection。

    示例:
        >>> from src.ingestion.storage.bm25_indexer import BM25Indexer
        >>> from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        >>>
        >>> settings = Settings.load('config/settings.yaml')
        >>> bm25_indexer = BM25Indexer(index_dir="data/db/bm25")
        >>> bm25_indexer.load("default")
        >>> vector_store = VectorStoreFactory.create(settings)
        >>>
        >>> retriever = SparseRetriever(
        ...     settings=settings,
        ...     bm25_indexer=bm25_indexer,
        ...     vector_store=vector_store
        ... )
        >>> results = retriever.retrieve(["RAG", "retrieval"], top_k=5)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None,
        default_top_k: int = 10,
        default_collection: str = "default",
    ) -> None:
        """使用依赖初始化 SparseRetriever。

        参数:
            settings: 应用配置。未提供 default_top_k 时用于提取该值。
            bm25_indexer: 用于关键词检索的 BM25 索引器。
                          执行实际检索操作时必需。
            vector_store: 用于获取文本与元数据的 Vector Store。
                          执行实际检索操作时必需。
            default_top_k: 默认返回的结果数量（默认：10）。
                           可被 settings.retrieval.sparse_top_k 覆盖。
            default_collection: 默认的 BM25 索引 collection 名称（默认："default"）。

        注意:
            依赖可被注入用于测试（配合 mock）或生产使用
            （使用工厂创建的真实实现）。
        """
        self.bm25_indexer = bm25_indexer
        self.vector_store = vector_store
        self.default_collection = default_collection

        # 若配置可用，则从中提取 default_top_k
        self.default_top_k = default_top_k
        if settings is not None:
            retrieval_config = getattr(settings, 'retrieval', None)
            if retrieval_config is not None:
                self.default_top_k = getattr(
                    retrieval_config, 'sparse_top_k', default_top_k
                )

        logger.info(
            f"SparseRetriever initialized with default_top_k={self.default_top_k}, "
            f"default_collection='{self.default_collection}'"
        )

    def retrieve(
        self,
        keywords: List[str],
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """使用 BM25 检索匹配给定关键词的 Chunk。

        参数:
            keywords: 要搜索的关键词列表（通常来自 QueryProcessor）。
            top_k: 返回的最大结果数量。为 None 时使用 default_top_k。
            collection: 要查询的 BM25 索引 collection。为 None 时使用 default_collection。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。

        返回:
            RetrievalResult 对象列表，按 BM25 分数降序排序。
            每个结果包含 chunk_id、score、text 与 metadata。

        异常:
            ValueError: 关键词列表为空时。
            RuntimeError: 未配置 bm25_indexer 或 vector_store，
                          或检索操作失败时。

        示例:
            >>> results = retriever.retrieve(["Azure", "OpenAI", "配置"])
            >>> for result in results:
            ...     print(f"[{result.score:.2f}] {result.chunk_id}: {result.text[:50]}...")
        """
        # 校验输入
        self._validate_keywords(keywords)
        self._validate_dependencies()

        # 未指定时使用默认值
        effective_top_k = top_k if top_k is not None else self.default_top_k
        effective_collection = collection if collection is not None else self.default_collection

        logger.debug(
            f"Retrieving for keywords={keywords[:5]}{'...' if len(keywords) > 5 else ''}, "
            f"top_k={effective_top_k}, collection='{effective_collection}'"
        )

        # 步骤 1：确保索引已加载
        if not self._ensure_index_loaded(effective_collection):
            logger.warning(
                f"BM25 index for collection '{effective_collection}' not available. "
                "Returning empty results."
            )
            return []

        # 步骤 2：查询 BM25 索引
        try:
            bm25_results = self.bm25_indexer.query(
                query_terms=keywords,
                top_k=effective_top_k,
                trace=trace,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query BM25 index: {e}. "
                "Check index availability and query terms."
            ) from e

        # 无匹配时提前返回
        if not bm25_results:
            logger.debug("BM25 query returned no results")
            return []

        # 步骤 3：从 Vector Store 获取文本与元数据
        chunk_ids = [r["chunk_id"] for r in bm25_results]
        try:
            records = self.vector_store.get_by_ids(chunk_ids, trace=trace)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch records from vector store: {e}. "
                "Check vector store configuration and data availability."
            ) from e

        # 步骤 4：合并 BM25 分数与文本/元数据
        results = self._merge_results(bm25_results, records)

        logger.debug(f"Retrieved {len(results)} results for keywords")
        return results

    def _validate_keywords(self, keywords: List[str]) -> None:
        """校验关键词列表。

        参数:
            keywords: 待校验的关键词列表。

        异常:
            ValueError: 关键词为空或不是列表时。
        """
        if not isinstance(keywords, list):
            raise ValueError(
                f"Keywords must be a list, got {type(keywords).__name__}"
            )
        if not keywords:
            raise ValueError("Keywords list cannot be empty")
        # 过滤空字符串但允许调用继续
        # （空字符串自然不会匹配任何内容）

    def _validate_dependencies(self) -> None:
        """校验所需依赖已配置。

        异常:
            RuntimeError: bm25_indexer 或 vector_store 为 None 时。
        """
        if self.bm25_indexer is None:
            raise RuntimeError(
                "SparseRetriever requires a bm25_indexer. "
                "Provide one during initialization or via setter."
            )
        if self.vector_store is None:
            raise RuntimeError(
                "SparseRetriever requires a vector_store. "
                "Provide one during initialization or via setter."
            )

    def _ensure_index_loaded(self, collection: str) -> bool:
        """确保给定 collection 的 BM25 索引已加载。

        每次都从磁盘重新加载，因为索引可能已被其他进程
        （例如 Dashboard 的数据摄入）更新。相比整体查询，
        加载过程很快（仅读取单个 JSON 文件）。

        参数:
            collection: 要加载的 collection 名称。

        返回:
            索引已加载并可用时返回 True，否则返回 False。
        """
        try:
            loaded = self.bm25_indexer.load(collection=collection)
            return loaded
        except Exception as e:
            logger.warning(f"Failed to load BM25 index for collection '{collection}': {e}")
            return False

    def _merge_results(
        self,
        bm25_results: List[Dict[str, Any]],
        records: List[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """将 BM25 分数与来自 Vector Store 的文本和元数据合并。

        参数:
            bm25_results: BM25 查询结果，每项包含 'chunk_id' 与 'score'。
            records: Vector Store 的记录，每项包含 'id'、'text'、'metadata'。

        返回:
            包含完整信息的 RetrievalResult 对象列表。
        """
        results = []

        for bm25_result, record in zip(bm25_results, records):
            chunk_id = bm25_result["chunk_id"]
            score = bm25_result["score"]

            # 处理未找到记录的情况
            if not record:
                logger.warning(
                    f"No record found in vector store for chunk_id='{chunk_id}'. "
                    "Skipping this result."
                )
                continue

            # 校验记录包含预期字段
            text = record.get('text', '')
            metadata = record.get('metadata', {})

            try:
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    text=str(text),
                    metadata=metadata,
                )
                results.append(result)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Failed to create RetrievalResult for chunk_id='{chunk_id}': {e}. "
                    "Skipping this result."
                )
                continue

        return results


def create_sparse_retriever(
    settings: Settings,
    bm25_indexer: Optional[BM25Indexer] = None,
    vector_store: Optional[BaseVectorStore] = None,
    index_dir: str = "data/db/bm25",
) -> SparseRetriever:
    """创建 SparseRetriever 的工厂函数，支持可选的依赖注入。

    该函数简化了 SparseRetriever 的创建：未提供依赖时，
    会自动从工厂创建依赖。

    参数:
        settings: 应用配置。
        bm25_indexer: 可选，预先配置好的 BM25 索引器。
                      为 None 时使用默认 index_dir 创建。
        vector_store: 可选，预先配置好的 Vector Store。
                      为 None 时从 VectorStoreFactory 创建。
        index_dir: BM25 索引文件所在目录（默认："data/db/bm25"）。

    返回:
        配置好的 SparseRetriever 实例。

    示例:
        >>> settings = Settings.load('config/settings.yaml')
        >>> retriever = create_sparse_retriever(settings)
    """
    # 延迟导入以避免循环依赖
    if bm25_indexer is None:
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        bm25_indexer = BM25Indexer(index_dir=index_dir)
    
    if vector_store is None:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        vector_store = VectorStoreFactory.create(settings)
    
    return SparseRetriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
