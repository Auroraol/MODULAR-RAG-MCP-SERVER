"""ChromaDB 的 VectorStore 实现。

本模块使用 ChromaDB 提供 BaseVectorStore 的具体实现。
ChromaDB 是一个轻量、开源的 Embedding 数据库，面向本地优先部署。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from src.core.settings import resolve_path
from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class ChromaStore(BaseVectorStore):
    """VectorStore 的 ChromaDB 实现。

    此类使用 ChromaDB 提供本地优先的持久化向量存储。
    支持 upsert、query 与元数据过滤操作。

    应用的设计原则：
    - 可插拔：实现 BaseVectorStore 接口，可与其他 provider 互换。
    - 配置驱动：所有设置（persist_directory、collection_name）均来自 settings.yaml。
    - 幂等：相同 ID 的 upsert 操作会覆盖已有记录。
    - 可观测：接受可选的 TraceContext（预留给阶段 F）。
    - 快速失败：初始化时校验依赖与配置。

    属性:
        client: ChromaDB 客户端实例。
        collection: 用于存储向量的 ChromaDB collection。
        collection_name: collection 名称。
        persist_directory: 持久化存储的目录路径。

    示例:
        >>> settings = Settings.load('config/settings.yaml')
        >>> store = ChromaStore(settings=settings)
        >>> records = [
        ...     {
        ...         'id': 'doc1_chunk0',
        ...         'vector': [0.1, 0.2, 0.3],
        ...         'metadata': {'source': 'doc1.pdf'}
        ...     }
        ... ]
        >>> store.upsert(records)
        >>> results = store.query([0.1, 0.2, 0.3], top_k=5)
    """
    
    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """使用配置初始化 ChromaStore。

        参数:
            settings: 包含 vector_store 配置的应用设置。
            **kwargs: 可选的 collection_name 或 persist_directory 覆盖。

        异常:
            ImportError: 当未安装 chromadb 包时。
            ValueError: 当缺少必需配置时。
            RuntimeError: 当 ChromaDB 客户端初始化失败时。
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb package is required for ChromaStore. "
                "Install it with: pip install chromadb"
            )
        
        # 提取配置
        try:
            vector_store_config = settings.vector_store
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vector_store. "
                "Please ensure 'vector_store' section exists in settings.yaml"
            ) from e
        
        # collection 名称（允许覆盖）
        self.collection_name = kwargs.get(
            'collection_name',
            getattr(vector_store_config, 'collection_name', 'knowledge_hub')
        )
        
        # 持久化目录（允许覆盖）
        persist_dir_str = kwargs.get(
            'persist_directory',
            getattr(vector_store_config, 'persist_directory', './data/db/chroma')
        )
        self.persist_directory = resolve_path(persist_dir_str)
        
        # 确保持久化目录存在
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initializing ChromaStore: collection='{self.collection_name}', "
            f"persist_directory='{self.persist_directory}'"
        )
        
        # 初始化带持久化存储的 ChromaDB 客户端
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{self.persist_directory}': {e}"
            ) from e
        
        # 获取或创建 collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get or create collection '{self.collection_name}': {e}"
            ) from e
        
        logger.info(
            f"ChromaStore initialized successfully. "
            f"Collection count: {self.collection.count()}"
        )
    
    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """在 ChromaDB 中插入或更新记录。

        参数:
            records: 待 upsert 的记录列表。每条记录必须包含：
                - 'id'：唯一标识符（str）
                - 'vector'：Embedding 向量（List[float]）
                - 'metadata'：可选的元数据 dict
            trace: 可选的 TraceContext，用于可观测性（预留给阶段 F）。
            **kwargs: provider 特定参数（Chroma 中未使用）。

        异常:
            ValueError: 当记录列表为空或包含无效项时。
            RuntimeError: 当 upsert 操作失败时。
        """
        # 校验记录
        self.validate_records(records)

        # 为 ChromaDB 准备数据
        ids = []
        embeddings = []
        metadatas = []
        documents = []  # ChromaDB 要求 documents 字段

        for record in records:
            ids.append(str(record['id']))
            embeddings.append(record['vector'])

            # 元数据：提取或默认为空 dict
            metadata = record.get('metadata', {})
            # 确保所有元数据值都可被 JSON 序列化
            # ChromaDB 要求值为 str、int、float 或 bool
            sanitized_metadata = self._sanitize_metadata(metadata)

            # ChromaDB 要求元数据 dict 非空
            if not sanitized_metadata:
                sanitized_metadata = {'_placeholder': 'true'}

            metadatas.append(sanitized_metadata)

            # 文档：若可用则使用 metadata.text，否则使用 id
            document = metadata.get('text', record['id'])
            documents.append(str(document))

        # 执行 upsert（ChromaDB 的 add() 对相同 ID 是幂等的）
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            logger.debug(f"Successfully upserted {len(records)} records to ChromaDB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to upsert {len(records)} records to ChromaDB: {e}"
            ) from e
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """在 ChromaDB 中查询相似向量。

        参数:
            vector: 待检索的查询向量（Embedding）。
            top_k: 返回结果的最大数量。
            filters: 可选的元数据过滤条件（如 {'source': 'doc1.pdf'}）。
            trace: 可选的 TraceContext，用于可观测性（预留给阶段 F）。
            **kwargs: provider 特定参数（Chroma 中未使用）。

        返回:
            匹配的记录列表，按相似度降序排列。
            每条记录包含：
                - 'id'：记录标识符
                - 'score'：相似度分数（1.0 = 相同，0.0 = 正交）
                - 'metadata'：关联的元数据

        异常:
            ValueError: 当向量为空或 top_k 无效时。
            RuntimeError: 当查询操作失败时。
        """
        # 校验查询参数
        self.validate_query_vector(vector, top_k)

        # 根据过滤条件构建 ChromaDB where 子句
        where_clause = self._build_where_clause(filters) if filters else None

        # 执行查询
        try:
            results = self.collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "distances", "documents"]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query ChromaDB with top_k={top_k}: {e}"
            ) from e
        
        # 将结果转换为标准格式
        # ChromaDB 返回嵌套列表：[[id1, id2, ...]]
        output = []
        
        if results and results['ids'] and results['ids'][0]:
            ids = results['ids'][0]
            distances = results['distances'][0] if 'distances' in results else [0.0] * len(ids)
            metadatas = results['metadatas'][0] if 'metadatas' in results else [{}] * len(ids)
            documents = results['documents'][0] if 'documents' in results else [''] * len(ids)
            
            for i, record_id in enumerate(ids):
                # 将距离转换为相似度分数
                # ChromaDB 返回余弦距离（0=相同，2=相反）
                # 转换为相似度：score = 1 - (distance / 2)
                distance = distances[i]
                score = 1.0 - (distance / 2.0)

                output.append({
                    'id': record_id,
                    'score': max(0.0, score),  # 限制到 [0, 1]
                    'text': documents[i] if documents[i] else '',  # 包含 documents 中的文本
                    'metadata': metadatas[i] if metadatas[i] else {}
                })
        
        logger.debug(f"Query returned {len(output)} results")
        return output
    
    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """按 ID 从 ChromaDB 中删除记录。

        参数:
            ids: 待删除的记录 ID 列表。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: provider 特定参数。

        异常:
            ValueError: 当 ids 列表为空时。
            RuntimeError: 当删除操作失败时。
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")
        
        try:
            self.collection.delete(ids=[str(id_) for id_ in ids])
            logger.debug(f"Successfully deleted {len(ids)} records from ChromaDB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete {len(ids)} records from ChromaDB: {e}"
            ) from e
    
    def clear(
        self,
        collection_name: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """清空 ChromaDB collection 中的所有记录。

        参数:
            collection_name: 可选的待清空 collection 名称。若为 None，则清空当前 collection。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: provider 特定参数。

        异常:
            RuntimeError: 当清空操作失败时。
        """
        try:
            target_collection = collection_name or self.collection_name

            # 删除并重建 collection（Chroma 中最高效的清空方式）
            self.client.delete_collection(name=target_collection)
            self.collection = self.client.get_or_create_collection(
                name=target_collection,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Successfully cleared collection '{target_collection}'")
        except Exception as e:
            raise RuntimeError(
                f"Failed to clear collection '{collection_name or self.collection_name}': {e}"
            ) from e

    def delete_by_metadata(
        self,
        filter_dict: Dict[str, Any],
        trace: Optional[Any] = None,
    ) -> int:
        """删除匹配元数据过滤条件的记录。

        参数:
            filter_dict: 待匹配的元数据键值对
                （如 ``{"source_hash": "abc123"}``）。
            trace: 可选的 TraceContext，用于可观测性。

        返回:
            已删除的记录数量。

        异常:
            ValueError: 当 *filter_dict* 为空时。
            RuntimeError: 当操作失败时。
        """
        if not filter_dict:
            raise ValueError("filter_dict cannot be empty")

        try:
            where = self._build_where_clause(filter_dict)
            # 先查询匹配的 ID
            results = self.collection.get(where=where, include=[])
            matching_ids = results.get("ids", [])

            if not matching_ids:
                logger.debug(f"delete_by_metadata: no records matched {filter_dict}")
                return 0

            self.collection.delete(ids=matching_ids)
            logger.info(
                f"delete_by_metadata: deleted {len(matching_ids)} records "
                f"matching {filter_dict}"
            )
            return len(matching_ids)
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete by metadata {filter_dict}: {e}"
            ) from e
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """清洗元数据以确保 ChromaDB 兼容。

        ChromaDB 要求元数据值为 str、int、float 或 bool。
        此方法会转换或过滤掉不兼容的类型。

        参数:
            metadata: 原始元数据 dict。

        返回:
            清洗后的元数据 dict。
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                # 跳过 None 值
                continue
            elif isinstance(value, (list, tuple)):
                # 转换为逗号分隔的字符串
                sanitized[key] = ",".join(str(v) for v in value)
            else:
                # 回退为字符串
                sanitized[key] = str(value)

        return sanitized

    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """根据过滤条件构建 ChromaDB where 子句。

        将标准过滤 dict 转换为 ChromaDB 的查询格式。

        参数:
            filters: 标准过滤 dict（如 {'source': 'doc1.pdf'}）。

        返回:
            ChromaDB where 子句 dict。

        注意:
            ChromaDB 支持 $eq、$ne、$gt、$lt、$in 等操作符。
            为简化起见，目前仅支持精确相等匹配。
            未来增强：支持复杂过滤条件。
        """
        # 简单实现：仅精确相等匹配
        # 对于复杂过滤条件（如 {'score': {'$gt': 0.5}}），扩展此方法
        where = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                # 已是 ChromaDB 操作符格式（如 {'$eq': 'value'}）
                where[key] = value
            else:
                # 简单相等
                where[key] = value

        return where

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取当前 collection 的统计信息。

        返回:
            包含 collection 统计信息的 dict：
                - count：collection 中的记录数量
                - name：collection 名称
                - metadata：collection 元数据
        """
        return {
            'count': self.collection.count(),
            'name': self.collection_name,
            'metadata': self.collection.metadata
        }
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """按 ID 从 ChromaDB 中检索记录。

        此方法由 SparseRetriever 使用，用于获取 BM25 检索匹配到的 Chunk
        的文本与元数据。

        参数:
            ids: 待检索的记录 ID 列表。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: provider 特定参数（Chroma 中未使用）。

        返回:
            与输入 ids 顺序一致的记录列表。
            每条记录包含：
                - 'id'：记录标识符
                - 'text'：存储的文本内容
                - 'metadata'：关联的元数据
            若某个 ID 未找到，则对应位置返回空 dict。

        异常:
            ValueError: 当 ids 列表为空时。
            RuntimeError: 当检索操作失败时。
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")

        # 确保所有 ID 均为字符串
        str_ids = [str(id_) for id_ in ids]

        try:
            # ChromaDB 的 get 方法按 ID 检索记录
            results = self.collection.get(
                ids=str_ids,
                include=["metadatas", "documents"]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get records by IDs from ChromaDB: {e}"
            ) from e

        # 构建 ID 到结果的映射，以实现 O(1) 查找
        id_to_result: Dict[str, Dict[str, Any]] = {}
        
        if results and results.get('ids'):
            result_ids = results['ids']
            documents = results.get('documents', [None] * len(result_ids))
            metadatas = results.get('metadatas', [{}] * len(result_ids))
            
            for i, record_id in enumerate(result_ids):
                id_to_result[record_id] = {
                    'id': record_id,
                    'text': documents[i] if documents and documents[i] else '',
                    'metadata': metadatas[i] if metadatas and metadatas[i] else {}
                }
        
        # 按输入 ids 的顺序返回结果
        output = []
        for id_ in str_ids:
            if id_ in id_to_result:
                output.append(id_to_result[id_])
            else:
                # 未找到 ID，返回空 dict
                output.append({})
        
        logger.debug(f"Retrieved {len([r for r in output if r])} of {len(ids)} records by IDs")
        return output
