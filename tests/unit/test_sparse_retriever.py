"""SparseRetriever 的单元测试。

测试覆盖：
- 初始化与配置
- 关键词校验
- 依赖校验
- BM25 查询 + Vector Store get_by_ids 集成
- 结果合并与转换
- 错误处理与边界情况
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from src.core.query_engine.retriever.sparse_retriever import SparseRetriever, create_sparse_retriever
from src.core.types import RetrievalResult


# ============================================================================
# Mock 类
# ============================================================================

class MockBM25Indexer:
    """用于测试的 Mock BM25 索引器。"""
    
    def __init__(self, index_dir: str = "data/db/bm25"):
        self.index_dir = index_dir
        self._index = {}
        self._metadata = {}
        self._loaded_collection = None
        
    def load(self, collection: str = "default", trace: Optional[Any] = None) -> bool:
        """模拟加载索引。"""
        self._loaded_collection = collection
        self._metadata = {"collection": collection}
        return True
    
    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """返回 Mock 的 BM25 结果。"""
        # 根据查询词项返回预设结果
        results = [
            {"chunk_id": "chunk_001", "score": 2.5},
            {"chunk_id": "chunk_002", "score": 1.8},
            {"chunk_id": "chunk_003", "score": 1.2},
        ]
        return results[:top_k]


class MockBM25IndexerEmpty:
    """返回空结果的 Mock BM25 索引器。"""
    
    def __init__(self):
        self._metadata = {}
    
    def load(self, collection: str = "default", trace: Optional[Any] = None) -> bool:
        self._metadata = {"collection": collection}
        return True
    
    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        return []


class MockBM25IndexerFailing:
    """在加载或查询时失败的 Mock BM25 索引器。"""
    
    def __init__(self, fail_on_load: bool = False, fail_on_query: bool = False):
        self.fail_on_load = fail_on_load
        self.fail_on_query = fail_on_query
        self._metadata = {}
    
    def load(self, collection: str = "default", trace: Optional[Any] = None) -> bool:
        if self.fail_on_load:
            raise RuntimeError("Simulated load failure")
        self._metadata = {"collection": collection}
        return True
    
    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        if self.fail_on_query:
            raise RuntimeError("Simulated query failure")
        return []


class MockVectorStore:
    """用于测试的 Mock Vector Store。"""
    
    def __init__(self):
        self._records = {
            "chunk_001": {
                "id": "chunk_001",
                "text": "This is the first chunk about machine learning.",
                "metadata": {"source_path": "doc1.pdf", "chunk_index": 0}
            },
            "chunk_002": {
                "id": "chunk_002",
                "text": "This is the second chunk about neural networks.",
                "metadata": {"source_path": "doc1.pdf", "chunk_index": 1}
            },
            "chunk_003": {
                "id": "chunk_003",
                "text": "This is the third chunk about deep learning.",
                "metadata": {"source_path": "doc2.pdf", "chunk_index": 0}
            },
        }
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """按 ID 返回 Mock 记录。"""
        results = []
        for id_ in ids:
            if id_ in self._records:
                results.append(self._records[id_])
            else:
                results.append({})  # 未找到
        return results


class MockVectorStoreFailing:
    """在 get_by_ids 时失败的 Mock Vector Store。"""
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        raise RuntimeError("Simulated vector store failure")


class MockSettings:
    """用于测试的 Mock 配置。"""
    
    def __init__(self, sparse_top_k: int = 15):
        self.retrieval = MagicMock()
        self.retrieval.sparse_top_k = sparse_top_k


# ============================================================================
# 测试：初始化
# ============================================================================

class TestSparseRetrieverInit:
    """SparseRetriever 初始化的测试。"""

    def test_init_with_defaults(self):
        """测试使用默认参数初始化。"""
        retriever = SparseRetriever()
        
        assert retriever.bm25_indexer is None
        assert retriever.vector_store is None
        assert retriever.default_top_k == 10
        assert retriever.default_collection == "default"
    
    def test_init_with_custom_top_k(self):
        """测试使用自定义 default_top_k 初始化。"""
        retriever = SparseRetriever(default_top_k=20)
        
        assert retriever.default_top_k == 20
    
    def test_init_with_settings(self):
        """测试从配置中提取 top_k 进行初始化。"""
        settings = MockSettings(sparse_top_k=25)
        retriever = SparseRetriever(settings=settings)
        
        assert retriever.default_top_k == 25
    
    def test_init_with_dependencies(self):
        """测试使用注入的依赖初始化。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        
        retriever = SparseRetriever(
            bm25_indexer=bm25,
            vector_store=vs,
            default_collection="my_collection"
        )
        
        assert retriever.bm25_indexer is bm25
        assert retriever.vector_store is vs
        assert retriever.default_collection == "my_collection"


# ============================================================================
# 测试：输入校验
# ============================================================================

class TestSparseRetrieverValidation:
    """输入校验的测试。"""

    def test_retrieve_raises_on_empty_keywords(self):
        """测试空关键词会抛出 ValueError。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve([])
    
    def test_retrieve_raises_on_non_list_keywords(self):
        """测试非列表类型的关键词会抛出 ValueError。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        with pytest.raises(ValueError, match="must be a list"):
            retriever.retrieve("not a list")  # type: ignore
    
    def test_retrieve_raises_without_bm25_indexer(self):
        """测试缺少 bm25_indexer 会抛出 RuntimeError。"""
        vs = MockVectorStore()
        retriever = SparseRetriever(vector_store=vs)
        
        with pytest.raises(RuntimeError, match="requires a bm25_indexer"):
            retriever.retrieve(["keyword"])
    
    def test_retrieve_raises_without_vector_store(self):
        """测试缺少 vector_store 会抛出 RuntimeError。"""
        bm25 = MockBM25Indexer()
        retriever = SparseRetriever(bm25_indexer=bm25)
        
        with pytest.raises(RuntimeError, match="requires a vector_store"):
            retriever.retrieve(["keyword"])


# ============================================================================
# 测试：基本检索
# ============================================================================

class TestSparseRetrieverRetrieve:
    """retrieve() 方法的测试。"""

    def test_retrieve_returns_results(self):
        """测试基本检索会返回 RetrievalResult 对象。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["machine", "learning"])
        
        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)
    
    def test_retrieve_results_have_correct_fields(self):
        """测试结果包含所有必需字段。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"])
        
        first = results[0]
        assert first.chunk_id == "chunk_001"
        assert first.score == 2.5
        assert "machine learning" in first.text
        assert first.metadata["source_path"] == "doc1.pdf"
    
    def test_retrieve_respects_top_k(self):
        """测试 top_k 参数会限制结果数量。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"], top_k=2)
        
        assert len(results) == 2
    
    def test_retrieve_uses_default_top_k(self):
        """测试未指定 top_k 时会使用 default_top_k。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(
            bm25_indexer=bm25, 
            vector_store=vs, 
            default_top_k=2
        )
        
        results = retriever.retrieve(["test"])
        
        assert len(results) == 2
    
    def test_retrieve_with_custom_collection(self):
        """测试使用自定义 collection 参数进行检索。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        # 不应抛出异常，collection 会传递给 bm25_indexer
        results = retriever.retrieve(["test"], collection="custom_collection")
        
        assert len(results) > 0
    
    def test_retrieve_preserves_result_order(self):
        """测试结果保持 BM25 分数排序。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"])
        
        # 结果应按分数降序排列（来自 BM25）
        assert results[0].score > results[1].score > results[2].score


# ============================================================================
# 测试：空结果
# ============================================================================

class TestSparseRetrieverEmptyResults:
    """空结果处理的测试。"""

    def test_retrieve_returns_empty_when_no_matches(self):
        """测试 BM25 无匹配时返回空列表。"""
        bm25 = MockBM25IndexerEmpty()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["nonexistent"])
        
        assert results == []
    
    def test_retrieve_returns_empty_when_index_not_loaded(self):
        """测试索引加载失败时能优雅处理。"""
        bm25 = MockBM25IndexerFailing(fail_on_load=True)
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        # 应返回空列表，而不是抛出异常
        results = retriever.retrieve(["test"])
        
        assert results == []


# ============================================================================
# 测试：错误处理
# ============================================================================

class TestSparseRetrieverErrorHandling:
    """错误处理的测试。"""

    def test_retrieve_raises_on_bm25_query_failure(self):
        """测试 BM25 查询失败会抛出 RuntimeError。"""
        bm25 = MockBM25IndexerFailing(fail_on_query=True)
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        with pytest.raises(RuntimeError, match="Failed to query BM25"):
            retriever.retrieve(["test"])
    
    def test_retrieve_raises_on_vector_store_failure(self):
        """测试 Vector Store 失败会抛出 RuntimeError。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStoreFailing()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        with pytest.raises(RuntimeError, match="Failed to fetch records"):
            retriever.retrieve(["test"])


# ============================================================================
# 测试：缺失记录
# ============================================================================

class TestSparseRetrieverMissingRecords:
    """缺失记录处理的测试。"""

    def test_retrieve_skips_missing_records(self):
        """测试 Vector Store 中缺失的记录会被跳过。"""
        bm25 = MockBM25Indexer()

        # 创建只包含部分记录的 Vector Store
        vs = MockVectorStore()
        vs._records = {
            "chunk_001": {
                "id": "chunk_001",
                "text": "First chunk.",
                "metadata": {"source_path": "doc.pdf"}
            },
            # chunk_002 和 chunk_003 缺失
        }
        
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        results = retriever.retrieve(["test"])
        
        # 应只返回找到的那条记录
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_001"


# ============================================================================
# 测试：结果转换
# ============================================================================

class TestSparseRetrieverResultTransformation:
    """结果转换的测试。"""

    def test_retrieve_result_is_serializable(self):
        """测试结果可以序列化为字典。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"], top_k=1)
        
        result_dict = results[0].to_dict()
        assert "chunk_id" in result_dict
        assert "score" in result_dict
        assert "text" in result_dict
        assert "metadata" in result_dict
    
    def test_retrieve_result_score_is_float(self):
        """测试分数会被转换为 float。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"])
        
        for result in results:
            assert isinstance(result.score, float)


# ============================================================================
# 测试：工厂函数
# ============================================================================

class TestCreateSparseRetriever:
    """create_sparse_retriever 工厂函数的测试。"""

    def test_create_with_injected_dependencies(self):
        """测试使用注入依赖的工厂函数。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        settings = MockSettings()
        
        retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25,
            vector_store=vs,
        )
        
        assert retriever.bm25_indexer is bm25
        assert retriever.vector_store is vs


# ============================================================================
# 测试：索引加载
# ============================================================================

class TestSparseRetrieverIndexLoading:
    """BM25 索引加载行为的测试。"""

    def test_index_loaded_once_per_collection(self):
        """测试每个 collection 的索引只加载一次。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)

        # 首次调用应加载索引
        retriever.retrieve(["test"], collection="test_col")
        assert bm25._loaded_collection == "test_col"

        # 使用相同 collection 的第二次调用不应重新加载
        bm25._loaded_collection = "test_col"  # 标记为已加载
        retriever.retrieve(["test"], collection="test_col")
        assert bm25._loaded_collection == "test_col"
    
    def test_index_reloaded_for_different_collection(self):
        """测试切换到不同 collection 时会重新加载索引。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)

        # 加载第一个 collection
        retriever.retrieve(["test"], collection="col_a")
        assert bm25._metadata.get("collection") == "col_a"

        # 加载不同的 collection
        retriever.retrieve(["test"], collection="col_b")
        assert bm25._metadata.get("collection") == "col_b"


# ============================================================================
# 测试：与真实类型的集成
# ============================================================================

class TestSparseRetrieverTypeIntegration:
    """与 core 类型集成的测试。"""

    def test_retrieve_result_type_matches_dense_retriever(self):
        """测试 SparseRetriever 返回的类型与 DenseRetriever 相同。"""
        from src.core.types import RetrievalResult as TypedRetrievalResult
        
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"])
        
        for result in results:
            assert isinstance(result, TypedRetrievalResult)
    
    def test_retrieve_result_compatible_with_from_dict(self):
        """测试结果可以从字典重建。"""
        bm25 = MockBM25Indexer()
        vs = MockVectorStore()
        retriever = SparseRetriever(bm25_indexer=bm25, vector_store=vs)
        
        results = retriever.retrieve(["test"], top_k=1)
        result_dict = results[0].to_dict()
        
        # 从字典重建
        recreated = RetrievalResult.from_dict(result_dict)
        
        assert recreated.chunk_id == results[0].chunk_id
        assert recreated.score == results[0].score
        assert recreated.text == results[0].text
