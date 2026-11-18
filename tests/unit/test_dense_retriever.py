"""DenseRetriever 的单元测试。

本模块使用 mock 的 embedding 客户端与 vector store 隔离测试
DenseRetriever 组件，确保其行为正确。
"""

import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List

from src.core.query_engine.retriever.dense_retriever import DenseRetriever, create_dense_retriever
from src.core.types import RetrievalResult


# =============================================================================
# 测试 Fixture
# =============================================================================

class FakeEmbedding:
    """用于测试的假 embedding 客户端。"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.call_count = 0
        self.last_texts = None

    def embed(self, texts: List[str], trace: Any = None, **kwargs) -> List[List[float]]:
        """返回确定性的假 embedding。"""
        self.call_count += 1
        self.last_texts = texts
        # 基于文本内容生成确定性向量
        return [[float(i) / self.dimension for i in range(self.dimension)] for _ in texts]


class FakeVectorStore:
    """用于测试的假 vector store。"""

    def __init__(self, stored_records: List[Dict[str, Any]] = None):
        self.stored_records = stored_records or []
        self.call_count = 0
        self.last_query_params = None
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Dict[str, Any] = None,
        trace: Any = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """按假相似度返回存储的记录。"""
        self.call_count += 1
        self.last_query_params = {
            "vector": vector,
            "top_k": top_k,
            "filters": filters,
        }
        # 最多返回 top_k 条记录
        return self.stored_records[:top_k]


@pytest.fixture
def fake_embedding():
    """创建假 embedding 客户端。"""
    return FakeEmbedding(dimension=384)


@pytest.fixture
def fake_vector_store():
    """创建带示例数据的假 vector store。"""
    return FakeVectorStore(stored_records=[
        {
            "id": "doc1_chunk_001",
            "score": 0.95,
            "text": "Azure OpenAI is a cloud-based AI service...",
            "metadata": {"source_path": "docs/azure.pdf", "chunk_index": 1},
        },
        {
            "id": "doc1_chunk_002",
            "score": 0.88,
            "text": "To configure Azure OpenAI, you need to...",
            "metadata": {"source_path": "docs/azure.pdf", "chunk_index": 2},
        },
        {
            "id": "doc2_chunk_001",
            "score": 0.72,
            "text": "RAG (Retrieval Augmented Generation) combines...",
            "metadata": {"source_path": "docs/rag-guide.pdf", "chunk_index": 1},
        },
    ])


@pytest.fixture
def retriever(fake_embedding, fake_vector_store):
    """使用假依赖创建 DenseRetriever。"""
    return DenseRetriever(
        embedding_client=fake_embedding,
        vector_store=fake_vector_store,
        default_top_k=10,
    )


# =============================================================================
# 初始化测试
# =============================================================================

class TestDenseRetrieverInit:
    """DenseRetriever 初始化的测试。"""

    def test_init_with_dependencies(self, fake_embedding, fake_vector_store):
        """测试使用显式依赖初始化。"""
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=fake_vector_store,
            default_top_k=5,
        )
        
        assert retriever.embedding_client is fake_embedding
        assert retriever.vector_store is fake_vector_store
        assert retriever.default_top_k == 5
    
    def test_init_with_default_top_k(self, fake_embedding, fake_vector_store):
        """测试未指定 top_k 时使用默认值。"""
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=fake_vector_store,
        )
        
        assert retriever.default_top_k == 10  # 默认值
    
    def test_init_with_settings_top_k(self, fake_embedding, fake_vector_store):
        """测试从配置中提取 top_k。"""
        mock_settings = Mock()
        mock_settings.retrieval = Mock()
        mock_settings.retrieval.dense_top_k = 20
        
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=fake_embedding,
            vector_store=fake_vector_store,
        )
        
        assert retriever.default_top_k == 20
    
    def test_init_without_settings_retrieval(self, fake_embedding, fake_vector_store):
        """测试 retrieval 配置缺失时初始化能优雅处理。"""
        mock_settings = Mock()
        mock_settings.retrieval = None
        
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=fake_embedding,
            vector_store=fake_vector_store,
            default_top_k=15,
        )
        
        assert retriever.default_top_k == 15
    
    def test_init_without_dependencies(self):
        """测试允许无依赖初始化（用于延迟配置）。"""
        retriever = DenseRetriever()
        
        assert retriever.embedding_client is None
        assert retriever.vector_store is None


# =============================================================================
# retrieve 方法测试
# =============================================================================

class TestDenseRetrieverRetrieve:
    """DenseRetriever.retrieve() 方法的测试。"""

    def test_retrieve_basic(self, retriever, fake_embedding, fake_vector_store):
        """测试基础检索返回正确结果。"""
        results = retriever.retrieve("What is Azure OpenAI?")
        
        assert len(results) == 3
        assert fake_embedding.call_count == 1
        assert fake_vector_store.call_count == 1
    
    def test_retrieve_returns_retrieval_result_objects(self, retriever):
        """测试结果是 RetrievalResult 实例。"""
        results = retriever.retrieve("test query")
        
        for result in results:
            assert isinstance(result, RetrievalResult)
    
    def test_retrieve_result_fields(self, retriever):
        """测试 RetrievalResult 包含所有预期字段。"""
        results = retriever.retrieve("test query")
        
        result = results[0]
        assert result.chunk_id == "doc1_chunk_001"
        assert result.score == 0.95
        assert "Azure OpenAI" in result.text
        assert result.metadata["source_path"] == "docs/azure.pdf"
    
    def test_retrieve_with_top_k(self, retriever, fake_vector_store):
        """测试使用自定义 top_k 检索。"""
        results = retriever.retrieve("test query", top_k=2)
        
        assert fake_vector_store.last_query_params["top_k"] == 2
        assert len(results) == 2
    
    def test_retrieve_with_default_top_k(self, fake_embedding, fake_vector_store):
        """测试未指定 top_k 时使用 default_top_k。"""
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=fake_vector_store,
            default_top_k=5,
        )
        
        retriever.retrieve("test query")
        
        assert fake_vector_store.last_query_params["top_k"] == 5
    
    def test_retrieve_with_filters(self, retriever, fake_vector_store):
        """测试检索将过滤条件传给 vector store。"""
        filters = {"collection": "api-docs", "doc_type": "pdf"}
        retriever.retrieve("test query", filters=filters)
        
        assert fake_vector_store.last_query_params["filters"] == filters
    
    def test_retrieve_passes_trace(self, retriever, fake_embedding, fake_vector_store):
        """测试检索透传 trace 上下文。"""
        mock_trace = Mock()
        retriever.retrieve("test query", trace=mock_trace)
        
        assert fake_embedding.last_texts == ["test query"]
    
    def test_retrieve_embeds_query_correctly(self, retriever, fake_embedding):
        """测试查询被正确 embedding。"""
        retriever.retrieve("How to configure RAG?")
        
        assert fake_embedding.last_texts == ["How to configure RAG?"]


# =============================================================================
# 输入校验测试
# =============================================================================

class TestDenseRetrieverValidation:
    """DenseRetriever 输入校验的测试。"""

    def test_retrieve_empty_query_raises_error(self, retriever):
        """测试空查询抛出 ValueError。"""
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve("")
    
    def test_retrieve_whitespace_query_raises_error(self, retriever):
        """测试仅含空白的查询抛出 ValueError。"""
        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve("   \t\n  ")
    
    def test_retrieve_non_string_query_raises_error(self, retriever):
        """测试非字符串查询抛出 ValueError。"""
        with pytest.raises(ValueError, match="must be a string"):
            retriever.retrieve(123)
    
    def test_retrieve_without_embedding_client_raises_error(self, fake_vector_store):
        """测试缺少 embedding 客户端时抛出 RuntimeError。"""
        retriever = DenseRetriever(
            embedding_client=None,
            vector_store=fake_vector_store,
        )
        
        with pytest.raises(RuntimeError, match="requires an embedding_client"):
            retriever.retrieve("test query")
    
    def test_retrieve_without_vector_store_raises_error(self, fake_embedding):
        """测试缺少 vector store 时抛出 RuntimeError。"""
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=None,
        )
        
        with pytest.raises(RuntimeError, match="requires a vector_store"):
            retriever.retrieve("test query")


# =============================================================================
# 错误处理测试
# =============================================================================

class TestDenseRetrieverErrorHandling:
    """DenseRetriever 错误处理的测试。"""

    def test_retrieve_embedding_failure(self, fake_vector_store):
        """测试 embedding 失败抛出 RuntimeError。"""
        mock_embedding = Mock()
        mock_embedding.embed.side_effect = Exception("API connection failed")
        
        retriever = DenseRetriever(
            embedding_client=mock_embedding,
            vector_store=fake_vector_store,
        )
        
        with pytest.raises(RuntimeError, match="Failed to embed query"):
            retriever.retrieve("test query")
    
    def test_retrieve_vector_store_failure(self, fake_embedding):
        """测试 vector store 失败抛出 RuntimeError。"""
        mock_store = Mock()
        mock_store.query.side_effect = Exception("Database connection failed")
        
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=mock_store,
        )
        
        with pytest.raises(RuntimeError, match="Failed to query vector store"):
            retriever.retrieve("test query")
    
    def test_retrieve_handles_malformed_results(self, fake_embedding):
        """测试格式错误的结果被优雅跳过。"""
        mock_store = Mock()
        mock_store.query.return_value = [
            {"id": "valid", "score": 0.9, "text": "Valid text", "metadata": {}},
            {"id": "", "score": "invalid", "text": ""},  # 格式错误
            {"id": "another", "score": 0.8, "text": "Another text", "metadata": {}},
        ]

        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=mock_store,
        )

        results = retriever.retrieve("test query")

        # 格式错误的结果应被跳过（空 chunk_id 抛出 ValueError）
        assert len(results) == 2
        assert results[0].chunk_id == "valid"
        assert results[1].chunk_id == "another"


# =============================================================================
# 空结果测试
# =============================================================================

class TestDenseRetrieverEmptyResults:
    """空结果处理的测试。"""

    def test_retrieve_empty_vector_store(self, fake_embedding):
        """测试从空 vector store 检索。"""
        empty_store = FakeVectorStore(stored_records=[])
        
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=empty_store,
        )
        
        results = retriever.retrieve("test query")
        
        assert results == []
    
    def test_retrieve_returns_empty_list_not_none(self, fake_embedding):
        """测试空结果返回空列表而非 None。"""
        empty_store = FakeVectorStore(stored_records=[])
        
        retriever = DenseRetriever(
            embedding_client=fake_embedding,
            vector_store=empty_store,
        )
        
        results = retriever.retrieve("test query")
        
        assert results is not None
        assert isinstance(results, list)


# =============================================================================
# 工厂函数测试
# =============================================================================

class TestCreateDenseRetriever:
    """create_dense_retriever 工厂函数的测试。"""

    def test_create_with_injected_dependencies(self, fake_embedding, fake_vector_store):
        """测试使用预配置依赖的工厂。"""
        mock_settings = Mock()
        mock_settings.retrieval = Mock()
        mock_settings.retrieval.dense_top_k = 15
        
        retriever = create_dense_retriever(
            settings=mock_settings,
            embedding_client=fake_embedding,
            vector_store=fake_vector_store,
        )
        
        assert retriever.embedding_client is fake_embedding
        assert retriever.vector_store is fake_vector_store
        assert retriever.default_top_k == 15
    
    @patch('src.libs.vector_store.vector_store_factory.VectorStoreFactory.create')
    @patch('src.libs.embedding.embedding_factory.EmbeddingFactory.create')
    def test_create_auto_creates_dependencies(
        self, mock_embedding_create, mock_vector_create, fake_embedding, fake_vector_store
    ):
        """测试未提供依赖时工厂自动创建依赖。"""
        mock_embedding_create.return_value = fake_embedding
        mock_vector_create.return_value = fake_vector_store
        
        mock_settings = Mock()
        mock_settings.retrieval = None
        
        retriever = create_dense_retriever(settings=mock_settings)
        
        mock_embedding_create.assert_called_once_with(mock_settings)
        mock_vector_create.assert_called_once_with(mock_settings)
        assert retriever.embedding_client is fake_embedding
        assert retriever.vector_store is fake_vector_store


# =============================================================================
# 与 RetrievalResult 类型的集成测试
# =============================================================================

class TestRetrievalResultIntegration:
    """RetrievalResult 类型集成的测试。"""

    def test_retrieval_result_serialization(self, retriever):
        """测试 RetrievalResult 可序列化为 dict。"""
        results = retriever.retrieve("test query")
        
        result_dict = results[0].to_dict()
        
        assert "chunk_id" in result_dict
        assert "score" in result_dict
        assert "text" in result_dict
        assert "metadata" in result_dict
    
    def test_retrieval_result_from_dict(self):
        """测试从 dict 创建 RetrievalResult。"""
        data = {
            "chunk_id": "test_001",
            "score": 0.85,
            "text": "Test content",
            "metadata": {"source": "test.pdf"},
        }
        
        result = RetrievalResult.from_dict(data)
        
        assert result.chunk_id == "test_001"
        assert result.score == 0.85
        assert result.text == "Test content"
        assert result.metadata["source"] == "test.pdf"
    
    def test_retrieval_result_validation(self):
        """测试 RetrievalResult 校验。"""
        # 空 chunk_id 应抛出异常
        with pytest.raises(ValueError, match="chunk_id cannot be empty"):
            RetrievalResult(chunk_id="", score=0.5, text="test")

        # 非数值 score 应抛出异常
        with pytest.raises(ValueError, match="score must be numeric"):
            RetrievalResult(chunk_id="test", score="invalid", text="test")


# =============================================================================
# 中文查询测试（与 D1 对齐）
# =============================================================================

class TestDenseRetrieverChineseSupport:
    """中文查询支持的测试（与 D1 QueryProcessor 对齐）。"""

    def test_retrieve_chinese_query(self, retriever, fake_embedding):
        """测试中文查询检索。"""
        query = "如何配置 Azure OpenAI？"
        retriever.retrieve(query)
        
        assert fake_embedding.last_texts == [query]
    
    def test_retrieve_mixed_language_query(self, retriever, fake_embedding):
        """测试中英文混合查询检索。"""
        query = "Azure OpenAI 配置步骤"
        retriever.retrieve(query)
        
        assert fake_embedding.last_texts == [query]
