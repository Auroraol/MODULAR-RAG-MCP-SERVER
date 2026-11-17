"""基于 Cross-Encoder 的 Reranker 实现的测试。"""

from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from src.core.settings import RerankSettings, Settings
from src.libs.reranker.cross_encoder_reranker import (
    CrossEncoderRerankError,
    CrossEncoderReranker,
)


class MockCrossEncoder:
    """用于确定性测试的 mock Cross-Encoder 模型。"""

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.call_count = 0
        self.last_pairs = None

    def predict(self, pairs: List[tuple[str, str]]) -> List[float]:
        """返回用于测试的确定性分数。

        打分策略：根据段落中关键词的出现情况进行打分。
        """
        self.call_count += 1
        self.last_pairs = pairs

        scores = []
        for query, passage in pairs:
            # 简单打分：统计关键词匹配数量
            score = 0.0
            query_words = query.lower().split()
            passage_lower = passage.lower()
            
            for word in query_words:
                if word in passage_lower:
                    score += 0.3
            
            scores.append(score)
        
        return scores


@pytest.fixture
def mock_settings():
    """创建用于测试的 mock 配置。"""
    settings = Mock(spec=Settings)
    settings.rerank = Mock(spec=RerankSettings)
    settings.rerank.model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    settings.rerank.enabled = True
    settings.rerank.provider = "cross_encoder"
    return settings


@pytest.fixture
def sample_candidates():
    """用于重排序的示例候选列表。"""
    return [
        {"id": "chunk_1", "text": "Python is a programming language.", "score": 0.8},
        {"id": "chunk_2", "text": "Machine learning uses neural networks.", "score": 0.75},
        {"id": "chunk_3", "text": "RAG combines retrieval and generation.", "score": 0.9},
        {"id": "chunk_4", "text": "Embeddings represent text as vectors.", "score": 0.7},
    ]


class TestCrossEncoderRerankerInit:
    """测试 CrossEncoderReranker 初始化。"""

    def test_init_with_mock_model(self, mock_settings):
        """测试使用注入的 mock 模型初始化。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model,
            timeout=5.0
        )
        
        assert reranker.settings == mock_settings
        assert reranker.model == mock_model
        assert reranker.timeout == 5.0
    
    def test_init_missing_model_config(self):
        """测试模型配置缺失时初始化失败。"""
        settings = Mock(spec=Settings)
        settings.rerank = Mock(spec=RerankSettings)
        settings.rerank.model = None
        
        with pytest.raises(CrossEncoderRerankError, match="Failed to initialize"):
            CrossEncoderReranker(settings=settings)
    
    def test_init_invalid_model_type(self):
        """测试模型类型无效时初始化失败。"""
        settings = Mock(spec=Settings)
        settings.rerank = Mock(spec=RerankSettings)
        settings.rerank.model = 123  # 不是字符串
        
        with pytest.raises(CrossEncoderRerankError, match="Failed to initialize"):
            CrossEncoderReranker(settings=settings)
    
    def test_get_model_name_from_settings(self, mock_settings):
        """测试从配置中提取模型名称。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        model_name = reranker._get_model_name_from_settings(mock_settings)
        assert model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    @patch('src.libs.reranker.cross_encoder_reranker.CrossEncoderReranker._load_cross_encoder_model')
    def test_init_loads_model_from_settings(self, mock_load, mock_settings):
        """测试未注入模型时从配置加载模型。"""
        mock_load.return_value = MockCrossEncoder()
        
        reranker = CrossEncoderReranker(settings=mock_settings)
        
        mock_load.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert reranker.model is not None


class TestCrossEncoderRerankerValidation:
    """测试输入校验。"""

    def test_validate_query_success(self, mock_settings):
        """测试有效查询通过校验。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        # 不应抛出异常
        reranker.validate_query("What is RAG?")
    
    def test_validate_query_empty(self, mock_settings):
        """测试空查询校验失败。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        with pytest.raises(ValueError, match="cannot be empty"):
            reranker.validate_query("")
    
    def test_validate_candidates_success(self, mock_settings, sample_candidates):
        """测试有效列表通过候选校验。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        # 不应抛出异常
        reranker.validate_candidates(sample_candidates)
    
    def test_validate_candidates_empty_list(self, mock_settings):
        """测试空列表候选校验失败。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        with pytest.raises(ValueError, match="cannot be empty"):
            reranker.validate_candidates([])


class TestCrossEncoderRerankerPairPreparation:
    """测试查询-段落对的准备。"""

    def test_prepare_pairs_with_text_field(self, mock_settings):
        """测试使用 'text' 字段准备查询-段落对。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        candidates = [
            {"id": "1", "text": "First passage"},
            {"id": "2", "text": "Second passage"},
        ]
        
        pairs = reranker._prepare_pairs("query", candidates)
        
        assert len(pairs) == 2
        assert pairs[0] == ("query", "First passage")
        assert pairs[1] == ("query", "Second passage")
    
    def test_prepare_pairs_with_content_field(self, mock_settings):
        """测试以 'content' 字段作为回退准备查询-段落对。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        candidates = [
            {"id": "1", "content": "First passage"},
            {"id": "2", "content": "Second passage"},
        ]
        
        pairs = reranker._prepare_pairs("query", candidates)
        
        assert len(pairs) == 2
        assert pairs[0] == ("query", "First passage")
        assert pairs[1] == ("query", "Second passage")
    
    def test_prepare_pairs_missing_text(self, mock_settings):
        """测试缺少 text 字段时准备查询-段落对。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        candidates = [
            {"id": "1"},  # 无 text 或 content
        ]

        pairs = reranker._prepare_pairs("query", candidates)

        assert len(pairs) == 1
        assert pairs[0] == ("query", "")  # 空字符串回退


class TestCrossEncoderRerankerScoring:
    """测试打分功能。"""

    def test_score_pairs_success(self, mock_settings):
        """测试成功地对查询-段落对打分。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        pairs = [
            ("machine learning", "Machine learning is a field of AI"),
            ("machine learning", "Python is a programming language"),
        ]
        
        scores = reranker._score_pairs(pairs)
        
        assert len(scores) == 2
        assert isinstance(scores[0], float)
        assert isinstance(scores[1], float)
        # 第一个段落应得分更高（包含两个关键词）
        assert scores[0] > scores[1]
    
    def test_score_pairs_model_called(self, mock_settings):
        """测试以正确的查询-段落对调用 model.predict。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        pairs = [("query", "passage")]
        
        reranker._score_pairs(pairs)
        
        assert mock_model.call_count == 1
        assert mock_model.last_pairs == pairs


class TestCrossEncoderRerankerSorting:
    """测试分数附加与排序。"""

    def test_attach_scores_and_sort(self, mock_settings):
        """测试附加分数并按相关性排序。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        candidates = [
            {"id": "1", "text": "Low"},
            {"id": "2", "text": "High"},
            {"id": "3", "text": "Medium"},
        ]
        scores = [0.1, 0.9, 0.5]
        
        result = reranker._attach_scores_and_sort(candidates, scores, top_k=3)
        
        assert len(result) == 3
        assert result[0]["id"] == "2"
        assert result[0]["rerank_score"] == 0.9
        assert result[1]["id"] == "3"
        assert result[1]["rerank_score"] == 0.5
        assert result[2]["id"] == "1"
        assert result[2]["rerank_score"] == 0.1
    
    def test_attach_scores_top_k_limit(self, mock_settings):
        """测试 top_k 限制输出数量。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        candidates = [
            {"id": "1", "text": "A"},
            {"id": "2", "text": "B"},
            {"id": "3", "text": "C"},
        ]
        scores = [0.3, 0.9, 0.6]
        
        result = reranker._attach_scores_and_sort(candidates, scores, top_k=2)
        
        assert len(result) == 2
        assert result[0]["id"] == "2"
        assert result[1]["id"] == "3"
    
    def test_attach_scores_preserves_original(self, mock_settings):
        """测试原始候选对象不被修改。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        candidates = [
            {"id": "1", "text": "Test", "metadata": {"key": "value"}},
        ]
        scores = [0.5]
        
        result = reranker._attach_scores_and_sort(candidates, scores, top_k=1)
        
        # 原始对象不应包含 rerank_score
        assert "rerank_score" not in candidates[0]
        # 结果应包含 rerank_score
        assert "rerank_score" in result[0]
        # 其他字段被保留
        assert result[0]["metadata"] == {"key": "value"}


class TestCrossEncoderRerankerEndToEnd:
    """测试端到端重排序。"""

    def test_rerank_success(self, mock_settings, sample_candidates):
        """测试成功的端到端重排序。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        query = "machine learning neural networks"
        result = reranker.rerank(query, sample_candidates)
        
        # 应返回重排序后的候选
        assert len(result) == 4
        assert all("rerank_score" in c for c in result)

        # 候选 2 应排名最高（包含两个关键词）
        assert result[0]["id"] == "chunk_2"
        assert result[0]["text"] == "Machine learning uses neural networks."
    
    def test_rerank_with_top_k(self, mock_settings, sample_candidates):
        """测试带 top_k 参数的重排序。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        query = "machine learning"
        result = reranker.rerank(query, sample_candidates, top_k=2)
        
        assert len(result) == 2
        # 应返回相关性最高的前 2 个
        assert result[0]["id"] == "chunk_2"
    
    def test_rerank_invalid_query(self, mock_settings, sample_candidates):
        """测试使用无效查询重排序。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        with pytest.raises(ValueError, match="Query"):
            reranker.rerank("", sample_candidates)
    
    def test_rerank_invalid_candidates(self, mock_settings):
        """测试使用无效候选重排序。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        with pytest.raises(ValueError, match="Candidates"):
            reranker.rerank("query", [])
    
    def test_rerank_invalid_top_k(self, mock_settings, sample_candidates):
        """测试使用无效 top_k 参数重排序。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        with pytest.raises(ValueError, match="top_k"):
            reranker.rerank("query", sample_candidates, top_k=0)
    
    def test_rerank_single_candidate(self, mock_settings):
        """测试单个候选的重排序。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        candidates = [{"id": "1", "text": "Single passage"}]
        result = reranker.rerank("query", candidates)
        
        assert len(result) == 1
        assert result[0]["id"] == "1"
        assert "rerank_score" in result[0]


class TestCrossEncoderRerankerIntegration:
    """测试集成场景。"""

    def test_rerank_with_trace_context(self, mock_settings, sample_candidates):
        """测试带 trace 上下文的重排序。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        mock_trace = Mock()
        query = "machine learning"
        
        result = reranker.rerank(query, sample_candidates, trace=mock_trace)
        
        # 应成功完成并传递 trace
        assert len(result) == 4
        assert all("rerank_score" in c for c in result)
    
    def test_rerank_preserves_all_fields(self, mock_settings):
        """测试重排序保留所有原始候选字段。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        candidates = [
            {
                "id": "1",
                "text": "Machine learning text",
                "score": 0.8,
                "metadata": {"source": "doc1", "page": 5},
                "custom_field": "custom_value"
            }
        ]
        
        result = reranker.rerank("machine learning", candidates)
        
        assert result[0]["id"] == "1"
        assert result[0]["score"] == 0.8
        assert result[0]["metadata"] == {"source": "doc1", "page": 5}
        assert result[0]["custom_field"] == "custom_value"
        assert result[0]["rerank_score"] > 0
    
    def test_rerank_deterministic(self, mock_settings):
        """测试使用 mock 时重排序结果是确定性的。"""
        mock_model = MockCrossEncoder()
        reranker = CrossEncoderReranker(
            settings=mock_settings,
            model=mock_model
        )
        
        candidates = [
            {"id": "1", "text": "First"},
            {"id": "2", "text": "Second"},
        ]
        
        result1 = reranker.rerank("query", candidates)
        result2 = reranker.rerank("query", candidates)
        
        # 应产生完全相同的结果
        assert result1[0]["id"] == result2[0]["id"]
        assert result1[0]["rerank_score"] == result2[0]["rerank_score"]
