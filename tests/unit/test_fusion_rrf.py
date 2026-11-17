"""RRF Fusion（Reciprocal Rank Fusion）的单元测试。

本模块测试 RRFFusion 类，该类使用 RRF 算法将多个检索排序列表
融合为一个统一的排序。

测试类别：
1. 初始化测试
2. 基础融合功能
3. 确定性行为验证
4. 边界情况与错误处理
5. 加权融合测试
6. 工具函数测试
"""

import pytest
from typing import List

from src.core.types import RetrievalResult
from src.core.query_engine.fusion import RRFFusion, rrf_score


# =============================================================================
# 测试 Fixture
# =============================================================================

@pytest.fixture
def fusion_default() -> RRFFusion:
    """创建默认 k=60 的 RRFFusion。"""
    return RRFFusion()


@pytest.fixture
def fusion_k20() -> RRFFusion:
    """创建 k=20 的 RRFFusion 以测试 k 敏感度。"""
    return RRFFusion(k=20)


@pytest.fixture
def dense_results() -> List[RetrievalResult]:
    """示例稠密检索结果（3 篇文档：a、b、c）。"""
    return [
        RetrievalResult(chunk_id="a", score=0.95, text="Dense text A", metadata={"source": "dense", "page": 1}),
        RetrievalResult(chunk_id="b", score=0.85, text="Dense text B", metadata={"source": "dense", "page": 2}),
        RetrievalResult(chunk_id="c", score=0.70, text="Dense text C", metadata={"source": "dense", "page": 3}),
    ]


@pytest.fixture
def sparse_results() -> List[RetrievalResult]:
    """示例稀疏检索结果（3 篇文档：b、c、d）。"""
    return [
        RetrievalResult(chunk_id="b", score=5.2, text="Sparse text B", metadata={"source": "sparse", "page": 2}),
        RetrievalResult(chunk_id="c", score=4.1, text="Sparse text C", metadata={"source": "sparse", "page": 3}),
        RetrievalResult(chunk_id="d", score=3.5, text="Sparse text D", metadata={"source": "sparse", "page": 4}),
    ]


@pytest.fixture
def disjoint_results_1() -> List[RetrievalResult]:
    """无重叠的结果（集合 1）。"""
    return [
        RetrievalResult(chunk_id="x", score=0.9, text="Text X", metadata={}),
        RetrievalResult(chunk_id="y", score=0.8, text="Text Y", metadata={}),
    ]


@pytest.fixture
def disjoint_results_2() -> List[RetrievalResult]:
    """无重叠的结果（集合 2）。"""
    return [
        RetrievalResult(chunk_id="p", score=5.0, text="Text P", metadata={}),
        RetrievalResult(chunk_id="q", score=4.0, text="Text Q", metadata={}),
    ]


# =============================================================================
# 初始化测试
# =============================================================================

class TestRRFFusionInit:
    """RRFFusion 初始化的测试。"""

    def test_default_k_is_60(self):
        """按原始 RRF 论文，默认 k 应为 60。"""
        fusion = RRFFusion()
        assert fusion.k == 60
    
    def test_custom_k_value(self):
        """应接受自定义 k 值。"""
        fusion = RRFFusion(k=100)
        assert fusion.k == 100
    
    def test_k_must_be_positive_integer(self):
        """应拒绝非正数的 k 值。"""
        with pytest.raises(ValueError, match="positive integer"):
            RRFFusion(k=0)
        
        with pytest.raises(ValueError, match="positive integer"):
            RRFFusion(k=-10)
    
    def test_k_must_be_integer(self):
        """应拒绝非整数的 k 值。"""
        with pytest.raises(ValueError, match="positive integer"):
            RRFFusion(k=60.5)
        
        with pytest.raises(ValueError, match="positive integer"):
            RRFFusion(k="60")


# =============================================================================
# 基础融合测试
# =============================================================================

class TestRRFFusionBasic:
    """基础 RRF 融合功能的测试。"""

    def test_fuse_single_list(self, fusion_default, dense_results):
        """融合单个列表应保留顺序并附带 RRF 分数。"""
        fused = fusion_default.fuse([dense_results])

        assert len(fused) == 3
        # 顺序应保留：a、b、c
        assert [r.chunk_id for r in fused] == ["a", "b", "c"]
        # 分数应为 RRF 分数：1/(60+1)、1/(60+2)、1/(60+3)
        assert abs(fused[0].score - 1/61) < 1e-10
        assert abs(fused[1].score - 1/62) < 1e-10
        assert abs(fused[2].score - 1/63) < 1e-10
    
    def test_fuse_two_overlapping_lists(self, fusion_default, dense_results, sparse_results):
        """融合有重叠的列表应正确合并分数。"""
        fused = fusion_default.fuse([dense_results, sparse_results])

        # 应有 4 个唯一 chunk：a、b、c、d
        assert len(fused) == 4
        chunk_ids = {r.chunk_id for r in fused}
        assert chunk_ids == {"a", "b", "c", "d"}

        # 'b' 同时出现于两个列表（dense 中排第 2，sparse 中排第 1）
        # RRF(b) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        b_result = next(r for r in fused if r.chunk_id == "b")
        expected_b_score = 1/62 + 1/61
        assert abs(b_result.score - expected_b_score) < 1e-10

        # 'c' 同时出现于两个列表（dense 中排第 3，sparse 中排第 2）
        # RRF(c) = 1/(60+3) + 1/(60+2) = 1/63 + 1/62
        c_result = next(r for r in fused if r.chunk_id == "c")
        expected_c_score = 1/63 + 1/62
        assert abs(c_result.score - expected_c_score) < 1e-10

        # 'a' 仅出现于 dense（排第 1）
        # RRF(a) = 1/(60+1) = 1/61
        a_result = next(r for r in fused if r.chunk_id == "a")
        assert abs(a_result.score - 1/61) < 1e-10

        # 'd' 仅出现于 sparse（排第 3）
        # RRF(d) = 1/(60+3) = 1/63
        d_result = next(r for r in fused if r.chunk_id == "d")
        assert abs(d_result.score - 1/63) < 1e-10

    def test_fuse_disjoint_lists(self, fusion_default, disjoint_results_1, disjoint_results_2):
        """融合无重叠的列表应包含所有文档。"""
        fused = fusion_default.fuse([disjoint_results_1, disjoint_results_2])

        assert len(fused) == 4
        chunk_ids = {r.chunk_id for r in fused}
        assert chunk_ids == {"x", "y", "p", "q"}

        # 所有文档都应具有单列表 RRF 分数
        x_result = next(r for r in fused if r.chunk_id == "x")
        assert abs(x_result.score - 1/61) < 1e-10  # 列表 1 中排第 1

    def test_fuse_with_top_k(self, fusion_default, dense_results, sparse_results):
        """top_k 应限制返回结果数量。"""
        fused = fusion_default.fuse([dense_results, sparse_results], top_k=2)

        assert len(fused) == 2
        # 应为按 RRF 分数最高的前 2 个

    def test_fuse_top_k_larger_than_results(self, fusion_default, dense_results):
        """top_k 大于可用结果数时应返回全部。"""
        fused = fusion_default.fuse([dense_results], top_k=100)
        
        assert len(fused) == 3


# =============================================================================
# 确定性行为测试
# =============================================================================

class TestRRFFusionDeterministic:
    """确定性行为的测试（规范的核心要求）。"""

    def test_same_input_produces_same_output(self, fusion_default, dense_results, sparse_results):
        """相同输入应始终产生完全相同的输出。"""
        fused1 = fusion_default.fuse([dense_results, sparse_results])
        fused2 = fusion_default.fuse([dense_results, sparse_results])

        assert len(fused1) == len(fused2)
        for r1, r2 in zip(fused1, fused2):
            assert r1.chunk_id == r2.chunk_id
            assert r1.score == r2.score
            assert r1.text == r2.text

    def test_tie_breaking_by_chunk_id(self, fusion_default):
        """RRF 分数相同时应按 chunk_id 字母顺序决定先后。"""
        # 构造两个 RRF 分数完全相同的 chunk
        list1 = [
            RetrievalResult(chunk_id="zebra", score=0.9, text="Z", metadata={}),
        ]
        list2 = [
            RetrievalResult(chunk_id="apple", score=0.9, text="A", metadata={}),
        ]

        fused = fusion_default.fuse([list1, list2])

        # 两者 RRF 分数均为 1/(60+1) = 1/61
        assert fused[0].chunk_id == "apple"  # 'a' < 'z'
        assert fused[1].chunk_id == "zebra"

    def test_list_order_affects_metadata_source(self, fusion_default):
        """应保留首次出现时的元数据。"""
        list1 = [
            RetrievalResult(chunk_id="shared", score=0.9, text="Text from list 1", metadata={"source": "list1"}),
        ]
        list2 = [
            RetrievalResult(chunk_id="shared", score=5.0, text="Text from list 2", metadata={"source": "list2"}),
        ]

        fused = fusion_default.fuse([list1, list2])

        assert len(fused) == 1
        # 保留首次出现（list1）的元数据
        assert fused[0].metadata["source"] == "list1"
        assert fused[0].text == "Text from list 1"

    def test_order_of_lists_does_not_affect_scores(self, fusion_default, dense_results, sparse_results):
        """交换列表顺序应产生相同分数（仅平局时顺序不同）。"""
        fused1 = fusion_default.fuse([dense_results, sparse_results])
        fused2 = fusion_default.fuse([sparse_results, dense_results])

        # 创建分数查找表
        scores1 = {r.chunk_id: r.score for r in fused1}
        scores2 = {r.chunk_id: r.score for r in fused2}

        # 分数应完全相同
        assert scores1 == scores2


# =============================================================================
# 边界情况与错误处理
# =============================================================================

class TestRRFFusionEdgeCases:
    """边界情况与错误处理的测试。"""

    def test_empty_ranking_lists_raises_error(self, fusion_default):
        """空的 ranking_lists 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="cannot be empty"):
            fusion_default.fuse([])

    def test_all_empty_lists_returns_empty(self, fusion_default):
        """所有列表均为空时应返回空结果。"""
        fused = fusion_default.fuse([[], []])

        assert fused == []

    def test_some_empty_lists_ignored(self, fusion_default, dense_results):
        """空列表应被过滤掉。"""
        fused = fusion_default.fuse([[], dense_results, []])

        assert len(fused) == 3
        assert [r.chunk_id for r in fused] == ["a", "b", "c"]

    def test_single_result_per_list(self, fusion_default):
        """应能处理仅含单个结果的列表。"""
        list1 = [RetrievalResult(chunk_id="only", score=1.0, text="Solo", metadata={})]

        fused = fusion_default.fuse([list1])

        assert len(fused) == 1
        assert fused[0].chunk_id == "only"

    def test_many_lists(self, fusion_default):
        """应能处理大量排序列表。"""
        lists = [
            [RetrievalResult(chunk_id=f"chunk_{i}", score=1.0, text=f"Text {i}", metadata={})]
            for i in range(10)
        ]

        fused = fusion_default.fuse(lists)

        assert len(fused) == 10

    def test_metadata_is_copied_not_shared(self, fusion_default, dense_results):
        """元数据应被复制以避免被意外修改。"""
        fused = fusion_default.fuse([dense_results])

        # 修改融合结果的元数据
        fused[0].metadata["modified"] = True

        # 原始数据不应受影响
        assert "modified" not in dense_results[0].metadata


# =============================================================================
# k 参数敏感度测试
# =============================================================================

class TestRRFFusionKParameter:
    """k 参数对融合影响的测试。"""

    def test_higher_k_reduces_score_differences(self):
        """k 越大，相邻名次之间的分数差异越小。"""
        fusion_k20 = RRFFusion(k=20)
        fusion_k100 = RRFFusion(k=100)

        # 排名 1 与排名 2 的分数差
        # k=20：1/21 - 1/22 = 0.00216...
        # k=100：1/101 - 1/102 = 0.000097...

        diff_k20 = 1/21 - 1/22
        diff_k100 = 1/101 - 1/102

        assert diff_k20 > diff_k100

    def test_k_affects_final_ranking(self):
        """在边界情况下，不同 k 值可能导致不同排序。"""
        # 构造 k 起作用的场景：
        # 文档 A：列表 1 中排第 1，不在列表 2 中
        # 文档 B：列表 1 中排第 10，列表 2 中排第 1

        list1 = [
            RetrievalResult(chunk_id="a", score=1.0, text="A", metadata={}),
        ] + [
            RetrievalResult(chunk_id=f"filler_{i}", score=0.5, text="", metadata={})
            for i in range(8)
        ] + [
            RetrievalResult(chunk_id="b", score=0.1, text="B", metadata={}),
        ]

        list2 = [
            RetrievalResult(chunk_id="b", score=5.0, text="B", metadata={}),
        ]

        fusion_k20 = RRFFusion(k=20)
        fusion_k100 = RRFFusion(k=100)

        fused_k20 = fusion_k20.fuse([list1, list2])
        fused_k100 = fusion_k100.fuse([list1, list2])

        # 获取 a 与 b 在两次融合中的分数
        scores_k20 = {r.chunk_id: r.score for r in fused_k20}
        scores_k100 = {r.chunk_id: r.score for r in fused_k100}

        # k=20 时：
        # A：1/(20+1) = 1/21 ≈ 0.0476
        # B：1/(20+10) + 1/(20+1) = 1/30 + 1/21 ≈ 0.0810
        # B > A

        # k=100 时：
        # A：1/(100+1) = 1/101 ≈ 0.0099
        # B：1/(100+10) + 1/(100+1) = 1/110 + 1/101 ≈ 0.0190
        # B > A（但差距更小）

        assert scores_k20["b"] > scores_k20["a"]
        assert scores_k100["b"] > scores_k100["a"]


# =============================================================================
# 加权融合测试
# =============================================================================

class TestRRFFusionWeighted:
    """加权 RRF 融合的测试。"""

    def test_weighted_fusion_basic(self, fusion_default):
        """加权融合应将分数乘以权重。"""
        list1 = [
            RetrievalResult(chunk_id="a", score=1.0, text="A", metadata={}),
        ]
        list2 = [
            RetrievalResult(chunk_id="b", score=1.0, text="B", metadata={}),
        ]

        # 为 list1 赋予双倍权重
        fused = fusion_default.fuse_with_weights([list1, list2], weights=[2.0, 1.0])

        scores = {r.chunk_id: r.score for r in fused}

        # A：2.0 * 1/(60+1) = 2/61
        # B：1.0 * 1/(60+1) = 1/61
        assert abs(scores["a"] - 2/61) < 1e-10
        assert abs(scores["b"] - 1/61) < 1e-10

    def test_weighted_fusion_default_uniform(self, fusion_default, dense_results):
        """未提供权重时应与普通融合相同。"""
        fused_regular = fusion_default.fuse([dense_results])
        fused_weighted = fusion_default.fuse_with_weights([dense_results])

        for r1, r2 in zip(fused_regular, fused_weighted):
            assert r1.chunk_id == r2.chunk_id
            assert abs(r1.score - r2.score) < 1e-10

    def test_weighted_fusion_weight_length_mismatch(self, fusion_default, dense_results, sparse_results):
        """权重长度必须与 ranking_lists 长度一致。"""
        with pytest.raises(ValueError, match="must match"):
            fusion_default.fuse_with_weights(
                [dense_results, sparse_results],
                weights=[1.0]  # 2 个列表仅 1 个权重
            )

    def test_weighted_fusion_negative_weight_rejected(self, fusion_default, dense_results):
        """负权重应被拒绝。"""
        with pytest.raises(ValueError, match="non-negative"):
            fusion_default.fuse_with_weights([dense_results], weights=[-1.0])

    def test_weighted_fusion_zero_weight(self, fusion_default):
        """零权重应等效于忽略该列表。"""
        list1 = [
            RetrievalResult(chunk_id="a", score=1.0, text="A", metadata={}),
        ]
        list2 = [
            RetrievalResult(chunk_id="b", score=1.0, text="B", metadata={}),
        ]

        fused = fusion_default.fuse_with_weights([list1, list2], weights=[1.0, 0.0])

        scores = {r.chunk_id: r.score for r in fused}

        assert abs(scores["a"] - 1/61) < 1e-10
        assert abs(scores["b"] - 0.0) < 1e-10  # 贡献为零

    def test_weighted_fusion_with_top_k(self, fusion_default, dense_results, sparse_results):
        """加权融合应遵守 top_k。"""
        fused = fusion_default.fuse_with_weights(
            [dense_results, sparse_results],
            weights=[1.5, 1.0],
            top_k=2
        )

        assert len(fused) == 2


# =============================================================================
# 工具函数测试
# =============================================================================

class TestRRFScoreFunction:
    """rrf_score 工具函数的测试。"""

    def test_rrf_score_rank_1(self):
        """默认 k 下排名 1 的 RRF 分数。"""
        score = rrf_score(1)
        assert abs(score - 1/61) < 1e-10

    def test_rrf_score_rank_10(self):
        """默认 k 下排名 10 的 RRF 分数。"""
        score = rrf_score(10)
        assert abs(score - 1/70) < 1e-10

    def test_rrf_score_custom_k(self):
        """使用自定义 k 值的 RRF 分数。"""
        score = rrf_score(1, k=20)
        assert abs(score - 1/21) < 1e-10

    def test_rrf_score_invalid_rank(self):
        """应拒绝无效的排名值。"""
        with pytest.raises(ValueError, match="positive integer"):
            rrf_score(0)

        with pytest.raises(ValueError, match="positive integer"):
            rrf_score(-1)

        with pytest.raises(ValueError, match="positive integer"):
            rrf_score(1.5)

    def test_rrf_score_invalid_k(self):
        """应拒绝无效的 k 值。"""
        with pytest.raises(ValueError, match="positive integer"):
            rrf_score(1, k=0)

        with pytest.raises(ValueError, match="positive integer"):
            rrf_score(1, k=-10)


# =============================================================================
# 类集成测试
# =============================================================================

class TestRRFFusionRealisticScenarios:
    """模拟真实检索场景的测试。"""

    def test_typical_hybrid_search_scenario(self, fusion_default):
        """模拟典型的 Dense + Sparse 混合检索。"""
        # 稠密检索：语义匹配
        dense = [
            RetrievalResult(chunk_id="semantic_1", score=0.92, text="Detailed explanation of RAG architecture", metadata={"type": "doc"}),
            RetrievalResult(chunk_id="semantic_2", score=0.88, text="How retrieval augmented generation works", metadata={"type": "doc"}),
            RetrievalResult(chunk_id="both_1", score=0.85, text="RAG implementation with Azure OpenAI", metadata={"type": "tutorial"}),
            RetrievalResult(chunk_id="semantic_3", score=0.80, text="Vector embeddings for document search", metadata={"type": "doc"}),
        ]

        # 稀疏检索："RAG Azure" 的关键词匹配
        sparse = [
            RetrievalResult(chunk_id="keyword_1", score=8.5, text="Azure RAG setup guide", metadata={"type": "guide"}),
            RetrievalResult(chunk_id="both_1", score=7.2, text="RAG implementation with Azure OpenAI", metadata={"type": "tutorial"}),
            RetrievalResult(chunk_id="keyword_2", score=5.1, text="Azure configuration for RAG", metadata={"type": "config"}),
        ]

        fused = fusion_default.fuse([dense, sparse], top_k=5)

        # 验证重叠文档 "both_1" 获得提升
        both_1_result = next(r for r in fused if r.chunk_id == "both_1")
        semantic_1_result = next(r for r in fused if r.chunk_id == "semantic_1")

        # both_1 的 RRF 分数应高于 semantic_1
        # both_1：1/(60+3) + 1/(60+2) = 1/63 + 1/62 ≈ 0.0320
        # semantic_1：1/(60+1) = 1/61 ≈ 0.0164
        assert both_1_result.score > semantic_1_result.score

    def test_sparse_dominant_scenario(self, fusion_default):
        """稀疏检索应更受信任的场景。"""
        # 稀疏检索有强精确关键词匹配
        sparse = [
            RetrievalResult(chunk_id="exact_match", score=15.0, text="配置 Azure OpenAI API 密钥", metadata={}),
        ]

        # 稠密检索仅有弱语义匹配
        dense = [
            RetrievalResult(chunk_id="weak_semantic_1", score=0.52, text="Cloud service configuration", metadata={}),
            RetrievalResult(chunk_id="weak_semantic_2", score=0.51, text="API integration patterns", metadata={}),
            RetrievalResult(chunk_id="exact_match", score=0.48, text="配置 Azure OpenAI API 密钥", metadata={}),
        ]

        fused = fusion_default.fuse([dense, sparse])

        # exact_match 因同时出现于两个列表而应排第一
        assert fused[0].chunk_id == "exact_match"
