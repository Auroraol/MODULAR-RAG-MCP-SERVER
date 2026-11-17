"""BM25Indexer 的单元测试。

测试涵盖：
- 基于词项统计构建索引
- IDF 计算准确性
- 使用 BM25 打分的查询功能
- 索引持久化（保存/加载往返）
- 重建与增量更新场景
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from src.ingestion.storage.bm25_indexer import BM25Indexer


class TestBM25IndexerBasics:
    """测试 BM25Indexer 的基础功能。"""

    def test_indexer_initialization_default(self):
        """测试默认初始化。"""
        indexer = BM25Indexer()
        
        assert indexer.k1 == 1.5
        assert indexer.b == 0.75
        assert os.path.normpath(str(indexer.index_dir)) == os.path.normpath("data/db/bm25")
    
    def test_indexer_initialization_custom(self):
        """测试使用自定义参数初始化。"""
        indexer = BM25Indexer(
            index_dir="custom/path",
            k1=2.0,
            b=0.5
        )
        
        assert indexer.k1 == 2.0
        assert indexer.b == 0.5
        assert os.path.normpath(str(indexer.index_dir)) == os.path.normpath("custom/path")
    
    def test_indexer_initialization_invalid_k1(self):
        """测试无效 k1 抛出 ValueError。"""
        with pytest.raises(ValueError, match="k1 must be > 0"):
            BM25Indexer(k1=0)
        
        with pytest.raises(ValueError, match="k1 must be > 0"):
            BM25Indexer(k1=-1.0)
    
    def test_indexer_initialization_invalid_b(self):
        """测试无效 b 抛出 ValueError。"""
        with pytest.raises(ValueError, match="b must be in"):
            BM25Indexer(b=-0.1)
        
        with pytest.raises(ValueError, match="b must be in"):
            BM25Indexer(b=1.5)


class TestBM25IndexBuilding:
    """测试索引构建功能。"""

    def test_build_simple_index(self, tmp_path):
        """测试根据词项统计构建简单索引。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {
                "chunk_id": "doc1",
                "term_frequencies": {"hello": 2, "world": 1},
                "doc_length": 3
            },
            {
                "chunk_id": "doc2",
                "term_frequencies": {"hello": 1, "python": 1},
                "doc_length": 2
            }
        ]
        
        indexer.build(term_stats, collection="test")
        
        # 检查元数据
        assert indexer._metadata["num_docs"] == 2
        assert indexer._metadata["avg_doc_length"] == 2.5
        assert indexer._metadata["total_terms"] == 3  # hello、world、python

        # 检查索引结构
        assert "hello" in indexer._index
        assert "world" in indexer._index
        assert "python" in indexer._index

        # 检查 hello 的倒排列表（在文档中出现于两篇）
        hello_data = indexer._index["hello"]
        assert hello_data["df"] == 2
        assert len(hello_data["postings"]) == 2
    
    def test_build_empty_term_stats_raises(self, tmp_path):
        """测试使用空 term_stats 构建时抛出 ValueError。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        with pytest.raises(ValueError, match="Cannot build index from empty"):
            indexer.build([])
    
    def test_build_invalid_structure_raises(self, tmp_path):
        """测试无效的 term_stats 结构抛出 ValueError。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))

        # 缺少 chunk_id
        with pytest.raises(ValueError, match="missing required field: chunk_id"):
            indexer.build([{"term_frequencies": {}, "doc_length": 0}])

        # 缺少 term_frequencies
        with pytest.raises(ValueError, match="missing required field: term_frequencies"):
            indexer.build([{"chunk_id": "1", "doc_length": 0}])

        # 缺少 doc_length
        with pytest.raises(ValueError, match="missing required field: doc_length"):
            indexer.build([{"chunk_id": "1", "term_frequencies": {}}])
    
    def test_build_invalid_types_raises(self, tmp_path):
        """测试无效的字段类型抛出 ValueError。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))

        # term_frequencies 不是 dict
        with pytest.raises(ValueError, match="term_frequencies.*must be dict"):
            indexer.build([
                {"chunk_id": "1", "term_frequencies": "invalid", "doc_length": 0}
            ])

        # doc_length 不是 int
        with pytest.raises(ValueError, match="doc_length.*must be non-negative int"):
            indexer.build([
                {"chunk_id": "1", "term_frequencies": {}, "doc_length": "invalid"}
            ])

        # doc_length 为负数
        with pytest.raises(ValueError, match="doc_length.*must be non-negative int"):
            indexer.build([
                {"chunk_id": "1", "term_frequencies": {}, "doc_length": -1}
            ])


class TestIDFCalculation:
    """测试 IDF 计算准确性。"""

    def test_idf_calculation_formula(self, tmp_path):
        """测试使用 BM25 公式正确计算 IDF。

        公式：IDF(term) = log((N - df + 0.5) / (df + 0.5))
        """
        indexer = BM25Indexer(index_dir=str(tmp_path))

        # N=3 篇文档，df=1（词项出现于 1 篇文档）
        # 期望 IDF = log((3 - 1 + 0.5) / (1 + 0.5)) = log(2.5 / 1.5) = log(1.6667)
        import math
        expected_idf = math.log((3 - 1 + 0.5) / (1 + 0.5))
        
        actual_idf = indexer._calculate_idf(num_docs=3, df=1)
        
        assert abs(actual_idf - expected_idf) < 0.0001
    
    def test_idf_rare_vs_common_terms(self, tmp_path):
        """测试稀有词项 IDF 高于常见词项。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "1", "term_frequencies": {"rare": 1, "common": 1}, "doc_length": 2},
            {"chunk_id": "2", "term_frequencies": {"common": 1}, "doc_length": 1},
            {"chunk_id": "3", "term_frequencies": {"common": 1}, "doc_length": 1},
        ]
        
        indexer.build(term_stats, collection="test")
        
        # "rare" 出现于 1/3 篇文档，"common" 出现于 3/3 篇文档
        rare_idf = indexer._index["rare"]["idf"]
        common_idf = indexer._index["common"]["idf"]

        # 稀有词项的 IDF 应更高
        assert rare_idf > common_idf


class TestBM25Querying:
    """测试 BM25 查询功能。"""

    def test_query_returns_sorted_results(self, tmp_path):
        """测试 query 返回按 BM25 分数排序的结果。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))

        # 使用更清晰的场景：一篇文档包含查询词，其他文档不包含
        term_stats = [
            {"chunk_id": "relevant", "term_frequencies": {"machine": 3, "learning": 2}, "doc_length": 5},
            {"chunk_id": "partial", "term_frequencies": {"machine": 1}, "doc_length": 3},
            {"chunk_id": "irrelevant", "term_frequencies": {"python": 2}, "doc_length": 2},
        ]
        
        indexer.build(term_stats, collection="test")
        
        results = indexer.query(["machine", "learning"], top_k=3)
        
        # 应返回 2 个结果（relevant 和 partial 包含查询词）
        assert len(results) == 2

        # 结果应按分数降序排列
        assert results[0]["score"] >= results[1]["score"]

        # "relevant" 包含两个词项，应排第一
        assert results[0]["chunk_id"] == "relevant"
        # "partial" 仅包含一个词项，应排第二
        assert results[1]["chunk_id"] == "partial"
    
    def test_query_respects_top_k(self, tmp_path):
        """测试 query 遵守 top_k 参数。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": f"doc{i}", "term_frequencies": {"test": 1}, "doc_length": 1}
            for i in range(10)
        ]
        
        indexer.build(term_stats, collection="test")
        
        results = indexer.query(["test"], top_k=3)
        
        assert len(results) == 3
    
    def test_query_term_not_in_corpus(self, tmp_path):
        """测试查询语料库中不存在的词项返回空结果。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "doc1", "term_frequencies": {"hello": 1}, "doc_length": 1}
        ]
        
        indexer.build(term_stats, collection="test")
        
        results = indexer.query(["nonexistent"], top_k=10)
        
        assert len(results) == 0
    
    def test_query_before_build_raises(self):
        """测试构建前查询抛出 ValueError。"""
        indexer = BM25Indexer()
        
        with pytest.raises(ValueError, match="Index not loaded"):
            indexer.query(["test"])
    
    def test_query_empty_terms_raises(self, tmp_path):
        """测试使用空词项查询抛出 ValueError。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "doc1", "term_frequencies": {"test": 1}, "doc_length": 1}
        ]
        indexer.build(term_stats, collection="test")
        
        with pytest.raises(ValueError, match="query_terms cannot be empty"):
            indexer.query([])


class TestIndexPersistence:
    """测试索引保存/加载功能。"""

    def test_save_and_load_roundtrip(self, tmp_path):
        """测试保存的索引可加载并产生相同结果。"""
        indexer1 = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "doc1", "term_frequencies": {"hello": 2, "world": 1}, "doc_length": 3},
            {"chunk_id": "doc2", "term_frequencies": {"hello": 1, "python": 1}, "doc_length": 2}
        ]
        
        indexer1.build(term_stats, collection="test")
        
        # 使用第一个索引器查询
        results1 = indexer1.query(["hello"], top_k=2)

        # 创建新索引器并加载
        indexer2 = BM25Indexer(index_dir=str(tmp_path))
        loaded = indexer2.load(collection="test")

        assert loaded is True

        # 使用已加载的索引器查询
        results2 = indexer2.query(["hello"], top_k=2)

        # 结果应完全相同
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1["chunk_id"] == r2["chunk_id"]
            assert abs(r1["score"] - r2["score"]) < 0.0001
    
    def test_load_nonexistent_index(self, tmp_path):
        """测试加载不存在的索引返回 False。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        loaded = indexer.load(collection="nonexistent")
        
        assert loaded is False
    
    def test_load_corrupted_index_raises(self, tmp_path):
        """测试加载损坏的索引抛出 ValueError。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))

        # 创建损坏的索引文件
        index_path = tmp_path / "corrupted_bm25.json"
        index_path.write_text("not valid json{")
        
        with pytest.raises(ValueError, match="Corrupted index file"):
            indexer.load(collection="corrupted")
    
    def test_load_invalid_structure_raises(self, tmp_path):
        """测试加载结构无效的索引抛出 ValueError。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))

        # 创建缺少字段的索引文件
        index_path = tmp_path / "invalid_bm25.json"
        index_path.write_text(json.dumps({"metadata": {}}))  # 缺少 "index"
        
        with pytest.raises(ValueError, match="Invalid index file structure"):
            indexer.load(collection="invalid")
    
    def test_index_file_created_in_correct_location(self, tmp_path):
        """测试索引文件被创建在正确的目录中。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "doc1", "term_frequencies": {"test": 1}, "doc_length": 1}
        ]
        
        indexer.build(term_stats, collection="my_collection")
        
        expected_path = tmp_path / "my_collection_bm25.json"
        assert expected_path.exists()
        
        # 验证其为有效的 JSON
        with open(expected_path) as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert "index" in data


class TestRebuildFunctionality:
    """测试重建与更新场景。"""

    def test_rebuild_replaces_old_index(self, tmp_path):
        """测试重建会完全替换旧索引。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))

        # 构建初始索引
        term_stats1 = [
            {"chunk_id": "doc1", "term_frequencies": {"old": 1}, "doc_length": 1}
        ]
        indexer.build(term_stats1, collection="test")
        
        # 使用新数据重建
        term_stats2 = [
            {"chunk_id": "doc2", "term_frequencies": {"new": 1}, "doc_length": 1}
        ]
        indexer.rebuild(term_stats2, collection="test")

        # 旧词项应不存在
        results_old = indexer.query(["old"], top_k=10)
        assert len(results_old) == 0

        # 新词项应存在
        results_new = indexer.query(["new"], top_k=10)
        assert len(results_new) == 1
        assert results_new[0]["chunk_id"] == "doc2"
    
    def test_rebuild_is_deterministic(self, tmp_path):
        """测试重建相同数据产生相同索引。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "doc1", "term_frequencies": {"test": 2}, "doc_length": 2},
            {"chunk_id": "doc2", "term_frequencies": {"test": 1}, "doc_length": 1}
        ]
        
        # 构建两次
        indexer.build(term_stats, collection="test1")
        results1 = indexer.query(["test"], top_k=2)

        indexer.rebuild(term_stats, collection="test2")
        results2 = indexer.query(["test"], top_k=2)

        # 结果应完全相同
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1["chunk_id"] == r2["chunk_id"]
            assert abs(r1["score"] - r2["score"]) < 0.0001


class TestEdgeCases:
    """测试边界情况与临界条件。"""

    def test_single_document_corpus(self, tmp_path):
        """测试单文档语料库的索引与查询。

        注意: 在单文档语料库中，词项出现于所有文档时，
        按 BM25 公式 IDF 可以为负: log((N - df + 0.5) / (df + 0.5))
        当 N=1, df=1 时: log((1-1+0.5)/(1+0.5)) = log(0.5/1.5) = log(0.33) < 0
        这是 BM25 的预期行为。
        """
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "only_doc", "term_frequencies": {"hello": 1}, "doc_length": 1}
        ]
        
        indexer.build(term_stats, collection="test")
        results = indexer.query(["hello"], top_k=1)
        
        assert len(results) == 1
        assert results[0]["chunk_id"] == "only_doc"
        # 单文档语料库中分数可以为负（BM25 的预期行为）
        assert isinstance(results[0]["score"], float)
    
    def test_empty_document(self, tmp_path):
        """测试处理零词项文档（边界情况）。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {"chunk_id": "empty", "term_frequencies": {}, "doc_length": 0},
            {"chunk_id": "normal", "term_frequencies": {"test": 1}, "doc_length": 1}
        ]
        
        # 不应抛出异常
        indexer.build(term_stats, collection="test")

        # 查询应仍可正常进行
        results = indexer.query(["test"], top_k=10)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "normal"
    
    def test_very_long_posting_list(self, tmp_path):
        """测试处理出现于大量文档中的词项。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))

        # 创建 100 篇均包含 "common" 的文档
        term_stats = [
            {"chunk_id": f"doc{i}", "term_frequencies": {"common": 1}, "doc_length": 1}
            for i in range(100)
        ]
        
        indexer.build(term_stats, collection="test")
        
        # 应能处理大型倒排列表
        assert indexer._index["common"]["df"] == 100
        assert len(indexer._index["common"]["postings"]) == 100

        # 查询应正常工作
        results = indexer.query(["common"], top_k=10)
        assert len(results) == 10  # 遵守 top_k
    
    def test_special_characters_in_terms(self, tmp_path):
        """测试处理含特殊字符的词项。"""
        indexer = BM25Indexer(index_dir=str(tmp_path))
        
        term_stats = [
            {
                "chunk_id": "doc1",
                "term_frequencies": {"hello-world": 1, "test_case": 1},
                "doc_length": 2
            }
        ]
        
        indexer.build(term_stats, collection="test")
        
        # 应能使用特殊字符进行查询
        results = indexer.query(["hello-world"], top_k=1)
        assert len(results) == 1
        
        results = indexer.query(["test_case"], top_k=1)
        assert len(results) == 1
