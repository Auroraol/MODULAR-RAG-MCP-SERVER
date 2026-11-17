"""SparseEncoder 的单元测试。

测试覆盖：
- 构造函数参数校验
- 基本编码功能
- 分词行为
- 边界情况（空文本、特殊字符）
- 语料库统计计算
- 确定性行为
"""

import pytest
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.core.types import Chunk


# ============================================================================
# 构造函数测试
# ============================================================================

def test_constructor_default():
    """测试默认构造函数。"""
    encoder = SparseEncoder()
    assert encoder.min_term_length == 2
    assert encoder.lowercase is True


def test_constructor_custom_min_term_length():
    """测试自定义 min_term_length。"""
    encoder = SparseEncoder(min_term_length=3)
    assert encoder.min_term_length == 3


def test_constructor_custom_lowercase():
    """测试自定义 lowercase 设置。"""
    encoder = SparseEncoder(lowercase=False)
    assert encoder.lowercase is False


def test_constructor_rejects_zero_min_term_length():
    """测试 min_term_length=0 会被拒绝。"""
    with pytest.raises(ValueError, match="min_term_length must be >= 1"):
        SparseEncoder(min_term_length=0)


def test_constructor_rejects_negative_min_term_length():
    """测试负的 min_term_length 会被拒绝。"""
    with pytest.raises(ValueError, match="min_term_length must be >= 1"):
        SparseEncoder(min_term_length=-1)


# ============================================================================
# 基本编码测试
# ============================================================================

def test_encode_single_chunk():
    """测试编码单个 chunk。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="hello world", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    assert len(results) == 1
    assert results[0]["chunk_id"] == "1"
    assert results[0]["term_frequencies"]["hello"] == 1
    assert results[0]["term_frequencies"]["world"] == 1
    assert results[0]["doc_length"] == 2
    assert results[0]["unique_terms"] == 2


def test_encode_multiple_chunks():
    """测试编码多个 chunk。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="machine learning", metadata={"source_path": "test.txt"}),
        Chunk(id="2", text="deep learning networks", metadata={"source_path": "test.txt"}),
    ]
    
    results = encoder.encode(chunks)
    
    assert len(results) == 2
    assert results[0]["chunk_id"] == "1"
    assert results[1]["chunk_id"] == "2"


def test_encode_with_repeated_terms():
    """测试词频统计是否正确。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="hello world hello hello", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    assert results[0]["term_frequencies"]["hello"] == 3
    assert results[0]["term_frequencies"]["world"] == 1
    assert results[0]["doc_length"] == 4
    assert results[0]["unique_terms"] == 2


# ============================================================================
# 分词测试
# ============================================================================

def test_tokenize_lowercases_by_default():
    """测试默认情况下词项会被转为小写。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="Hello World HELLO", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    assert "hello" in results[0]["term_frequencies"]
    assert "world" in results[0]["term_frequencies"]
    assert results[0]["term_frequencies"]["hello"] == 2


def test_tokenize_preserves_case_when_configured():
    """测试 lowercase=False 时保留大小写。"""
    encoder = SparseEncoder(lowercase=False)
    chunks = [
        Chunk(id="1", text="Hello World", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    assert "Hello" in results[0]["term_frequencies"]
    assert "World" in results[0]["term_frequencies"]
    assert "hello" not in results[0]["term_frequencies"]


def test_tokenize_filters_by_min_term_length():
    """测试短词项会被过滤掉。"""
    encoder = SparseEncoder(min_term_length=3)
    chunks = [
        Chunk(id="1", text="I am learning Python AI", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    # 应过滤掉 "I"、"am" 和 "AI"（长度 < 3）
    assert "learning" in results[0]["term_frequencies"]
    assert "python" in results[0]["term_frequencies"]
    assert results[0]["unique_terms"] == 2  # learning、python


def test_tokenize_handles_punctuation():
    """测试标点符号被正确处理。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="Hello, world! How are you?", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    # 标点符号应被移除
    assert "hello" in results[0]["term_frequencies"]
    assert "world" in results[0]["term_frequencies"]
    assert "how" in results[0]["term_frequencies"]
    assert "," not in results[0]["term_frequencies"]
    assert "!" not in results[0]["term_frequencies"]


def test_tokenize_handles_hyphens_and_underscores():
    """测试连字符和下划线会被保留。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="machine-learning deep_learning", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    assert "machine-learning" in results[0]["term_frequencies"]
    assert "deep_learning" in results[0]["term_frequencies"]


def test_tokenize_handles_numbers():
    """测试数字会被分词。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="Python 3.11 and GPT-4", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    # 数字应作为字母数字词项被保留
    assert "python" in results[0]["term_frequencies"]
    # "3.11" 可能根据分词器不同被拆分为 "3" 和 "11"
    # "gpt-4" 应作为带连字符的词项被保留
    assert "gpt-4" in results[0]["term_frequencies"]


# ============================================================================
# 边界情况
# ============================================================================

def test_encode_rejects_empty_chunks_list():
    """测试空的 chunk 列表会被拒绝。"""
    encoder = SparseEncoder()
    
    with pytest.raises(ValueError, match="Cannot encode empty chunks list"):
        encoder.encode([])


def test_encode_rejects_chunk_with_empty_text():
    """测试文本为空的 chunk 会被拒绝。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="", metadata={"source_path": "test.txt"})
    ]
    
    with pytest.raises(ValueError, match="empty or whitespace-only text"):
        encoder.encode(chunks)


def test_encode_rejects_chunk_with_whitespace_only_text():
    """测试文本仅含空白的 chunk 会被拒绝。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="   \n\t  ", metadata={"source_path": "test.txt"})
    ]
    
    with pytest.raises(ValueError, match="empty or whitespace-only text"):
        encoder.encode(chunks)


def test_encode_handles_special_characters():
    """测试编码含特殊字符的文本。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="C++ and C# programming @2024", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    # 应提取出字母数字词项
    assert "programming" in results[0]["term_frequencies"]
    assert "2024" in results[0]["term_frequencies"]


def test_encode_handles_unicode():
    """测试编码含 Unicode 字符的文本。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="café résumé naïve", metadata={"source_path": "test.txt"})
    ]
    
    results = encoder.encode(chunks)
    
    # Unicode 字符应被正确处理
    assert results[0]["doc_length"] > 0
    assert results[0]["unique_terms"] > 0


# ============================================================================
# 确定性测试
# ============================================================================

def test_encode_is_deterministic():
    """测试编码结果是确定性的。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="machine learning deep learning", metadata={"source_path": "test.txt"})
    ]
    
    results1 = encoder.encode(chunks)
    results2 = encoder.encode(chunks)
    
    assert results1 == results2


def test_encode_preserves_chunk_order():
    """测试输出顺序与输入顺序一致。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="first chunk", metadata={"source_path": "test.txt"}),
        Chunk(id="2", text="second chunk", metadata={"source_path": "test.txt"}),
        Chunk(id="3", text="third chunk", metadata={"source_path": "test.txt"}),
    ]
    
    results = encoder.encode(chunks)
    
    assert len(results) == 3
    assert results[0]["chunk_id"] == "1"
    assert results[1]["chunk_id"] == "2"
    assert results[2]["chunk_id"] == "3"


# ============================================================================
# 语料库统计测试
# ============================================================================

def test_get_corpus_stats_single_document():
    """测试单文档的语料库统计。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="hello world", metadata={"source_path": "test.txt"})
    ]
    
    encoded = encoder.encode(chunks)
    stats = encoder.get_corpus_stats(encoded)
    
    assert stats["num_docs"] == 1
    assert stats["avg_doc_length"] == 2.0
    assert stats["document_frequency"]["hello"] == 1
    assert stats["document_frequency"]["world"] == 1


def test_get_corpus_stats_multiple_documents():
    """测试多文档的语料库统计。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="machine learning", metadata={"source_path": "test.txt"}),
        Chunk(id="2", text="deep learning networks", metadata={"source_path": "test.txt"}),
        Chunk(id="3", text="machine learning algorithms", metadata={"source_path": "test.txt"}),
    ]
    
    encoded = encoder.encode(chunks)
    stats = encoder.get_corpus_stats(encoded)
    
    assert stats["num_docs"] == 3
    assert stats["avg_doc_length"] == (2 + 3 + 3) / 3
    # "learning" 出现在全部 3 个文档中
    assert stats["document_frequency"]["learning"] == 3
    # "machine" 出现在 2 个文档中
    assert stats["document_frequency"]["machine"] == 2
    # "deep" 出现在 1 个文档中
    assert stats["document_frequency"]["deep"] == 1


def test_get_corpus_stats_calculates_average_doc_length():
    """测试平均文档长度计算是否正确。"""
    encoder = SparseEncoder()
    chunks = [
        Chunk(id="1", text="short", metadata={"source_path": "test.txt"}),
        Chunk(id="2", text="this is a longer document", metadata={"source_path": "test.txt"}),
    ]
    
    encoded = encoder.encode(chunks)
    stats = encoder.get_corpus_stats(encoded)
    
    # 第一个文档：1 个词项（"short"），第二个文档：4 个词项（"this"、"is"、"longer"、"document" - "a" 被过滤），平均值 = 2.5
    assert stats["avg_doc_length"] == 2.5


def test_get_corpus_stats_handles_empty_list():
    """测试编码 chunk 列表为空时的语料库统计。"""
    encoder = SparseEncoder()
    
    stats = encoder.get_corpus_stats([])
    
    assert stats["num_docs"] == 0
    assert stats["avg_doc_length"] == 0.0
    assert stats["document_frequency"] == {}


# ============================================================================
# 集成测试
# ============================================================================

def test_realistic_encoding_scenario():
    """测试内容多样的真实编码场景。"""
    encoder = SparseEncoder()
    
    chunks = [
        Chunk(
            id="doc1_chunk0",
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"source_path": "textbook.pdf"}
        ),
        Chunk(
            id="doc1_chunk1",
            text="Deep learning uses neural networks with multiple layers.",
            metadata={"source_path": "textbook.pdf"}
        ),
        Chunk(
            id="doc2_chunk0",
            text="Natural language processing (NLP) enables machines to understand text.",
            metadata={"source_path": "paper.pdf"}
        ),
    ]
    
    results = encoder.encode(chunks)
    
    # 校验结构
    assert len(results) == 3
    for i, result in enumerate(results):
        assert "chunk_id" in result
        assert "term_frequencies" in result
        assert "doc_length" in result
        assert "unique_terms" in result
        assert result["chunk_id"] == chunks[i].id
        assert result["doc_length"] > 0
        assert result["unique_terms"] > 0
        assert len(result["term_frequencies"]) == result["unique_terms"]
    
    # 获取语料库统计
    corpus_stats = encoder.get_corpus_stats(results)
    assert corpus_stats["num_docs"] == 3
    assert corpus_stats["avg_doc_length"] > 0
    assert len(corpus_stats["document_frequency"]) > 0

