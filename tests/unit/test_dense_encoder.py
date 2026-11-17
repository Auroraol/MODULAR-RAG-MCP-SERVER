"""DenseEncoder 的单元测试。

使用 mock 的 BaseEmbedding provider 隔离测试 DenseEncoder 类。
验证批处理、错误处理与输出的正确性。
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List

from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.core.types import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding


class FakeEmbedding(BaseEmbedding):
    """用于测试的假 embedding provider。

    根据文本长度返回确定性向量。
    """

    def __init__(self, dimension: int = 1536, fail_on_call: bool = False):
        self.dimension = dimension
        self.fail_on_call = fail_on_call
        self.call_count = 0
        self.call_history: List[List[str]] = []

    def embed(self, texts: List[str], trace=None, **kwargs) -> List[List[float]]:
        """生成假 embedding。"""
        self.call_count += 1
        self.call_history.append(texts)

        if self.fail_on_call:
            raise RuntimeError("Simulated embedding failure")

        # 校验输入（与真实 provider 行为一致）
        self.validate_texts(texts)

        # 根据文本长度生成确定性向量
        vectors = []
        for text in texts:
            # 使用文本长度生成确定但各异的向量
            base_value = len(text) / 1000.0
            vector = [base_value + (i * 0.001) for i in range(self.dimension)]
            vectors.append(vector)
        
        return vectors


# ============================================================================
# 构造函数测试
# ============================================================================

def test_constructor_valid():
    """测试使用有效参数初始化 DenseEncoder。"""
    embedding = FakeEmbedding()
    encoder = DenseEncoder(embedding, batch_size=32)
    
    assert encoder.embedding is embedding
    assert encoder.batch_size == 32


def test_constructor_default_batch_size():
    """测试未指定时 DenseEncoder 使用默认 batch_size。"""
    embedding = FakeEmbedding()
    encoder = DenseEncoder(embedding)
    
    assert encoder.batch_size == 100


def test_constructor_rejects_zero_batch_size():
    """测试 batch_size=0 被拒绝。"""
    embedding = FakeEmbedding()
    
    with pytest.raises(ValueError, match="batch_size must be positive"):
        DenseEncoder(embedding, batch_size=0)


def test_constructor_rejects_negative_batch_size():
    """测试负的 batch_size 被拒绝。"""
    embedding = FakeEmbedding()
    
    with pytest.raises(ValueError, match="batch_size must be positive"):
        DenseEncoder(embedding, batch_size=-1)


# ============================================================================
# 基础编码测试
# ============================================================================

def test_encode_single_chunk():
    """测试编码单个 chunk。"""
    embedding = FakeEmbedding(dimension=4)
    encoder = DenseEncoder(embedding, batch_size=10)
    
    chunks = [Chunk(id="1", text="Hello world", metadata={"source_path": "test.pdf"})]
    vectors = encoder.encode(chunks)
    
    assert len(vectors) == 1
    assert len(vectors[0]) == 4
    assert embedding.call_count == 1
    assert embedding.call_history[0] == ["Hello world"]


def test_encode_multiple_chunks():
    """测试在单个批次中编码多个 chunk。"""
    embedding = FakeEmbedding(dimension=8)
    encoder = DenseEncoder(embedding, batch_size=10)
    
    chunks = [
        Chunk(id="1", text="First chunk", metadata={"source_path": "test.pdf"}),
        Chunk(id="2", text="Second chunk", metadata={"source_path": "test.pdf"}),
        Chunk(id="3", text="Third chunk", metadata={"source_path": "test.pdf"}),
    ]
    vectors = encoder.encode(chunks)
    
    assert len(vectors) == 3
    assert all(len(v) == 8 for v in vectors)
    assert embedding.call_count == 1  # 全部在一个批次中


def test_encode_preserves_chunk_order():
    """测试输出向量保持输入 chunk 的顺序。"""
    embedding = FakeEmbedding(dimension=4)
    encoder = DenseEncoder(embedding, batch_size=10)
    
    chunks = [
        Chunk(id="1", text="A" * 10, metadata={"source_path": "test.pdf"}),
        Chunk(id="2", text="B" * 20, metadata={"source_path": "test.pdf"}),
        Chunk(id="3", text="C" * 30, metadata={"source_path": "test.pdf"}),
    ]
    vectors = encoder.encode(chunks)
    
    # 向量应按文本长度排序（因为 FakeEmbedding 使用长度）
    assert vectors[0][0] < vectors[1][0] < vectors[2][0]


# ============================================================================
# 批处理测试
# ============================================================================

def test_encode_respects_batch_size():
    """测试编码遵守配置的 batch_size。"""
    embedding = FakeEmbedding(dimension=4)
    encoder = DenseEncoder(embedding, batch_size=2)
    
    chunks = [
        Chunk(id="1", text="Chunk 1", metadata={"source_path": "test.pdf"}),
        Chunk(id="2", text="Chunk 2", metadata={"source_path": "test.pdf"}),
        Chunk(id="3", text="Chunk 3", metadata={"source_path": "test.pdf"}),
        Chunk(id="4", text="Chunk 4", metadata={"source_path": "test.pdf"}),
        Chunk(id="5", text="Chunk 5", metadata={"source_path": "test.pdf"}),
    ]
    vectors = encoder.encode(chunks)
    
    # 应调用 3 次：[0:2]、[2:4]、[4:5]
    assert embedding.call_count == 3
    assert len(embedding.call_history[0]) == 2  # 第一批
    assert len(embedding.call_history[1]) == 2  # 第二批
    assert len(embedding.call_history[2]) == 1  # 最后一批

    # 所有向量均被返回
    assert len(vectors) == 5


def test_encode_exact_batch_boundary():
    """测试 chunk 数量恰为 batch_size 整数倍时的编码。"""
    embedding = FakeEmbedding(dimension=4)
    encoder = DenseEncoder(embedding, batch_size=3)
    
    chunks = [Chunk(id=str(i), text=f"Chunk {i}", metadata={"source_path": "test.pdf"}) for i in range(6)]
    vectors = encoder.encode(chunks)
    
    assert embedding.call_count == 2  # 恰好 2 个批次
    assert len(vectors) == 6


def test_encode_large_batch():
    """测试 batch size 大于 chunk 数量时的编码。"""
    embedding = FakeEmbedding(dimension=4)
    encoder = DenseEncoder(embedding, batch_size=100)
    
    chunks = [Chunk(id=str(i), text=f"Chunk {i}", metadata={"source_path": "test.pdf"}) for i in range(10)]
    vectors = encoder.encode(chunks)
    
    assert embedding.call_count == 1  # 单个批次
    assert len(vectors) == 10


# ============================================================================
# 输入校验测试
# ============================================================================

def test_encode_rejects_empty_chunks_list():
    """测试 encode() 拒绝空的 chunks 列表。"""
    embedding = FakeEmbedding()
    encoder = DenseEncoder(embedding)
    
    with pytest.raises(ValueError, match="Cannot encode empty chunks list"):
        encoder.encode([])


def test_encode_rejects_chunk_with_empty_text():
    """测试文本为空的 chunk 被拒绝。"""
    embedding = FakeEmbedding()
    encoder = DenseEncoder(embedding)
    
    chunks = [
        Chunk(id="1", text="Valid text", metadata={"source_path": "test.pdf"}),
        Chunk(id="2", text="", metadata={"source_path": "test.pdf"}),  # 空文本
    ]
    
    with pytest.raises(ValueError, match="Chunk at index 1.*has empty"):
        encoder.encode(chunks)


def test_encode_rejects_chunk_with_whitespace_only_text():
    """测试仅含空白字符的 chunk 被拒绝。"""
    embedding = FakeEmbedding()
    encoder = DenseEncoder(embedding)
    
    chunks = [
        Chunk(id="1", text="   \n\t  ", metadata={"source_path": "test.pdf"}),  # 仅空白字符
    ]
    
    with pytest.raises(ValueError, match="has empty or whitespace-only text"):
        encoder.encode(chunks)


# ============================================================================
# 错误处理测试
# ============================================================================

def test_encode_handles_embedding_provider_failure():
    """测试 embedding provider 的失败被正确暴露。"""
    embedding = FakeEmbedding(fail_on_call=True)
    encoder = DenseEncoder(embedding, batch_size=10)
    
    chunks = [Chunk(id="1", text="Test", metadata={"source_path": "test.pdf"})]
    
    with pytest.raises(RuntimeError, match="Failed to encode batch.*Simulated embedding failure"):
        encoder.encode(chunks)


def test_encode_failure_includes_batch_range():
    """测试错误消息包含批次范围以便调试。"""
    embedding = FakeEmbedding(fail_on_call=True)
    encoder = DenseEncoder(embedding, batch_size=2)
    
    chunks = [
        Chunk(id="1", text="Chunk 1", metadata={"source_path": "test.pdf"}),
        Chunk(id="2", text="Chunk 2", metadata={"source_path": "test.pdf"}),
        Chunk(id="3", text="Chunk 3", metadata={"source_path": "test.pdf"}),
    ]
    
    with pytest.raises(RuntimeError, match=r"Failed to encode batch 0-2"):
        encoder.encode(chunks)


def test_encode_validates_vector_count():
    """测试检测到向量数量不匹配。"""
    embedding = Mock(spec=BaseEmbedding)
    # 返回错误的向量数量
    embedding.embed.return_value = [[0.1, 0.2]]  # 只有 1 个向量
    
    encoder = DenseEncoder(embedding, batch_size=10)
    chunks = [
        Chunk(id="1", text="Chunk 1", metadata={"source_path": "test.pdf"}),
        Chunk(id="2", text="Chunk 2", metadata={"source_path": "test.pdf"}),
    ]
    
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 texts"):
        encoder.encode(chunks)


def test_encode_validates_vector_dimensions():
    """测试检测到向量维度不一致。"""
    embedding = Mock(spec=BaseEmbedding)
    # 返回维度不一致的向量
    embedding.embed.return_value = [
        [0.1, 0.2, 0.3],  # 3 维
        [0.4, 0.5],       # 2 维（不一致！）
    ]
    
    encoder = DenseEncoder(embedding, batch_size=10)
    chunks = [
        Chunk(id="1", text="Chunk 1", metadata={"source_path": "test.pdf"}),
        Chunk(id="2", text="Chunk 2", metadata={"source_path": "test.pdf"}),
    ]
    
    with pytest.raises(RuntimeError, match="Inconsistent vector dimensions"):
        encoder.encode(chunks)


# ============================================================================
# 工具方法测试
# ============================================================================

def test_get_batch_count_single_batch():
    """测试单个批次的批次数计算。"""
    embedding = FakeEmbedding()
    encoder = DenseEncoder(embedding, batch_size=10)
    
    assert encoder.get_batch_count(5) == 1
    assert encoder.get_batch_count(10) == 1


def test_get_batch_count_multiple_batches():
    """测试多个批次的批次数计算。"""
    embedding = FakeEmbedding()
    encoder = DenseEncoder(embedding, batch_size=10)
    
    assert encoder.get_batch_count(11) == 2
    assert encoder.get_batch_count(20) == 2
    assert encoder.get_batch_count(21) == 3


def test_get_batch_count_zero_chunks():
    """测试零个 chunk 的批次数。"""
    embedding = FakeEmbedding()
    encoder = DenseEncoder(embedding, batch_size=10)
    
    assert encoder.get_batch_count(0) == 0


def test_get_batch_count_with_different_batch_sizes():
    """测试批次数随 batch_size 变化。"""
    embedding = FakeEmbedding()
    
    encoder_small = DenseEncoder(embedding, batch_size=2)
    encoder_large = DenseEncoder(embedding, batch_size=100)
    
    assert encoder_small.get_batch_count(10) == 5
    assert encoder_large.get_batch_count(10) == 1


# ============================================================================
# 类集成测试（仍使用 mock）
# ============================================================================

def test_encode_realistic_scenario():
    """测试包含多个批次的真实编码场景。"""
    embedding = FakeEmbedding(dimension=1536)
    encoder = DenseEncoder(embedding, batch_size=32)

    # 创建 100 个 chunk
    chunks = [
        Chunk(
            id=f"chunk_{i}",
            text=f"This is chunk number {i} with some content",
            metadata={"source_path": "test.pdf", "page": i // 10}
        )
        for i in range(100)
    ]
    
    vectors = encoder.encode(chunks)
    
    # 验证输出
    assert len(vectors) == 100
    assert all(len(v) == 1536 for v in vectors)

    # 验证分批（100 / 32 = 4 个批次）
    assert embedding.call_count == 4

    # 验证所有 chunk 均被处理
    total_processed = sum(len(batch) for batch in embedding.call_history)
    assert total_processed == 100


def test_encode_with_trace_context():
    """测试 trace 上下文被传递给 embedding provider。"""
    embedding = Mock(spec=BaseEmbedding)
    embedding.embed.return_value = [[0.1, 0.2, 0.3]]
    
    encoder = DenseEncoder(embedding, batch_size=10)
    chunks = [Chunk(id="1", text="Test", metadata={"source_path": "test.pdf"})]
    
    mock_trace = {"trace_id": "test_trace"}
    encoder.encode(chunks, trace=mock_trace)
    
    # 验证 trace 被传给 embedding provider
    embedding.embed.assert_called_once()
    call_kwargs = embedding.embed.call_args.kwargs
    assert call_kwargs["trace"] == mock_trace
