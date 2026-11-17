"""BatchProcessor 的单元测试。

测试批处理编排、耗时指标与错误处理。
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

from src.core.types import Chunk
from src.ingestion.embedding.batch_processor import BatchProcessor, BatchResult
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.core.trace.trace_context import TraceContext


class FakeDenseEncoder:
    """用于测试的 DenseEncoder 假实现，不进行真实 embedding 调用。"""
    
    def __init__(self, vector_dim: int = 3, should_fail: bool = False):
        self.vector_dim = vector_dim
        self.should_fail = should_fail
        self.encode_call_count = 0
    
    def encode(self, chunks: List[Chunk], trace=None) -> List[List[float]]:
        """返回确定性的假向量。"""
        self.encode_call_count += 1
        
        if self.should_fail:
            raise RuntimeError("Dense encoder failed")
        
        # 基于 chunk ID 生成确定性向量
        vectors = []
        for chunk in chunks:
            # 使用配置维度生成简单的确定性向量
            chunk_id_hash = float(hash(chunk.id) % 100) / 100.0
            # 第一个元素基于哈希，其余为 0.5
            vec = [chunk_id_hash] + [0.5] * (self.vector_dim - 1)
            vectors.append(vec)
        return vectors


class FakeSparseEncoder:
    """用于测试的 SparseEncoder 假实现，不进行真实分词。"""
    
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.encode_call_count = 0
    
    def encode(self, chunks: List[Chunk], trace=None) -> List[Dict[str, Any]]:
        """返回确定性的假统计信息。"""
        self.encode_call_count += 1
        
        if self.should_fail:
            raise RuntimeError("Sparse encoder failed")
        
        stats = []
        for chunk in chunks:
            stats.append({
                "chunk_id": chunk.id,
                "term_frequencies": {"fake": 1, "terms": 1},
                "doc_length": 2,
                "unique_terms": 2
            })
        return stats


# ============================================================================
# 测试 BatchProcessor 初始化
# ============================================================================

def test_batch_processor_initialization():
    """测试 BatchProcessor 可使用必需组件初始化。"""
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()
    processor = BatchProcessor(
        dense_encoder=dense,
        sparse_encoder=sparse,
        batch_size=10
    )
    
    assert processor.dense_encoder is dense
    assert processor.sparse_encoder is sparse
    assert processor.batch_size == 10


def test_batch_processor_initialization_with_default_batch_size():
    """测试默认 batch size 为 100。"""
    processor = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder()
    )
    assert processor.batch_size == 100


def test_batch_processor_rejects_invalid_batch_size():
    """测试无效 batch_size 时初始化失败。"""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        BatchProcessor(
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder(),
            batch_size=0
        )
    
    with pytest.raises(ValueError, match="batch_size must be positive"):
        BatchProcessor(
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder(),
            batch_size=-5
        )


# ============================================================================
# 测试批次创建
# ============================================================================

def test_create_batches_divides_evenly():
    """测试 chunk 被均分为多个批次。"""
    processor = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=2
    )
    
    chunks = [
        Chunk(id="1", text="a", metadata={"source_path": "test.pdf"}),
        Chunk(id="2", text="b", metadata={"source_path": "test.pdf"}),
        Chunk(id="3", text="c", metadata={"source_path": "test.pdf"}),
        Chunk(id="4", text="d", metadata={"source_path": "test.pdf"})
    ]
    
    batches = processor._create_batches(chunks)
    
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert batches[0][0].id == "1"
    assert batches[1][1].id == "4"


def test_create_batches_handles_remainder():
    """测试存在余数 chunk 时的分批（5 个 chunk，batch_size=2 -> 3 个批次）。"""
    processor = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=2
    )
    
    chunks = [Chunk(id=f"{i}", text=f"text{i}", metadata={"source_path": "doc.pdf"}) for i in range(5)]
    batches = processor._create_batches(chunks)
    
    assert len(batches) == 3, "5 chunks with batch_size=2 should create 3 batches"
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 1, "Last batch should have 1 chunk"


def test_create_batches_preserves_order():
    """测试批次创建保持 chunk 顺序。"""
    processor = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=2
    )
    
    chunks = [Chunk(id=f"chunk_{i}", text="", metadata={"source_path": "test.pdf"}) for i in range(5)]
    batches = processor._create_batches(chunks)
    
    # 展平批次并验证顺序
    flattened = [chunk for batch in batches for chunk in batch]
    assert [c.id for c in flattened] == [f"chunk_{i}" for i in range(5)]


def test_create_batches_single_chunk():
    """测试单个 chunk 的分批。"""
    processor = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=10
    )
    
    chunks = [Chunk(id="1", text="single", metadata={"source_path": "test.pdf"})]
    batches = processor._create_batches(chunks)
    
    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert batches[0][0].id == "1"


def test_get_batch_count():
    """测试批次数量计算工具。"""
    processor = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=2
    )
    
    assert processor.get_batch_count(0) == 0
    assert processor.get_batch_count(1) == 1
    assert processor.get_batch_count(2) == 1
    assert processor.get_batch_count(3) == 2
    assert processor.get_batch_count(4) == 2
    assert processor.get_batch_count(5) == 3


# ============================================================================
# 测试 process 方法 —— 正常路径
# ============================================================================

def test_process_encodes_all_chunks():
    """测试 process() 通过两个编码器编码所有 chunk。"""
    dense = FakeDenseEncoder(vector_dim=3)
    sparse = FakeSparseEncoder()
    processor = BatchProcessor(
        dense_encoder=dense,
        sparse_encoder=sparse,
        batch_size=2
    )
    
    chunks = [Chunk(id=f"{i}", text=f"text {i}", metadata={"source_path": "test.pdf"}) for i in range(5)]
    result = processor.process(chunks)
    
    # 验证结果
    assert len(result.dense_vectors) == 5
    assert len(result.sparse_stats) == 5
    assert result.successful_chunks == 5
    assert result.failed_chunks == 0
    assert result.batch_count == 3  # 5 个 chunk / batch_size=2 -> 3 个批次


def test_process_maintains_chunk_order():
    """测试输出顺序与输入 chunk 顺序一致。"""
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()
    processor = BatchProcessor(
        dense_encoder=dense,
        sparse_encoder=sparse,
        batch_size=2
    )
    
    chunks = [Chunk(id=f"chunk_{i}", text=f"text {i}", metadata={"source_path": "test.pdf"}) for i in range(5)]
    result = processor.process(chunks)
    
    # 验证 sparse 统计保持顺序
    assert [stat["chunk_id"] for stat in result.sparse_stats] == [f"chunk_{i}" for i in range(5)]


def test_process_returns_correct_batch_count():
    """测试 BatchResult 包含正确的批次数量。"""
    processor = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=3
    )
    
    chunks = [Chunk(id=f"{i}", text="", metadata={"source_path": "test.pdf"}) for i in range(10)]
    result = processor.process(chunks)
    
    assert result.batch_count == 4  # 10 / 3 = 4 个批次


def test_process_records_timing():
    """测试 process() 记录总处理耗时。"""
    processor = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=2
    )
    
    chunks = [Chunk(id=f"{i}", text="", metadata={"source_path": "test.pdf"}) for i in range(3)]
    result = processor.process(chunks)
    
    assert result.total_time > 0.0
    assert isinstance(result.total_time, float)


# ============================================================================
# 测试带 TraceContext 的 process
# ============================================================================

def test_process_with_trace_records_batch_info():
    """测试 process() 将批次信息记录到 TraceContext。"""
    processor = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=2
    )
    
    chunks = [Chunk(id=f"{i}", text="", metadata={"source_path": "test.pdf"}) for i in range(5)]
    trace = TraceContext()
    result = processor.process(chunks, trace=trace)
    
    # 验证已记录 batch_processing 阶段
    batch_data = trace.get_stage_data("batch_processing")
    assert batch_data is not None
    assert batch_data["total_chunks"] == 5
    assert batch_data["batch_count"] == 3
    assert batch_data["batch_size"] == 2
    assert batch_data["successful_chunks"] == 5
    assert batch_data["failed_chunks"] == 0


def test_process_with_trace_records_individual_batches():
    """测试记录各个批次的耗时。"""
    processor = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=2
    )
    
    chunks = [Chunk(id=f"{i}", text="", metadata={"source_path": "test.pdf"}) for i in range(5)]
    trace = TraceContext()
    processor.process(chunks, trace=trace)
    
    # 验证各个批次阶段
    for batch_idx in range(3):
        batch_data = trace.get_stage_data(f"batch_{batch_idx}")
        assert batch_data is not None
        assert "duration_seconds" in batch_data
        assert "chunks_processed" in batch_data


# ============================================================================
# 测试错误处理
# ============================================================================

def test_process_rejects_empty_chunks():
    """测试空 chunks 列表时 process() 抛出 ValueError。"""
    processor = BatchProcessor(
        dense_encoder=FakeDenseEncoder(),
        sparse_encoder=FakeSparseEncoder(),
        batch_size=2
    )
    
    with pytest.raises(ValueError, match="Cannot process empty chunks list"):
        processor.process([])


def test_process_continues_on_batch_failure():
    """测试某个批次失败后 process() 继续处理。"""
    # 创建在第二次调用时失败的编码器
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()

    # 让 dense 编码器在第二批失败
    original_encode = dense.encode
    call_count = [0]

    def failing_encode(chunks, trace=None):
        call_count[0] += 1
        if call_count[0] == 2:  # 在第二批失败
            raise RuntimeError("Simulated batch failure")
        return original_encode(chunks, trace)
    
    dense.encode = failing_encode
    
    processor = BatchProcessor(
        dense_encoder=dense,
        sparse_encoder=sparse,
        batch_size=2
    )
    
    chunks = [Chunk(id=f"{i}", text="", metadata={"source_path": "test.pdf"}) for i in range(6)]
    result = processor.process(chunks)
    
    # 应成功处理批次 1 和 3，批次 2 失败
    assert result.successful_chunks == 4  # 批次 1 的 2 个，批次 3 的 2 个
    assert result.failed_chunks == 2  # 批次 2


def test_process_records_batch_errors_to_trace():
    """测试批次错误被记录到 TraceContext。"""
    dense = FakeDenseEncoder(should_fail=True)
    sparse = FakeSparseEncoder()
    
    processor = BatchProcessor(
        dense_encoder=dense,
        sparse_encoder=sparse,
        batch_size=2
    )
    
    chunks = [Chunk(id=f"{i}", text="", metadata={"source_path": "test.pdf"}) for i in range(3)]
    trace = TraceContext()
    result = processor.process(chunks, trace=trace)
    
    # 验证错误已被记录
    assert result.failed_chunks == 3
    batch_0_error = trace.get_stage_data("batch_0_error")
    assert batch_0_error is not None
    assert "Dense encoder failed" in batch_0_error["error"]


# ============================================================================
# 测试 BatchResult 数据类
# ============================================================================

def test_batch_result_structure():
    """测试 BatchResult 包含所有必需字段。"""
    result = BatchResult(
        dense_vectors=[[0.1, 0.2]],
        sparse_stats=[{"chunk_id": "1", "term_frequencies": {}}],
        batch_count=1,
        total_time=0.5,
        successful_chunks=1,
        failed_chunks=0
    )
    
    assert len(result.dense_vectors) == 1
    assert len(result.sparse_stats) == 1
    assert result.batch_count == 1
    assert result.total_time == 0.5
    assert result.successful_chunks == 1
    assert result.failed_chunks == 0


# ============================================================================
# 与真实编码器的集成测试（使用假实现）
# ============================================================================

def test_process_integration_with_encoders():
    """测试 BatchProcessor 与编码器接口正确集成。"""
    dense = FakeDenseEncoder(vector_dim=4)
    sparse = FakeSparseEncoder()
    
    processor = BatchProcessor(
        dense_encoder=dense,
        sparse_encoder=sparse,
        batch_size=10
    )
    
    chunks = [
        Chunk(id="doc1_chunk0", text="machine learning systems", metadata={"source_path": "doc1.pdf", "source": "doc1"}),
        Chunk(id="doc1_chunk1", text="natural language processing", metadata={"source_path": "doc1.pdf", "source": "doc1"}),
        Chunk(id="doc2_chunk0", text="computer vision models", metadata={"source_path": "doc2.pdf", "source": "doc2"})
    ]
    
    result = processor.process(chunks)
    
    # 验证 dense 向量
    assert len(result.dense_vectors) == 3
    assert all(len(vec) == 4 for vec in result.dense_vectors)

    # 验证 sparse 统计
    assert len(result.sparse_stats) == 3
    assert all("chunk_id" in stat for stat in result.sparse_stats)
    assert result.sparse_stats[0]["chunk_id"] == "doc1_chunk0"

    # 验证指标
    assert result.batch_count == 1  # 全部属于一个批次
    assert result.successful_chunks == 3
    assert result.failed_chunks == 0
    assert result.total_time > 0


def test_process_deterministic_output():
    """测试相同 chunk 产生相同输出。"""
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()
    processor = BatchProcessor(
        dense_encoder=dense,
        sparse_encoder=sparse,
        batch_size=2
    )
    
    chunks = [Chunk(id=f"chunk_{i}", text=f"text {i}", metadata={"source_path": "test.pdf"}) for i in range(3)]
    
    result1 = processor.process(chunks)
    result2 = processor.process(chunks)
    
    # dense 向量应完全相同（基于 chunk ID 的确定性结果）
    assert result1.dense_vectors == result2.dense_vectors

    # sparse 统计应完全相同
    assert result1.sparse_stats == result2.sparse_stats

