"""VectorUpserter 幂等性与正确性的单元测试。

测试覆盖：
1. 幂等性：同一 chunk 重复 upsert 会产生相同 ID
2. 确定性：chunk ID 生成稳定且可复现
3. 内容敏感性：不同内容产生不同 ID
4. 批量操作：多 chunk 时的顺序与正确性
5. 错误处理：校验与失败场景
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.core.types import Chunk
from src.core.settings import Settings


# ============================================================================
# 测试夹具
# ============================================================================

@pytest.fixture
def mock_settings():
    """创建带 vector_store 配置的 Mock Settings 对象。"""
    settings = Mock(spec=Settings)
    settings.vector_store = Mock()
    settings.vector_store.backend = "chroma"
    settings.vector_store.persist_path = "data/db/chroma"
    return settings


@pytest.fixture
def mock_vector_store():
    """创建可捕获 upsert 调用的 Mock VectorStore。"""
    mock_store = Mock()
    mock_store.upsert = Mock()
    return mock_store


@pytest.fixture
def upserter_with_mock_store(mock_settings, mock_vector_store):
    """创建使用 Mock Vector Store 的 VectorUpserter。"""
    with patch("src.ingestion.storage.vector_upserter.VectorStoreFactory") as mock_factory:
        mock_factory.create.return_value = mock_vector_store
        upserter = VectorUpserter(mock_settings)
        return upserter, mock_vector_store


@pytest.fixture
def sample_chunk():
    """创建用于测试的示例 Chunk。"""
    return Chunk(
        id="temp_id",  # 将被生成的 ID 替换
        text="This is a test chunk for vector storage.",
        metadata={
            "source_path": "data/documents/test.pdf",
            "chunk_index": 0,
            "source_ref": "doc_abc123",
        },
    )


@pytest.fixture
def sample_vector():
    """创建示例 Embedding 向量。"""
    return [0.1, 0.2, 0.3, 0.4, 0.5]


# ============================================================================
# Chunk ID 生成测试
# ============================================================================

def test_chunk_id_deterministic(upserter_with_mock_store, sample_chunk):
    """测试同一 chunk 每次都会产生相同 ID。"""
    upserter, _ = upserter_with_mock_store

    # 多次生成 ID
    id1 = upserter._generate_chunk_id(sample_chunk)
    id2 = upserter._generate_chunk_id(sample_chunk)
    id3 = upserter._generate_chunk_id(sample_chunk)
    
    assert id1 == id2 == id3, "Chunk ID must be deterministic"


def test_chunk_id_format(upserter_with_mock_store, sample_chunk):
    """测试 chunk ID 遵循预期的格式。"""
    upserter, _ = upserter_with_mock_store
    
    chunk_id = upserter._generate_chunk_id(sample_chunk)
    
    # 格式：{source_hash}_{index:04d}_{content_hash}
    parts = chunk_id.split("_")
    
    assert len(parts) == 3, "Chunk ID must have 3 parts"
    assert len(parts[0]) == 8, "Source hash must be 8 characters"
    assert parts[1] == "0000", "Index must be zero-padded to 4 digits"
    assert len(parts[2]) == 8, "Content hash must be 8 characters"


def test_chunk_id_changes_with_content(upserter_with_mock_store, sample_chunk):
    """测试不同内容会产生不同 ID。"""
    upserter, _ = upserter_with_mock_store

    id1 = upserter._generate_chunk_id(sample_chunk)

    # 修改内容
    sample_chunk.text = "Different content"
    id2 = upserter._generate_chunk_id(sample_chunk)
    
    assert id1 != id2, "Different content must produce different IDs"
    
    # 但来源哈希与索引应保持不变
    assert id1.split("_")[0] == id2.split("_")[0], "Source hash should be same"
    assert id1.split("_")[1] == id2.split("_")[1], "Index should be same"


def test_chunk_id_changes_with_index(upserter_with_mock_store, sample_chunk):
    """测试不同索引会产生不同 ID。"""
    upserter, _ = upserter_with_mock_store

    id1 = upserter._generate_chunk_id(sample_chunk)

    # 修改索引
    sample_chunk.metadata["chunk_index"] = 5
    id2 = upserter._generate_chunk_id(sample_chunk)
    
    assert id1 != id2, "Different index must produce different IDs"
    assert id1.split("_")[1] == "0000", "First ID should have index 0000"
    assert id2.split("_")[1] == "0005", "Second ID should have index 0005"


def test_chunk_id_changes_with_source_path(upserter_with_mock_store, sample_chunk):
    """测试不同来源路径会产生不同 ID。"""
    upserter, _ = upserter_with_mock_store

    id1 = upserter._generate_chunk_id(sample_chunk)

    # 修改来源路径
    sample_chunk.metadata["source_path"] = "data/documents/other.pdf"
    id2 = upserter._generate_chunk_id(sample_chunk)
    
    assert id1 != id2, "Different source path must produce different IDs"
    assert id1.split("_")[0] != id2.split("_")[0], "Source hash should be different"


def test_chunk_id_generation_missing_source_path(upserter_with_mock_store):
    """测试缺少 source_path 时在 Chunk 创建阶段抛出 ValueError。"""
    upserter, _ = upserter_with_mock_store

    # Chunk 校验会在初始化阶段捕获此问题
    with pytest.raises(ValueError, match="source_path"):
        chunk = Chunk(
            id="temp",
            text="Test",
            metadata={"chunk_index": 0},  # 缺少 source_path
        )


def test_chunk_id_generation_missing_chunk_index(upserter_with_mock_store):
    """测试缺少 chunk_index 会抛出 ValueError。"""
    upserter, _ = upserter_with_mock_store

    chunk = Chunk(
        id="temp",
        text="Test",
        metadata={"source_path": "test.pdf"},  # 缺少 chunk_index
    )
    
    with pytest.raises(ValueError, match="chunk_index"):
        upserter._generate_chunk_id(chunk)


# ============================================================================
# Upsert 测试
# ============================================================================

def test_upsert_single_chunk(upserter_with_mock_store, sample_chunk, sample_vector):
    """测试 upsert 单个带向量的 chunk。"""
    upserter, mock_store = upserter_with_mock_store

    chunk_ids = upserter.upsert([sample_chunk], [sample_vector])

    # 验证返回的 ID
    assert len(chunk_ids) == 1
    assert isinstance(chunk_ids[0], str)
    
    # 验证 Vector Store 被调用
    assert mock_store.upsert.called
    call_args = mock_store.upsert.call_args
    records = call_args[0][0]
    
    assert len(records) == 1
    assert records[0]["id"] == chunk_ids[0]
    assert records[0]["vector"] == sample_vector
    assert records[0]["metadata"]["text"] == sample_chunk.text


def test_upsert_multiple_chunks(upserter_with_mock_store):
    """测试 upsert 多个 chunk 时保持顺序。"""
    upserter, mock_store = upserter_with_mock_store

    chunks = [
        Chunk(
            id=f"temp{i}",
            text=f"Chunk {i}",
            metadata={"source_path": "test.pdf", "chunk_index": i},
        )
        for i in range(5)
    ]
    vectors = [[float(i)] * 5 for i in range(5)]

    chunk_ids = upserter.upsert(chunks, vectors)

    # 验证数量与顺序
    assert len(chunk_ids) == 5

    # 验证 Vector Store 收到正确的记录
    records = mock_store.upsert.call_args[0][0]
    assert len(records) == 5
    
    for i, record in enumerate(records):
        assert record["id"] == chunk_ids[i]
        assert record["vector"] == vectors[i]
        assert record["metadata"]["text"] == f"Chunk {i}"


def test_upsert_idempotency(upserter_with_mock_store, sample_chunk, sample_vector):
    """测试重复 upsert 会产生相同 ID。"""
    upserter, mock_store = upserter_with_mock_store

    # 第一次 upsert
    ids1 = upserter.upsert([sample_chunk], [sample_vector])

    # 第二次 upsert（同一 chunk）
    ids2 = upserter.upsert([sample_chunk], [sample_vector])

    # ID 应完全相同
    assert ids1 == ids2, "Idempotent upsert must produce same IDs"


def test_upsert_preserves_metadata(upserter_with_mock_store, sample_chunk, sample_vector):
    """测试所有元数据都会被保留到存储中。"""
    upserter, mock_store = upserter_with_mock_store

    # 添加额外的元数据
    sample_chunk.metadata["custom_field"] = "custom_value"
    sample_chunk.metadata["tags"] = ["tag1", "tag2"]
    
    upserter.upsert([sample_chunk], [sample_vector])
    
    records = mock_store.upsert.call_args[0][0]
    metadata = records[0]["metadata"]
    
    # 验证所有原始元数据都被保留
    assert metadata["source_path"] == "data/documents/test.pdf"
    assert metadata["chunk_index"] == 0
    assert metadata["source_ref"] == "doc_abc123"
    assert metadata["custom_field"] == "custom_value"
    assert metadata["tags"] == ["tag1", "tag2"]
    
    # 验证文本已存储
    assert metadata["text"] == sample_chunk.text


def test_upsert_empty_chunks_raises_error(upserter_with_mock_store):
    """测试空 chunk 列表会抛出 ValueError。"""
    upserter, _ = upserter_with_mock_store
    
    with pytest.raises(ValueError, match="empty chunks"):
        upserter.upsert([], [])


def test_upsert_mismatched_lengths_raises_error(upserter_with_mock_store, sample_chunk):
    """测试 chunks 与 vectors 长度不一致会抛出错误。"""
    upserter, _ = upserter_with_mock_store

    chunks = [sample_chunk, sample_chunk]
    vectors = [[0.1, 0.2]]  # 2 个 chunk 只有 1 个向量
    
    with pytest.raises(ValueError, match="must match"):
        upserter.upsert(chunks, vectors)


def test_upsert_vector_store_failure(upserter_with_mock_store, sample_chunk, sample_vector):
    """测试 Vector Store 失败会被正确包装。"""
    upserter, mock_store = upserter_with_mock_store

    # 模拟 Vector Store 失败
    mock_store.upsert.side_effect = Exception("Connection failed")
    
    with pytest.raises(RuntimeError, match="Vector store upsert failed"):
        upserter.upsert([sample_chunk], [sample_vector])


def test_upsert_with_trace_context(upserter_with_mock_store, sample_chunk, sample_vector):
    """测试 trace 上下文会被传递给 Vector Store。"""
    upserter, mock_store = upserter_with_mock_store

    mock_trace = Mock()
    upserter.upsert([sample_chunk], [sample_vector], trace=mock_trace)

    # 验证 trace 被传递给 Vector Store
    call_kwargs = mock_store.upsert.call_args[1]
    assert call_kwargs["trace"] == mock_trace


# ============================================================================
# 批量 Upsert 测试
# ============================================================================

def test_upsert_batch_single_batch(upserter_with_mock_store):
    """测试 upsert 单个批次。"""
    upserter, mock_store = upserter_with_mock_store

    chunks = [
        Chunk(id="t1", text="C1", metadata={"source_path": "test.pdf", "chunk_index": 0}),
        Chunk(id="t2", text="C2", metadata={"source_path": "test.pdf", "chunk_index": 1}),
    ]
    vectors = [[0.1, 0.2], [0.3, 0.4]]

    batches = [(chunks, vectors)]
    chunk_ids = upserter.upsert_batch(batches)
    
    assert len(chunk_ids) == 2
    assert mock_store.upsert.call_count == 1


def test_upsert_batch_multiple_batches(upserter_with_mock_store):
    """测试 upsert 多个批次时保持顺序。"""
    upserter, mock_store = upserter_with_mock_store

    batch1 = (
        [
            Chunk(id="t1", text="C1", metadata={"source_path": "test.pdf", "chunk_index": 0}),
            Chunk(id="t2", text="C2", metadata={"source_path": "test.pdf", "chunk_index": 1}),
        ],
        [[0.1, 0.2], [0.3, 0.4]],
    )

    batch2 = (
        [
            Chunk(id="t3", text="C3", metadata={"source_path": "test.pdf", "chunk_index": 2}),
        ],
        [[0.5, 0.6]],
    )

    chunk_ids = upserter.upsert_batch([batch1, batch2])

    # 总共应有 3 个 chunk
    assert len(chunk_ids) == 3

    # 应只调用一次 upsert，且数据已展平
    assert mock_store.upsert.call_count == 1
    records = mock_store.upsert.call_args[0][0]
    assert len(records) == 3


def test_upsert_batch_empty_batches(upserter_with_mock_store):
    """测试空的批次列表会抛出错误。"""
    upserter, _ = upserter_with_mock_store
    
    with pytest.raises(ValueError, match="empty"):
        upserter.upsert_batch([])


# ============================================================================
# 边界情况
# ============================================================================

def test_chunk_with_unicode_text(upserter_with_mock_store):
    """测试处理 chunk 文本中的 Unicode 字符。"""
    upserter, mock_store = upserter_with_mock_store

    chunk = Chunk(
        id="temp",
        text="测试中文 🚀 émojis αβγ",
        metadata={"source_path": "test.pdf", "chunk_index": 0},
    )

    chunk_id = upserter._generate_chunk_id(chunk)

    # 不应抛出异常，ID 应为合法的 ASCII
    assert isinstance(chunk_id, str)
    assert chunk_id.isascii()


def test_chunk_with_long_source_path(upserter_with_mock_store):
    """测试处理超长的来源路径。"""
    upserter, _ = upserter_with_mock_store

    long_path = "a" * 500 + ".pdf"
    chunk = Chunk(
        id="temp",
        text="Test",
        metadata={"source_path": long_path, "chunk_index": 0},
    )

    chunk_id = upserter._generate_chunk_id(chunk)

    # ID 长度仍应合理（哈希会保证其简短）
    assert len(chunk_id) < 50


def test_chunk_with_large_index(upserter_with_mock_store):
    """测试处理较大的 chunk 索引。"""
    upserter, _ = upserter_with_mock_store

    chunk = Chunk(
        id="temp",
        text="Test",
        metadata={"source_path": "test.pdf", "chunk_index": 9999},
    )

    chunk_id = upserter._generate_chunk_id(chunk)

    # 索引应补零到 4 位
    assert "_9999_" in chunk_id
