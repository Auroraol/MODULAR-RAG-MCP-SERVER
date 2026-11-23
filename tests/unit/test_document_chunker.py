"""DocumentChunker 的单元测试 —— Document 到 Chunk 的适配器。

本测试套件验证 DocumentChunker 的五项核心增值特性：
1. 唯一且确定性的 chunk ID 生成
2. 从 Document 到 Chunk 的完整元数据继承
3. 用于记录顺序位置的 chunk_index 追踪
4. source_ref 建立父子可追溯关系
5. 类型契约合规（core.types.Chunk）

所有测试使用 FakeSplitter 将 DocumentChunker 的业务逻辑与底层文本切分实现隔离，
确保不依赖外部组件。
"""

import pytest
from unittest.mock import Mock

from src.core.types import Document, Chunk
from src.core.settings import Settings
from src.ingestion.chunking import DocumentChunker
from src.libs.splitter.base_splitter import BaseSplitter


class FakeSplitter(BaseSplitter):
    """用于测试的假 splitter —— 返回可预测的 chunk。

    该假实现使我们能够在测试 DocumentChunker 业务逻辑时，
    不依赖真实的文本切分算法或外部库。
    """

    def __init__(self, chunk_size: int = 100, overlap: int = 0, **kwargs):
        """为兼容性初始化，忽略所有参数。"""
        pass

    def split_text(self, text: str) -> list[str]:
        """按双换行切分文本（基于段落切分）。"""
        # 测试用的简单切分：按双换行切分
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]


@pytest.fixture
def fake_settings():
    """提供用于测试的最小配置。"""
    settings = Mock(spec=Settings)
    settings.splitter = Mock()
    settings.splitter.provider = "fake"
    settings.splitter.chunk_size = 100
    settings.splitter.overlap = 0
    return settings


@pytest.fixture
def chunker(fake_settings, monkeypatch):
    """提供注入 FakeSplitter 的 DocumentChunker。"""
    # monkey-patch SplitterFactory 使其返回我们的 FakeSplitter
    from src.libs.splitter import splitter_factory
    
    original_create = splitter_factory.SplitterFactory.create
    
    def mock_create(settings):
        return FakeSplitter()
    
    monkeypatch.setattr(splitter_factory.SplitterFactory, "create", mock_create)
    
    return DocumentChunker(fake_settings)


@pytest.fixture
def sample_document():
    """提供用于测试的示例文档。"""
    return Document(
        id="doc_sample_001",
        text="First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph content.",
        metadata={
            "source_path": "data/documents/sample.pdf",
            "doc_type": "pdf",
            "title": "Sample Document",
            "page_count": 3
        }
    )


# =============================================================================
# 测试 1：Chunk ID 生成 —— 唯一性与确定性
# =============================================================================

def test_chunk_ids_are_unique(chunker, sample_document):
    """测试每个 chunk 获得唯一 ID。"""
    chunks = chunker.split_document(sample_document)

    # 提取所有 chunk ID
    chunk_ids = [chunk.id for chunk in chunks]

    # 验证唯一性：无重复
    assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs must be unique"


def test_chunk_ids_are_deterministic(chunker, sample_document):
    """测试同一文档切分两次产生相同的 ID。"""
    # 切分两次
    chunks_first = chunker.split_document(sample_document)
    chunks_second = chunker.split_document(sample_document)

    # 提取 ID
    ids_first = [c.id for c in chunks_first]
    ids_second = [c.id for c in chunks_second]

    # 验证确定性：多次运行 ID 一致
    assert ids_first == ids_second, "Chunk IDs must be deterministic"


def test_chunk_id_format(chunker, sample_document):
    """测试 chunk ID 遵循预期格式：{doc_id}_{index:04d}_{hash}。"""
    chunks = chunker.split_document(sample_document)

    for i, chunk in enumerate(chunks):
        # 预期格式：doc_sample_001_0000_{hash}、doc_sample_001_0001_{hash} 等
        assert chunk.id.startswith(f"doc_sample_001_{i:04d}_"), \
            f"Chunk ID should start with 'doc_sample_001_{i:04d}_', got: {chunk.id}"

        # 哈希部分应为 8 个字符
        hash_part = chunk.id.split("_")[-1]
        assert len(hash_part) == 8, f"Hash should be 8 chars, got: {hash_part}"


def test_chunk_id_changes_with_content(chunker):
    """测试内容变化时 chunk ID 随之变化。"""
    doc1 = Document(
        id="doc_001",
        text="Content A",
        metadata={"source_path": "file.pdf"}
    )
    doc2 = Document(
        id="doc_001",  # 相同 doc_id
        text="Content B",  # 不同内容
        metadata={"source_path": "file.pdf"}
    )
    
    chunks1 = chunker.split_document(doc1)
    chunks2 = chunker.split_document(doc2)
    
    # ID 应因内容哈希不同而不同
    assert chunks1[0].id != chunks2[0].id, \
        "Chunk ID should change when content changes"


# =============================================================================
# 测试 2：元数据继承 —— 完整传递
# =============================================================================

def test_metadata_inheritance(chunker, sample_document):
    """测试所有文档元数据被 chunk 继承。"""
    chunks = chunker.split_document(sample_document)

    for chunk in chunks:
        # 所有文档元数据都应存在
        assert chunk.metadata["source_path"] == "data/documents/sample.pdf"
        assert chunk.metadata["doc_type"] == "pdf"
        assert chunk.metadata["title"] == "Sample Document"
        assert chunk.metadata["page_count"] == 3


def test_metadata_independence(chunker, sample_document):
    """测试每个 chunk 拥有独立的元数据 dict（非共享引用）。"""
    chunks = chunker.split_document(sample_document)

    # 修改第一个 chunk 的元数据
    chunks[0].metadata["custom_field"] = "test_value"

    # 其他 chunk 不应受影响
    assert "custom_field" not in chunks[1].metadata, \
        "Chunks should have independent metadata dicts"


def test_metadata_with_empty_document_metadata(chunker):
    """测试文档仅有最少元数据时的切分。"""
    doc = Document(
        id="doc_minimal",
        text="Paragraph 1.\n\nParagraph 2.",
        metadata={"source_path": "minimal.txt"}  # 仅含必需字段
    )

    chunks = chunker.split_document(doc)

    # 最少元数据下仍应正常工作
    assert len(chunks) == 2
    assert chunks[0].metadata["source_path"] == "minimal.txt"


# =============================================================================
# 测试 3：chunk_index —— 顺序位置追踪
# =============================================================================

def test_chunk_index_sequential(chunker, sample_document):
    """测试 chunk_index 从 0 开始并依次递增。"""
    chunks = chunker.split_document(sample_document)

    # 验证索引连续：0、1、2、...
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i, \
            f"Chunk at position {i} should have chunk_index={i}"


def test_chunk_index_added_to_all_chunks(chunker, sample_document):
    """测试每个 chunk 都包含 chunk_index 字段。"""
    chunks = chunker.split_document(sample_document)
    
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata, \
            "All chunks must have chunk_index field"
        assert isinstance(chunk.metadata["chunk_index"], int), \
            "chunk_index must be an integer"


# =============================================================================
# 测试 4：source_ref —— 父子可追溯性
# =============================================================================

def test_source_ref_points_to_document(chunker, sample_document):
    """测试 source_ref 正确引用父文档 ID。"""
    chunks = chunker.split_document(sample_document)

    for chunk in chunks:
        assert chunk.metadata["source_ref"] == sample_document.id, \
            f"source_ref should point to document ID '{sample_document.id}'"


def test_source_ref_added_to_all_chunks(chunker, sample_document):
    """测试每个 chunk 都有 source_ref 字段。"""
    chunks = chunker.split_document(sample_document)

    for chunk in chunks:
        assert "source_ref" in chunk.metadata, \
            "All chunks must have source_ref field"


# =============================================================================
# 测试 5：类型契约 —— 符合 core.types.Chunk
# =============================================================================

def test_chunks_are_chunk_type(chunker, sample_document):
    """测试输出项为 Chunk 对象。"""
    chunks = chunker.split_document(sample_document)

    for chunk in chunks:
        assert isinstance(chunk, Chunk), \
            f"Output should be Chunk objects, got: {type(chunk)}"


def test_chunk_serialization(chunker, sample_document):
    """测试 chunk 可序列化为 dict。"""
    chunks = chunker.split_document(sample_document)

    for chunk in chunks:
        chunk_dict = chunk.to_dict()

        # 验证 dict 结构
        assert "id" in chunk_dict
        assert "text" in chunk_dict
        assert "metadata" in chunk_dict
        assert isinstance(chunk_dict["metadata"], dict)


def test_chunk_fields_complete(chunker, sample_document):
    """测试 chunk 具备所有必需的 Chunk 字段。"""
    chunks = chunker.split_document(sample_document)

    for chunk in chunks:
        # 必需的 Chunk 字段
        assert hasattr(chunk, "id") and chunk.id
        assert hasattr(chunk, "text") and chunk.text
        assert hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict)


# =============================================================================
# 测试 6：配置驱动的行为
# =============================================================================

def test_different_splitter_config_produces_different_chunks(fake_settings, monkeypatch):
    """测试修改 splitter 配置会影响 chunk 输出。"""
    # 本测试验证 DocumentChunker 遵守 splitter 配置

    from src.libs.splitter import splitter_factory

    # 创建两个行为不同的假 splitter
    class SmallChunkSplitter(BaseSplitter):
        def split_text(self, text: str) -> list[str]:
            # 按句子（句号）切分
            return [s.strip() + "." for s in text.split(".") if s.strip()]
    
    class LargeChunkSplitter(BaseSplitter):
        def split_text(self, text: str) -> list[str]:
            # 将整个文本作为一个 chunk 返回
            return [text]
    
    document = Document(
        id="doc_test",
        text="First sentence. Second sentence. Third sentence.",
        metadata={"source_path": "test.txt"}
    )
    
    # 使用小 chunk 测试
    def mock_create_small(settings):
        return SmallChunkSplitter()

    monkeypatch.setattr(splitter_factory.SplitterFactory, "create", mock_create_small)
    chunker_small = DocumentChunker(fake_settings)
    chunks_small = chunker_small.split_document(document)

    # 使用大 chunk 测试
    def mock_create_large(settings):
        return LargeChunkSplitter()

    monkeypatch.setattr(splitter_factory.SplitterFactory, "create", mock_create_large)
    chunker_large = DocumentChunker(fake_settings)
    chunks_large = chunker_large.split_document(document)

    # 不同配置应产生不同数量的 chunk
    assert len(chunks_small) != len(chunks_large), \
        "Different splitter configs should produce different chunk counts"


# =============================================================================
# 测试 7：边界情况与错误处理
# =============================================================================

def test_empty_document_raises_error(chunker):
    """测试空文档抛出明确错误。"""
    doc = Document(
        id="doc_empty",
        text="",
        metadata={"source_path": "empty.txt"}
    )
    
    with pytest.raises(ValueError, match="has no text content"):
        chunker.split_document(doc)


def test_whitespace_only_document_raises_error(chunker):
    """测试仅含空白的文档抛出错误。"""
    doc = Document(
        id="doc_whitespace",
        text="   \n\n   \t  ",
        metadata={"source_path": "whitespace.txt"}
    )
    
    with pytest.raises(ValueError, match="has no text content"):
        chunker.split_document(doc)


def test_splitter_returns_empty_list_raises_error(chunker, monkeypatch):
    """测试 splitter 不返回 chunk 时抛出明确错误。"""
    from src.libs.splitter import splitter_factory

    class EmptySplitter(BaseSplitter):
        def split_text(self, text: str) -> list[str]:
            return []  # 不返回任何 chunk

    def mock_create(settings):
        return EmptySplitter()

    monkeypatch.setattr(splitter_factory.SplitterFactory, "create", mock_create)

    doc = Document(
        id="doc_test",
        text="Some content",
        metadata={"source_path": "test.txt"}
    )

    chunker_empty = DocumentChunker(chunker._settings)

    with pytest.raises(ValueError, match="Splitter returned no chunks"):
        chunker_empty.split_document(doc)


# =============================================================================
# 测试 8：集成冒烟测试
# =============================================================================

def test_end_to_end_smoke(chunker, sample_document):
    """验证完整端到端切分流程的冒烟测试。"""
    # 执行完整切分
    chunks = chunker.split_document(sample_document)

    # 基础健全性检查
    assert len(chunks) > 0, "Should produce at least one chunk"

    # 验证每个 chunk 满足所有要求
    for i, chunk in enumerate(chunks):
        # ID 要求
        assert chunk.id
        assert chunk.id.startswith(sample_document.id)

        # 文本要求
        assert chunk.text
        assert chunk.text.strip()

        # 元数据要求
        assert chunk.metadata["source_path"] == sample_document.metadata["source_path"]
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["source_ref"] == sample_document.id

        # 类型要求
        assert isinstance(chunk, Chunk)


# =============================================================================
# 测试 9：图片占位符 → Chunk 级图片元数据
# =============================================================================

def test_images_resolved_from_placeholders(chunker):
    """引用图片的 Chunk 回填 images / page_num，未引用的 Chunk 不继承。"""
    doc = Document(
        id="doc_img",
        text="Intro with [IMAGE: img_001].\n\nPlain paragraph.",
        metadata={
            "source_path": "data/documents/sample.docx",
            "images": [
                {"id": "img_001", "path": "data/images/doc_img/img_001.png", "page": 3}
            ],
        },
    )

    chunks = chunker.split_document(doc)

    assert chunks[0].metadata["image_refs"] == ["img_001"]
    assert chunks[0].metadata["images"][0]["path"] == "data/images/doc_img/img_001.png"
    assert chunks[0].metadata["page_num"] == 3

    assert chunks[1].metadata["image_refs"] == []
    assert "images" not in chunks[1].metadata
    assert "page_num" not in chunks[1].metadata
