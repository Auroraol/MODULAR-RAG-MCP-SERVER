"""使用真实 PDF 文件的 PDF Loader 集成测试。

这些测试使用 fixtures/sample_documents 中的真实 PDF 文件，
以校验 PDF loader 的端到端功能。
"""

from pathlib import Path

import pytest

from src.core.types import Document
from src.libs.loader.pdf_loader import PdfLoader


# Fixture 路径
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_documents"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"
IMAGES_PDF = FIXTURES_DIR / "with_images.pdf"


class TestPdfLoaderWithRealFiles:
    """使用真实 PDF 文件的集成测试。"""

    def test_simple_pdf_exists(self):
        """校验测试 fixture 存在。"""
        assert SIMPLE_PDF.exists(), f"Test fixture not found: {SIMPLE_PDF}"

    def test_images_pdf_exists(self):
        """校验含图像的测试 fixture 存在。"""
        assert IMAGES_PDF.exists(), f"Test fixture not found: {IMAGES_PDF}"

    def test_load_simple_pdf(self):
        """加载一个简单的纯文本 PDF 并校验 Document 结构。"""
        loader = PdfLoader()
        doc = loader.load(SIMPLE_PDF)

        # 校验 Document 结构
        assert isinstance(doc, Document)
        assert doc.id.startswith("doc_")
        assert len(doc.text) > 0

        # 校验必需的 metadata
        assert doc.metadata["source_path"] == str(SIMPLE_PDF)
        assert doc.metadata["doc_type"] == "pdf"
        assert "doc_hash" in doc.metadata

        # 校验标题提取（该 PDF 的标题为 "Sample Document"）
        assert "title" in doc.metadata
        assert len(doc.metadata["title"]) > 0
        assert "sample" in doc.metadata["title"].lower() or "document" in doc.metadata["title"].lower()

        # 校验文本内容包含预期关键词
        assert "sample" in doc.text.lower() or "test" in doc.text.lower()

    def test_load_pdf_with_images(self):
        """加载含图像的 PDF 并校验 Document 结构。"""
        loader = PdfLoader(extract_images=True)
        doc = loader.load(IMAGES_PDF)

        # 校验 Document 结构
        assert isinstance(doc, Document)
        assert doc.id.startswith("doc_")
        assert len(doc.text) > 0

        # 校验必需的 metadata
        assert doc.metadata["source_path"] == str(IMAGES_PDF)
        assert doc.metadata["doc_type"] == "pdf"
        assert "doc_hash" in doc.metadata

        # 注意: 当前实现返回空图像列表（桩实现）
        # 完整的图像提取将在后续实现
        if "images" in doc.metadata:
            assert isinstance(doc.metadata["images"], list)

    def test_load_simple_pdf_without_image_extraction(self):
        """在禁用图像提取的情况下加载 PDF。"""
        loader = PdfLoader(extract_images=False)
        doc = loader.load(SIMPLE_PDF)

        assert isinstance(doc, Document)
        assert doc.metadata["doc_type"] == "pdf"
        # 禁用提取时不应有 images metadata
        assert "images" not in doc.metadata or doc.metadata.get("images") == []

    def test_document_is_serializable(self):
        """校验加载的 Document 可序列化为 dict/JSON。"""
        loader = PdfLoader()
        doc = loader.load(SIMPLE_PDF)

        doc_dict = doc.to_dict()
        assert isinstance(doc_dict, dict)
        assert "id" in doc_dict
        assert "text" in doc_dict
        assert "metadata" in doc_dict

        # 校验可从 dict 重建
        doc_recreated = Document.from_dict(doc_dict)
        assert doc_recreated.id == doc.id
        assert doc_recreated.text == doc.text

    def test_file_hash_consistency(self):
        """校验同一文件产生相同的 hash。"""
        loader = PdfLoader()

        doc1 = loader.load(SIMPLE_PDF)
        doc2 = loader.load(SIMPLE_PDF)

        # 同一文件应产生相同的 doc_hash
        assert doc1.metadata["doc_hash"] == doc2.metadata["doc_hash"]

    def test_different_files_different_hash(self):
        """校验不同文件产生不同的 hash。"""
        loader = PdfLoader()

        doc1 = loader.load(SIMPLE_PDF)
        doc2 = loader.load(IMAGES_PDF)

        # 不同文件应产生不同的 hash
        assert doc1.metadata["doc_hash"] != doc2.metadata["doc_hash"]

    def test_custom_image_storage_dir(self):
        """校验自定义图像存储目录被正确使用。"""
        custom_dir = "custom/images"
        loader = PdfLoader(extract_images=True, image_storage_dir=custom_dir)

        assert loader.image_storage_dir == Path(custom_dir)
