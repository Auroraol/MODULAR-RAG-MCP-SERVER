"""PDF Loader 契约与行为的单元测试。

测试验证：
- BaseLoader 抽象接口
- PdfLoader 初始化与配置
- 辅助方法（哈希计算、标题提取等）
- 无效输入的错误处理
- 使用真实测试文件的核心 PDF 转换功能

注意：更多集成测试位于 tests/integration/test_pdf_loader_integration.py
"""

from pathlib import Path

import pytest

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader


# 测试夹具路径
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_documents"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"
IMAGES_PDF = FIXTURES_DIR / "with_images.pdf"


class TestBaseLoader:
    """BaseLoader 抽象接口的测试。"""

    def test_cannot_instantiate_abstract_class(self):
        """BaseLoader 不能直接实例化。"""
        with pytest.raises(TypeError, match="abstract"):
            BaseLoader()
    
    def test_validate_file_existing_file(self, tmp_path):
        """_validate_file 对已存在的文件返回 Path。"""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")
        
        # 通过子类访问静态方法
        validated = PdfLoader._validate_file(test_file)
        assert validated.exists()
        assert validated.is_file()
    
    def test_validate_file_nonexistent(self):
        """_validate_file 对缺失文件抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            PdfLoader._validate_file("nonexistent_file.pdf")
    
    def test_validate_file_directory(self, tmp_path):
        """_validate_file 对目录抛出 ValueError。"""
        with pytest.raises(ValueError, match="not a file"):
            PdfLoader._validate_file(tmp_path)


class TestPdfLoaderInitialization:
    """PdfLoader 初始化的测试。"""

    def test_default_initialization(self):
        """PdfLoader 可使用默认值初始化。"""
        loader = PdfLoader()
        assert loader.extract_images is True
        assert loader.image_storage_dir == Path("data/images")
    
    def test_custom_initialization(self):
        """PdfLoader 遵循自定义配置。"""
        loader = PdfLoader(
            extract_images=False,
            image_storage_dir="custom/path"
        )
        assert loader.extract_images is False
        assert loader.image_storage_dir == Path("custom/path")
    
    def test_markitdown_available(self):
        """PdfLoader 要求 MarkItDown 可用。"""
        loader = PdfLoader()
        assert loader._markitdown is not None


class TestPdfLoaderValidation:
    """输入校验的测试。"""

    def test_load_requires_pdf_extension(self, tmp_path):
        """load() 对非 PDF 文件抛出 ValueError。"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")
        
        loader = PdfLoader()
        with pytest.raises(ValueError, match="not a PDF"):
            loader.load(txt_file)
    
    def test_load_nonexistent_file(self):
        """load() 对缺失文件抛出 FileNotFoundError。"""
        loader = PdfLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.pdf")


class TestPdfLoaderHelperMethods:
    """辅助方法的测试。"""

    def test_compute_file_hash_consistency(self, tmp_path):
        """_compute_file_hash 对相同内容返回一致的哈希。"""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"consistent content")
        
        loader = PdfLoader()
        hash1 = loader._compute_file_hash(test_file)
        hash2 = loader._compute_file_hash(test_file)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 十六进制长度

    def test_compute_file_hash_differs_for_different_content(self, tmp_path):
        """_compute_file_hash 对不同内容返回不同哈希。"""
        file1 = tmp_path / "file1.pdf"
        file2 = tmp_path / "file2.pdf"
        file1.write_bytes(b"content 1")
        file2.write_bytes(b"content 2")
        
        loader = PdfLoader()
        hash1 = loader._compute_file_hash(file1)
        hash2 = loader._compute_file_hash(file2)
        
        assert hash1 != hash2
    
    def test_extract_title_from_markdown_heading(self):
        """_extract_title 查找首个 Markdown 标题。"""
        loader = PdfLoader()
        
        text = "# Main Title\n\nParagraph text\n\n## Subtitle"
        title = loader._extract_title(text)
        assert title == "Main Title"
    
    def test_extract_title_from_first_line(self):
        """_extract_title 回退使用首个非空行。"""
        loader = PdfLoader()
        
        text = "First Line Title\n\nParagraph text"
        title = loader._extract_title(text)
        assert title == "First Line Title"
    
    def test_extract_title_handles_empty_text(self):
        """_extract_title 对空文本返回 None。"""
        loader = PdfLoader()
        
        text = ""
        title = loader._extract_title(text)
        assert title is None
    
    def test_generate_image_id_format(self):
        """_generate_image_id 生成一致的 ID 格式。"""
        image_id = PdfLoader._generate_image_id("abc123def456", 2, 0)
        assert image_id == "abc123de_2_0"


class TestPdfConversionCore:
    """使用真实 PDF 文件的核心 PDF 转换功能测试。"""

    def test_convert_simple_pdf_to_text(self):
        """将简单 PDF 转换为文本——校验核心 Markdown 转换。"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")
        
        loader = PdfLoader()
        doc = loader.load(SIMPLE_PDF)
        
        # 校验 Document 结构
        assert isinstance(doc, Document)
        assert doc.id.startswith("doc_")
        
        # 校验文本内容已被提取
        assert len(doc.text) > 0
        assert isinstance(doc.text, str)
        
        # 校验预期内容存在（来自生成的 PDF）
        text_lower = doc.text.lower()
        assert "sample" in text_lower or "document" in text_lower
        assert "test" in text_lower or "pdf" in text_lower
        
        # 校验 metadata
        assert doc.metadata["source_path"] == str(SIMPLE_PDF)
        assert doc.metadata["doc_type"] == "pdf"
        assert "doc_hash" in doc.metadata
    
    def test_extract_title_from_pdf(self):
        """校验从真实 PDF 提取标题。"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")
        
        loader = PdfLoader()
        doc = loader.load(SIMPLE_PDF)
        
        # 应提取标题（来自标题或首行）
        assert "title" in doc.metadata
        assert doc.metadata["title"] is not None
        assert len(doc.metadata["title"]) > 0
        
        # 标题应包含相关关键词
        title_lower = doc.metadata["title"].lower()
        assert "sample" in title_lower or "document" in title_lower
    
    def test_pdf_with_images_structure(self):
        """校验含图片的 PDF 在启用图片提取时被正确处理。"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")
        
        loader = PdfLoader(extract_images=True)
        doc = loader.load(IMAGES_PDF)
        
        # 校验基本结构
        assert isinstance(doc, Document)
        assert len(doc.text) > 0
        
        # 校验图片已被提取
        assert "images" in doc.metadata
        assert isinstance(doc.metadata["images"], list)
        
        if len(doc.metadata["images"]) > 0:
            # 校验图片 metadata 结构
            for img in doc.metadata["images"]:
                assert "id" in img
                assert "path" in img
                assert "page" in img
                assert "text_offset" in img
                assert "text_length" in img
                assert "position" in img
                
                # 校验图片文件存在
                img_path = Path(img["path"])
                assert img_path.exists(), f"Image file should exist: {img_path}"

                # 校验文本中存在占位符
                placeholder = f"[IMAGE: {img['id']}]"
                assert placeholder in doc.text, f"Placeholder {placeholder} should be in text"
    
    def test_image_extraction_disabled(self):
        """校验图片提取可被禁用。"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")
        
        loader = PdfLoader(extract_images=False)
        doc = loader.load(IMAGES_PDF)
        
        # 仍应提取文本
        assert len(doc.text) > 0

        # 不应包含 images metadata
        assert "images" not in doc.metadata or doc.metadata.get("images") == []
    
    def test_document_hash_consistency(self):
        """校验相同 PDF 产生相同的 Document 哈希。"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")
        
        loader = PdfLoader()
        
        # 加载同一文件两次
        doc1 = loader.load(SIMPLE_PDF)
        doc2 = loader.load(SIMPLE_PDF)

        # 应产生相同的哈希（以保证幂等性）
        assert doc1.metadata["doc_hash"] == doc2.metadata["doc_hash"]
        assert doc1.id == doc2.id
    
    def test_document_serialization(self):
        """校验加载的 Document 可被序列化。"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")
        
        loader = PdfLoader()
        doc = loader.load(SIMPLE_PDF)
        
        # 序列化为字典
        doc_dict = doc.to_dict()
        assert isinstance(doc_dict, dict)
        assert "id" in doc_dict
        assert "text" in doc_dict
        assert "metadata" in doc_dict
        
        # 校验 metadata 完整
        assert "source_path" in doc_dict["metadata"]
        assert "doc_type" in doc_dict["metadata"]
        assert doc_dict["metadata"]["doc_type"] == "pdf"
        
        # 校验可从字典重新创建
        doc_recreated = Document.from_dict(doc_dict)
        assert doc_recreated.id == doc.id
        assert doc_recreated.text == doc.text
        assert doc_recreated.metadata == doc.metadata
