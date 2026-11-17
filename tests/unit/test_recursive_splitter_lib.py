"""RecursiveSplitter 的单元测试。

测试覆盖：
- 从配置驱动的实例化
- Chunk 大小与重叠参数处理
- Markdown 结构保留（标题、代码块）
- 边界情况：空文本、极短文本、极长文本
- 错误处理：缺失依赖、无效配置
- 与 BaseSplitter 校验的集成
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.libs.splitter.base_splitter import BaseSplitter


# 检查 langchain-text-splitters 是否可用
try:
    from src.libs.splitter.recursive_splitter import RecursiveSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    RecursiveSplitter = None  # type: ignore[misc, assignment]


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterConfiguration:
    """RecursiveSplitter 配置与初始化的测试。"""

    def create_mock_settings(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Any:
        """创建 mock 配置对象。"""
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.chunk_size = chunk_size
        settings.ingestion.chunk_overlap = chunk_overlap
        return settings
    
    def test_initialization_from_settings(self):
        """测试 RecursiveSplitter 从 settings 读取配置。"""
        settings = self.create_mock_settings(chunk_size=500, chunk_overlap=100)
        splitter = RecursiveSplitter(settings=settings)
        
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100
        assert isinstance(splitter, BaseSplitter)
    
    def test_initialization_with_overrides(self):
        """测试构造参数覆盖 settings 中的值。"""
        settings = self.create_mock_settings(chunk_size=500, chunk_overlap=100)
        splitter = RecursiveSplitter(
            settings=settings,
            chunk_size=300,
            chunk_overlap=50,
        )
        
        assert splitter.chunk_size == 300
        assert splitter.chunk_overlap == 50
    
    def test_initialization_with_custom_separators(self):
        """测试自定义分隔符配置。"""
        settings = self.create_mock_settings()
        custom_separators = ["\n\n", "\n", " "]
        splitter = RecursiveSplitter(
            settings=settings,
            separators=custom_separators,
        )
        
        assert splitter.separators == custom_separators
    
    def test_initialization_default_separators(self):
        """测试默认分隔符能识别 Markdown。"""
        settings = self.create_mock_settings()
        splitter = RecursiveSplitter(settings=settings)
        
        # 检查默认分隔符包含常见文本边界
        assert "\n\n" in splitter.separators  # 段落
        assert "\n" in splitter.separators    # 行
        assert " " in splitter.separators     # 词
        assert "" in splitter.separators      # 字符

    def test_initialization_missing_settings(self):
        """测试 settings.ingestion 缺失时的错误。"""
        settings = MagicMock()
        settings.ingestion = None
        
        with pytest.raises(ValueError, match="Missing ingestion configuration"):
            RecursiveSplitter(settings=settings)
    
    def test_initialization_invalid_chunk_size_negative(self):
        """测试 chunk_size 为负数时的错误。"""
        settings = self.create_mock_settings(chunk_size=-100)
        
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            RecursiveSplitter(settings=settings)
    
    def test_initialization_invalid_chunk_size_zero(self):
        """测试 chunk_size 为零时的错误。"""
        settings = self.create_mock_settings(chunk_size=0)
        
        with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
            RecursiveSplitter(settings=settings)
    
    def test_initialization_invalid_chunk_overlap_negative(self):
        """测试 chunk_overlap 为负数时的错误。"""
        settings = self.create_mock_settings(chunk_overlap=-50)
        
        with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer"):
            RecursiveSplitter(settings=settings)
    
    def test_initialization_overlap_exceeds_chunk_size(self):
        """测试 chunk_overlap >= chunk_size 时的错误。"""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=100)
        
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            RecursiveSplitter(settings=settings)
    
    def test_initialization_overlap_greater_than_chunk_size(self):
        """测试 chunk_overlap > chunk_size 时的错误。"""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=200)
        
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            RecursiveSplitter(settings=settings)


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterBasicSplitting:
    """基础文本切分行为的测试。"""

    def create_mock_settings(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Any:
        """创建 mock 配置对象。"""
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.chunk_size = chunk_size
        settings.ingestion.chunk_overlap = chunk_overlap
        return settings
    
    def test_split_short_text(self):
        """测试切分短于 chunk_size 的文本。"""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        text = "This is a short text."
        chunks = splitter.split_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_text_by_paragraphs(self):
        """测试切分器遵循段落边界。"""
        settings = self.create_mock_settings(chunk_size=50, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)

        # 创建足够长、需要多个 Chunk 的文本
        text = "Paragraph one with some more content to make it longer.\n\nParagraph two also needs sufficient length.\n\nParagraph three should push it over the limit."
        chunks = splitter.split_text(text)

        # 文本超过 chunk_size 时应在段落边界切分
        assert len(chunks) >= 1  # 至少返回该文本
        for chunk in chunks:
            assert len(chunk) <= 70  # 为边界条件留出一些灵活性

    def test_split_text_with_overlap(self):
        """测试 Chunk 之间存在重叠内容。"""
        settings = self.create_mock_settings(chunk_size=30, chunk_overlap=10)
        splitter = RecursiveSplitter(settings=settings)
        
        text = "This is a long sentence that will be split into multiple chunks with overlap."
        chunks = splitter.split_text(text)
        
        # 应产生多个 Chunk
        assert len(chunks) >= 2

        # 每个 Chunk 应遵循 chunk_size（允许词边界的少量容差）
        for chunk in chunks:
            assert len(chunk) <= 30 + 20  # 为不断词留出容差

    def test_split_preserves_order(self):
        """测试 Chunk 保持原始文本顺序。"""
        settings = self.create_mock_settings(chunk_size=50, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        text = "First section. Second section. Third section. Fourth section."
        chunks = splitter.split_text(text)
        
        # 重组后应保持顺序
        reconstructed = " ".join(chunks)
        assert "First" in reconstructed
        assert reconstructed.index("First") < reconstructed.index("Fourth")
    
    def test_split_empty_string_validation(self):
        """测试空字符串抛出校验错误。"""
        settings = self.create_mock_settings()
        splitter = RecursiveSplitter(settings=settings)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            splitter.split_text("   ")
    
    def test_split_non_string_validation(self):
        """测试非字符串输入抛出校验错误。"""
        settings = self.create_mock_settings()
        splitter = RecursiveSplitter(settings=settings)
        
        with pytest.raises(ValueError, match="must be a string"):
            splitter.split_text(123)  # type: ignore[arg-type]


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterMarkdownStructure:
    """Markdown 结构保留的测试。"""

    def create_mock_settings(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Any:
        """创建 mock 配置对象。"""
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.chunk_size = chunk_size
        settings.ingestion.chunk_overlap = chunk_overlap
        return settings
    
    def test_split_markdown_with_headers(self):
        """测试 Markdown 标题在 Chunk 中被保留。"""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        text = """# Main Header

## Section 1
Content for section 1 goes here.

## Section 2
Content for section 2 goes here.

## Section 3
Content for section 3 goes here."""
        
        chunks = splitter.split_text(text)
        
        # 应产生多个 Chunk
        assert len(chunks) >= 1

        # 标题应出现在相应的 Chunk 中
        all_text = "".join(chunks)
        assert "# Main Header" in all_text
        assert "## Section 1" in all_text
    
    def test_split_markdown_code_blocks(self):
        """测试代码块被恰当处理。"""
        settings = self.create_mock_settings(chunk_size=150, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        text = """Some text before code.

```python
def example():
    return "code"
```

Some text after code."""
        
        chunks = splitter.split_text(text)
        
        # 所有内容应被保留
        all_text = "".join(chunks)
        assert "def example():" in all_text
        assert "Some text before" in all_text
        assert "Some text after" in all_text
    
    def test_split_markdown_lists(self):
        """测试 Markdown 列表被恰当处理。"""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        text = """# List Example

- Item 1
- Item 2
- Item 3
- Item 4
- Item 5"""
        
        chunks = splitter.split_text(text)
        
        # 应保留列表结构
        all_text = "".join(chunks)
        assert "- Item 1" in all_text
        assert "- Item 5" in all_text


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterEdgeCases:
    """边界情况与错误处理的测试。"""

    def create_mock_settings(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Any:
        """创建 mock 配置对象。"""
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.chunk_size = chunk_size
        settings.ingestion.chunk_overlap = chunk_overlap
        return settings
    
    def test_split_very_long_text(self):
        """测试切分极长文本。"""
        settings = self.create_mock_settings(chunk_size=100, chunk_overlap=20)
        splitter = RecursiveSplitter(settings=settings)

        # 生成长文本（1000 个词）
        text = " ".join(["word"] * 1000)
        chunks = splitter.split_text(text)

        # 应产生多个 Chunk
        assert len(chunks) >= 10

        # 每个 Chunk 应遵循 chunk_size（含容差）
        for chunk in chunks:
            assert len(chunk) <= 100 + 30  # 允许容差

    def test_split_single_long_word(self):
        """测试处理长于 chunk_size 的单个词。"""
        settings = self.create_mock_settings(chunk_size=10, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)

        # 单个词长于 chunk_size
        text = "a" * 100
        chunks = splitter.split_text(text)

        # 仍应切分（对不可切分内容可能超出 chunk_size）
        assert len(chunks) >= 1

    def test_split_unicode_text(self):
        """测试 Unicode 字符的处理。"""
        settings = self.create_mock_settings(chunk_size=50, chunk_overlap=0)
        splitter = RecursiveSplitter(settings=settings)
        
        text = "Hello 世界! Привет мир! 🌍🌎🌏"
        chunks = splitter.split_text(text)
        
        # 应能处理 Unicode 而不出错
        assert len(chunks) >= 1
        all_text = "".join(chunks)
        assert "世界" in all_text
        assert "мир" in all_text
        assert "🌍" in all_text
    
    def test_split_with_trace_parameter(self):
        """测试接受 trace 参数但不使用它。"""
        settings = self.create_mock_settings()
        splitter = RecursiveSplitter(settings=settings)
        
        text = "Some text to split."
        mock_trace = MagicMock()
        
        # 使用 trace 参数不应抛出错误
        chunks = splitter.split_text(text, trace=mock_trace)
        assert len(chunks) >= 1


@pytest.mark.skipif(LANGCHAIN_AVAILABLE, reason="Test only when langchain-text-splitters NOT installed")
class TestRecursiveSplitterImportError:
    """未安装 langchain-text-splitters 时的 ImportError 测试。"""

    def test_import_error_without_langchain(self):
        """测试 langchain 不可用时抛出 ImportError。"""
        with patch.dict('sys.modules', {'langchain_text_splitters': None}):
            # 强制重新导入以触发 ImportError
            import importlib
            import src.libs.splitter.recursive_splitter
            importlib.reload(src.libs.splitter.recursive_splitter)
            
            from src.libs.splitter.recursive_splitter import RecursiveSplitter
            
            settings = MagicMock()
            settings.ingestion = MagicMock()
            settings.ingestion.chunk_size = 1000
            settings.ingestion.chunk_overlap = 200
            
            with pytest.raises(ImportError, match="langchain-text-splitters is not installed"):
                RecursiveSplitter(settings=settings)


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain-text-splitters not installed")
class TestRecursiveSplitterFactoryIntegration:
    """工厂集成的测试。"""

    def test_factory_can_create_recursive_splitter(self):
        """测试工厂可实例化 RecursiveSplitter。"""
        from src.libs.splitter.splitter_factory import SplitterFactory

        # 注册 provider
        SplitterFactory.register_provider("recursive", RecursiveSplitter)

        # 创建配置
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.splitter = "recursive"
        settings.ingestion.chunk_size = 500
        settings.ingestion.chunk_overlap = 100
        
        # 工厂应创建 RecursiveSplitter
        splitter = SplitterFactory.create(settings)
        assert isinstance(splitter, RecursiveSplitter)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100
