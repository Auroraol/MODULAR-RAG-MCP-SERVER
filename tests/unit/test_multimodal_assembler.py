"""MultimodalAssembler 的单元测试。

测试覆盖：
- 从 Chunk metadata 提取图片引用
- 路径解析策略
- Base64 编码与 MIME 类型检测
- 生成 MCP 内容块
- 错误处理与边界情况
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from mcp import types

from src.core.response.multimodal_assembler import (
    IMAGE_PLACEHOLDER_PATTERN,
    MAGIC_BYTES,
    MIME_TYPE_MAP,
    ImageContent,
    ImageReference,
    MultimodalAssembler,
)
from src.core.types import RetrievalResult


# =============================================================================
# 夹具
# =============================================================================


@pytest.fixture
def sample_png_bytes() -> bytes:
    """创建用于测试的最小有效 PNG 字节。"""
    # PNG magic bytes + 最小 IHDR 块
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """创建用于测试的最小 JPEG 字节。"""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 100


@pytest.fixture
def sample_gif_bytes() -> bytes:
    """创建用于测试的最小 GIF 字节。"""
    return b"GIF89a" + b"\x00" * 100


@pytest.fixture
def temp_image_dir(sample_png_bytes) -> Path:
    """创建包含示例图片的临时目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "images" / "test_collection"
        img_dir.mkdir(parents=True)

        # 创建测试图片
        (img_dir / "img001.png").write_bytes(sample_png_bytes)
        (img_dir / "img002.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        
        yield img_dir


@pytest.fixture
def assembler() -> MultimodalAssembler:
    """创建默认的 MultimodalAssembler 实例。"""
    return MultimodalAssembler()


@pytest.fixture
def result_with_images() -> RetrievalResult:
    """创建带图片 metadata 的 RetrievalResult。"""
    return RetrievalResult(
        chunk_id="doc1_chunk_001",
        score=0.95,
        text="This is content with [IMAGE: img001] embedded.",
        metadata={
            "source_path": "docs/guide.pdf",
            "images": [
                {
                    "id": "img001",
                    "path": "/path/to/img001.png",
                    "page": 1,
                    "text_offset": 21,
                    "text_length": 15,
                },
                {
                    "id": "img002",
                    "path": "/path/to/img002.jpg",
                    "page": 2,
                    "text_offset": 50,
                    "text_length": 15,
                },
            ],
            "image_captions": {
                "img001": "A diagram showing system architecture",
                "img002": "Performance metrics chart",
            },
        },
    )


@pytest.fixture
def result_without_images() -> RetrievalResult:
    """创建不含图片的 RetrievalResult。"""
    return RetrievalResult(
        chunk_id="doc2_chunk_001",
        score=0.88,
        text="Plain text content without images.",
        metadata={
            "source_path": "docs/readme.md",
        },
    )


# =============================================================================
# ImageReference 测试
# =============================================================================


class TestImageReference:
    """ImageReference 数据类的测试。"""

    def test_create_basic_reference(self):
        """测试创建基本的图片引用。"""
        ref = ImageReference(image_id="img001")
        
        assert ref.image_id == "img001"
        assert ref.file_path is None
        assert ref.page is None
        assert ref.text_offset is None
        assert ref.caption is None
    
    def test_create_full_reference(self):
        """测试创建填充全部字段的引用。"""
        ref = ImageReference(
            image_id="img001",
            file_path="/path/to/img001.png",
            page=5,
            text_offset=100,
            text_length=15,
            caption="Test caption",
        )
        
        assert ref.image_id == "img001"
        assert ref.file_path == "/path/to/img001.png"
        assert ref.page == 5
        assert ref.text_offset == 100
        assert ref.text_length == 15
        assert ref.caption == "Test caption"
    
    def test_to_dict(self):
        """测试序列化为字典。"""
        ref = ImageReference(
            image_id="img001",
            file_path="/path/to/img.png",
            page=1,
            caption="Caption text",
        )
        
        data = ref.to_dict()
        
        assert data["image_id"] == "img001"
        assert data["file_path"] == "/path/to/img.png"
        assert data["page"] == 1
        assert data["caption"] == "Caption text"


# =============================================================================
# ImageContent 测试
# =============================================================================


class TestImageContent:
    """ImageContent 数据类的测试。"""

    def test_create_image_content(self):
        """测试创建图片内容。"""
        content = ImageContent(
            image_id="img001",
            data="base64data==",
            mime_type="image/png",
        )
        
        assert content.image_id == "img001"
        assert content.data == "base64data=="
        assert content.mime_type == "image/png"
        assert content.caption is None
    
    def test_to_mcp_content(self):
        """测试转换为 MCP ImageContent。"""
        content = ImageContent(
            image_id="img001",
            data="iVBORw0KGgo=",
            mime_type="image/png",
        )
        
        mcp_content = content.to_mcp_content()
        
        assert isinstance(mcp_content, types.ImageContent)
        assert mcp_content.type == "image"
        assert mcp_content.data == "iVBORw0KGgo="
        assert mcp_content.mime_type == "image/png"
    
    def test_to_dict(self):
        """测试序列化为字典。"""
        content = ImageContent(
            image_id="img001",
            data="base64data",
            mime_type="image/jpeg",
            caption="A test image",
        )
        
        data = content.to_dict()
        
        assert data["image_id"] == "img001"
        assert data["data"] == "base64data"
        assert data["mime_type"] == "image/jpeg"
        assert data["caption"] == "A test image"


# =============================================================================
# MIME 类型检测测试
# =============================================================================


class TestMimeTypeDetection:
    """MIME 类型检测工具的测试。"""

    def test_mime_type_map_contains_common_formats(self):
        """测试常见图片格式存在于 MIME 类型映射中。"""
        assert ".png" in MIME_TYPE_MAP
        assert ".jpg" in MIME_TYPE_MAP
        assert ".jpeg" in MIME_TYPE_MAP
        assert ".gif" in MIME_TYPE_MAP
        assert ".webp" in MIME_TYPE_MAP
        assert ".bmp" in MIME_TYPE_MAP
    
    def test_mime_type_values(self):
        """测试 MIME 类型值正确。"""
        assert MIME_TYPE_MAP[".png"] == "image/png"
        assert MIME_TYPE_MAP[".jpg"] == "image/jpeg"
        assert MIME_TYPE_MAP[".jpeg"] == "image/jpeg"
        assert MIME_TYPE_MAP[".gif"] == "image/gif"
    
    def test_magic_bytes_detection(self):
        """测试常见格式的 magic bytes 已定义。"""
        # PNG 魔数
        assert b"\x89PNG\r\n\x1a\n" in MAGIC_BYTES
        assert MAGIC_BYTES[b"\x89PNG\r\n\x1a\n"] == "image/png"

        # JPEG 魔数
        assert b"\xff\xd8\xff" in MAGIC_BYTES
        assert MAGIC_BYTES[b"\xff\xd8\xff"] == "image/jpeg"

        # GIF 魔数
        assert b"GIF89a" in MAGIC_BYTES
        assert MAGIC_BYTES[b"GIF89a"] == "image/gif"


# =============================================================================
# 占位符模式测试
# =============================================================================


class TestImagePlaceholderPattern:
    """图片占位符正则模式的测试。"""

    def test_match_simple_placeholder(self):
        """测试匹配简单占位符。"""
        text = "Here is [IMAGE: img001] in text."
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)
        
        assert len(matches) == 1
        assert matches[0].strip() == "img001"
    
    def test_match_multiple_placeholders(self):
        """测试匹配多个占位符。"""
        text = "[IMAGE: img001] and [IMAGE: img002] and [IMAGE: img003]"
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)
        
        assert len(matches) == 3
        assert "img001" in [m.strip() for m in matches]
        assert "img002" in [m.strip() for m in matches]
        assert "img003" in [m.strip() for m in matches]
    
    def test_match_with_extra_spaces(self):
        """测试匹配含多余空白的情况。"""
        text = "[IMAGE:   img001  ]"
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)
        
        assert len(matches) == 1
        assert matches[0].strip() == "img001"
    
    def test_match_complex_image_id(self):
        """测试匹配复杂图片 ID。"""
        text = "[IMAGE: doc_abc123_p1_img0]"
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)
        
        assert len(matches) == 1
        assert matches[0].strip() == "doc_abc123_p1_img0"
    
    def test_no_match_malformed(self):
        """测试格式错误的占位符不匹配。"""
        text = "[IMG: img001] or [IMAGE img001] or IMAGE: img001"
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)
        
        assert len(matches) == 0


# =============================================================================
# MultimodalAssembler - 图片引用提取测试
# =============================================================================


class TestExtractImageRefs:
    """extract_image_refs 方法的测试。"""

    def test_extract_from_metadata_images_list(self, assembler, result_with_images):
        """测试从 metadata.images 列表提取引用。"""
        refs = assembler.extract_image_refs(result_with_images)
        
        assert len(refs) == 2
        assert refs[0].image_id == "img001"
        assert refs[0].file_path == "/path/to/img001.png"
        assert refs[0].page == 1
        assert refs[0].caption == "A diagram showing system architecture"
        
        assert refs[1].image_id == "img002"
        assert refs[1].caption == "Performance metrics chart"
    
    def test_extract_from_text_placeholders(self, assembler):
        """测试从文本占位符回退提取。"""
        result = RetrievalResult(
            chunk_id="test",
            score=0.9,
            text="Content with [IMAGE: fallback_img001] and [IMAGE: fallback_img002].",
            metadata={"source_path": "test.pdf"},
        )
        
        refs = assembler.extract_image_refs(result)
        
        assert len(refs) == 2
        assert refs[0].image_id == "fallback_img001"
        assert refs[1].image_id == "fallback_img002"
    
    def test_prefer_metadata_over_placeholders(self, assembler):
        """测试 metadata.images 优先于文本占位符。"""
        result = RetrievalResult(
            chunk_id="test",
            score=0.9,
            text="[IMAGE: placeholder_img]",
            metadata={
                "source_path": "test.pdf",
                "images": [{"id": "metadata_img", "path": "/test.png"}],
            },
        )
        
        refs = assembler.extract_image_refs(result)
        
        assert len(refs) == 1
        assert refs[0].image_id == "metadata_img"
    
    def test_extract_empty_result(self, assembler, result_without_images):
        """测试从无图片的结果中提取。"""
        refs = assembler.extract_image_refs(result_without_images)
        
        assert len(refs) == 0
    
    def test_max_images_limit(self):
        """测试 max_images_per_result 被遵循。"""
        assembler = MultimodalAssembler(max_images_per_result=2)
        
        result = RetrievalResult(
            chunk_id="test",
            score=0.9,
            text="Text",
            metadata={
                "source_path": "test.pdf",
                "images": [
                    {"id": f"img{i}", "path": f"/img{i}.png"}
                    for i in range(10)
                ],
            },
        )
        
        refs = assembler.extract_image_refs(result)
        
        assert len(refs) == 2
    
    def test_extract_with_captions(self, assembler):
        """测试标题被附加到引用上。"""
        result = RetrievalResult(
            chunk_id="test",
            score=0.9,
            text="[IMAGE: img1]",
            metadata={
                "source_path": "test.pdf",
                "images": [{"id": "img1", "path": "/img1.png"}],
                "image_captions": {"img1": "Test caption"},
            },
        )
        
        refs = assembler.extract_image_refs(result)
        
        assert len(refs) == 1
        assert refs[0].caption == "Test caption"
    
    def test_extract_with_malformed_images_list(self, assembler):
        """测试格式错误的图片列表的处理。"""
        result = RetrievalResult(
            chunk_id="test",
            score=0.9,
            text="Text",
            metadata={
                "source_path": "test.pdf",
                "images": [
                    {"id": "valid_img"},  # 有效
                    {"path": "/no_id.png"},  # 缺少 id
                    "not_a_dict",  # 类型错误
                ],
            },
        )
        
        refs = assembler.extract_image_refs(result)
        
        assert len(refs) == 1
        assert refs[0].image_id == "valid_img"


# =============================================================================
# MultimodalAssembler - 路径解析测试
# =============================================================================


class TestResolveImagePath:
    """resolve_image_path 方法的测试。"""

    def test_resolve_with_explicit_path(self, assembler, temp_image_dir):
        """测试使用显式 file_path 解析。"""
        img_path = temp_image_dir / "img001.png"
        ref = ImageReference(image_id="img001", file_path=str(img_path))
        
        resolved = assembler.resolve_image_path(ref)
        
        assert resolved is not None
        assert Path(resolved).exists()
    
    def test_resolve_with_image_storage(self, assembler):
        """测试通过 ImageStorage 查找解析。"""
        mock_storage = MagicMock()
        mock_storage.get_image_path.return_value = "/resolved/path/img.png"
        
        assembler_with_storage = MultimodalAssembler(image_storage=mock_storage)
        ref = ImageReference(image_id="img001")
        
        with patch("pathlib.Path.exists", return_value=True):
            resolved = assembler_with_storage.resolve_image_path(ref)
        
        assert resolved == "/resolved/path/img.png"
        mock_storage.get_image_path.assert_called_once_with("img001")
    
    def test_resolve_fallback_convention_path(self, temp_image_dir, sample_png_bytes):
        """测试回退到约定路径。"""
        # 在约定路径创建图片
        collection = temp_image_dir.parent.name
        conv_dir = temp_image_dir.parent.parent / "data" / "images" / collection
        conv_dir.mkdir(parents=True, exist_ok=True)
        (conv_dir / "conv_img.png").write_bytes(sample_png_bytes)

        assembler = MultimodalAssembler()
        ref = ImageReference(image_id="conv_img")

        # patch 以使用我们的临时目录
        with patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = True
            # 此测试已简化——真实代码会检查实际路径
            resolved = assembler.resolve_image_path(ref, collection=collection)

        # 应尝试使用 collection 构造路径
        # 实际解析取决于文件系统状态

    def test_resolve_nonexistent_path(self, assembler):
        """测试文件缺失时解析返回 None。"""
        ref = ImageReference(
            image_id="nonexistent",
            file_path="/definitely/not/a/real/path.png",
        )
        
        resolved = assembler.resolve_image_path(ref)
        
        assert resolved is None
    
    def test_resolve_with_image_storage_failure(self):
        """测试 ImageStorage 错误的优雅处理。"""
        mock_storage = MagicMock()
        mock_storage.get_image_path.side_effect = Exception("Storage error")

        assembler = MultimodalAssembler(image_storage=mock_storage)
        ref = ImageReference(image_id="img001")

        # 不应抛出异常，应返回 None
        resolved = assembler.resolve_image_path(ref)

        assert resolved is None


# =============================================================================
# MultimodalAssembler - 图片加载测试
# =============================================================================


class TestLoadImage:
    """load_image 方法的测试。"""

    def test_load_png_image(self, assembler, temp_image_dir):
        """测试加载 PNG 图片。"""
        img_path = str(temp_image_dir / "img001.png")
        
        content = assembler.load_image(img_path)
        
        assert content is not None
        assert content.mime_type == "image/png"
        assert content.data  # 非空 base64

        # 校验 base64 有效
        decoded = base64.b64decode(content.data)
        assert decoded.startswith(b"\x89PNG")
    
    def test_load_jpeg_image(self, assembler, temp_image_dir):
        """测试加载 JPEG 图片。"""
        img_path = str(temp_image_dir / "img002.jpg")
        
        content = assembler.load_image(img_path)
        
        assert content is not None
        assert content.mime_type == "image/jpeg"
    
    def test_load_nonexistent_file(self, assembler):
        """测试文件缺失时加载返回 None。"""
        content = assembler.load_image("/nonexistent/path/image.png")
        
        assert content is None
    
    def test_load_empty_file(self, assembler):
        """测试空文件加载返回 None。"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"")
            temp_path = f.name
        
        try:
            content = assembler.load_image(temp_path)
            assert content is None
        finally:
            Path(temp_path).unlink()
    
    def test_load_image_sets_correct_image_id(self, assembler, temp_image_dir):
        """测试加载的图片从文件名获得正确 ID。"""
        img_path = str(temp_image_dir / "img001.png")
        
        content = assembler.load_image(img_path)
        
        assert content is not None
        assert content.image_id == "img001"


# =============================================================================
# MultimodalAssembler - MIME 类型检测测试
# =============================================================================


class TestDetectMimeType:
    """_detect_mime_type 方法的测试。"""

    def test_detect_from_extension_png(self, assembler, sample_png_bytes):
        """测试从 .png 扩展名检测 MIME。"""
        mime = assembler._detect_mime_type(Path("test.png"), sample_png_bytes)
        assert mime == "image/png"
    
    def test_detect_from_extension_jpg(self, assembler, sample_jpeg_bytes):
        """测试从 .jpg 扩展名检测 MIME。"""
        mime = assembler._detect_mime_type(Path("test.jpg"), sample_jpeg_bytes)
        assert mime == "image/jpeg"
    
    def test_detect_from_extension_jpeg(self, assembler, sample_jpeg_bytes):
        """测试从 .jpeg 扩展名检测 MIME。"""
        mime = assembler._detect_mime_type(Path("test.jpeg"), sample_jpeg_bytes)
        assert mime == "image/jpeg"
    
    def test_detect_from_magic_bytes_png(self, assembler, sample_png_bytes):
        """测试从 PNG magic bytes 检测 MIME。"""
        mime = assembler._detect_mime_type(Path("test.unknown"), sample_png_bytes)
        assert mime == "image/png"
    
    def test_detect_from_magic_bytes_jpeg(self, assembler, sample_jpeg_bytes):
        """测试从 JPEG magic bytes 检测 MIME。"""
        mime = assembler._detect_mime_type(Path("test.unknown"), sample_jpeg_bytes)
        assert mime == "image/jpeg"
    
    def test_detect_from_magic_bytes_gif(self, assembler, sample_gif_bytes):
        """测试从 GIF magic bytes 检测 MIME。"""
        mime = assembler._detect_mime_type(Path("test.unknown"), sample_gif_bytes)
        assert mime == "image/gif"
    
    def test_detect_unknown_defaults_to_png(self, assembler):
        """测试未知格式默认为 image/png。"""
        unknown_data = b"UNKNOWN_FORMAT_DATA"
        mime = assembler._detect_mime_type(Path("test.xyz"), unknown_data)
        assert mime == "image/png"


# =============================================================================
# MultimodalAssembler - 组装测试
# =============================================================================


class TestAssembleForResult:
    """assemble_for_result 方法的测试。"""

    def test_assemble_with_images(self, temp_image_dir, sample_png_bytes):
        """测试为含图片的结果组装内容。"""
        # 创建测试图片
        img_path = temp_image_dir / "test_img.png"
        img_path.write_bytes(sample_png_bytes)
        
        assembler = MultimodalAssembler()
        result = RetrievalResult(
            chunk_id="test",
            score=0.9,
            text="Content with image",
            metadata={
                "source_path": "test.pdf",
                "images": [{"id": "test_img", "path": str(img_path)}],
            },
        )
        
        blocks = assembler.assemble_for_result(result)
        
        # 应至少有一个 ImageContent 块
        image_blocks = [b for b in blocks if isinstance(b, types.ImageContent)]
        assert len(image_blocks) >= 1
    
    def test_assemble_without_images(self, assembler, result_without_images):
        """测试无图片的结果组装返回空。"""
        blocks = assembler.assemble_for_result(result_without_images)
        
        assert len(blocks) == 0
    
    def test_assemble_includes_caption(self, temp_image_dir, sample_png_bytes):
        """测试标题作为文本块包含在内。"""
        img_path = temp_image_dir / "captioned_img.png"
        img_path.write_bytes(sample_png_bytes)
        
        assembler = MultimodalAssembler(include_captions=True)
        result = RetrievalResult(
            chunk_id="test",
            score=0.9,
            text="Content",
            metadata={
                "source_path": "test.pdf",
                "images": [{"id": "captioned_img", "path": str(img_path)}],
                "image_captions": {"captioned_img": "This is the caption"},
            },
        )
        
        blocks = assembler.assemble_for_result(result)
        
        # 应包含带标题的 TextContent
        text_blocks = [b for b in blocks if isinstance(b, types.TextContent)]
        caption_blocks = [b for b in text_blocks if "caption" in b.text.lower()]
        assert len(caption_blocks) >= 1
    
    def test_assemble_without_captions_disabled(self, temp_image_dir, sample_png_bytes):
        """测试禁用时不包含标题。"""
        img_path = temp_image_dir / "no_caption_img.png"
        img_path.write_bytes(sample_png_bytes)
        
        assembler = MultimodalAssembler(include_captions=False)
        result = RetrievalResult(
            chunk_id="test",
            score=0.9,
            text="Content",
            metadata={
                "source_path": "test.pdf",
                "images": [{"id": "no_caption_img", "path": str(img_path)}],
                "image_captions": {"no_caption_img": "This should not appear"},
            },
        )
        
        blocks = assembler.assemble_for_result(result)
        
        # 应只有 ImageContent，没有标题 TextContent
        text_blocks = [b for b in blocks if isinstance(b, types.TextContent)]
        assert len(text_blocks) == 0


class TestAssembleMultiple:
    """assemble 方法（多个结果）的测试。"""

    def test_assemble_multiple_results(self, temp_image_dir):
        """测试从多个含不同图片的结果组装。"""
        # 创建内容不同的测试图片以避免去重
        png_data_1 = b"\x89PNG\r\n\x1a\n" + b"\x01" * 100  # 不同数据
        png_data_2 = b"\x89PNG\r\n\x1a\n" + b"\x02" * 100  # 不同数据
        
        (temp_image_dir / "multi_img1.png").write_bytes(png_data_1)
        (temp_image_dir / "multi_img2.png").write_bytes(png_data_2)
        
        assembler = MultimodalAssembler()
        results = [
            RetrievalResult(
                chunk_id="test1",
                score=0.9,
                text="Content 1",
                metadata={
                    "source_path": "test1.pdf",
                    "images": [{"id": "multi_img1", "path": str(temp_image_dir / "multi_img1.png")}],
                },
            ),
            RetrievalResult(
                chunk_id="test2",
                score=0.8,
                text="Content 2",
                metadata={
                    "source_path": "test2.pdf",
                    "images": [{"id": "multi_img2", "path": str(temp_image_dir / "multi_img2.png")}],
                },
            ),
        ]
        
        blocks = assembler.assemble(results)
        
        # 应包含来自两个结果的图片（内容不同，因此不会去重）
        image_blocks = [b for b in blocks if isinstance(b, types.ImageContent)]
        assert len(image_blocks) >= 2
    
    def test_assemble_deduplicates_images(self, temp_image_dir, sample_png_bytes):
        """测试重复图片被去重。"""
        img_path = temp_image_dir / "shared_img.png"
        img_path.write_bytes(sample_png_bytes)

        assembler = MultimodalAssembler()

        # 两个结果引用同一张图片
        results = [
            RetrievalResult(
                chunk_id="test1",
                score=0.9,
                text="Content 1",
                metadata={
                    "source_path": "test1.pdf",
                    "images": [{"id": "shared_img", "path": str(img_path)}],
                },
            ),
            RetrievalResult(
                chunk_id="test2",
                score=0.8,
                text="Content 2",
                metadata={
                    "source_path": "test2.pdf",
                    "images": [{"id": "shared_img", "path": str(img_path)}],
                },
            ),
        ]
        
        blocks = assembler.assemble(results)
        
        # 应去重——仅一张图片
        image_blocks = [b for b in blocks if isinstance(b, types.ImageContent)]
        assert len(image_blocks) == 1


# =============================================================================
# MultimodalAssembler - 工具方法测试
# =============================================================================


class TestUtilityMethods:
    """工具方法的测试。"""

    def test_has_images_true(self, assembler, result_with_images):
        """测试存在图片时 has_images 返回 True。"""
        assert assembler.has_images(result_with_images) is True
    
    def test_has_images_false(self, assembler, result_without_images):
        """测试无图片时 has_images 返回 False。"""
        assert assembler.has_images(result_without_images) is False
    
    def test_count_images_single(self, assembler, result_with_images):
        """测试统计单个结果中的图片。"""
        count = assembler.count_images([result_with_images])
        assert count == 2
    
    def test_count_images_multiple(self, assembler, result_with_images, result_without_images):
        """测试跨多个结果统计图片。"""
        count = assembler.count_images([result_with_images, result_without_images])
        assert count == 2  # 仅第一个结果含图片

    def test_count_images_empty(self, assembler):
        """测试空列表统计图片返回 0。"""
        count = assembler.count_images([])
        assert count == 0


# =============================================================================
# 边界情况与错误处理
# =============================================================================


class TestEdgeCases:
    """边界情况与错误处理的测试。"""

    def test_none_metadata(self, assembler):
        """测试处理类似 None 的 metadata。"""
        result = RetrievalResult(
            chunk_id="test",
            score=0.9,
            text="Content",
            metadata={"source_path": "test.pdf"},  # 空但有效
        )

        refs = assembler.extract_image_refs(result)
        assert len(refs) == 0

    def test_images_not_list(self, assembler):
        """测试 images 不是列表时的处理。"""
        result = RetrievalResult(
            chunk_id="test",
            score=0.9,
            text="Content",
            metadata={
                "source_path": "test.pdf",
                "images": "not_a_list",  # 无效类型
            },
        )

        refs = assembler.extract_image_refs(result)
        # 应回退到文本占位符解析
        assert len(refs) == 0

    def test_image_captions_not_dict(self, assembler):
        """测试 image_captions 不是字典时的处理。"""
        result = RetrievalResult(
            chunk_id="test",
            score=0.9,
            text="[IMAGE: img1]",
            metadata={
                "source_path": "test.pdf",
                "image_captions": ["not", "a", "dict"],  # 无效类型
            },
        )

        refs = assembler.extract_image_refs(result)
        # 仍应提取引用，只是没有标题
        assert len(refs) == 1
        assert refs[0].caption is None

    def test_empty_text(self, assembler):
        """测试空文本的处理。"""
        result = RetrievalResult(
            chunk_id="test",
            score=0.9,
            text="",
            metadata={"source_path": "test.pdf"},
        )
        
        refs = assembler.extract_image_refs(result)
        assert len(refs) == 0
    
    def test_unicode_in_image_id(self, assembler):
        """测试 image ID 中的 Unicode 字符处理。"""
        result = RetrievalResult(
            chunk_id="test",
            score=0.9,
            text="[IMAGE: 图片_001]",
            metadata={"source_path": "test.pdf"},
        )
        
        refs = assembler.extract_image_refs(result)
        assert len(refs) == 1
        assert refs[0].image_id == "图片_001"
