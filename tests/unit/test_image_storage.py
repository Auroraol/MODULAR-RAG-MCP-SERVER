"""ImageStorage 模块的单元测试。"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ingestion.storage.image_storage import ImageStorage


@pytest.fixture
def temp_storage():
    """创建使用临时目录的 ImageStorage。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "image_index.db")
        images_root = str(Path(tmpdir) / "images")
        storage = ImageStorage(db_path=db_path, images_root=images_root)
        yield storage
        storage.close()


@pytest.fixture
def sample_image_data():
    """生成类似 PNG 的示例二进制数据。"""
    # 简单的 PNG 头（并非有效 PNG，但对测试已足够）
    return b'\x89PNG\r\n\x1a\n' + b'\x00' * 100


class TestImageStorageInitialization:
    """ImageStorage 初始化与设置的测试。"""

    def test_creates_database_file(self):
        """初始化时应创建数据库文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            storage = ImageStorage(db_path=db_path)
            
            assert Path(db_path).exists()
            storage.close()
    
    def test_creates_images_directory(self):
        """应创建图片根目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            images_root = str(Path(tmpdir) / "images")
            storage = ImageStorage(images_root=images_root)
            
            assert Path(images_root).exists()
            storage.close()
    
    def test_creates_nested_directories(self):
        """应创建嵌套的父目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "nested" / "path" / "db.db")
            storage = ImageStorage(db_path=db_path)
            
            assert Path(db_path).exists()
            storage.close()
    
    def test_database_schema_created(self, temp_storage):
        """数据库应具有正确的 schema。"""
        conn = sqlite3.connect(temp_storage.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='image_index'"
        )
        assert cursor.fetchone() is not None
        conn.close()
    
    def test_database_indexes_created(self, temp_storage):
        """数据库应具有 collection 和 doc_hash 索引。"""
        conn = sqlite3.connect(temp_storage.db_path)

        # 检查 idx_collection
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_collection'"
        )
        assert cursor.fetchone() is not None

        # 检查 idx_doc_hash
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_doc_hash'"
        )
        assert cursor.fetchone() is not None
        
        conn.close()
    
    def test_wal_mode_enabled(self, temp_storage):
        """数据库应使用 WAL 模式以支持并发。"""
        conn = sqlite3.connect(temp_storage.db_path)
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        conn.close()
        
        assert mode.lower() == "wal"


class TestSaveImage:
    """图片保存功能的测试。"""

    def test_save_image_from_bytes(self, temp_storage, sample_image_data):
        """应保存来自字节数据的图片。"""
        path = temp_storage.save_image(
            image_id="test_img_1",
            image_data=sample_image_data,
            collection="test_collection"
        )
        
        assert path is not None
        assert Path(path).exists()
        assert Path(path).read_bytes() == sample_image_data
    
    def test_save_image_from_file(self, temp_storage):
        """应通过复制源文件保存图片。"""
        # 创建源文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(b"test_image_data")
            source_path = tmp.name
        
        try:
            path = temp_storage.save_image(
                image_id="test_img_2",
                image_data=source_path,
                collection="test_collection"
            )
            
            assert path is not None
            assert Path(path).exists()
            assert Path(path).read_bytes() == b"test_image_data"
        finally:
            Path(source_path).unlink(missing_ok=True)
    
    def test_save_image_creates_collection_directory(self, temp_storage, sample_image_data):
        """collection 目录不存在时应创建。"""
        path = temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data,
            collection="new_collection"
        )
        
        assert "new_collection" in path
        collection_dir = Path(temp_storage.images_root) / "new_collection"
        assert collection_dir.exists()
    
    def test_save_image_with_custom_extension(self, temp_storage, sample_image_data):
        """应遵循自定义文件扩展名。"""
        path = temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data,
            collection="test",
            extension="jpg"
        )
        
        assert path.endswith(".jpg")
    
    def test_save_image_registers_in_database(self, temp_storage, sample_image_data):
        """应在数据库中登记图片 metadata。"""
        temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data,
            collection="test",
            doc_hash="abc123",
            page_num=5
        )
        
        conn = sqlite3.connect(temp_storage.db_path)
        cursor = conn.execute(
            "SELECT image_id, collection, doc_hash, page_num FROM image_index WHERE image_id = ?",
            ("img1",)
        )
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == "img1"
        assert row[1] == "test"
        assert row[2] == "abc123"
        assert row[3] == 5
    
    def test_save_image_idempotent(self, temp_storage, sample_image_data):
        """重复保存相同 image_id 应更新而非重复插入。"""
        # 首次保存
        temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data,
            collection="coll1"
        )
        
        # 使用不同 collection 再次保存
        new_data = b"updated_data"
        temp_storage.save_image(
            image_id="img1",
            image_data=new_data,
            collection="coll2"
        )

        # 应仅有一条记录
        conn = sqlite3.connect(temp_storage.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM image_index WHERE image_id = ?", ("img1",))
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

        # 文件应被更新
        path = temp_storage.get_image_path("img1")
        assert Path(path).read_bytes() == new_data

    def test_save_image_empty_id_raises_error(self, temp_storage, sample_image_data):
        """空 image_id 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="image_id cannot be empty"):
            temp_storage.save_image("", sample_image_data)
        
        with pytest.raises(ValueError, match="image_id cannot be empty"):
            temp_storage.save_image("   ", sample_image_data)
    
    def test_save_image_missing_source_file_raises_error(self, temp_storage):
        """源文件不存在时应抛出错误。"""
        with pytest.raises(IOError, match="Failed to save image"):
            temp_storage.save_image(
                image_id="img1",
                image_data="/nonexistent/path.png"
            )
    
    def test_save_image_uses_default_collection(self, temp_storage, sample_image_data):
        """未指定 collection 时应使用 'default'。"""
        path = temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data
        )
        
        assert "default" in path


class TestGetImagePath:
    """图片路径检索的测试。"""

    def test_get_existing_image_path(self, temp_storage, sample_image_data):
        """应返回已存在图片的路径。"""
        original_path = temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data
        )
        
        retrieved_path = temp_storage.get_image_path("img1")
        
        assert retrieved_path == original_path
    
    def test_get_nonexistent_image_path_returns_none(self, temp_storage):
        """不存在的图片应返回 None。"""
        path = temp_storage.get_image_path("nonexistent_id")
        
        assert path is None


class TestImageExists:
    """图片存在性检查的测试。"""

    def test_image_exists_returns_true(self, temp_storage, sample_image_data):
        """已存在的图片应返回 True。"""
        temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data
        )
        
        assert temp_storage.image_exists("img1") is True
    
    def test_image_exists_returns_false(self, temp_storage):
        """不存在的图片应返回 False。"""
        assert temp_storage.image_exists("nonexistent") is False


class TestListImages:
    """图片列表功能的测试。"""

    def test_list_all_images(self, temp_storage, sample_image_data):
        """无过滤条件时应列出所有图片。"""
        temp_storage.save_image("img1", sample_image_data, "coll1")
        temp_storage.save_image("img2", sample_image_data, "coll2")
        temp_storage.save_image("img3", sample_image_data, "coll1")
        
        images = temp_storage.list_images()
        
        assert len(images) == 3
        image_ids = {img["image_id"] for img in images}
        assert image_ids == {"img1", "img2", "img3"}
    
    def test_list_images_by_collection(self, temp_storage, sample_image_data):
        """应按 collection 过滤图片。"""
        temp_storage.save_image("img1", sample_image_data, "coll1")
        temp_storage.save_image("img2", sample_image_data, "coll2")
        temp_storage.save_image("img3", sample_image_data, "coll1")
        
        images = temp_storage.list_images(collection="coll1")
        
        assert len(images) == 2
        image_ids = {img["image_id"] for img in images}
        assert image_ids == {"img1", "img3"}
    
    def test_list_images_by_doc_hash(self, temp_storage, sample_image_data):
        """应按 Document 哈希过滤图片。"""
        temp_storage.save_image("img1", sample_image_data, "coll1", doc_hash="hash1")
        temp_storage.save_image("img2", sample_image_data, "coll1", doc_hash="hash2")
        temp_storage.save_image("img3", sample_image_data, "coll1", doc_hash="hash1")
        
        images = temp_storage.list_images(doc_hash="hash1")
        
        assert len(images) == 2
        image_ids = {img["image_id"] for img in images}
        assert image_ids == {"img1", "img3"}
    
    def test_list_images_with_both_filters(self, temp_storage, sample_image_data):
        """应组合 collection 与 doc_hash 过滤条件。"""
        temp_storage.save_image("img1", sample_image_data, "coll1", doc_hash="hash1")
        temp_storage.save_image("img2", sample_image_data, "coll2", doc_hash="hash1")
        temp_storage.save_image("img3", sample_image_data, "coll1", doc_hash="hash2")
        
        images = temp_storage.list_images(collection="coll1", doc_hash="hash1")
        
        assert len(images) == 1
        assert images[0]["image_id"] == "img1"
    
    def test_list_images_returns_complete_metadata(self, temp_storage, sample_image_data):
        """应返回每张图片的完整 metadata。"""
        temp_storage.save_image(
            image_id="img1",
            image_data=sample_image_data,
            collection="test",
            doc_hash="abc123",
            page_num=5
        )
        
        images = temp_storage.list_images()
        
        assert len(images) == 1
        img = images[0]
        assert img["image_id"] == "img1"
        assert img["collection"] == "test"
        assert img["doc_hash"] == "abc123"
        assert img["page_num"] == 5
        assert "file_path" in img
        assert "created_at" in img
    
    def test_list_images_empty_when_no_matches(self, temp_storage):
        """无匹配图片时应返回空列表。"""
        images = temp_storage.list_images(collection="nonexistent")
        
        assert images == []


class TestDeleteImage:
    """图片删除的测试。"""

    def test_delete_image_removes_from_database(self, temp_storage, sample_image_data):
        """应从数据库中移除图片。"""
        temp_storage.save_image("img1", sample_image_data)
        
        deleted = temp_storage.delete_image("img1")
        
        assert deleted is True
        assert temp_storage.image_exists("img1") is False
    
    def test_delete_image_removes_file(self, temp_storage, sample_image_data):
        """默认应删除图片文件。"""
        path = temp_storage.save_image("img1", sample_image_data)
        
        temp_storage.delete_image("img1", remove_file=True)
        
        assert not Path(path).exists()
    
    def test_delete_image_keeps_file_when_requested(self, temp_storage, sample_image_data):
        """remove_file=False 时应保留文件。"""
        path = temp_storage.save_image("img1", sample_image_data)

        temp_storage.delete_image("img1", remove_file=False)

        assert Path(path).exists()  # 文件仍然存在
        assert temp_storage.image_exists("img1") is False  # 但不在数据库中

    def test_delete_nonexistent_image_returns_false(self, temp_storage):
        """删除不存在的图片应返回 False。"""
        deleted = temp_storage.delete_image("nonexistent")
        
        assert deleted is False


class TestGetCollectionStats:
    """collection 统计的测试。"""

    def test_get_collection_stats_counts_images(self, temp_storage, sample_image_data):
        """应统计 collection 中的图片总数。"""
        temp_storage.save_image("img1", sample_image_data, "coll1")
        temp_storage.save_image("img2", sample_image_data, "coll1")
        temp_storage.save_image("img3", sample_image_data, "coll2")
        
        stats = temp_storage.get_collection_stats("coll1")
        
        assert stats["total_images"] == 2
    
    def test_get_collection_stats_calculates_size(self, temp_storage, sample_image_data):
        """应计算图片总大小。"""
        temp_storage.save_image("img1", sample_image_data, "coll1")
        temp_storage.save_image("img2", sample_image_data, "coll1")
        
        stats = temp_storage.get_collection_stats("coll1")
        
        # 应约为 2 * len(sample_image_data)
        assert stats["total_size_bytes"] > 0
        assert stats["total_size_bytes"] == 2 * len(sample_image_data)
    
    def test_get_collection_stats_empty_collection(self, temp_storage):
        """应处理空 collection。"""
        stats = temp_storage.get_collection_stats("empty")
        
        assert stats["total_images"] == 0
        assert stats["total_size_bytes"] == 0


class TestConcurrency:
    """并发访问模式的测试。"""

    def test_multiple_connections_can_read(self, temp_storage, sample_image_data):
        """WAL 模式应允许并发读取。"""
        temp_storage.save_image("img1", sample_image_data)

        # 模拟多个读取者
        path1 = temp_storage.get_image_path("img1")
        path2 = temp_storage.get_image_path("img1")

        assert path1 == path2

    def test_can_write_while_reading(self, temp_storage, sample_image_data):
        """WAL 模式应允许读取期间写入。"""
        temp_storage.save_image("img1", sample_image_data)

        # 读取已有图片
        path1 = temp_storage.get_image_path("img1")

        # 写入新图片
        temp_storage.save_image("img2", sample_image_data)

        # 读取应仍然可用
        path2 = temp_storage.get_image_path("img2")
        
        assert path1 is not None
        assert path2 is not None


class TestEdgeCases:
    """边界情况与错误处理的测试。"""

    def test_special_characters_in_image_id(self, temp_storage, sample_image_data):
        """应处理 image_id 中的特殊字符。"""
        # 注意：某些字符可能对文件系统不友好
        # 使用安全的特殊字符
        image_id = "doc_123-page_5.img_0"
        
        path = temp_storage.save_image(image_id, sample_image_data)
        
        assert temp_storage.image_exists(image_id)
        retrieved_path = temp_storage.get_image_path(image_id)
        assert retrieved_path == path
    
    def test_unicode_in_collection_name(self, temp_storage, sample_image_data):
        """应处理 collection 名称中的 Unicode。"""
        collection = "文档集合"
        
        path = temp_storage.save_image(
            "img1",
            sample_image_data,
            collection=collection
        )
        
        assert collection in path
        images = temp_storage.list_images(collection=collection)
        assert len(images) == 1
    
    def test_close_and_reopen(self, sample_image_data):
        """数据应在关闭/重新打开后持久保留。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            images_root = str(Path(tmpdir) / "images")

            # 创建并保存
            storage1 = ImageStorage(db_path=db_path, images_root=images_root)
            storage1.save_image("img1", sample_image_data, "coll1")
            storage1.close()

            # 重新打开并校验
            storage2 = ImageStorage(db_path=db_path, images_root=images_root)
            assert storage2.image_exists("img1")
            images = storage2.list_images()
            assert len(images) == 1
            storage2.close()
