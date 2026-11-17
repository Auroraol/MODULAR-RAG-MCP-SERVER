"""文件完整性校验模块的单元测试。

测试涵盖：
- SHA256 哈希计算一致性
- 跳过逻辑（mark_success 后 should_skip 返回 True）
- 数据库创建与持久化
- 并发写入支持（SQLite WAL 模式）
- 无效文件的错误处理
- 幂等操作
"""

import hashlib
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.libs.loader.file_integrity import (
    FileIntegrityChecker,
    SQLiteIntegrityChecker,
)


@pytest.fixture
def temp_db():
    """创建用于测试的临时数据库。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield str(db_path)


@pytest.fixture
def temp_file():
    """创建临时测试文件。"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Test content for hashing")
        temp_path = f.name
    
    yield temp_path
    
    # 清理
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def checker(temp_db):
    """创建文件完整性校验器实例。"""
    return SQLiteIntegrityChecker(db_path=temp_db)


class TestSQLiteIntegrityChecker:
    """SQLiteIntegrityChecker 的测试套件。"""

    def test_init_creates_database(self, temp_db):
        """测试初始化会创建数据库文件。"""
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        
        assert Path(temp_db).exists()
        assert Path(temp_db).is_file()
    
    def test_init_creates_parent_directories(self):
        """测试初始化会创建父目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "nested" / "test.db"
            checker = SQLiteIntegrityChecker(db_path=str(db_path))
            
            assert db_path.exists()
            assert db_path.parent.exists()
    
    def test_database_schema_created(self, temp_db, checker):
        """测试数据库 schema 被正确初始化。"""
        conn = sqlite3.connect(temp_db)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ingestion_history'"
            )
            assert cursor.fetchone() is not None
            
            # 检查 WAL 模式已启用
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            conn.close()
    
    def test_compute_sha256_consistency(self, checker, temp_file):
        """测试两次计算哈希得到相同结果。"""
        hash1 = checker.compute_sha256(temp_file)
        hash2 = checker.compute_sha256(temp_file)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 产生 64 个十六进制字符
        assert isinstance(hash1, str)
    
    def test_compute_sha256_matches_hashlib(self, checker, temp_file):
        """测试计算的哈希与直接使用 hashlib 的结果一致。"""
        # 使用校验器计算
        checker_hash = checker.compute_sha256(temp_file)

        # 直接使用 hashlib 计算
        with open(temp_file, "rb") as f:
            content = f.read()
            expected_hash = hashlib.sha256(content).hexdigest()
        
        assert checker_hash == expected_hash
    
    def test_compute_sha256_file_not_found(self, checker):
        """测试对不存在的文件计算哈希抛出错误。"""
        with pytest.raises(FileNotFoundError, match="File not found"):
            checker.compute_sha256("/nonexistent/file.txt")
    
    def test_compute_sha256_directory_raises_error(self, checker, temp_db):
        """测试对目录计算哈希抛出错误。"""
        dir_path = Path(temp_db).parent
        
        with pytest.raises(IOError, match="Path is not a file"):
            checker.compute_sha256(str(dir_path))
    
    def test_should_skip_new_file(self, checker):
        """测试新文件哈希的 should_skip 返回 False。"""
        fake_hash = "a" * 64
        assert checker.should_skip(fake_hash) is False
    
    def test_mark_success_and_should_skip(self, checker, temp_file):
        """测试标记成功后 should_skip 返回 True。"""
        file_hash = checker.compute_sha256(temp_file)

        # 初始时不应跳过
        assert checker.should_skip(file_hash) is False

        # 标记成功
        checker.mark_success(file_hash, temp_file)

        # 现在应跳过
        assert checker.should_skip(file_hash) is True
    
    def test_mark_success_with_collection(self, checker, temp_file):
        """测试带 collection 名称标记成功。"""
        file_hash = checker.compute_sha256(temp_file)
        collection = "test_collection"

        checker.mark_success(file_hash, temp_file, collection=collection)

        # 验证 collection 已被存储
        conn = sqlite3.connect(checker.db_path)
        try:
            cursor = conn.execute(
                "SELECT collection FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == collection
        finally:
            conn.close()
    
    def test_mark_success_idempotent(self, checker, temp_file):
        """测试多次标记成功是安全的。"""
        file_hash = checker.compute_sha256(temp_file)

        # 标记成功三次
        checker.mark_success(file_hash, temp_file)
        checker.mark_success(file_hash, temp_file)
        checker.mark_success(file_hash, temp_file)

        # 应仍然跳过
        assert checker.should_skip(file_hash) is True

        # 验证仅存在一行记录
        conn = sqlite3.connect(checker.db_path)
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            count = cursor.fetchone()[0]
            assert count == 1
        finally:
            conn.close()
    
    def test_mark_failed_does_not_skip(self, checker, temp_file):
        """测试标记失败后不会跳过。"""
        file_hash = checker.compute_sha256(temp_file)

        checker.mark_failed(file_hash, temp_file, "Test error")

        # 失败的文件不应被跳过（允许重试）
        assert checker.should_skip(file_hash) is False
    
    def test_mark_failed_stores_error_message(self, checker, temp_file):
        """测试失败文件的错误消息被存储。"""
        file_hash = checker.compute_sha256(temp_file)
        error_msg = "Test error message"

        checker.mark_failed(file_hash, temp_file, error_msg)

        # 验证错误消息已存储
        conn = sqlite3.connect(checker.db_path)
        try:
            cursor = conn.execute(
                "SELECT error_msg, status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == error_msg
            assert result[1] == "failed"
        finally:
            conn.close()
    
    def test_mark_success_after_failure_clears_error(self, checker, temp_file):
        """测试失败后标记成功会清除错误消息。"""
        file_hash = checker.compute_sha256(temp_file)

        # 先标记失败
        checker.mark_failed(file_hash, temp_file, "Initial error")
        assert checker.should_skip(file_hash) is False

        # 再标记成功
        checker.mark_success(file_hash, temp_file)
        assert checker.should_skip(file_hash) is True

        # 验证错误已清除且状态为成功
        conn = sqlite3.connect(checker.db_path)
        try:
            cursor = conn.execute(
                "SELECT error_msg, status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            assert result is not None
            assert result[0] is None  # error_msg 应为 NULL
            assert result[1] == "success"
        finally:
            conn.close()
    
    def test_timestamps_are_recorded(self, checker, temp_file):
        """测试 processed_at 与 updated_at 时间戳被记录。"""
        file_hash = checker.compute_sha256(temp_file)

        checker.mark_success(file_hash, temp_file)
        
        conn = sqlite3.connect(checker.db_path)
        try:
            cursor = conn.execute(
                "SELECT processed_at, updated_at FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            assert result is not None
            assert result[0] is not None  # processed_at
            assert result[1] is not None  # updated_at
        finally:
            conn.close()
    
    def test_multiple_files_independent(self, checker):
        """测试多个文件可独立追踪。"""
        # 创建多个临时文件
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f1:
            f1.write("Content 1")
            file1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f2:
            f2.write("Content 2")
            file2 = f2.name
        
        try:
            hash1 = checker.compute_sha256(file1)
            hash2 = checker.compute_sha256(file2)
            
            # 哈希应不同
            assert hash1 != hash2

            # 仅将第一个文件标记为成功
            checker.mark_success(hash1, file1)

            # 检查跳过状态
            assert checker.should_skip(hash1) is True
            assert checker.should_skip(hash2) is False
        finally:
            Path(file1).unlink(missing_ok=True)
            Path(file2).unlink(missing_ok=True)
    
    def test_compute_sha256_large_file(self, checker):
        """测试能正确处理大文件（分块读取）。"""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            # 写入 1MB 数据
            data = b"x" * (1024 * 1024)
            f.write(data)
            large_file = f.name
        
        try:
            # 不应抛出内存错误
            file_hash = checker.compute_sha256(large_file)

            # 验证哈希
            expected_hash = hashlib.sha256(data).hexdigest()
            assert file_hash == expected_hash
        finally:
            Path(large_file).unlink(missing_ok=True)
    
    def test_database_persists_across_instances(self, temp_db, temp_file):
        """测试创建新校验器实例时数据仍然持久化。"""
        # 创建第一个校验器并标记文件
        checker1 = SQLiteIntegrityChecker(db_path=temp_db)
        file_hash = checker1.compute_sha256(temp_file)
        checker1.mark_success(file_hash, temp_file)

        # 创建新的校验器实例

        # 应仍然跳过
        checker2 = SQLiteIntegrityChecker(db_path=temp_db)
        assert checker2.should_skip(file_hash) is True

    def test_abstract_base_class_cannot_instantiate(self):
        """测试 FileIntegrityChecker 抽象类无法实例化。"""
        with pytest.raises(TypeError):
            FileIntegrityChecker()
    
    def test_concurrent_writes_supported(self, checker, temp_file):
        """测试 WAL 模式支持并发操作（基础检查）。"""
        file_hash = checker.compute_sha256(temp_file)

        # 序列执行的多个操作应正常工作
        checker.mark_success(file_hash, temp_file)
        result1 = checker.should_skip(file_hash)
        checker.mark_success(file_hash, temp_file, collection="test")
        result2 = checker.should_skip(file_hash)
        
        assert result1 is True
        assert result2 is True


class TestHashConsistency:
    """哈希计算一致性的测试。"""

    def test_empty_file_hash(self, checker):
        """测试空文件的哈希计算。"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            empty_file = f.name
        
        try:
            file_hash = checker.compute_sha256(empty_file)
            
            # 空文件应具有一致的哈希
            expected = hashlib.sha256(b"").hexdigest()
            assert file_hash == expected
        finally:
            Path(empty_file).unlink(missing_ok=True)
    
    def test_same_content_different_files(self, checker):
        """测试相同内容的文件产生相同哈希。"""
        content = "Identical content"
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f1:
            f1.write(content)
            file1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f2:
            f2.write(content)
            file2 = f2.name
        
        try:
            hash1 = checker.compute_sha256(file1)
            hash2 = checker.compute_sha256(file2)
            
            assert hash1 == hash2
        finally:
            Path(file1).unlink(missing_ok=True)
            Path(file2).unlink(missing_ok=True)
    
    def test_different_content_different_hash(self, checker):
        """测试不同内容产生不同哈希。"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f1:
            f1.write("Content A")
            file1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f2:
            f2.write("Content B")
            file2 = f2.name
        
        try:
            hash1 = checker.compute_sha256(file1)
            hash2 = checker.compute_sha256(file2)
            
            assert hash1 != hash2
        finally:
            Path(file1).unlink(missing_ok=True)
            Path(file2).unlink(missing_ok=True)


class TestErrorHandling:
    """错误处理场景的测试。"""

    def test_should_skip_with_corrupted_db_raises_error(self, temp_db):
        """测试数据库损坏时的行为。"""
        # 先创建有效的校验器
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        checker.close()  # 显式关闭连接

        # 写入无效数据以损坏数据库
        with open(temp_db, "wb") as f:
            f.write(b"This is not a valid SQLite database")

        # 使用损坏的数据库创建新校验器 —— 会在初始化时失败
        with pytest.raises(sqlite3.DatabaseError):
            corrupted_checker = SQLiteIntegrityChecker(db_path=temp_db)
    
    def test_mark_success_with_readonly_db_raises_error(self, temp_db, temp_file):
        """测试数据库只读时的错误处理。"""
        # 创建校验器并将数据库设为只读
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        Path(temp_db).chmod(0o444)

        try:
            file_hash = checker.compute_sha256(temp_file)

            # 写入时应抛出 RuntimeError
            with pytest.raises(RuntimeError, match="Failed to mark success"):
                checker.mark_success(file_hash, temp_file)
        finally:
            # 恢复权限以便清理
            Path(temp_db).chmod(0o644)
