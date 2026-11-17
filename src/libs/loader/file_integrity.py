"""用于增量摄入的文件完整性检查器。

本模块提供基于 SHA256 的文件完整性跟踪，以支持增量摄入。
已成功处理的文件可在后续摄入运行时跳过。

设计原则：
- 幂等：同一文件的多次摄入运行是安全的
- 持久化：基于 SQLite 的存储可跨进程重启保留
- 并发：WAL 模式支持并发读写操作
- 优雅：失败的摄入会被记录，但不会阻止重试
"""

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class FileIntegrityChecker(ABC):
    """文件完整性检查的抽象基类。

    各实现跟踪哪些文件已成功处理，以支持增量摄入。
    """
    
    @abstractmethod
    def compute_sha256(self, file_path: str) -> str:
        """计算文件的 SHA256 哈希。

        参数:
            file_path: 待哈希文件的路径。

        返回:
            十六进制 SHA256 哈希字符串（64 个字符）。

        异常:
            FileNotFoundError: 当文件不存在时抛出。
            IOError: 当路径不是文件或无法读取时抛出。
        """
        pass
    
    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """根据哈希判断文件是否应被跳过。

        参数:
            file_hash: 文件的 SHA256 哈希。

        返回:
            若文件此前已成功处理则返回 True，否则返回 False。
        """
        pass
    
    @abstractmethod
    def mark_success(
        self, 
        file_hash: str, 
        file_path: str, 
        collection: Optional[str] = None
    ) -> None:
        """将文件标记为处理成功。

        参数:
            file_hash: 文件的 SHA256 哈希。
            file_path: 原始文件路径（用于跟踪）。
            collection: 可选的 collection/命名空间标识符。

        异常:
            RuntimeError: 当数据库操作失败时抛出。
        """
        pass
    
    @abstractmethod
    def mark_failed(
        self, 
        file_hash: str, 
        file_path: str, 
        error_msg: str
    ) -> None:
        """将文件处理标记为失败。

        失败的文件会被跟踪，但不会在后续运行中被跳过，从而允许重试。

        参数:
            file_hash: 文件的 SHA256 哈希。
            file_path: 原始文件路径（用于跟踪）。
            error_msg: 描述失败原因的错误信息。

        异常:
            RuntimeError: 当数据库操作失败时抛出。
        """
        pass

    @abstractmethod
    def remove_record(self, file_hash: str) -> bool:
        """根据文件哈希删除一条摄入记录。

        参数:
            file_hash: 用于标识该记录的 SHA256 哈希。

        返回:
            若删除了记录则返回 True，未找到则返回 False。
        """
        pass

    @abstractmethod
    def list_processed(
        self, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """列出已成功处理的文件。

        参数:
            collection: 可选的 collection 过滤条件。当为 *None* 时
                返回所有成功记录。

        返回:
            字典列表，键为：file_hash、file_path、collection、
            processed_at、updated_at。
        """
        pass


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """基于 SQLite 的文件完整性检查器。

    将摄入历史存储在使用 WAL 模式以支持并发访问的 SQLite 数据库中。

    数据库表结构:
        ingestion_history (
            file_hash TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            status TEXT NOT NULL,  -- 'success' or 'failed'
            collection TEXT,
            error_msg TEXT,
            processed_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )

    参数:
        db_path: SQLite 数据库文件路径（若不存在则会创建）。

    异常:
        sqlite3.DatabaseError: 当数据库文件损坏时抛出。
    """
    
    def __init__(self, db_path: str):
        """初始化检查器，并在需要时创建数据库。

        参数:
            db_path: SQLite 数据库文件路径。
        """
        self.db_path = db_path
        self._conn = None
        self._ensure_database()
    
    def close(self) -> None:
        """若数据库连接已打开则关闭。"""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __del__(self):
        """清理：对象删除时关闭连接。"""
        self.close()
    
    def _ensure_database(self) -> None:
        """若数据库文件和表结构不存在则创建。"""
        # 按需创建父目录
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 连接并初始化表结构
        conn = sqlite3.connect(self.db_path)
        try:
            # 启用 WAL 模式以支持并发访问
            conn.execute("PRAGMA journal_mode=WAL")
            
            # 若表不存在则创建
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    collection TEXT,
                    error_msg TEXT,
                    processed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # 在 status 上创建索引以加快查询
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON ingestion_history(status)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def compute_sha256(self, file_path: str) -> str:
        """以分块读取方式计算文件的 SHA256 哈希。

        使用 64KB 分块，以便在不将整个文件载入内存的情况下处理大文件。

        参数:
            file_path: 待哈希文件的路径。

        返回:
            十六进制 SHA256 哈希字符串（64 个字符）。

        异常:
            FileNotFoundError: 当文件不存在时抛出。
            IOError: 当路径不是文件或无法读取时抛出。
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise IOError(f"Path is not a file: {file_path}")
        
        # 使用分块读取计算哈希
        sha256_hash = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                # 按 64KB 分块读取
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(chunk)
        except Exception as e:
            raise IOError(f"Failed to read file {file_path}: {e}")
        
        return sha256_hash.hexdigest()
    
    def should_skip(self, file_hash: str) -> bool:
        """判断文件是否应被跳过。

        只有 status='success' 的文件才会被跳过。失败的文件可以重试。

        参数:
            file_hash: 文件的 SHA256 哈希。

        返回:
            若文件 status='success' 则返回 True，否则返回 False。
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            
            if result is None:
                return False
            
            return result[0] == "success"
        finally:
            conn.close()
    
    def mark_success(
        self, 
        file_hash: str, 
        file_path: str, 
        collection: Optional[str] = None
    ) -> None:
        """将文件标记为处理成功。

        使用 INSERT OR REPLACE 以保证幂等操作。

        参数:
            file_hash: 文件的 SHA256 哈希。
            file_path: 原始文件路径（用于跟踪）。
            collection: 可选的 collection/命名空间标识符。

        异常:
            RuntimeError: 当数据库操作失败时抛出。
        """
        now = datetime.now(timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            # 检查记录是否存在，以保留 processed_at
            cursor = conn.execute(
                "SELECT processed_at FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            
            if result:
                # 更新已有记录
                conn.execute("""
                    UPDATE ingestion_history 
                    SET file_path = ?,
                        status = 'success',
                        collection = ?,
                        error_msg = NULL,
                        updated_at = ?
                    WHERE file_hash = ?
                """, (file_path, collection, now, file_hash))
            else:
                # 插入新记录
                conn.execute("""
                    INSERT INTO ingestion_history 
                    (file_hash, file_path, status, collection, error_msg, processed_at, updated_at)
                    VALUES (?, ?, 'success', ?, NULL, ?, ?)
                """, (file_hash, file_path, collection, now, now))
            
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark success for {file_path}: {e}")
        finally:
            conn.close()
    
    def mark_failed(
        self, 
        file_hash: str, 
        file_path: str, 
        error_msg: str
    ) -> None:
        """将文件处理标记为失败。

        失败的文件不会被跳过，从而允许重试。

        参数:
            file_hash: 文件的 SHA256 哈希。
            file_path: 原始文件路径（用于跟踪）。
            error_msg: 描述失败原因的错误信息。

        异常:
            RuntimeError: 当数据库操作失败时抛出。
        """
        now = datetime.now(timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            # 检查记录是否存在，以保留 processed_at
            cursor = conn.execute(
                "SELECT processed_at FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()
            
            if result:
                # 更新已有记录
                conn.execute("""
                    UPDATE ingestion_history 
                    SET file_path = ?,
                        status = 'failed',
                        error_msg = ?,
                        updated_at = ?
                    WHERE file_hash = ?
                """, (file_path, error_msg, now, file_hash))
            else:
                # 插入新记录
                conn.execute("""
                    INSERT INTO ingestion_history 
                    (file_hash, file_path, status, collection, error_msg, processed_at, updated_at)
                    VALUES (?, ?, 'failed', NULL, ?, ?, ?)
                """, (file_hash, file_path, error_msg, now, now))
            
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark failure for {file_path}: {e}")
        finally:
            conn.close()

    def remove_record(self, file_hash: str) -> bool:
        """根据文件哈希删除一条摄入记录。

        参数:
            file_hash: 用于标识该记录的 SHA256 哈希。

        返回:
            若删除了记录则返回 True，未找到则返回 False。
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            )
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to remove record {file_hash}: {e}")
        finally:
            conn.close()

    def list_processed(
        self, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """列出已成功处理的文件。

        参数:
            collection: 可选的 collection 过滤条件。

        返回:
            字典列表，键为：file_hash、file_path、collection、
            processed_at、updated_at。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = (
                "SELECT file_hash, file_path, collection, processed_at, updated_at "
                "FROM ingestion_history WHERE status = 'success'"
            )
            params: list[str] = []
            if collection is not None:
                query += " AND collection = ?"
                params.append(collection)
            query += " ORDER BY processed_at ASC"

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
