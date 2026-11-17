"""跨存储后端的文档生命周期管理。

本模块提供统一的入口，用于在四个存储后端（ChromaDB、BM25、
ImageStorage、FileIntegrityChecker）之间列出、查看和删除文档。

设计原则：
- 协调性：一次调用级联到所有相关存储。
- 故障安全：部分失败会被报告，但不会中断其余存储的处理。
- 只读安全：list / stats / detail 方法绝不修改数据。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 结果数据类
# ---------------------------------------------------------------------------

@dataclass
class DocumentInfo:
    """已摄入文档的摘要信息。"""

    source_path: str
    source_hash: str
    collection: Optional[str] = None
    chunk_count: int = 0
    image_count: int = 0
    processed_at: Optional[str] = None


@dataclass
class DocumentDetail(DocumentInfo):
    """扩展的文档信息，包含 Chunk ID 和 Image ID。"""

    chunk_ids: List[str] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)


@dataclass
class DeleteResult:
    """delete_document 操作的结果。"""

    success: bool
    chunks_deleted: int = 0
    bm25_removed: bool = False
    images_deleted: int = 0
    integrity_removed: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class CollectionStats:
    """某个 Collection 的聚合统计信息。"""

    collection: Optional[str] = None
    document_count: int = 0
    chunk_count: int = 0
    image_count: int = 0


# ---------------------------------------------------------------------------
# DocumentManager
# ---------------------------------------------------------------------------

class DocumentManager:
    """跨所有存储后端协调文档生命周期。

    参数:
        chroma_store: ChromaStore 实例（向量存储）。
        bm25_indexer: BM25Indexer 实例（稀疏索引）。
        image_storage: ImageStorage 实例（图片文件 + SQLite 索引）。
        file_integrity: SQLiteIntegrityChecker 实例（摄入历史记录）。
    """

    def __init__(
        self,
        chroma_store: Any,
        bm25_indexer: Any,
        image_storage: Any,
        file_integrity: Any,
    ) -> None:
        self.chroma = chroma_store
        self.bm25 = bm25_indexer
        self.images = image_storage
        self.integrity = file_integrity

    # ------------------------------------------------------------------
    # list_documents
    # ------------------------------------------------------------------

    def list_documents(
        self, collection: Optional[str] = None
    ) -> List[DocumentInfo]:
        """返回已摄入文档的列表。

        合并来自 integrity checker 的信息（source_path、
        hash、processed_at）与来自 ChromaDB 和 ImageStorage 的计数。

        参数:
            collection: 可选的 Collection 过滤条件。

        返回:
            ``DocumentInfo`` 对象列表。
        """
        records = self.integrity.list_processed(collection)

        docs: List[DocumentInfo] = []
        for rec in records:
            source_hash = rec["file_hash"]
            source_path = rec["file_path"]
            coll = rec.get("collection")

            # 统计 Chroma 中的 Chunk 数量
            chunk_count = self._count_chunks(source_hash)

            # 统计图片数量
            image_count = self._count_images(source_hash)

            docs.append(
                DocumentInfo(
                    source_path=source_path,
                    source_hash=source_hash,
                    collection=coll,
                    chunk_count=chunk_count,
                    image_count=image_count,
                    processed_at=rec.get("processed_at"),
                )
            )

        return docs

    # ------------------------------------------------------------------
    # get_document_detail
    # ------------------------------------------------------------------

    def get_document_detail(self, doc_id: str) -> Optional[DocumentDetail]:
        """获取单个文档的详细信息。

        *doc_id* 会与 integrity checker 中存储的 ``source_hash`` 进行匹配。

        参数:
            doc_id: 文档的 source_hash。

        返回:
            包含 Chunk / Image ID 的 ``DocumentDetail``；若未找到则返回 *None*。
        """
        # 查找 integrity 记录
        all_records = self.integrity.list_processed()
        record = None
        for rec in all_records:
            if rec["file_hash"] == doc_id:
                record = rec
                break

        if record is None:
            return None

        source_hash = record["file_hash"]

        # 从 Chroma 收集 Chunk ID
        chunk_ids = self._get_chunk_ids(source_hash)

        # 收集 Image ID
        image_ids = self._get_image_ids(source_hash)

        return DocumentDetail(
            source_path=record["file_path"],
            source_hash=source_hash,
            collection=record.get("collection"),
            chunk_count=len(chunk_ids),
            image_count=len(image_ids),
            processed_at=record.get("processed_at"),
            chunk_ids=chunk_ids,
            image_ids=image_ids,
        )

    # ------------------------------------------------------------------
    # delete_document
    # ------------------------------------------------------------------

    def delete_document(
        self,
        source_path: str,
        collection: str = "default",
        source_hash: Optional[str] = None,
    ) -> DeleteResult:
        """从所有存储后端删除文档。

        协调 ChromaDB、BM25、ImageStorage 和 FileIntegrity
        上的删除操作。部分失败会记录在
        ``DeleteResult.errors`` 中，但不会阻止其余存储的清理。

        文档通过其 *source_hash* 标识。未提供 hash 时，
        方法会尝试从文件计算；若文件已不存在，
        则回退到按路径从 integrity 记录中查找 hash。

        参数:
            source_path: 文档的原始文件系统路径。
            collection: 文档所属的 Collection。
            source_hash: 预先计算好的 SHA-256 哈希。提供时
                方法不会尝试读取源文件。

        返回:
            概述清理结果的 ``DeleteResult``。
        """
        result = DeleteResult(success=True)

        # 解析 hash —— 优先使用调用方提供的值，其次读取文件，最后查库
        if source_hash is None:
            try:
                source_hash = self.integrity.compute_sha256(source_path)
            except Exception as e:
                source_hash = self._hash_from_path(source_path)
                if source_hash is None:
                    result.success = False
                    result.errors.append(f"Cannot identify document: {e}")
                    return result

        # 1. ChromaDB —— 删除匹配 source_hash 的 Chunk
        try:
            count = self.chroma.delete_by_metadata(
                {"doc_hash": source_hash}
            )
            result.chunks_deleted = count
        except Exception as e:
            result.errors.append(f"ChromaDB delete failed: {e}")

        # 2. BM25 —— 移除该文档的倒排记录
        try:
            result.bm25_removed = self.bm25.remove_document(
                source_hash, collection
            )
        except Exception as e:
            result.errors.append(f"BM25 remove failed: {e}")

        # 3. ImageStorage —— 按 doc_hash 删除图片
        try:
            images = self.images.list_images(doc_hash=source_hash)
            deleted_imgs = 0
            for img in images:
                if self.images.delete_image(img["image_id"]):
                    deleted_imgs += 1
            result.images_deleted = deleted_imgs
        except Exception as e:
            result.errors.append(f"ImageStorage delete failed: {e}")

        # 4. FileIntegrity —— 移除摄入记录
        try:
            result.integrity_removed = self.integrity.remove_record(
                source_hash
            )
        except Exception as e:
            result.errors.append(f"FileIntegrity remove failed: {e}")

        if result.errors:
            result.success = False

        return result

    # ------------------------------------------------------------------
    # get_collection_stats
    # ------------------------------------------------------------------

    def get_collection_stats(
        self, collection: Optional[str] = None
    ) -> CollectionStats:
        """返回某个 Collection 的聚合统计信息。

        参数:
            collection: Collection 名称。为 *None* 时统计覆盖
                所有 Collection。

        返回:
            ``CollectionStats`` dataclass。
        """
        docs = self.list_documents(collection)
        chunk_total = sum(d.chunk_count for d in docs)
        image_total = sum(d.image_count for d in docs)

        return CollectionStats(
            collection=collection,
            document_count=len(docs),
            chunk_count=chunk_total,
            image_count=image_total,
        )

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _count_chunks(self, source_hash: str) -> int:
        """统计 Chroma 中属于 *source_hash* 的 Chunk 数量。"""
        try:
            results = self.chroma.collection.get(
                where={"doc_hash": source_hash}, include=[]
            )
            return len(results.get("ids", []))
        except Exception:
            return 0

    def _get_chunk_ids(self, source_hash: str) -> List[str]:
        """返回 Chroma 中匹配 *source_hash* 的 Chunk ID。"""
        try:
            results = self.chroma.collection.get(
                where={"doc_hash": source_hash}, include=[]
            )
            return results.get("ids", [])
        except Exception:
            return []

    def _count_images(self, source_hash: str) -> int:
        """统计属于 *source_hash* 的图片数量。"""
        try:
            return len(self.images.list_images(doc_hash=source_hash))
        except Exception:
            return 0

    def _get_image_ids(self, source_hash: str) -> List[str]:
        """返回属于 *source_hash* 的 Image ID。"""
        try:
            imgs = self.images.list_images(doc_hash=source_hash)
            return [img["image_id"] for img in imgs]
        except Exception:
            return []

    def _hash_from_path(self, source_path: str) -> Optional[str]:
        """尝试按路径从 integrity 记录中查找 source_hash。"""
        try:
            for rec in self.integrity.list_processed():
                if rec["file_path"] == source_path:
                    return rec["file_hash"]
        except Exception:
            pass
        return None
