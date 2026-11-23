"""Modular RAG MCP Server 的 Ingestion Pipeline 编排器。

本模块实现主 Pipeline，编排完整的文档摄入流程：
    1. 文件完整性检查（SHA256 跳过检查）
    2. 文档加载（PDF → Document）
    3. 切分（Document → Chunks）
    4. Transform（Refine + Enrich + Caption）
    5. 编码（稠密 + 稀疏向量）
    6. 存储（VectorStore + BM25 索引 + ImageStorage）

设计原则：
- 配置驱动：所有组件通过 settings.yaml 配置
- 可观测：记录进度与阶段完成日志
- 优雅降级：LLM 失败不会阻塞 Pipeline
- 幂等：基于 SHA256 跳过未变更的文件
"""

from pathlib import Path
from typing import Callable, List, Optional, Dict, Any
import time

from src.core.settings import Settings, load_settings, resolve_path
from src.core.types import Document, Chunk
from src.core.trace.trace_context import TraceContext
from src.observability.logger import get_logger

# Libs 层导入
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

# Ingestion 层导入
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.ingestion.storage.image_storage import ImageStorage

logger = get_logger(__name__)


class PipelineResult:
    """Pipeline 执行结果，包含详细统计信息。

    属性:
        success: Pipeline 是否成功完成
        file_path: 被处理文件的路径
        doc_id: 文档 ID（SHA256 哈希）
        chunk_count: 生成的 Chunk 数量
        image_count: 处理的图片数量
        vector_ids: 已存储的向量 ID 列表
        error: Pipeline 失败时的错误信息
        stages: 阶段名称到各阶段结果的字典
    """
    
    def __init__(
        self,
        success: bool,
        file_path: str,
        doc_id: Optional[str] = None,
        chunk_count: int = 0,
        image_count: int = 0,
        vector_ids: Optional[List[str]] = None,
        error: Optional[str] = None,
        stages: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.file_path = file_path
        self.doc_id = doc_id
        self.chunk_count = chunk_count
        self.image_count = image_count
        self.vector_ids = vector_ids or []
        self.error = error
        self.stages = stages or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便序列化。"""
        return {
            "success": self.success,
            "file_path": self.file_path,
            "doc_id": self.doc_id,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "vector_ids_count": len(self.vector_ids),
            "error": self.error,
            "stages": self.stages
        }


class IngestionPipeline:
    """文档摄入的主 Pipeline 编排器。

    该类协调摄入过程的所有阶段：
    - 文件完整性检查，用于增量处理
    - 文档加载（PDF 含图片抽取）
    - 可配置切分器的文本切分
    - Chunk 精炼（规则 + LLM）
    - 元数据增强（规则 + LLM）
    - 图片描述生成（Vision LLM）
    - 稠密 Embedding（Azure text-embedding-ada-002）
    - 稀疏编码（BM25 词项统计）
    - 向量存储（ChromaDB）
    - BM25 索引构建

    示例:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings("config/settings.yaml")
        >>> pipeline = IngestionPipeline(settings)
        >>> result = pipeline.run("documents/report.pdf", collection="contracts")
        >>> print(f"Processed {result.chunk_count} chunks")
    """
    
    def __init__(
        self,
        settings: Settings,
        collection: str = "default",
        force: bool = False
    ):
        """初始化 Pipeline 及其所有组件。

        参数:
            settings: 来自 settings.yaml 的应用配置
            collection: 用于组织文档的 Collection 名称
            force: 为 True 时即使文件此前已处理过也重新处理
        """
        self.settings = settings
        self.collection = collection
        self.force = force
        
        # 初始化所有组件
        logger.info("Initializing Ingestion Pipeline components...")
        
        # 阶段 1: 文件完整性检查(用于增量摄入的文件完整性检查器)
        self.integrity_checker = SQLiteIntegrityChecker(db_path=str(resolve_path("data/db/ingestion_history.db")))
        logger.info("  ✓ FileIntegrityChecker initialized")
        
        # 阶段 2: 加载器(按扩展名分发: pdf / txt / md / csv / json / docx / pptx / xlsx / html)
        # 1.转换为带图片占位符的标准 Markdown 格式, 2.保存解析出来的图片到指定的本地目录
        self.loader = LoaderFactory(
            extract_images=True,
            image_storage_dir=str(resolve_path(f"data/images/{collection}"))
        )
        logger.info(f"  ✓ LoaderFactory initialized (supported: {', '.join(sorted(LoaderFactory.supported_extensions()))})")
        
        # 阶段 3: 切分器
        self.chunker = DocumentChunker(settings)
        logger.info("  ✓ DocumentChunker initialized")
        
        # 阶段 4: Transform 组件
        self.chunk_refiner = ChunkRefiner(settings)
        logger.info(f"  ✓ ChunkRefiner initialized (use_llm={self.chunk_refiner.use_llm})")
        
        self.metadata_enricher = MetadataEnricher(settings)
        logger.info(f"  ✓ MetadataEnricher initialized (use_llm={self.metadata_enricher.use_llm})")
        
        self.image_captioner = ImageCaptioner(settings)
        has_vision = self.image_captioner.llm is not None
        logger.info(f"  ✓ ImageCaptioner initialized (vision_enabled={has_vision})")
        
        # 阶段 5: 编码器(向量化)
        embedding = EmbeddingFactory.create(settings)
        batch_size = settings.ingestion.batch_size if settings.ingestion else 100
        self.dense_encoder = DenseEncoder(embedding, batch_size=batch_size)
        logger.info(f"  ✓ DenseEncoder initialized (provider={settings.embedding.provider})")
        
        self.sparse_encoder = SparseEncoder()
        logger.info("  ✓ SparseEncoder initialized")
        
        self.batch_processor = BatchProcessor(
            dense_encoder=self.dense_encoder,
            sparse_encoder=self.sparse_encoder,
            batch_size=batch_size
        )
        logger.info(f"  ✓ BatchProcessor initialized (batch_size={batch_size})")
        
        # 阶段 6: 存储
        self.vector_upserter = VectorUpserter(settings, collection_name=collection)
        logger.info(f"  ✓ VectorUpserter initialized (provider={settings.vector_store.provider}, collection={collection})")
        
        self.bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
        logger.info("  ✓ BM25Indexer initialized")
        
        self.image_storage = ImageStorage(
            db_path=str(resolve_path("data/db/image_index.db")),
            images_root=str(resolve_path("data/images"))
        )
        logger.info("  ✓ ImageStorage initialized")
        
        logger.info("Pipeline initialization complete!")
    
    def run(
        self,
        file_path: str,
        trace: Optional[TraceContext] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> PipelineResult:
        """对文件执行完整的摄入 Pipeline。

        参数:
            file_path: 待处理文件的路径（例如 PDF）
            trace: 可选的 Trace 上下文，用于可观测性
            on_progress: 可选回调 ``(stage_name, current, total)``，
                在每个 Pipeline 阶段完成时调用。*current* 是
                已完成阶段的 1 基索引；*total* 是阶段总数
                （当前为 6）。

        返回:
            包含成功状态与统计信息的 PipelineResult
        """
        file_path = Path(file_path)
        stages: Dict[str, Any] = {}
        _total_stages = 6

        # 内部闭包封装
        def _notify(stage_name: str, step: int) -> None:
            if on_progress is not None:
                on_progress(stage_name, step, _total_stages)
        
        logger.info(f"=" * 60)
        logger.info(f"Starting Ingestion Pipeline for: {file_path}")
        logger.info(f"Collection: {self.collection}")
        logger.info(f"=" * 60)
        
        try:
            # ─────────────────────────────────────────────────────────────
            # 阶段 1: 文件完整性检查
            # ─────────────────────────────────────────────────────────────
            logger.info("\n📋 Stage 1: File Integrity Check")
            _notify("integrity", 1)
            
            file_hash = self.integrity_checker.compute_sha256(str(file_path))
            logger.info(f"  File hash: {file_hash[:16]}...")
            
            if not self.force and self.integrity_checker.should_skip(file_hash):
                logger.info(f"  ⏭️  File already processed, skipping (use force=True to reprocess)")
                return PipelineResult(
                    success=True,
                    file_path=str(file_path),
                    doc_id=file_hash,
                    stages={"integrity": {"skipped": True, "reason": "already_processed"}}
                )
            
            stages["integrity"] = {"file_hash": file_hash, "skipped": False}
            logger.info("  ✓ File needs processing")
            
            # ─────────────────────────────────────────────────────────────
            # 阶段 2: 文档加载(转换为带图片占位符的标准 Markdown 格式)
            # ─────────────────────────────────────────────────────────────
            logger.info("\n📄 Stage 2: Document Loading")
            _notify("load", 2)
            
            _t0 = time.monotonic()
            loader = self.loader.get_loader(str(file_path))
            document = loader.load(str(file_path))
            _elapsed = (time.monotonic() - _t0) * 1000.0
            
            text_preview = document.text[:200].replace('\n', ' ') + "..." if len(document.text) > 200 else document.text
            image_count = len(document.metadata.get("images", []))
            
            logger.info(f"  文档 ID: {document.id}")
            logger.info(f"  文本长度: {len(document.text)} 字符")
            logger.info(f"  提取图片数: {image_count}")
            logger.info(f"  内容预览: {text_preview[:100]}...")
            
            stages["loading"] = {
                "doc_id": document.id,
                "text_length": len(document.text),
                "image_count": image_count
            }
            # trace记录加载阶段
            if trace is not None:
                trace.record_stage("load", {
                    "method": type(loader).__name__,
                    "doc_id": document.id,
                    "text_length": len(document.text),
                    "image_count": image_count,
                    "text_preview": document.text,
                }, elapsed_ms=_elapsed)
            
            # ─────────────────────────────────────────────────────────────
            # 阶段 3: 切分
            # ─────────────────────────────────────────────────────────────
            logger.info("\n✂️  Stage 3: Document Chunking")
            _notify("split", 3)
            
            _t0 = time.monotonic()
            # 切块 + 填充图片到元数据中
            chunks = self.chunker.split_document(document)
            _elapsed = (time.monotonic() - _t0) * 1000.0
            
            logger.info(f"  Chunks generated: {len(chunks)}")
            if chunks:
                logger.info(f"  First chunk ID: {chunks[0].id}")
                logger.info(f"  First chunk preview: {chunks[0].text[:100]}...")
            
            stages["chunking"] = {
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0
            }
            if trace is not None:
                trace.record_stage("split", {
                    "method": "recursive",
                    "chunk_count": len(chunks),
                    "avg_chunk_size": sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0,
                    "chunks": [
                        {
                            "chunk_id": c.id,
                            "text": c.text,
                            "char_len": len(c.text),
                            "chunk_index": c.metadata.get("chunk_index", i),
                        }
                        for i, c in enumerate(chunks)
                    ],
                }, elapsed_ms=_elapsed)
            
            # ─────────────────────────────────────────────────────────────
            # 阶段 4: Transform 流水线(增强)
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🔄 Stage 4: Transform Pipeline")
            _notify("transform", 4)

            logger.info("  4a. Chunk Refinement...")
            _t0_transform = time.monotonic()
            # 精炼前快照
            _pre_refine_texts = {c.id: c.text for c in chunks}
            # 4a: Chunk refiner 精炼
            chunks = self.chunk_refiner.transform(chunks, trace)
            refined_by_llm = sum(1 for c in chunks if c.metadata.get("refined_by") == "llm")
            refined_by_rule = sum(1 for c in chunks if c.metadata.get("refined_by") == "rule")
            logger.info(f"      LLM refined: {refined_by_llm}, Rule refined: {refined_by_rule}")
            
            # 4b: 元数据增强
            logger.info("  4b. Metadata Enrichment...")
            chunks = self.metadata_enricher.transform(chunks, trace)
            enriched_by_llm = sum(1 for c in chunks if c.metadata.get("enriched_by") == "llm")
            enriched_by_rule = sum(1 for c in chunks if c.metadata.get("enriched_by") == "rule")
            logger.info(f"      LLM enriched: {enriched_by_llm}, Rule enriched: {enriched_by_rule}")
            
            # 4c: 图片描述生成
            logger.info("  4c. Image Captioning...")
            chunks = self.image_captioner.transform(chunks, trace)
            captioned = sum(1 for c in chunks if c.metadata.get("image_captions"))
            logger.info(f"      Chunks with captions: {captioned}")
            
            stages["transform"] = {
                "chunk_refiner": {"llm": refined_by_llm, "rule": refined_by_rule},
                "metadata_enricher": {"llm": enriched_by_llm, "rule": enriched_by_rule},
                "image_captioner": {"captioned_chunks": captioned}
            }
            _elapsed_transform = (time.monotonic() - _t0_transform) * 1000.0
            if trace is not None:
                trace.record_stage("transform", {
                    "method": "refine+enrich+caption",
                    "refined_by_llm": refined_by_llm,
                    "refined_by_rule": refined_by_rule,
                    "enriched_by_llm": enriched_by_llm,
                    "enriched_by_rule": enriched_by_rule,
                    "captioned_chunks": captioned,
                    "chunks": [
                        {
                            "chunk_id": c.id,
                            "text_before": _pre_refine_texts.get(c.id, ""),
                            "text_after": c.text,
                            "char_len": len(c.text),
                            "refined_by": c.metadata.get("refined_by", ""),
                            "enriched_by": c.metadata.get("enriched_by", ""),
                            "title": c.metadata.get("title", ""),
                            "tags": c.metadata.get("tags", []),
                            "summary": c.metadata.get("summary", ""),
                        }
                        for c in chunks
                    ],
                }, elapsed_ms=_elapsed_transform)
            
            # ─────────────────────────────────────────────────────────────
            # 阶段 5: 编码
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🔢 Stage 5: Encoding")
            _notify("embed", 5)
            
            # 通过 BatchProcessor 处理
            _t0 = time.perf_counter()
            batch_result = self.batch_processor.process(chunks, trace)
            _elapsed = (time.perf_counter() - _t0) * 1000.0

            dense_vectors = batch_result.dense_vectors
            sparse_stats = batch_result.sparse_stats

            # BatchProcessor 会吞掉单批异常并继续下一批，这里把失败
            # 上升为明确的错误 —— 否则后续 upsert 只会报
            # "Chunk count (N) must match vector count (0)" 这种看不出根因的错误。
            # 注意：仅当失败计数确实为正整数时才报错（便于测试注入 Mock）。
            failed_chunks = getattr(batch_result, "failed_chunks", 0)
            if isinstance(failed_chunks, int) and failed_chunks > 0:
                first_error = next(
                    (
                        entry["data"]["error"]
                        for entry in (trace.stages if trace is not None else [])
                        if entry["stage"].endswith("_error")
                        and isinstance(entry.get("data"), dict)
                        and "error" in entry["data"]
                    ),
                    "unknown error",
                )
                raise RuntimeError(
                    f"Encoding failed for {batch_result.failed_chunks}/"
                    f"{len(chunks)} chunks: {first_error}"
                )

            logger.info(f"  Dense vectors: {len(dense_vectors)} (dim={len(dense_vectors[0]) if dense_vectors else 0})")
            logger.info(f"  Sparse stats: {len(sparse_stats)} documents")
            
            stages["encoding"] = {
                "dense_vector_count": len(dense_vectors),
                "dense_dimension": len(dense_vectors[0]) if dense_vectors else 0,
                "sparse_doc_count": len(sparse_stats)
            }
            if trace is not None:
                # 构建每个 Chunk 的编码详情（稠密与稀疏）
                chunk_details = []
                for idx, c in enumerate(chunks):
                    detail: dict = {
                        "chunk_id": c.id,
                        "char_len": len(c.text),
                    }
                    # 稠密：向量维度（所有 Chunk 相同，但逐个确认）
                    if idx < len(dense_vectors):
                        detail["dense_dim"] = len(dense_vectors[idx])
                    # 稀疏：BM25 词项统计
                    if idx < len(sparse_stats):
                        ss = sparse_stats[idx]
                        detail["doc_length"] = ss.get("doc_length", 0)
                        detail["unique_terms"] = ss.get("unique_terms", 0)
                        # 按词频排序的 Top-10 词项，用于检查
                        tf = ss.get("term_frequencies", {})
                        top_terms = sorted(tf.items(), key=lambda x: x[1], reverse=True)[:10]
                        detail["top_terms"] = [{"term": t, "freq": f} for t, f in top_terms]
                    chunk_details.append(detail)

                trace.record_stage("embed", {
                    "method": "batch_processor",
                    "dense_vector_count": len(dense_vectors),
                    "dense_dimension": len(dense_vectors[0]) if dense_vectors else 0,
                    "sparse_doc_count": len(sparse_stats),
                    "chunks": chunk_details,
                }, elapsed_ms=_elapsed)
            
            # ─────────────────────────────────────────────────────────────
            # 阶段 6: 存储
            # ─────────────────────────────────────────────────────────────
            logger.info("\n💾 Stage 6: Storage")
            _notify("upsert", 6)
            
            # 6a: 向量写入
            logger.info("  6a. Vector Storage (ChromaDB)...")
            _t0_storage = time.monotonic()
            vector_ids = self.vector_upserter.upsert(chunks, dense_vectors, trace)
            logger.info(f"      Stored {len(vector_ids)} vectors")

            # 将 BM25 的 chunk_id 与 Chroma 向量 ID 对齐，以便
            # SparseRetriever 在检索后能在向量存储中查到 BM25 命中项。
            for stat, vid in zip(sparse_stats, vector_ids):
                stat["chunk_id"] = vid

            # 6b: BM25 索引
            logger.info("  6b. BM25 Index...")
            self.bm25_indexer.add_documents(
                sparse_stats,
                collection=self.collection,
                doc_id=document.id,
                trace=trace,
            )
            logger.info(f"      Index built for {len(sparse_stats)} documents")
            
            # 6c: 在图片存储索引中登记图片
            # 注意：图片已由 PdfLoader 保存，这里只需为其建立索引
            logger.info("  6c. Image Storage Index...")
            images = document.metadata.get("images", [])
            for img in images:
                img_path = Path(img["path"])
                if img_path.exists():
                    self.image_storage.register_image(
                        image_id=img["id"],
                        file_path=img_path,
                        collection=self.collection,
                        doc_hash=file_hash,
                        page_num=img.get("page", 0)
                    )
            logger.info(f"      Indexed {len(images)} images")
            
            stages["storage"] = {
                "vector_count": len(vector_ids),
                "bm25_docs": len(sparse_stats),
                "images_indexed": len(images)
            }
            _elapsed_storage = (time.monotonic() - _t0_storage) * 1000.0
            if trace is not None:
                # 每个 Chunk 的存储映射：chunk_id → vector_id
                chunk_storage = [
                    {
                        "chunk_id": c.id,
                        "vector_id": vector_ids[i] if i < len(vector_ids) else "—",
                        "collection": self.collection,
                        "store": "ChromaDB",
                    }
                    for i, c in enumerate(chunks)
                ]
                # 图片存储详情
                image_storage_details = [
                    {
                        "image_id": img["id"],
                        "file_path": str(img["path"]),
                        "page": img.get("page", 0),
                        "doc_hash": file_hash,
                    }
                    for img in images
                ]
                trace.record_stage("upsert", {
                    "method": "chroma+bm25+image_storage",
                    "dense_store": {
                        "backend": "ChromaDB",
                        "collection": self.collection,
                        "count": len(vector_ids),
                        "path": "data/db/chroma/",
                    },
                    "sparse_store": {
                        "backend": "BM25",
                        "collection": self.collection,
                        "count": len(sparse_stats),
                        "path": f"data/db/bm25/{self.collection}/",
                    },
                    "image_store": {
                        "backend": "ImageStorage (JSON index)",
                        "count": len(images),
                        "images": image_storage_details,
                    },
                    "chunk_mapping": chunk_storage,
                }, elapsed_ms=_elapsed_storage)
            
            # ─────────────────────────────────────────────────────────────
            # 标记成功
            # ─────────────────────────────────────────────────────────────
            self.integrity_checker.mark_success(file_hash, str(file_path), self.collection)
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ Pipeline completed successfully!")
            logger.info(f"   Chunks: {len(chunks)}")
            logger.info(f"   Vectors: {len(vector_ids)}")
            logger.info(f"   Images: {len(images)}")
            logger.info("=" * 60)
            
            return PipelineResult(
                success=True,
                file_path=str(file_path),
                doc_id=file_hash,
                chunk_count=len(chunks),
                image_count=len(images),
                vector_ids=vector_ids,
                stages=stages
            )
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            self.integrity_checker.mark_failed(file_hash, str(file_path), str(e))
            
            return PipelineResult(
                success=False,
                file_path=str(file_path),
                doc_id=file_hash if 'file_hash' in locals() else None,
                error=str(e),
                stages=stages
            )
    
    def close(self) -> None:
        """清理资源。"""
        self.image_storage.close()


def run_pipeline(
    file_path: str,
    settings_path: Optional[str] = None,
    collection: str = "default",
    force: bool = False
) -> PipelineResult:
    """便捷函数，用于运行 Pipeline。

    参数:
        file_path: 待处理文件的路径
        settings_path: settings.yaml 的路径（默认：<repo>/config/settings.yaml）
        collection: Collection 名称
        force: 强制重新处理

    返回:
        包含执行详情的 PipelineResult
    """
    settings = load_settings(settings_path)
    pipeline = IngestionPipeline(settings, collection=collection, force=force)
    
    try:
        return pipeline.run(file_path)
    finally:
        pipeline.close()
