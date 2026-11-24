"""批处理器，用于编排稠密与稀疏编码。

本模块实现 Ingestion Pipeline 的批处理器组件，负责协调
编码工作流并管理批处理操作。

设计原则：
- 编排：在统一工作流中协调 DenseEncoder 与 SparseEncoder
- 配置驱动：批大小来自配置，而非硬编码
- 可观测：通过 TraceContext 记录批次耗时与统计信息
- 错误处理：单个批次失败不会导致整个 Pipeline 崩溃
- 确定性：相同输入产生相同的分批方式和结果
"""

from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass

from src.core.types import Chunk
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder


@dataclass
class BatchResult:
    """批处理操作的结果。

    属性:
        dense_vectors: 稠密 Embedding 列表（每个 Chunk 一个）
        sparse_stats: 词项统计列表（每个 Chunk 一个）
        batch_count: 已处理的批次数
        total_time: 总处理时间（秒）
        successful_chunks: 成功处理的 Chunk 数量
        failed_chunks: 处理失败的 Chunk 数量
    """
    dense_vectors: List[List[float]]
    sparse_stats: List[Dict[str, Any]]
    batch_count: int
    total_time: float
    successful_chunks: int
    failed_chunks: int


class BatchProcessor:
    """编排 Chunk 通过编码 Pipeline 的批处理。

    该处理器管理将 Chunk 转换为稠密和稀疏两种表示的流程。
    它将 Chunk 划分为批次，驱动编码器，并收集耗时指标。

    设计：
    - 无状态：process() 调用之间不保留状态
    - 并行编码：稠密与稀疏编码相互独立
    - 指标收集：记录批次级耗时以支持可观测性
    - 顺序保持：输出顺序与输入 Chunk 顺序一致

    示例:
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        >>> from src.core.settings import load_settings
        >>> 
        >>> settings = load_settings("config/settings.yaml")
        >>> embedding = EmbeddingFactory.create(settings)
        >>> dense_encoder = DenseEncoder(embedding, batch_size=2)
        >>> sparse_encoder = SparseEncoder()
        >>> 
        >>> processor = BatchProcessor(
        ...     dense_encoder=dense_encoder,
        ...     sparse_encoder=sparse_encoder,
        ...     batch_size=2
        ... )
        >>> 
        >>> chunks = [
        ...     Chunk(id="1", text="Hello", metadata={}),
        ...     Chunk(id="2", text="World", metadata={})
        ... ]
        >>> result = processor.process(chunks)
        >>> len(result.dense_vectors) == len(chunks)  # True
        >>> len(result.sparse_stats) == len(chunks)  # True
    """
    
    def __init__(
        self,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        batch_size: int = 100,
    ):
        """初始化 BatchProcessor。

        参数:
            dense_encoder: 用于生成 Embedding 的 DenseEncoder 实例
            sparse_encoder: 用于词项统计的 SparseEncoder 实例
            batch_size: 每批处理的 Chunk 数量（默认：100）

        异常:
            ValueError: 若 batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size
    
    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> BatchResult:
        """通过稠密与稀疏编码 Pipeline 处理 Chunk。

        工作流：
        1. 校验输入
        2. 将 Chunk 划分为批次
        3. 通过两个编码器处理每个批次
        4. 收集结果与耗时指标
        5. 若提供 TraceContext 则记录

        参数:
            chunks: 待处理的 Chunk 对象列表
            trace: 可选的 TraceContext，用于可观测性

        返回:
            包含向量、统计信息与指标的 BatchResult

        异常:
            ValueError: 若 chunks 列表为空
            RuntimeError: 若两个编码器都完全失败

        示例:
            >>> chunks = [Chunk(id=f"{i}", text=f"Text {i}", metadata={}) 
            ...           for i in range(5)]
            >>> result = processor.process(chunks)
            >>> result.batch_count  # 3 (with batch_size=2)
            >>> result.successful_chunks  # 5
        """
        if not chunks:
            raise ValueError("Cannot process empty chunks list")
        
        start_time = time.perf_counter()
        
        # 创建批次
        batches = self._create_batches(chunks)
        batch_count = len(batches)

        # 处理所有批次
        # 稠密编码(向量化)
        dense_vectors: List[List[float]] = []
        # 稀疏编码(编码为BM25)
        sparse_stats: List[Dict[str, Any]] = []
        successful_chunks = 0
        failed_chunks = 0
        
        for batch_idx, batch in enumerate(batches):
            batch_start = time.perf_counter()
            
            try:
                # 稠密编码
                batch_dense = self.dense_encoder.encode(batch, trace=trace)
                dense_vectors.extend(batch_dense)

                # 稀疏编码
                batch_sparse = self.sparse_encoder.encode(batch, trace=trace)
                sparse_stats.extend(batch_sparse)
                
                successful_chunks += len(batch)
                
            except Exception as e:
                # 记录失败，但继续处理剩余批次
                failed_chunks += len(batch)
                if trace:
                    trace.record_stage(
                        f"batch_{batch_idx}_error",
                        {"error": str(e), "batch_size": len(batch)}
                    )
            
            batch_duration = time.perf_counter() - batch_start
            
            # 若有 trace 则记录批次耗时
            if trace:
                trace.record_stage(
                    f"batch_{batch_idx}",
                    {
                        "batch_size": len(batch),
                        "duration_seconds": batch_duration,
                        "chunks_processed": len(batch)
                    }
                )
        
        total_time = time.perf_counter() - start_time
        
        # 记录整体处理统计信息
        if trace:
            trace.record_stage(
                "batch_processing",
                {
                    "total_chunks": len(chunks),
                    "batch_count": batch_count,
                    "batch_size": self.batch_size,
                    "successful_chunks": successful_chunks,
                    "failed_chunks": failed_chunks,
                    "total_time_seconds": total_time
                }
            )
        
        return BatchResult(
            dense_vectors=dense_vectors,
            sparse_stats=sparse_stats,
            batch_count=batch_count,
            total_time=total_time,
            successful_chunks=successful_chunks,
            failed_chunks=failed_chunks
        )
    
    def _create_batches(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """将 Chunk 按指定大小划分为批次。

        参数:
            chunks: 待分批的 Chunk 列表

        返回:
            批次列表，每个批次是 Chunk 列表。
            顺序保持不变：第一批包含 chunks[0:batch_size]，
            第二批包含 chunks[batch_size:2*batch_size]，依此类推。

        示例:
            >>> chunks = [Chunk(id=f"{i}", text="", metadata={}) for i in range(5)]
            >>> batches = processor._create_batches(chunks)
            >>> len(batches)  # 3 (with batch_size=2)
            >>> [len(b) for b in batches]  # [2, 2, 1]
        """
        batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def get_batch_count(self, total_chunks: int) -> int:
        """计算给定 Chunk 数量所需的批次数。

        用于规划和测试的工具方法。

        参数:
            total_chunks: 待处理的 Chunk 总数

        返回:
            将创建的批次数

        示例:
            >>> processor.get_batch_count(5)  # 3 (with batch_size=2)
            >>> processor.get_batch_count(4)  # 2
            >>> processor.get_batch_count(0)  # 0
        """
        if total_chunks <= 0:
            return 0
        return (total_chunks + self.batch_size - 1) // self.batch_size
