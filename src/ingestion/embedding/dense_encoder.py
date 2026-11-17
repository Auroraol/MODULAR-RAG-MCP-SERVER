"""稠密编码器，用于从文本 Chunk 生成 Embedding。

本模块实现 Ingestion Pipeline 的稠密编码器组件，负责使用
可配置的 Embedding 提供方将文本 Chunk 转换为稠密向量表示。

设计原则：
- 配置驱动：通过工厂模式从配置中获取 Embedding 提供方
- 批处理：通过分批优化 API 调用
- 可观测：接受 TraceContext 以支持未来的可观测性集成
- 错误处理：单个失败不应导致整个批次崩溃
- 确定性：相同输入产生相同输出
"""

from typing import List, Optional, Any
from src.core.types import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding


class DenseEncoder:
    """使用 BaseEmbedding 提供方将文本 Chunk 编码为稠密向量。

    该编码器充当 Ingestion Pipeline 与可插拔 Embedding 层之间的桥梁。
    它负责批处理、错误恢复，并保持输入 Chunk 与输出向量之间的对齐。

    设计：
    - 依赖注入：接收 BaseEmbedding 实例（不直接调用工厂）
    - 批处理优先：以可配置的批大小处理所有 Chunk
    - 无状态：encode() 调用之间无内部状态

    示例:
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        >>> from src.core.settings import load_settings
        >>> 
        >>> settings = load_settings("config/settings.yaml")
        >>> embedding = EmbeddingFactory.create(settings)
        >>> encoder = DenseEncoder(embedding, batch_size=32)
        >>> 
        >>> chunks = [Chunk(id="1", text="Hello world", metadata={})]
        >>> vectors = encoder.encode(chunks)
        >>> print(len(vectors))  # 1
        >>> print(len(vectors[0]))  # dimension (e.g., 1536)
    """
    
    def __init__(
        self,
        embedding: BaseEmbedding,
        batch_size: int = 100,
    ):
        """初始化 DenseEncoder。

        参数:
            embedding: Embedding 提供方实例（来自 EmbeddingFactory）
            batch_size: 每次 API 调用处理的 Chunk 数量（默认：100）

        异常:
            ValueError: 若 batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        self.embedding = embedding
        self.batch_size = batch_size
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[List[float]]:
        """将 Chunk 编码为稠密向量。

        该方法：
        1. 从每个 Chunk 中提取文本
        2. 按 batch_size 对文本分批
        3. 对每个批次调用 embedding.embed()
        4. 拼接结果并保持 Chunk 顺序

        参数:
            chunks: 待编码的 Chunk 对象列表
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）

        返回:
            稠密向量列表（每个 Chunk 一个，顺序一致）。
            每个向量是与 Embedding 模型维度匹配的浮点数列表。

        异常:
            ValueError: 若 chunks 列表为空
            RuntimeError: 若 Embedding 提供方在所有批次上均失败

        示例:
            >>> chunks = [
            ...     Chunk(id="1", text="First chunk", metadata={}),
            ...     Chunk(id="2", text="Second chunk", metadata={})
            ... ]
            >>> vectors = encoder.encode(chunks)
            >>> len(vectors) == len(chunks)  # True
        """
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")
        
        # 从 Chunk 中提取文本
        texts = [chunk.text for chunk in chunks]

        # 校验所有文本均非空
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(
                    f"Chunk at index {i} (id={chunks[i].id}) has empty or whitespace-only text"
                )

        # 分批处理
        all_vectors: List[List[float]] = []
        
        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            try:
                # 调用 Embedding 提供方
                batch_vectors = self.embedding.embed(
                    texts=batch_texts,
                    trace=trace,
                )
                
                # 校验输出形状
                if len(batch_vectors) != len(batch_texts):
                    raise RuntimeError(
                        f"Embedding provider returned {len(batch_vectors)} vectors "
                        f"for {len(batch_texts)} texts in batch {batch_start}-{batch_end}"
                    )
                
                all_vectors.extend(batch_vectors)
                
            except Exception as e:
                # 重新抛出异常，并附带失败批次的信息
                raise RuntimeError(
                    f"Failed to encode batch {batch_start}-{batch_end}: {str(e)}"
                ) from e
        
        # 最终校验
        if len(all_vectors) != len(chunks):
            raise RuntimeError(
                f"Vector count mismatch: got {len(all_vectors)} vectors "
                f"for {len(chunks)} chunks"
            )
        
        # 校验向量维度一致
        if all_vectors:
            expected_dim = len(all_vectors[0])
            for i, vec in enumerate(all_vectors):
                if len(vec) != expected_dim:
                    raise RuntimeError(
                        f"Inconsistent vector dimensions: vector {i} has "
                        f"{len(vec)} dimensions, expected {expected_dim}"
                    )
        
        return all_vectors
    
    def get_batch_count(self, num_chunks: int) -> int:
        """计算给定 Chunk 数量所需的批次数。

        用于日志记录 / 进度跟踪的工具方法。

        参数:
            num_chunks: 待编码的 Chunk 数量

        返回:
            所需的批次数
        """
        if num_chunks <= 0:
            return 0
        return (num_chunks + self.batch_size - 1) // self.batch_size
