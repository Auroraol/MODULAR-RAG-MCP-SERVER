"""稀疏编码器，用于从文本 Chunk 生成 BM25 词项统计。

本模块实现 Ingestion Pipeline 的稀疏编码器组件，负责提取
BM25 索引所需的词项统计。

设计原则：
- 无状态处理：encode() 调用之间无内部状态
- 可观测：接受 TraceContext 以支持未来的可观测性集成
- 确定性：相同输入产生相同的词项统计
- 契约清晰：为下游 BM25Indexer 定义明确的输出结构
"""

from typing import List, Dict, Optional, Any
from collections import Counter

from src.core.types import Chunk
from src.core.text_utils import tokenize as shared_tokenize


class SparseEncoder:
    """将文本 Chunk 编码为 BM25 词项统计。

    该编码器准备 BM25 索引所需的词项级统计。
    实际的索引构建由 BM25Indexer（C12）负责。

    输出结构：
        对每个 Chunk 生成：
        {
            "chunk_id": str,
            "term_frequencies": Dict[str, int],  # 词项 -> 在此 Chunk 中的出现次数
            "doc_length": int,                    # Chunk 中的词项数
            "unique_terms": int                   # Chunk 中的词表大小
        }

    设计：
    - 分词：简单按空白切分 + 小写化（后续可增强）
    - 停用词：默认无（可在后续迭代中添加）
    - 确定性：相同 Chunk 文本始终产生相同统计

    示例:
        >>> from src.core.types import Chunk
        >>> encoder = SparseEncoder()
        >>> 
        >>> chunks = [Chunk(id="1", text="Hello world hello", metadata={})]
        >>> stats = encoder.encode(chunks)
        >>> stats[0]["term_frequencies"]["hello"]  # 2
        >>> stats[0]["doc_length"]  # 3
    """
    
    def __init__(
        self,
        min_term_length: int = 2,
        lowercase: bool = True,
    ):
        """初始化 SparseEncoder。

        参数:
            min_term_length: 词项的最小字符长度（默认：2）
            lowercase: 是否将词项转换为小写（默认：True）

        异常:
            ValueError: 若 min_term_length < 1
        """
        if min_term_length < 1:
            raise ValueError(f"min_term_length must be >= 1, got {min_term_length}")
        
        self.min_term_length = min_term_length
        self.lowercase = lowercase
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """将 Chunk 编码为 BM25 词项统计。

        对每个 Chunk 提取：
        - 词项频率（词项 -> 出现次数）
        - 文档长度（词项总数）
        - 唯一词项数

        参数:
            chunks: 待编码的 Chunk 对象列表
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）

        返回:
            统计字典列表（每个 Chunk 一个，顺序一致）。
            每个字典包含：chunk_id、term_frequencies、doc_length、unique_terms

        异常:
            ValueError: 若 chunks 列表为空
            ValueError: 若任一 Chunk 的文本为空

        示例:
            >>> chunks = [
            ...     Chunk(id="1", text="machine learning", metadata={}),
            ...     Chunk(id="2", text="deep learning networks", metadata={})
            ... ]
            >>> stats = encoder.encode(chunks)
            >>> len(stats) == len(chunks)  # True
            >>> stats[0]["term_frequencies"]["machine"]  # 1
            >>> stats[1]["doc_length"]  # 3
        """
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")
        
        results = []
        
        for i, chunk in enumerate(chunks):
            # 校验 Chunk 文本
            if not chunk.text or not chunk.text.strip():
                raise ValueError(
                    f"Chunk at index {i} (id={chunk.id}) has empty or whitespace-only text"
                )

            # 分词并统计词项
            terms = self._tokenize(chunk.text)
            term_frequencies = Counter(terms)

            # 构建统计字典
            stat_dict = {
                "chunk_id": chunk.id,
                "term_frequencies": dict(term_frequencies),  # 将 Counter 转为 dict
                "doc_length": len(terms),
                "unique_terms": len(term_frequencies),
            }
            
            results.append(stat_dict)
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """将文本分词为词项。

        复用 ``src.core.text_utils.tokenize``，确保与查询侧
        （QueryProcessor）的分词方式完全一致 —— 这是 BM25 匹配的必要条件。

        参数:
            text: 待分词的输入文本

        返回:
            有效词项列表
        """
        tokens = shared_tokenize(text)

        # 若配置开启则应用小写化
        if self.lowercase:
            tokens = [t.lower() for t in tokens]

        # 按最小长度过滤
        terms = [t for t in tokens if len(t) >= self.min_term_length]

        return terms
    
    def get_corpus_stats(
        self,
        encoded_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """从已编码的 Chunk 计算语料库级统计。

        供 BM25Indexer 使用的工具方法，用于计算：
        - 平均文档长度
        - 文档频率（包含每个词项的文档数量）
        - 文档总数

        参数:
            encoded_chunks: 来自 encode() 的统计字典列表

        返回:
            包含语料库级统计的字典：
            {
                "num_docs": int,
                "avg_doc_length": float,
                "document_frequency": Dict[str, int]  # 词项 -> 包含它的文档数
            }
        """
        if not encoded_chunks:
            return {
                "num_docs": 0,
                "avg_doc_length": 0.0,
                "document_frequency": {}
            }
        
        num_docs = len(encoded_chunks)
        total_length = sum(chunk["doc_length"] for chunk in encoded_chunks)
        avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0
        
        # 计算每个词项的文档频率（DF）
        doc_freq: Dict[str, int] = {}
        for chunk_stats in encoded_chunks:
            # 此 Chunk 中的每个唯一词项对 DF 贡献 1
            for term in chunk_stats["term_frequencies"].keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1
        
        return {
            "num_docs": num_docs,
            "avg_doc_length": avg_doc_length,
            "document_frequency": doc_freq,
        }
