"""用于合并多路检索结果的倒数排名融合（RRF）。

本模块实现 RRF 融合算法，将稠密检索与稀疏检索产生的排名列表
合并为统一的排名。RRF 是一种简单而有效的排名聚合方法，
无需进行分数归一化。

参考:
    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    "Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
    SIGIR '09.

RRF 倒数排名融合
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)


class RRFFusion:
    """用于合并多路排名列表的倒数排名融合（RRF）。

    RRF 使用以下公式合并来自多个来源的排名：
        RRF_score(d) = Σ 1 / (k + rank(d))

    其中：
        - d 是文档（Chunk）
        - k 是平滑常数（通常为 60）
        - rank(d) 是文档 d 在某个排名列表中的从 1 开始的排名

    关键特性：
    - 确定性：相同输入始终产生相同的输出顺序
    - 与分数无关：仅使用排名位置，不使用原始分数
    - 无需归一化：可处理不同量纲的分数
    - 兼容缺失文档：仅出现在一个列表中的文档仍会贡献分数

    遵循的设计原则：
    - 配置驱动：k 参数可配置（默认：60）
    - 类型安全：返回标准化的 RetrievalResult 对象
    - 确定性：稳定排序，并以 chunk_id 作为并列时的决胜条件
    - 可观测：为融合过程提供日志

    属性:
        k: RRF 公式的平滑常数（默认：60）。
           k 越大，低排名文档获得的权重越高。

    示例:
        >>> fusion = RRFFusion(k=60)
        >>> dense_results = [
        ...     RetrievalResult(chunk_id="a", score=0.9, text="...", metadata={}),
        ...     RetrievalResult(chunk_id="b", score=0.8, text="...", metadata={}),
        ... ]
        >>> sparse_results = [
        ...     RetrievalResult(chunk_id="b", score=5.2, text="...", metadata={}),
        ...     RetrievalResult(chunk_id="c", score=4.1, text="...", metadata={}),
        ... ]
        >>> fused = fusion.fuse([dense_results, sparse_results], top_k=5)
    """

    # 原始 RRF 论文推荐的默认平滑常数
    DEFAULT_K = 60

    def __init__(self, k: int = DEFAULT_K) -> None:
        """使用可配置的平滑常数初始化 RRF 融合。

        参数:
            k: RRF 公式的平滑常数（默认：60）。
               - 必须为正整数
               - 数值越大，排名差异的重要性越低
               - 常用取值：60（原始论文）、20、100

        异常:
            ValueError: k 不是正整数时。
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        
        self.k = k
        logger.info(f"RRFFusion initialized with k={k}")
    
    def fuse(
        self,
        ranking_lists: List[List[RetrievalResult]],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """使用倒数排名融合合并多个排名列表。

        参数:
            ranking_lists: 排名列表的列表，每个列表包含按相关性降序
                           排序的 RetrievalResult 对象。
                           通常为 [dense_results, sparse_results]。
            top_k: 返回的最大结果数量。为 None 时返回全部。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。

        返回:
            RetrievalResult 对象列表，按融合后的 RRF 分数降序排序。
            score 字段保存 RRF 分数，而非原始检索分数。
            text 与 metadata 保留自每个 Chunk 的首次出现。

        异常:
            ValueError: ranking_lists 为空时。

        注意:
            - 出现在多个列表中的文档会获得所有列表的贡献
            - 仅出现在一个列表中的文档仍会获得 RRF 分数
            - 并列处理：RRF 分数相同时，按 chunk_id 排序以保证稳定性

        示例:
            >>> fusion = RRFFusion(k=60)
            >>> fused = fusion.fuse([dense_results, sparse_results], top_k=10)
            >>> for r in fused:
            ...     print(f"[RRF={r.score:.4f}] {r.chunk_id}")
        """
        if not ranking_lists:
            raise ValueError("ranking_lists cannot be empty")

        # 过滤掉空列表
        non_empty_lists = [lst for lst in ranking_lists if lst]

        if not non_empty_lists:
            logger.debug("All ranking lists are empty, returning empty result")
            return []

        logger.debug(
            f"Fusing {len(non_empty_lists)} ranking lists with "
            f"sizes {[len(lst) for lst in non_empty_lists]}"
        )

        # 步骤 1：为每个唯一 Chunk 计算 RRF 分数
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, RetrievalResult] = {}  # 保留 text/metadata

        for list_idx, ranking_list in enumerate(non_empty_lists):
            for rank, result in enumerate(ranking_list, start=1):
                chunk_id = result.chunk_id

                # 计算 RRF 贡献：1 / (k + rank)
                rrf_contribution = 1.0 / (self.k + rank)

                # 累加分数
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    # 保存首次出现的数据（text、metadata）
                    chunk_data[chunk_id] = result

                rrf_scores[chunk_id] += rrf_contribution

        logger.debug(f"Computed RRF scores for {len(rrf_scores)} unique chunks")

        # 步骤 2：使用 RRF 分数构建融合结果
        fused_results = []
        for chunk_id, rrf_score in rrf_scores.items():
            original = chunk_data[chunk_id]
            fused_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=rrf_score,
                    text=original.text,
                    metadata=original.metadata.copy(),
                )
            )

        # 步骤 3：按 RRF 分数降序排序，再按 chunk_id 排序以保证稳定性
        fused_results.sort(key=lambda r: (-r.score, r.chunk_id))

        # 步骤 4：若指定则应用 top_k 限制
        if top_k is not None and top_k > 0:
            fused_results = fused_results[:top_k]
        
        logger.debug(
            f"Fusion complete: {len(fused_results)} results "
            f"(top_k={top_k if top_k else 'all'})"
        )
        
        return fused_results
    
    def fuse_with_weights(
        self,
        ranking_lists: List[List[RetrievalResult]],
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """使用可选的逐列表权重合并多个排名列表。

        这是 fuse() 的扩展版本，允许为不同的排名来源设置权重。
        例如，对语义查询给予稠密检索更高权重，
        或对关键词查询给予稀疏检索更高权重。

        参数:
            ranking_lists: 排名列表的列表，每个列表包含 RetrievalResult 对象。
            weights: 可选，每个排名列表的权重列表（默认：均匀）。
                     提供时必须与 ranking_lists 长度相同。
                     权重会与 RRF 贡献相乘。
            top_k: 返回的最大结果数量。为 None 时返回全部。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。

        返回:
            RetrievalResult 对象列表，按加权 RRF 分数降序排序。

        异常:
            ValueError: ranking_lists 为空或 weights 长度不匹配时。

        示例:
            >>> fusion = RRFFusion(k=60)
            >>> # 为稠密结果赋予 1.5 倍权重
            >>> fused = fusion.fuse_with_weights(
            ...     [dense_results, sparse_results],
            ...     weights=[1.5, 1.0],
            ...     top_k=10
            ... )
        """
        if not ranking_lists:
            raise ValueError("ranking_lists cannot be empty")

        # 默认使用均匀权重
        if weights is None:
            weights = [1.0] * len(ranking_lists)

        if len(weights) != len(ranking_lists):
            raise ValueError(
                f"weights length ({len(weights)}) must match "
                f"ranking_lists length ({len(ranking_lists)})"
            )

        # 校验权重
        for i, w in enumerate(weights):
            if not isinstance(w, (int, float)) or w < 0:
                raise ValueError(f"Weight at index {i} must be non-negative, got {w}")

        # 过滤掉空列表（保持权重对齐）
        filtered = [
            (lst, w) for lst, w in zip(ranking_lists, weights) if lst
        ]

        if not filtered:
            logger.debug("All ranking lists are empty, returning empty result")
            return []

        non_empty_lists, filtered_weights = zip(*filtered)

        logger.debug(
            f"Fusing {len(non_empty_lists)} ranking lists with "
            f"weights={list(filtered_weights)}"
        )

        # 计算加权 RRF 分数
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, RetrievalResult] = {}

        for list_idx, (ranking_list, weight) in enumerate(zip(non_empty_lists, filtered_weights)):
            for rank, result in enumerate(ranking_list, start=1):
                chunk_id = result.chunk_id

                # 加权 RRF 贡献
                rrf_contribution = weight * (1.0 / (self.k + rank))

                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    chunk_data[chunk_id] = result

                rrf_scores[chunk_id] += rrf_contribution

        # 构建并排序结果
        fused_results = [
            RetrievalResult(
                chunk_id=chunk_id,
                score=rrf_score,
                text=chunk_data[chunk_id].text,
                metadata=chunk_data[chunk_id].metadata.copy(),
            )
            for chunk_id, rrf_score in rrf_scores.items()
        ]

        fused_results.sort(key=lambda r: (-r.score, r.chunk_id))

        if top_k is not None and top_k > 0:
            fused_results = fused_results[:top_k]

        return fused_results


def rrf_score(rank: int, k: int = RRFFusion.DEFAULT_K) -> float:
    """计算单个排名位置的 RRF 分数贡献。

    这是一个用于计算单个 RRF 贡献的工具函数。

    参数:
        rank: 从 1 开始的排名位置（1 表示最高排名）
        k: 平滑常数（默认：60）

    返回:
        RRF 分数贡献：1 / (k + rank)

    异常:
        ValueError: rank 不是正整数或 k 不为正数时。

    示例:
        >>> rrf_score(1, k=60)  # 排名第一的文档
        0.01639344262295082
        >>> rrf_score(10, k=60)  # 排名第十的文档
        0.014285714285714285
    """
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"rank must be a positive integer, got {rank}")
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    
    return 1.0 / (k + rank)
