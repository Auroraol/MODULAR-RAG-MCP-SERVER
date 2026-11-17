"""基于 Cross-Encoder 的 Reranker 实现。

本模块使用 Cross-Encoder 模型实现重排，直接对 (query, passage) 对打分。
既支持通过 sentence-transformers 加载本地模型，也支持基于 API 的端点。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderRerankError(RuntimeError):
    """Cross-Encoder 重排失败时抛出。"""


class CrossEncoderReranker(BaseReranker):
    """基于 Cross-Encoder 的 Reranker，用于对 query-passage 对打分。

    此实现使用 Cross-Encoder 模型（如 ms-marco-MiniLM），
    直接编码并对 (query, passage) 对打分，相比 bi-encoder 方法
    能提供更准确的相关性分数，但计算开销更高。

    应用的设计原则：
    - 可插拔：可通过工厂替换为其他 Reranker 实现。
    - 配置驱动：模型名称与参数来自 settings.yaml。
    - 可观测：支持 TraceContext 以进行监控（阶段 F 集成）。
    - 可降级：为上游降级逻辑提供超时/失败信号。
    - 确定性测试：支持注入 mock scorer 以便测试。
    """
    
    def __init__(
        self,
        settings: Any,
        model: Optional[Any] = None,
        timeout: float = 10.0,
        **kwargs: Any
    ) -> None:
        """初始化 Cross-Encoder Reranker。

        参数:
            settings: 包含 rerank 配置的应用设置。
            model: 可选的已初始化 CrossEncoder 模型。若为 None，则从
                settings.rerank.model 创建。用于测试时注入 mock 模型。
            timeout: 等待重排完成的最长时间（秒）。默认 10 秒。
                用于在重排耗时过长时启用降级策略。
            **kwargs: 额外的 provider 特定参数。
        """
        self.settings = settings
        self.timeout = timeout
        self.kwargs = kwargs
        
        # 初始化或注入模型
        if model is not None:
            self.model = model
        else:
            try:
                model_name = self._get_model_name_from_settings(settings)
                self.model = self._load_cross_encoder_model(model_name)
            except Exception as e:
                raise CrossEncoderRerankError(
                    f"Failed to initialize Cross-Encoder model: {e}"
                ) from e
    
    def _get_model_name_from_settings(self, settings: Any) -> str:
        """从配置中提取模型名称。

        参数:
            settings: 应用设置。

        返回:
            模型名称字符串。

        异常:
            AttributeError: 当 rerank.model 未配置时。
        """
        try:
            model_name = settings.rerank.model
            if not model_name or not isinstance(model_name, str):
                raise ValueError("Model name must be a non-empty string")
            return model_name
        except AttributeError as e:
            raise AttributeError(
                "Missing configuration: settings.rerank.model. "
                "Please specify 'rerank.model' in settings.yaml"
            ) from e
    
    def _load_cross_encoder_model(self, model_name: str) -> Any:
        """加载 Cross-Encoder 模型。

        参数:
            model_name: Cross-Encoder 模型的名称或路径。

        返回:
            已初始化的 CrossEncoder 实例。

        异常:
            ImportError: 当未安装 sentence-transformers 时。
            RuntimeError: 当模型加载失败时。
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for Cross-Encoder reranking. "
                "Install it with: pip install sentence-transformers"
            ) from e
        
        try:
            logger.info(f"Loading Cross-Encoder model: {model_name}")
            model = CrossEncoder(model_name)
            logger.info(f"Cross-Encoder model loaded successfully: {model_name}")
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Cross-Encoder model '{model_name}': {e}"
            ) from e
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """使用 Cross-Encoder 打分对候选进行重排。

        参数:
            query: 用户查询字符串。
            candidates: 待重排的候选记录列表。每项必须包含
                'text' 或 'content' 字段用于打分。
            trace: 可选的 TraceContext，用于可观测性（阶段 F 集成）。
            **kwargs: 额外参数（如 top_k 用于限制输出数量）。

        返回:
            按相关性分数降序排列的重排后候选列表。
            每个候选会包含 'rerank_score' 字段，保存模型的分数。

        异常:
            ValueError: 当 query 或 candidates 无效时。
            CrossEncoderRerankError: 当打分失败或超时时。
        """
        # 校验输入
        self.validate_query(query)
        self.validate_candidates(candidates)

        # 提取 top_k 参数
        top_k = kwargs.get("top_k", len(candidates))
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        try:
            # 准备用于打分的 (query, passage) 对
            pairs = self._prepare_pairs(query, candidates)

            # 使用模型对各个对进行打分
            scores = self._score_pairs(pairs, trace=trace)

            # 将分数附加到候选并排序
            reranked = self._attach_scores_and_sort(candidates, scores, top_k)

            if trace:
                self._log_trace(trace, query, len(candidates), len(reranked))

            return reranked

        except Exception as e:
            logger.error(f"Cross-Encoder reranking failed: {e}", exc_info=True)
            # 为上游降级逻辑发出失败信号
            raise CrossEncoderRerankError(
                f"Cross-Encoder reranking failed: {e}"
            ) from e
    
    def _prepare_pairs(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[tuple[str, str]]:
        """为打分准备 (query, passage) 对。

        参数:
            query: 用户查询。
            candidates: 候选记录列表。

        返回:
            (query, passage_text) 元组列表。
        """
        pairs = []
        for candidate in candidates:
            # 从候选中提取文本（同时支持 'text' 和 'content' 键）
            text = candidate.get("text") or candidate.get("content", "")
            if not isinstance(text, str):
                text = str(text)
            pairs.append((query, text))
        return pairs
    
    def _score_pairs(
        self,
        pairs: List[tuple[str, str]],
        trace: Optional[Any] = None
    ) -> List[float]:
        """使用 Cross-Encoder 模型对 (query, passage) 对打分。

        参数:
            pairs: (query, passage) 元组列表。
            trace: 可选的 TraceContext，用于可观测性。

        返回:
            相关性分数列表（每个对一个分数）。

        异常:
            CrossEncoderRerankError: 当打分失败或超时时。
        """
        try:
            # 使用 model.predict() 批量对所有对打分
            scores = self.model.predict(pairs)

            # 必要时将 numpy 数组转换为列表
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            
            return scores
            
        except Exception as e:
            raise CrossEncoderRerankError(
                f"Failed to score pairs with Cross-Encoder: {e}"
            ) from e
    
    def _attach_scores_and_sort(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """将分数附加到候选并按相关性排序。

        参数:
            candidates: 原始候选列表。
            scores: 模型给出的相关性分数。
            top_k: 返回的候选数量上限。

        返回:
            排序后的前 top_k 个候选列表，并新增 'rerank_score' 字段。
        """
        # 将分数附加到候选
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            # 创建副本以避免修改原始数据
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = float(score)
            scored_candidates.append(candidate_copy)

        # 按分数降序排序并取前 top_k 个
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        
        return sorted_candidates[:top_k]
    
    def _log_trace(
        self,
        trace: Any,
        query: str,
        input_count: int,
        output_count: int
    ) -> None:
        """将重排操作记录到 trace 上下文。

        参数:
            trace: TraceContext 实例。
            query: 查询字符串。
            input_count: 输入候选数量。
            output_count: 输出候选数量。
        """
        # 阶段 F 集成的占位
        # 未来：trace.log_rerank_step(...)
        logger.debug(
            f"Cross-Encoder rerank: query='{query[:50]}...', "
            f"input={input_count}, output={output_count}"
        )
