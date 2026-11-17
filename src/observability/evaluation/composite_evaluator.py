"""组合多个评估器的复合评估器。

该评估器实现了 Composite 模式：持有一个 BaseEvaluator 实例列表，
依次运行它们，并将各自的指标字典合并为单一结果。

设计原则：
- 可插拔：任意 BaseEvaluator 均可组合。
- 配置驱动：`evaluation.backends: [ragas, custom]` 驱动组合。
- 可观测：记录各评估器成功/失败情况。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class CompositeEvaluator(BaseEvaluator):
    """组合多个评估器并合并指标的评估器。

    每个子评估器均以相同参数调用。结果合并到单一指标字典中。
    若两个评估器产生相同的指标键，则后者的值胜出（并记录警告）。

    允许部分失败：若某个子评估器失败，其错误会被记录，其余
    评估器仍照常执行。

    示例::

        composite = CompositeEvaluator(evaluators=[
            CustomEvaluator(metrics=["hit_rate", "mrr"]),
            RagasEvaluator(metrics=["faithfulness"]),
        ])
        metrics = composite.evaluate(
            query="test", retrieved_chunks=[...],
            generated_answer="...", ground_truth=[...]
        )
        # metrics == {"hit_rate": 1.0, "mrr": 0.5, "faithfulness": 0.92}
    """

    def __init__(
        self,
        evaluators: Optional[Sequence[BaseEvaluator]] = None,
        settings: Any = None,
        **kwargs: Any,
    ) -> None:
        """初始化 CompositeEvaluator。

        参数:
            evaluators: 预先构建的评估器实例。若为 None，则依据 settings 构建。
            settings: 应用配置（用于配置驱动的组合）。
            **kwargs: 转发给子评估器的额外参数。

        异常:
            ValueError: 未提供评估器且 settings 中未指定 backend 时抛出。
        """
        self.settings = settings
        self.kwargs = kwargs

        if evaluators is not None:
            self._evaluators: List[BaseEvaluator] = list(evaluators)
        else:
            self._evaluators = self._build_from_settings(settings, **kwargs)

        if not self._evaluators:
            raise ValueError(
                "CompositeEvaluator requires at least one sub-evaluator. "
                "Provide evaluators directly or configure "
                "'evaluation.backends' in settings.yaml."
            )

        logger.info(
            "CompositeEvaluator initialised with %d evaluator(s): %s",
            len(self._evaluators),
            [type(e).__name__ for e in self._evaluators],
        )

    @property
    def evaluators(self) -> List[BaseEvaluator]:
        """返回已组合的评估器列表。"""
        return list(self._evaluators)

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """运行所有子评估器并合并其指标。

        参数:
            query: 用户查询字符串。
            retrieved_chunks: 检索到的 chunk 或记录。
            generated_answer: 可选的生成回答文本。
            ground_truth: 可选的 ground truth 数据。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: 额外参数。

        返回:
            所有指标名称到浮点值的合并字典。

        异常:
            RuntimeError: 当所有子评估器均失败时抛出。
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        merged: Dict[str, float] = {}
        errors: List[str] = []

        for evaluator in self._evaluators:
            name = type(evaluator).__name__
            try:
                metrics = evaluator.evaluate(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    generated_answer=generated_answer,
                    ground_truth=ground_truth,
                    trace=trace,
                    **kwargs,
                )
                for key, value in metrics.items():
                    if key in merged:
                        logger.warning(
                            "Metric '%s' produced by multiple evaluators; "
                            "overwriting with value from %s",
                            key,
                            name,
                        )
                    merged[key] = value

                logger.debug(
                    "%s produced %d metric(s): %s",
                    name,
                    len(metrics),
                    list(metrics.keys()),
                )

            except Exception as exc:
                msg = f"{name} failed: {exc}"
                logger.warning(msg)
                errors.append(msg)

        if not merged and errors:
            raise RuntimeError(
                "All sub-evaluators failed:\n" + "\n".join(errors)
            )

        return merged

    # ── 配置驱动的构建器 ────────────────────────────────────────

    @staticmethod
    def _build_from_settings(
        settings: Any,
        **kwargs: Any,
    ) -> List[BaseEvaluator]:
        """依据 settings.evaluation.backends 构建子评估器。

        预期的配置格式::

            evaluation:
              enabled: true
              provider: composite
              backends:
                - ragas
                - custom
              metrics:
                - faithfulness
                - hit_rate
                - mrr

        参数:
            settings: 应用配置。
            **kwargs: 转发给各子评估器构造函数。

        返回:
            BaseEvaluator 实例列表。
        """
        if settings is None:
            return []

        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []

        backends = getattr(evaluation, "backends", None)
        if not backends:
            return []

        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        evaluators: List[BaseEvaluator] = []
        for backend_name in backends:
            backend_name = str(backend_name).strip().lower()
            if backend_name in {"composite", "none", "disabled"}:
                continue  # 避免无限递归 / 空操作

            try:
                # 创建一个覆盖了 provider 的 mock settings
                from unittest.mock import MagicMock

                sub_settings = MagicMock(wraps=settings)
                sub_eval = MagicMock()
                sub_eval.enabled = True
                sub_eval.provider = backend_name
                sub_eval.metrics = getattr(evaluation, "metrics", [])
                sub_eval.backends = []  # 防止递归
                sub_settings.evaluation = sub_eval

                evaluator = EvaluatorFactory.create(sub_settings, **kwargs)
                evaluators.append(evaluator)
                logger.info("CompositeEvaluator: loaded backend '%s'", backend_name)
            except Exception as exc:
                logger.warning(
                    "CompositeEvaluator: failed to load backend '%s': %s",
                    backend_name,
                    exc,
                )

        return evaluators
