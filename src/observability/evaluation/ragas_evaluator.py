"""基于 Ragas 的 RAG 质量评估器。

该评估器封装 Ragas 框架以计算 LLM-as-Judge 指标：
- Faithfulness：回答是否忠实于检索到的上下文？
- Answer Relevancy：回答是否与查询相关？
- Context Precision：检索到的 chunk 是否相关且排序合理？

设计原则：
- 可插拔：实现 BaseEvaluator 接口，可通过工厂替换。
- 配置驱动：LLM/Embedding 后端从 settings.yaml 读取。
- 优雅降级：未安装 ragas 时给出清晰的 ImportError。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# 指标名称常量
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"

SUPPORTED_METRICS = {FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION}


def _import_ragas() -> None:
    """校验 ragas 可导入，不可导入时抛出清晰的错误。"""
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ragas' package is required for RagasEvaluator. "
            "Install it with: pip install ragas datasets"
        ) from exc


class RagasEvaluator(BaseEvaluator):
    """使用 Ragas 框架计算 LLM-as-Judge 指标的评估器。

    Ragas 不需要 ground-truth 标签。它使用 LLM 结合检索到的上下文
    评判生成回答的质量。

    支持的指标：
        - faithfulness：衡量与上下文的事实一致性。
        - answer_relevancy：衡量回答与查询的相关程度。
        - context_precision：衡量检索到的 chunk 的相关性与排序。

    示例::

        evaluator = RagasEvaluator(settings=settings)
        metrics = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"id": "c1", "text": "RAG is ..."}],
            generated_answer="RAG stands for ...",
        )
        # metrics == {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}
    """

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 RagasEvaluator。

        参数:
            settings: 应用配置（用于配置 LLM 后端）。
            metrics: 要计算的指标名称。默认为所有支持的指标。
            **kwargs: 额外参数（保留）。

        异常:
            ImportError: 未安装 ragas 时抛出。
            ValueError: 请求了不支持的指标名称时抛出。
        """
        _import_ragas()

        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalised = [m.strip().lower() for m in (metrics or [])]
        if not normalised:
            normalised = sorted(SUPPORTED_METRICS)

        unsupported = [m for m in normalised if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )

        self._metric_names = normalised

    # ── 公共 API ──────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """使用 Ragas LLM-as-Judge 指标评估 RAG 质量。

        参数:
            query: 用户查询字符串。
            retrieved_chunks: 检索到的 chunk（含 'text' 键的 dict 或字符串）。
            generated_answer: 生成的回答文本。Ragas 必需。
            ground_truth: 被 Ragas 忽略（LLM-as-Judge 不需要）。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: 额外参数。

        返回:
            指标名称到浮点分数（0.0 – 1.0）的字典。

        异常:
            ValueError: query/chunk 无效或缺少 generated_answer 时抛出。
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        if not generated_answer or not generated_answer.strip():
            raise ValueError(
                "RagasEvaluator requires a non-empty 'generated_answer'. "
                "Ragas uses LLM-as-Judge and needs the answer text to evaluate."
            )

        contexts = self._extract_texts(retrieved_chunks)

        try:
            result = self._run_ragas(query, contexts, generated_answer)
        except Exception as exc:
            logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ragas evaluation failed: {exc}") from exc

        return result

    # ── 私有辅助方法 ─────────────────────────────────────────────

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        answer: str,
    ) -> Dict[str, float]:
        """执行 Ragas collections 指标并返回归一化分数。

        Ragas 0.4+ 的 collections 指标使用各指标自身的 ``score()``，
        而非旧版的 ``evaluate()`` 流水线。每个指标的签名不同：
        - Faithfulness / ContextPrecision: (user_input, response, retrieved_contexts)
        - AnswerRelevancy: (user_input, response)
        """
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
        )

        # 从配置构建 LLM / Embedding 封装
        llm, embeddings = self._build_wrappers()

        scores: Dict[str, float] = {}

        for metric_name in self._metric_names:
            if metric_name == FAITHFULNESS:
                m = Faithfulness(llm=llm)
                result = m.score(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            elif metric_name == ANSWER_RELEVANCY:
                m = AnswerRelevancy(llm=llm, embeddings=embeddings)
                result = m.score(user_input=query, response=answer)
            elif metric_name == CONTEXT_PRECISION:
                m = ContextPrecisionWithoutReference(llm=llm)
                result = m.score(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            else:
                continue

            scores[metric_name] = float(result.value) if result.value is not None else 0.0

        return scores

    def _build_wrappers(self) -> tuple:
        """基于项目配置构建 Ragas 的 LLM 与 Embedding 封装。

        使用 Ragas 0.4+ 原生 API（InstructorLLM + OpenAIEmbeddings），
        而不是已废弃的 LangchainLLMWrapper。

        返回:
            (llm_wrapper, embeddings_wrapper) 元组。
        """
        from openai import AsyncAzureOpenAI, AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import OpenAIEmbeddings

        if self.settings is None:
            raise ValueError("Settings required to create LLM for Ragas evaluation")

        # ── LLM ──
        llm_cfg = self.settings.llm
        provider = llm_cfg.provider.lower()
        llm_azure_endpoint = getattr(llm_cfg, "azure_endpoint", None)

        # Azure 兼容模式：若配置了 azure_endpoint，则即使 provider 为
        # "openai" 也使用 Azure 客户端（与项目约定一致）。
        use_azure_llm = (
            provider == "azure"
            or (provider == "openai" and llm_azure_endpoint)
        )

        if use_azure_llm:
            llm_client = AsyncAzureOpenAI(
                api_key=llm_cfg.api_key,
                azure_endpoint=llm_azure_endpoint or llm_cfg.azure_endpoint,
                api_version=getattr(llm_cfg, "api_version", None) or "2024-02-15-preview",
            )
        elif provider == "openai":
            llm_client = AsyncOpenAI(api_key=llm_cfg.api_key)
        else:
            raise ValueError(
                f"Unsupported LLM provider for Ragas: '{provider}'. "
                "Supported: azure, openai"
            )

        llm = llm_factory(llm_cfg.model, client=llm_client, max_tokens=8192)

        # ── Embeddings ──
        emb_cfg = self.settings.embedding
        emb_provider = emb_cfg.provider.lower()
        emb_azure_endpoint = getattr(emb_cfg, "azure_endpoint", None)

        # Embedding 同样采用 Azure 兼容模式检测
        use_azure_emb = (
            emb_provider == "azure"
            or (emb_provider == "openai" and emb_azure_endpoint)
        )

        if use_azure_emb:
            emb_client = AsyncAzureOpenAI(
                api_key=emb_cfg.api_key,
                azure_endpoint=emb_azure_endpoint or emb_cfg.azure_endpoint,
                api_version=getattr(emb_cfg, "api_version", None) or "2024-02-15-preview",
            )
        elif emb_provider == "openai":
            emb_client = AsyncOpenAI(api_key=emb_cfg.api_key)
        else:
            raise ValueError(
                f"Unsupported embedding provider for Ragas: '{emb_provider}'. "
                "Supported: azure, openai"
            )

        embeddings = OpenAIEmbeddings(model=emb_cfg.model, client=emb_client)

        return llm, embeddings

    def _extract_texts(self, chunks: List[Any]) -> List[str]:
        """从各种 chunk 表示形式中提取文本字符串。

        参数:
            chunks: chunk dict、字符串或带 .text 属性的对象列表。

        返回:
            文本字符串列表。
        """
        texts: List[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content", "")
                texts.append(str(text))
            elif hasattr(chunk, "text"):
                texts.append(str(getattr(chunk, "text")))
            else:
                texts.append(str(chunk))
        return texts

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """若可用，从配置中提取指标列表。"""
        if settings is None:
            return []
        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []
        raw_metrics = getattr(evaluation, "metrics", None)
        if raw_metrics is None:
            return []
        # 仅保留 ragas 支持的指标
        return [m for m in raw_metrics if m.lower() in SUPPORTED_METRICS]
