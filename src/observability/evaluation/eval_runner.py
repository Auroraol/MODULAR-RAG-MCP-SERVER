"""用于批量质量评估的评估运行器。

EvalRunner 读取 golden 测试集，对每个测试用例运行 HybridSearch，
可选地生成回答，然后调用配置的 Evaluator 生成结构化的
评估报告。

设计原则：
- 配置驱动：通过 settings.yaml 选择 Evaluator。
- 可观测：生成包含逐查询详情的 EvalReport。
- 解耦：可与任意 BaseEvaluator 实现配合使用。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


@dataclass
class GoldenTestCase:
    """golden 测试集中的单个评估测试用例。

    属性:
        query: 测试查询字符串。
        expected_chunk_ids: 用于 IR 指标的 ground-truth chunk ID。
        expected_sources: ground-truth 源文件名（可选）。
        reference_answer: 用于 LLM-as-Judge 的参考答案文本（可选）。
    """

    query: str
    expected_chunk_ids: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    reference_answer: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoldenTestCase:
        return cls(
            query=data["query"],
            expected_chunk_ids=data.get("expected_chunk_ids", []),
            expected_sources=data.get("expected_sources", []),
            reference_answer=data.get("reference_answer"),
        )


@dataclass
class QueryResult:
    """评估单个测试用例的结果。

    属性:
        query: 测试查询。
        retrieved_chunk_ids: 实际检索到的 chunk ID。
        generated_answer: 生成的回答（如适用）。
        metrics: 该查询的评估指标。
        elapsed_ms: 检索 + 评估所耗时间。
    """

    query: str
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    generated_answer: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class EvalReport:
    """跨所有测试用例的聚合评估报告。

    属性:
        query_results: 逐查询的评估结果。
        aggregate_metrics: 所有查询的平均指标。
        total_elapsed_ms: 整个评估的总耗时。
        evaluator_name: 所用评估器的名称。
        test_set_path: golden 测试集文件路径。
    """

    query_results: List[QueryResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0
    evaluator_name: str = ""
    test_set_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """将报告序列化为字典。"""
        return {
            "evaluator_name": self.evaluator_name,
            "test_set_path": self.test_set_path,
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "aggregate_metrics": {
                k: round(v, 4) for k, v in self.aggregate_metrics.items()
            },
            "query_count": len(self.query_results),
            "query_results": [
                {
                    "query": qr.query,
                    "retrieved_chunk_ids": qr.retrieved_chunk_ids,
                    "generated_answer": qr.generated_answer,
                    "metrics": {k: round(v, 4) for k, v in qr.metrics.items()},
                    "elapsed_ms": round(qr.elapsed_ms, 1),
                }
                for qr in self.query_results
            ],
        }


def load_test_set(path: str | Path) -> List[GoldenTestCase]:
    """从 JSON 文件加载 golden 测试集。

    参数:
        path: golden 测试集 JSON 文件路径。

    返回:
        TestCase 实例列表。

    异常:
        FileNotFoundError: 文件不存在时抛出。
        ValueError: 文件格式无效时抛出。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Golden test set not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "test_cases" not in data:
        raise ValueError(
            "Invalid golden test set format: missing 'test_cases' key."
        )

    return [GoldenTestCase.from_dict(tc) for tc in data["test_cases"]]


class EvalRunner:
    """针对 golden 测试集运行批量评估。

    该类负责编排：
    1. 加载 golden 测试集
    2. 为每个查询运行 HybridSearch
    3. 可选地生成回答
    4. 调用评估器为每个结果打分
    5. 将指标聚合为 EvalReport

    示例::

        runner = EvalRunner(
            settings=settings,
            hybrid_search=hybrid_search,
            evaluator=evaluator,
        )
        report = runner.run("tests/fixtures/golden_test_set.json")
        print(report.aggregate_metrics)
    """

    def __init__(
        self,
        settings: Any = None,
        hybrid_search: Any = None,
        evaluator: Optional[BaseEvaluator] = None,
        answer_generator: Any = None,
        answer_overrides: Optional[Dict[int, str]] = None,
        reranker: Any = None,
    ) -> None:
        """初始化 EvalRunner。

        参数:
            settings: 应用配置。
            hybrid_search: 用于检索的 HybridSearch 实例。
            evaluator: 用于打分的 BaseEvaluator 实例。
            answer_generator: 可选的可调用对象 (query, chunks) -> str，
                用于生成回答。若为 None，则使用简单的拼接作为
                占位实现。
            answer_overrides: 可选的 dict，将测试用例索引（从 0 开始）
                映射到用户提供的回答字符串。存在时，该测试用例将使用
                覆盖回答，而不是自动生成。
            reranker: 可选的 CoreReranker 实例，用于对结果重排序。
        """
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator
        self.answer_generator = answer_generator
        self.answer_overrides = answer_overrides or {}
        self.reranker = reranker

    def run(
        self,
        test_set_path: str | Path,
        top_k: int = 10,
        collection: Optional[str] = None,
    ) -> EvalReport:
        """对 golden 测试集运行评估。

        参数:
            test_set_path: golden_test_set.json 的路径。
            top_k: 每个查询检索的 chunk 数量。
            collection: 可选的 collection 名称过滤。

        返回:
            包含逐查询指标与聚合指标的 EvalReport。

        异常:
            FileNotFoundError: 测试集文件不存在时抛出。
            ValueError: 未设置 evaluator 或 hybrid_search 时抛出。
        """
        if self.evaluator is None:
            raise ValueError("EvalRunner requires an evaluator.")

        test_cases = load_test_set(test_set_path)
        if not test_cases:
            raise ValueError("Golden test set is empty.")

        logger.info(
            "Starting evaluation: %d test cases, evaluator=%s",
            len(test_cases),
            type(self.evaluator).__name__,
        )

        report = EvalReport(
            evaluator_name=type(self.evaluator).__name__,
            test_set_path=str(test_set_path),
        )

        t0 = time.perf_counter()

        for idx, tc in enumerate(test_cases):
            logger.info("Evaluating [%d/%d]: %s", idx + 1, len(test_cases), tc.query[:60])
            # 若该索引存在用户提供的回答覆盖，则使用之
            answer_override = self.answer_overrides.get(idx)
            qr = self._evaluate_single(
                tc, top_k=top_k, collection=collection,
                answer_override=answer_override,
            )
            report.query_results.append(qr)

        report.total_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        report.aggregate_metrics = self._aggregate_metrics(report.query_results)

        logger.info(
            "Evaluation complete: %d queries, aggregate=%s",
            len(report.query_results),
            report.aggregate_metrics,
        )

        return report

    def _evaluate_single(
        self,
        test_case: GoldenTestCase,
        top_k: int = 10,
        collection: Optional[str] = None,
        answer_override: Optional[str] = None,
    ) -> QueryResult:
        """评估单个测试用例。

        参数:
            test_case: 要评估的测试用例。
            top_k: 检索结果数量。
            collection: 可选的 collection 过滤。
            answer_override: 用户提供的回答文本。设置时，将替代由
                chunk 自动生成的回答。

        返回:
            包含该测试用例指标的 QueryResult。
        """
        t0 = time.perf_counter()
        qr = QueryResult(query=test_case.query)

        # 步骤 1：检索 chunk
        retrieved_chunks = self._retrieve(test_case.query, top_k, collection)
        qr.retrieved_chunk_ids = [
            self._get_chunk_id(c) for c in retrieved_chunks
        ]

        # 步骤 2：生成回答 —— 优先使用用户覆盖，其次生成器，最后回退
        if answer_override:
            answer = answer_override
        else:
            answer = self._generate_answer(test_case.query, retrieved_chunks)
        qr.generated_answer = answer

        # 步骤 3：构建 ground truth
        ground_truth = (
            {"ids": test_case.expected_chunk_ids}
            if test_case.expected_chunk_ids
            else None
        )

        # 步骤 4：评估
        try:
            metrics = self.evaluator.evaluate(  # type: ignore[union-attr]
                query=test_case.query,
                retrieved_chunks=retrieved_chunks,
                generated_answer=answer,
                ground_truth=ground_truth,
            )
            qr.metrics = metrics
        except Exception as exc:
            logger.warning("Evaluation failed for '%s': %s", test_case.query[:40], exc)
            qr.metrics = {}

        qr.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return qr

    def _retrieve(
        self,
        query: str,
        top_k: int,
        collection: Optional[str],
    ) -> List[Any]:
        """使用 HybridSearch + 可选 Reranking 检索 chunk。

        若未配置搜索则回退为空列表。
        """
        if self.hybrid_search is None:
            logger.warning("No HybridSearch configured; returning empty results.")
            return []

        try:
            # 若启用 reranker，则检索更多候选结果
            has_reranker = self.reranker is not None and getattr(self.reranker, 'is_enabled', False)
            initial_top_k = top_k * 2 if has_reranker else top_k

            results = self.hybrid_search.search(
                query=query,
                top_k=initial_top_k,
            )
            results = results if isinstance(results, list) else results.results

            # 若已启用则应用重排序
            if has_reranker and results:
                rerank_result = self.reranker.rerank(query=query, results=results, top_k=top_k)
                results = rerank_result.results

            return results
        except Exception as exc:
            logger.warning("Retrieval failed for '%s': %s", query[:40], exc)
            return []

    def _generate_answer(self, query: str, chunks: List[Any]) -> str:
        """从检索到的 chunk 生成回答。

        若提供了自定义 answer_generator，则使用它。
        否则，将 chunk 文本拼接作为简单的占位实现。
        """
        if self.answer_generator is not None:
            try:
                return self.answer_generator(query, chunks)
            except Exception as exc:
                logger.warning("Answer generation failed: %s", exc)

        # 回退方案：拼接 chunk 文本
        texts = []
        for c in chunks:
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, dict):
                texts.append(c.get("text", str(c)))
            elif hasattr(c, "text"):
                texts.append(str(getattr(c, "text")))
            else:
                texts.append(str(c))

        return " ".join(texts[:5])  # 前 5 个 chunk

    def _get_chunk_id(self, chunk: Any) -> str:
        """从各种表示形式中提取 chunk ID。"""
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            for key in ("id", "chunk_id"):
                if key in chunk:
                    return str(chunk[key])
            return str(chunk)
        if hasattr(chunk, "chunk_id"):
            return str(getattr(chunk, "chunk_id"))
        if hasattr(chunk, "id"):
            return str(getattr(chunk, "id"))
        return str(chunk)

    @staticmethod
    def _aggregate_metrics(results: List[QueryResult]) -> Dict[str, float]:
        """计算所有查询结果的平均指标。

        参数:
            results: 带逐查询指标的 QueryResult 列表。

        返回:
            平均指标值字典。
        """
        if not results:
            return {}

        # 收集所有指标键
        all_keys: set[str] = set()
        for qr in results:
            all_keys.update(qr.metrics.keys())

        # 计算每个指标的平均值
        averages: Dict[str, float] = {}
        for key in sorted(all_keys):
            values = [qr.metrics[key] for qr in results if key in qr.metrics]
            averages[key] = sum(values) / len(values) if values else 0.0

        return averages
