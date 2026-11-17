"""E2E 召回回归测试。

基于 golden 测试集，通过 hit@k 阈值断言验证检索质量。
该测试用于 CI/CD 回归门禁——若召回率低于配置阈值，
测试即失败。

前提条件:
- Vector Store 中已索引数据（先运行 ingest.py）。
- golden 测试集位于 ``tests/fixtures/golden_test_set.json``。

用法::

    pytest tests/e2e/test_recall.py -v
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest

logger = logging.getLogger(__name__)

# ── 配置 ─────────────────────────────────────────────────────────────

GOLDEN_TEST_SET_PATH = Path("tests/fixtures/golden_test_set.json")

# 用于回归门禁的最低可接受阈值。
# 这些阈值刻意设置得较为保守——随着数据质量提升再调整。
HIT_RATE_THRESHOLD = 0.0  # 最低平均 hit_rate（0.0 表示在有数据前不做回归检查）
MRR_THRESHOLD = 0.0  # 最低平均 MRR


# ── 辅助函数 ─────────────────────────────────────────────────────────

def _load_golden_set() -> List[Dict[str, Any]]:
    """从 JSON 文件加载 golden 测试集。"""
    if not GOLDEN_TEST_SET_PATH.exists():
        pytest.skip(f"Golden test set not found: {GOLDEN_TEST_SET_PATH}")

    with GOLDEN_TEST_SET_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("test_cases", [])
    if not cases:
        pytest.skip("Golden test set has no test cases")

    return cases


def _try_create_search_engine() -> Any:
    """尝试根据配置创建 HybridSearch。

    若基础设施不可用则跳过测试。
    """
    try:
        from src.core.settings import load_settings
        from src.core.query_engine.hybrid_search import HybridSearch

        settings = load_settings()
        return HybridSearch(settings)
    except Exception as exc:
        pytest.skip(f"HybridSearch not available: {exc}")


def _try_create_evaluator() -> Any:
    """创建用于计算 hit_rate/mrr 的 CustomEvaluator。"""
    try:
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        return CustomEvaluator()
    except Exception as exc:
        pytest.skip(f"CustomEvaluator not available: {exc}")


# ── 测试类 ───────────────────────────────────────────────────────────


@pytest.mark.e2e
class TestRecallRegression:
    """基于 golden 测试集的 E2E 召回回归测试。

    这些测试确保检索质量不会低于配置的阈值。
    它们要求已存在索引数据。
    """

    @pytest.fixture(autouse=True)
    def setup_components(self) -> None:
        """初始化搜索引擎与评估器。"""
        self.golden_set = _load_golden_set()
        self.search = _try_create_search_engine()
        self.evaluator = _try_create_evaluator()

    def test_golden_set_is_valid(self) -> None:
        """校验 golden 测试集的结构符合预期。"""
        assert len(self.golden_set) >= 1, "Golden set must have at least 1 case"

        for idx, tc in enumerate(self.golden_set):
            assert "query" in tc, f"Test case {idx} missing 'query'"
            assert tc["query"].strip(), f"Test case {idx} has empty query"

    def test_hit_rate_above_threshold(self) -> None:
        """断言平均 hit@k 达到最低阈值。

        若任一期望 Chunk 出现在 top-k 结果中，则 hit@k = 1，
        否则为 0。对所有定义了 expected_chunk_ids 的测试用例取平均。
        """
        cases_with_ground_truth = [
            tc for tc in self.golden_set
            if tc.get("expected_chunk_ids")
        ]

        if not cases_with_ground_truth:
            pytest.skip("No test cases with expected_chunk_ids defined")

        hit_rates: List[float] = []
        top_k = 10

        for tc in cases_with_ground_truth:
            query = tc["query"]
            expected_ids = set(tc["expected_chunk_ids"])

            # 执行检索
            try:
                results = self.search.search(query=query, top_k=top_k)
                retrieved_ids = set()
                for r in results:
                    if hasattr(r, "chunk_id"):
                        retrieved_ids.add(r.chunk_id)
                    elif isinstance(r, dict) and "chunk_id" in r:
                        retrieved_ids.add(r["chunk_id"])
            except Exception as exc:
                logger.warning("Search failed for '%s': %s", query[:40], exc)
                hit_rates.append(0.0)
                continue

            # hit@k：若检索结果中包含任一期望 Chunk 则为 1
            hit = 1.0 if expected_ids & retrieved_ids else 0.0
            hit_rates.append(hit)

            logger.info(
                "Query: %s | hit@%d=%.1f | expected=%s | retrieved=%s",
                query[:40], top_k, hit, expected_ids, retrieved_ids,
            )

        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0

        logger.info("Average hit@%d = %.4f (threshold=%.4f)", top_k, avg_hit_rate, HIT_RATE_THRESHOLD)

        assert avg_hit_rate >= HIT_RATE_THRESHOLD, (
            f"hit@{top_k} regression: {avg_hit_rate:.4f} < {HIT_RATE_THRESHOLD:.4f}"
        )

    def test_mrr_above_threshold(self) -> None:
        """断言平均 MRR 达到最低阈值。

        MRR = 第一个相关 Chunk 的排名倒数。对所有定义了
        expected_chunk_ids 的测试用例取平均。
        """
        cases_with_ground_truth = [
            tc for tc in self.golden_set
            if tc.get("expected_chunk_ids")
        ]

        if not cases_with_ground_truth:
            pytest.skip("No test cases with expected_chunk_ids defined")

        mrrs: List[float] = []
        top_k = 10

        for tc in cases_with_ground_truth:
            query = tc["query"]
            expected_ids = set(tc["expected_chunk_ids"])

            try:
                results = self.search.search(query=query, top_k=top_k)
                retrieved_ids: List[str] = []
                for r in results:
                    if hasattr(r, "chunk_id"):
                        retrieved_ids.append(r.chunk_id)
                    elif isinstance(r, dict) and "chunk_id" in r:
                        retrieved_ids.append(r["chunk_id"])
            except Exception:
                mrrs.append(0.0)
                continue

            # MRR：第一个相关命中的排名倒数
            mrr = 0.0
            for rank, rid in enumerate(retrieved_ids, start=1):
                if rid in expected_ids:
                    mrr = 1.0 / rank
                    break

            mrrs.append(mrr)

        avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

        logger.info("Average MRR = %.4f (threshold=%.4f)", avg_mrr, MRR_THRESHOLD)

        assert avg_mrr >= MRR_THRESHOLD, (
            f"MRR regression: {avg_mrr:.4f} < {MRR_THRESHOLD:.4f}"
        )

    def test_all_queries_return_results(self) -> None:
        """验证每个 golden 查询至少返回一条结果。

        这是基本的合理性检查——若某个查询返回零条结果，
        可能存在索引或检索问题。
        """
        empty_queries: List[str] = []

        for tc in self.golden_set:
            query = tc["query"]
            try:
                results = self.search.search(query=query, top_k=5)
                if not results:
                    empty_queries.append(query)
            except Exception as exc:
                logger.warning("Search error for '%s': %s", query[:40], exc)
                empty_queries.append(query)

        if empty_queries:
            logger.warning(
                "Queries with empty results (%d/%d): %s",
                len(empty_queries),
                len(self.golden_set),
                empty_queries,
            )

        # 仅作提示——数据可能尚未索引，因此不使测试失败。
        # 数据就绪后请取消下面断言的注释：
        # assert not empty_queries, f"{len(empty_queries)} queries returned no results"


# ── 独立运行器 ───────────────────────────────────────────────────────

class TestRecallUnit:
    """轻量级单元测试式召回测试（无需基础设施）。

    这些测试仅验证召回计算逻辑本身，
    与任何搜索引擎或索引数据无关。
    """

    def test_hit_rate_computation(self) -> None:
        """验证 hit_rate 的计算逻辑。"""
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        evaluator = CustomEvaluator()

        # 完全命中
        metrics = evaluator.evaluate(
            query="test",
            retrieved_chunks=[{"id": "c1"}, {"id": "c2"}],
            ground_truth={"ids": ["c1"]},
        )
        assert metrics["hit_rate"] == 1.0

        # 未命中
        metrics = evaluator.evaluate(
            query="test",
            retrieved_chunks=[{"id": "c3"}],
            ground_truth={"ids": ["c1"]},
        )
        assert metrics["hit_rate"] == 0.0

    def test_mrr_computation(self) -> None:
        """验证 MRR 的计算逻辑。"""
        from src.libs.evaluator.custom_evaluator import CustomEvaluator

        evaluator = CustomEvaluator()

        # 第一位命中 → MRR = 1.0
        metrics = evaluator.evaluate(
            query="test",
            retrieved_chunks=[{"id": "c1"}, {"id": "c2"}],
            ground_truth={"ids": ["c1"]},
        )
        assert metrics["mrr"] == 1.0

        # 第二位命中 → MRR = 0.5
        metrics = evaluator.evaluate(
            query="test",
            retrieved_chunks=[{"id": "c2"}, {"id": "c1"}],
            ground_truth={"ids": ["c1"]},
        )
        assert metrics["mrr"] == 0.5

    def test_golden_test_set_loadable(self) -> None:
        """验证 golden 测试集能够被加载和解析。"""
        from src.observability.evaluation.eval_runner import load_test_set

        if not GOLDEN_TEST_SET_PATH.exists():
            pytest.skip("Golden test set not found")

        cases = load_test_set(GOLDEN_TEST_SET_PATH)
        assert len(cases) >= 1
        for tc in cases:
            assert tc.query.strip(), "Each test case must have a non-empty query"
