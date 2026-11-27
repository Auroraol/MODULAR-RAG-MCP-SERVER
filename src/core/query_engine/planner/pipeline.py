"""查询规划流水线：意图识别 → 意图路由 → 查询改写。

对外只有一个入口 plan()，返回 QueryPlan。调用方按 plan.route 决定
直答还是检索，检索时用 plan.queries（可能有多条）。

示例:
    >>> pipeline = QueryPipeline()
    >>> plan = pipeline.plan("BOM25 怎么接线")
    >>> plan.route.value, plan.queries
    ('retrieve', ['BOM25 怎么接线'])
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from src.core.query_engine.planner.intent import Intent, IntentClassifier
from src.core.query_engine.planner.rewriter import BaseRewriter, NoOpRewriter
from src.core.query_engine.planner.router import (
    QueryRouter,
    RewriteMode,
    Route,
    RouteDecision,
)


@dataclass
class QueryPlan:
    """一次查询的执行计划。

    属性:
        query: 原始查询
        intent: 识别出的意图
        confidence: 意图置信度
        route: 执行走向
        rewrite_mode: 实际生效的改写模式（NoOpRewriter 时为 NONE）
        queries: 待检索查询列表；route != RETRIEVE 时为空
    """

    query: str
    intent: Intent
    confidence: float
    route: Route
    rewrite_mode: RewriteMode
    queries: List[str] = field(default_factory=list)


class QueryPipeline:
    """串起意图识别、路由与改写的规划器。

    组件全部依赖注入，便于替换与测试；未提供时使用规则/NoOp 默认实现。
    注意：本类不执行检索，只产出执行计划。
    """

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        router: Optional[QueryRouter] = None,
        rewriter: Optional[BaseRewriter] = None,
    ) -> None:
        self.classifier = classifier or IntentClassifier()
        self.router = router or QueryRouter()
        self.rewriter = rewriter or NoOpRewriter()

    def plan(
        self,
        query: str,
        history: Optional[Sequence[str]] = None,
        trace: Optional[Any] = None,
    ) -> QueryPlan:
        """生成查询执行计划。

        参数:
            query: 用户查询
            history: 之前的对话轮次
            trace: 可选的 TraceContext，记录规划阶段

        返回:
            QueryPlan
        """
        _t0 = time.monotonic()

        intent_result = self.classifier.classify(query, list(history) if history else None)
        decision: RouteDecision = self.router.route(intent_result.intent)

        queries: List[str] = []
        rewrite_mode = RewriteMode.NONE

        if decision.route == Route.RETRIEVE:
            rewrite_mode = decision.rewrite_mode
            if rewrite_mode == RewriteMode.NONE:
                queries = [query]
            else:
                queries = [q for q in self.rewriter.rewrite(query, history) if q and q.strip()]
                if not queries:
                    queries = [query]

        plan = QueryPlan(
            query=query,
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            route=decision.route,
            rewrite_mode=rewrite_mode,
            queries=queries,
        )

        if trace is not None:
            trace.record_stage(
                "query_planning",
                {
                    "intent": plan.intent.value,
                    "confidence": plan.confidence,
                    "route": plan.route.value,
                    "rewrite_mode": plan.rewrite_mode.value,
                    "queries": plan.queries,
                    "intent_metadata": intent_result.metadata,
                },
                elapsed_ms=(time.monotonic() - _t0) * 1000.0,
            )

        return plan