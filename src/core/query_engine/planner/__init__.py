"""查询规划层：意图识别 → 意图路由 → 查询改写。

这是检索之前的编排层，与 query_processor（检索前的分词/停用词处理）
职责不同：本层决定"要不要检索、用哪条查询检索"。

在检索入口处的典型用法::

    plan = pipeline.plan(query, history=history, trace=trace)
    if plan.route is Route.DIRECT:
        return direct_answer(plan.query)      # 问候 / 闲聊，不检索
    for q in plan.queries:                    # 通常只有一条
        results = hybrid_search.search(q, ...)
"""

from src.core.query_engine.planner.intent import (
    Intent,
    IntentClassifier,
    IntentResult,
)
from src.core.query_engine.planner.pipeline import QueryPipeline, QueryPlan
from src.core.query_engine.planner.rewriter import (
    BaseRewriter,
    LLMRewriter,
    NoOpRewriter,
    create_rewriter,
)
from src.core.query_engine.planner.router import (
    QueryRouter,
    RewriteMode,
    Route,
    RouteDecision,
    RoutingPolicy,
)

__all__ = [
    "Intent",
    "IntentClassifier",
    "IntentResult",
    "QueryPipeline",
    "QueryPlan",
    "BaseRewriter",
    "LLMRewriter",
    "NoOpRewriter",
    "create_rewriter",
    "QueryRouter",
    "RewriteMode",
    "Route",
    "RouteDecision",
    "RoutingPolicy",
]