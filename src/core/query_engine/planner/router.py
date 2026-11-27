"""意图路由：把意图映射为执行策略（检索 / 直答 / 兜底）与改写模式。

路由是一张可覆盖的策略表，不写 if/else 瀑布。新增意图或调整策略
只需改 RoutingPolicy，不动流程代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from src.core.query_engine.planner.intent import Intent


class Route(str, Enum):
    """查询走向。"""

    RETRIEVE = "retrieve"  # 走向量/BM25 检索
    DIRECT = "direct"      # 直接回答，不检索（问候、闲聊）
    FALLBACK = "fallback"  # 兜底 / 转人工


class RewriteMode(str, Enum):
    """检索前的查询改写模式。"""

    NONE = "none"        # 原样检索
    REWRITE = "rewrite"  # 结合历史改写成独立查询（指代消解）
    HYDE = "hyde"        # 生成假设文档代替查询
    SUBQUERY = "subquery"  # 拆成多个子查询


@dataclass
class RoutingPolicy:
    """路由策略表，可按需覆盖默认映射。

    属性:
        routes: 意图 → 走向
        rewrites: 意图 → 改写模式
        default_route: 表中未出现的意图的默认走向
        default_rewrite: 表中未出现的意图的默认改写模式
    """

    routes: Dict[Intent, Route] = field(
        default_factory=lambda: {
            Intent.GREETING: Route.DIRECT,
            Intent.FAQ: Route.RETRIEVE,
            Intent.FOLLOW_UP: Route.RETRIEVE,
            Intent.KNOWLEDGE: Route.RETRIEVE,
        }
    )
    rewrites: Dict[Intent, RewriteMode] = field(
        default_factory=lambda: {
            Intent.GREETING: RewriteMode.NONE,
            Intent.FAQ: RewriteMode.NONE,
            Intent.FOLLOW_UP: RewriteMode.REWRITE,
            Intent.KNOWLEDGE: RewriteMode.NONE,
        }
    )
    default_route: Route = Route.RETRIEVE
    default_rewrite: RewriteMode = RewriteMode.NONE


@dataclass
class RouteDecision:
    """单个意图的路由结果。"""

    route: Route
    rewrite_mode: RewriteMode


class QueryRouter:
    """基于策略表的意图路由器。

    示例:
        >>> router = QueryRouter()
        >>> router.route(Intent.GREETING).route
        <Route.DIRECT: 'direct'>
        >>> router.route(Intent.FOLLOW_UP).rewrite_mode
        <RewriteMode.REWRITE: 'rewrite'>
    """

    def __init__(self, policy: Optional[RoutingPolicy] = None) -> None:
        self.policy = policy or RoutingPolicy()

    def route(self, intent: Intent) -> RouteDecision:
        """按策略表给出该意图的执行方式。"""
        return RouteDecision(
            route=self.policy.routes.get(intent, self.policy.default_route),
            rewrite_mode=self.policy.rewrites.get(intent, self.policy.default_rewrite),
        )