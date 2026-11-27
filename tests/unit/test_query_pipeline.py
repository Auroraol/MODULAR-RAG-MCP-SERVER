"""查询规划层（意图识别 / 路由 / 改写）的单元测试。"""

import pytest

from src.core.query_engine.planner import (
    Intent,
    IntentClassifier,
    LLMRewriter,
    QueryPipeline,
    QueryRouter,
    RewriteMode,
    Route,
    RoutingPolicy,
)
from src.libs.llm.base_llm import ChatResponse


class _FakeLLM:
    """返回固定内容的假 LLM。"""

    def __init__(self, content):
        self.content = content

    def chat(self, messages, trace=None, **kwargs):
        return ChatResponse(content=self.content, model="fake")


class _BrokenLLM:
    def chat(self, messages, trace=None, **kwargs):
        raise RuntimeError("upstream error")


def test_greeting_routes_to_direct_without_queries():
    plan = QueryPipeline().plan("你好！")
    assert plan.intent is Intent.GREETING
    assert plan.route is Route.DIRECT
    assert plan.queries == []


def test_greeting_prefix_does_not_swallow_real_question():
    plan = QueryPipeline().plan("你好，BOM25 怎么接线")
    assert plan.intent is Intent.KNOWLEDGE
    assert plan.route is Route.RETRIEVE


def test_knowledge_query_passes_through_unchanged():
    plan = QueryPipeline().plan("BOM25 怎么接线")
    assert plan.intent is Intent.KNOWLEDGE
    assert plan.route is Route.RETRIEVE
    assert plan.rewrite_mode is RewriteMode.NONE
    assert plan.queries == ["BOM25 怎么接线"]


def test_follow_up_with_history_triggers_rewrite():
    rewriter = LLMRewriter(_FakeLLM("BOM25 主板怎么接线"), RewriteMode.REWRITE)
    plan = QueryPipeline(rewriter=rewriter).plan(
        "它怎么接线", history=["BOM25 是什么"]
    )
    assert plan.intent is Intent.FOLLOW_UP
    assert plan.rewrite_mode is RewriteMode.REWRITE
    assert plan.queries == ["BOM25 主板怎么接线"]


def test_follow_up_marker_without_history_is_knowledge():
    plan = QueryPipeline().plan("它怎么接线")
    assert plan.intent is Intent.KNOWLEDGE


def test_llm_failure_falls_back_to_original_query():
    rewriter = LLMRewriter(_BrokenLLM(), RewriteMode.HYDE)
    assert rewriter.rewrite("BOM25 怎么接线") == ["BOM25 怎么接线"]


def test_subquery_mode_parses_numbered_lines():
    rewriter = LLMRewriter(
        _FakeLLM("1. BOM25 接线\n2. BOM25 供电\n3. BOM25 参数"),
        RewriteMode.SUBQUERY,
        num_subqueries=2,
    )
    assert rewriter.rewrite("BOM25 怎么接线和供电") == ["BOM25 接线", "BOM25 供电"]


def test_custom_classifier_can_emit_faq():
    from src.core.query_engine.planner import IntentResult

    def faq_hit(query, history):
        if "保修" in query:
            return IntentResult(Intent.FAQ, 0.9, {"article_id": "faq_001"})
        return None

    plan = QueryPipeline(classifier=IntentClassifier(custom=faq_hit)).plan("保修多久？")
    assert plan.intent is Intent.FAQ
    assert plan.route is Route.RETRIEVE


def test_policy_override_changes_route():
    policy = RoutingPolicy(rewrites={Intent.KNOWLEDGE: RewriteMode.HYDE})
    router = QueryRouter(policy)
    assert router.route(Intent.KNOWLEDGE).rewrite_mode is RewriteMode.HYDE
    # 未覆盖的意图仍走默认映射
    assert router.route(Intent.GREETING).route is Route.DIRECT