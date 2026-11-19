"""Streamlit Dashboard 页面的 E2E 冒烟测试。

使用 Streamlit 的 ``AppTest`` 框架在无头模式下渲染每个页面的 ``render()``
函数，并验证：

1. 渲染过程中未抛出任何 Python 异常。
2. 每个页面至少产生一个预期的 UI 元素（header / info / metric）。

这些测试**不**依赖实时数据——在 Vector Store 为空的干净检出上
应当能够通过。

用法::

    pytest tests/e2e/test_dashboard_smoke.py -v
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


# ── 辅助函数 ─────────────────────────────────────────────────────────


def _mock_settings() -> MagicMock:
    """返回一个满足所有 Dashboard 页面需求的最小 mock Settings。"""
    s = MagicMock()
    s.llm.provider = "azure"
    s.llm.model = "gpt-4o"
    s.llm.temperature = 0.0
    s.llm.max_tokens = 4096

    s.embedding.provider = "azure"
    s.embedding.model = "text-embedding-ada-002"
    s.embedding.dimensions = 1536

    s.vector_store.provider = "chroma"
    s.vector_store.collection_name = "default"
    s.vector_store.persist_directory = "./data/db/chroma"

    s.retrieval.dense_top_k = 20
    s.retrieval.sparse_top_k = 20
    s.retrieval.fusion_top_k = 10

    s.rerank.enabled = False
    s.rerank.provider = "none"
    s.rerank.model = ""
    s.rerank.top_k = 5

    s.vision_llm.enabled = False
    s.vision_llm.provider = "azure"
    s.vision_llm.model = "gpt-4o"
    s.vision_llm.max_image_size = 2048

    s.observability.log_level = "INFO"
    s.observability.trace_enabled = True
    s.observability.trace_file = "./logs/traces.jsonl"
    s.observability.structured_logging = True

    s.ingestion.chunk_size = 1000
    s.ingestion.chunk_overlap = 200
    s.ingestion.splitter = "recursive"
    s.ingestion.batch_size = 100
    return s


def _collect_text(at: Any) -> str:
    """收集 AppTest 运行期间渲染出的全部文本，用于断言。"""
    parts: List[str] = []
    for attr in ("markdown", "header", "subheader", "info", "error", "title", "text", "success", "warning"):
        for el in getattr(at, attr, []):
            parts.append(str(getattr(el, "value", "")))
    return "\n".join(parts)


def _run_page(page_script: Any, language: str = "en", **patches: Any) -> Any:
    """在指定语言下无头渲染一个页面并返回 AppTest。

    参数:
        page_script: 无参函数，内部调用页面的 ``render()``。
        language: ``"zh"`` 或 ``"en"``，写入 ``st.session_state["lang"]``。
        **patches: ``{"target.attr": mock}``，逐个 patch 后再运行。
    """
    from contextlib import ExitStack

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(page_script, default_timeout=10)
    at.session_state["lang"] = language
    with ExitStack() as stack:
        for target, mock in patches.items():
            stack.enter_context(patch(target, return_value=mock))
        at.run()
    return at


def _has_any(text: str, *candidates: str) -> bool:
    """文本是否包含任一候选词（不区分大小写）。"""
    lowered = text.lower()
    return any(c.lower() in lowered for c in candidates)


# ── 测试 ─────────────────────────────────────────────────────────────


class TestDashboardSmoke:
    """冒烟测试：每个页面均能在无未捕获异常的情况下渲染。"""

    # ------------------------------------------------------------------
    # 1. Overview 页面
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        ("language", "markers"),
        [("en", ("overview", "system")), ("zh", ("总览", "系统"))],
    )
    def test_overview_page_renders(self, language: str, markers: tuple) -> None:
        """Overview 页面能够加载并显示系统概览标题。"""

        def page_script():
            from src.observability.dashboard.pages.overview import render
            render()

        at = _run_page(
            page_script,
            language,
            **{
                "src.observability.dashboard.services.config_service.load_settings":
                    _mock_settings(),
            },
        )

        assert not at.exception, (
            f"Overview page raised an exception: {at.exception}"
        )
        assert _has_any(_collect_text(at), *markers)

    # ------------------------------------------------------------------
    # 2. Data Browser 页面
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        ("language", "markers"),
        [("en", ("data", "browser", "document")), ("zh", ("数据", "浏览", "文档"))],
    )
    def test_data_browser_page_renders(self, language: str, markers: tuple) -> None:
        """Data Browser 页面能够加载（可能显示“no documents”提示）。"""
        mock_svc = MagicMock()
        mock_svc.list_documents.return_value = []
        mock_svc.list_collections.return_value = []

        def page_script():
            from src.observability.dashboard.pages.data_browser import render
            render()

        at = _run_page(
            page_script,
            language,
            **{"src.observability.dashboard.pages.data_browser.DataService": mock_svc},
        )

        assert not at.exception, (
            f"Data Browser page raised an exception: {at.exception}"
        )
        assert _has_any(_collect_text(at), *markers)

    # ------------------------------------------------------------------
    # 3. Ingestion Manager 页面
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    @pytest.mark.parametrize(("language",), [("en",), ("zh",)])
    def test_ingestion_manager_page_renders(self, language: str) -> None:
        """Ingestion Manager 页面在无错误的情况下加载。"""
        mock_svc = MagicMock()
        mock_svc.list_documents.return_value = []

        def page_script():
            from src.observability.dashboard.pages.ingestion_manager import render
            render()

        at = _run_page(
            page_script,
            language,
            **{"src.observability.dashboard.pages.ingestion_manager.DataService": mock_svc},
        )

        assert not at.exception, (
            f"Ingestion Manager page raised an exception: {at.exception}"
        )

    # ------------------------------------------------------------------
    # 4. Ingestion Traces 页面
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        ("language", "markers"),
        [("en", ("trace", "ingestion")), ("zh", ("trace", "摄取"))],
    )
    def test_ingestion_traces_page_renders(self, language: str, markers: tuple) -> None:
        """Ingestion Traces 页面能够加载（Trace 列表为空亦可）。"""
        mock_svc = MagicMock()
        mock_svc.list_traces.return_value = []

        def page_script():
            from src.observability.dashboard.pages.ingestion_traces import render
            render()

        at = _run_page(
            page_script,
            language,
            **{"src.observability.dashboard.pages.ingestion_traces.TraceService": mock_svc},
        )

        assert not at.exception, (
            f"Ingestion Traces page raised an exception: {at.exception}"
        )
        assert _has_any(_collect_text(at), *markers)

    # ------------------------------------------------------------------
    # 5. Query Traces 页面
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        ("language", "markers"),
        [("en", ("query", "trace")), ("zh", ("trace", "查询"))],
    )
    def test_query_traces_page_renders(self, language: str, markers: tuple) -> None:
        """Query Traces 页面能够加载（Trace 列表为空亦可）。"""
        mock_svc = MagicMock()
        mock_svc.list_traces.return_value = []

        def page_script():
            from src.observability.dashboard.pages.query_traces import render
            render()

        at = _run_page(
            page_script,
            language,
            **{"src.observability.dashboard.pages.query_traces.TraceService": mock_svc},
        )

        assert not at.exception, (
            f"Query Traces page raised an exception: {at.exception}"
        )
        assert _has_any(_collect_text(at), *markers)

    # ------------------------------------------------------------------
    # 6. Evaluation Panel 页面
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        ("language", "markers"),
        [("en", ("evaluation", "panel")), ("zh", ("评估",))],
    )
    def test_evaluation_panel_page_renders(self, language: str, markers: tuple) -> None:
        """Evaluation Panel 页面在无错误的情况下加载。"""

        def page_script():
            from src.observability.dashboard.pages.evaluation_panel import render
            render()

        at = _run_page(page_script, language)

        assert not at.exception, (
            f"Evaluation Panel page raised an exception: {at.exception}"
        )
        assert _has_any(_collect_text(at), *markers)

    # ------------------------------------------------------------------
    # 7. 导航与语言切换
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        ("language", "titles"),
        [
            ("en", ("Overview", "Data Browser", "Ingestion Manager")),
            ("zh", ("系统总览", "数据浏览", "摄取管理")),
        ],
    )
    def test_navigation_titles_follow_language(self, language: str, titles: tuple) -> None:
        """导航标题随语言切换。

        ``st.navigation`` 的页面链接由前端渲染，AppTest 无法断言，
        因此在 AppTest 运行时内构造 ``build_pages()`` 并把标题写出来。
        ``st.Page.title`` 依赖 Streamlit 运行时上下文，必须在脚本内访问。
        """
        from streamlit.testing.v1 import AppTest

        def page_script():
            import streamlit as st

            from src.observability.dashboard import app as dashboard_app
            for page in dashboard_app.build_pages():
                st.write(f"TITLE::{page.title}")

        at = AppTest.from_function(page_script, default_timeout=10)
        at.session_state["lang"] = language
        at.run()

        assert not at.exception, f"build_pages raised: {at.exception}"
        rendered = _collect_text(at)
        for title in titles:
            assert f"TITLE::{title}" in rendered, (
                f"Missing nav title {title!r} in {language} mode; got:\n{rendered}"
            )

    @pytest.mark.e2e
    @pytest.mark.parametrize("language", ["en", "zh"])
    def test_dashboard_app_renders(self, language: str) -> None:
        """整个 Dashboard（含侧边栏语言切换器）能无异常渲染。"""
        from streamlit.testing.v1 import AppTest

        # AppTest.from_function 会序列化函数体，函数内必须自行 import，
        # 不能依赖模块级全局名。
        def page_script():
            from src.observability.dashboard import app as dashboard_app
            dashboard_app.main()

        at = AppTest.from_function(page_script, default_timeout=10)
        at.session_state["lang"] = language
        at.run()

        assert not at.exception, f"Dashboard raised an exception: {at.exception}"
        assert len(at.sidebar.selectbox) >= 1, "Language selector not rendered"
