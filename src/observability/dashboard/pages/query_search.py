"""Query Search 页面 – 在 Dashboard 中直接发起在线检索。

复用 MCP 工具 ``query_knowledge_hub`` 的完整链路（混合检索 + 可选
重排序 + 引用格式化），所以这里看到的结果与 MCP 客户端一致。

布局：
1. Collection 选择器 + Top K
2. 查询输入框
3. 结果 Markdown（含引用）与相关图片
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import streamlit as st

from src.observability.dashboard.i18n import tr
from src.observability.dashboard.services.data_service import DataService

logger = logging.getLogger(__name__)

_RESPONSE_KEY = "query_search_response"


def render() -> None:
    """渲染 Query Search 页面。"""
    st.header(tr("query_search.header"))

    collections = DataService().list_collections()
    if "default" not in collections:
        collections.insert(0, "default")

    col_collection, col_top_k = st.columns([3, 1])
    with col_collection:
        collection = st.selectbox(
            tr("common.collection"),
            options=collections,
            key="qs_collection",
        )
    with col_top_k:
        top_k = st.number_input(
            tr("query_search.top_k"),
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="qs_top_k",
        )

    query = st.text_input(
        tr("query_search.query_label"),
        placeholder=tr("query_search.placeholder"),
        key="qs_query",
    )

    if st.button(tr("query_search.search"), type="primary", key="qs_search"):
        if not query.strip():
            st.warning(tr("query_search.empty_query"))
        else:
            with st.spinner(tr("query_search.searching")):
                try:
                    response = _run_query(query.strip(), int(top_k), collection)
                except Exception as exc:
                    logger.exception("Dashboard query failed")
                    st.error(tr("query_search.failed", exc=exc))
                    st.session_state.pop(_RESPONSE_KEY, None)
                else:
                    st.session_state[_RESPONSE_KEY] = {
                        "response": response,
                        "collection": collection,
                        "result_count": response.metadata.get(
                            "result_count", len(response.citations)
                        ),
                    }

    # 结果存 session_state，避免每次控件交互都重新检索
    state = st.session_state.get(_RESPONSE_KEY)
    if not state:
        return

    response = state["response"]
    st.divider()
    st.caption(
        tr(
            "query_search.results_caption",
            collection=state["collection"],
            count=state["result_count"],
        )
    )
    st.markdown(response.content)

    if response.image_contents:
        st.subheader(tr("query_search.images"))
        for index, block in enumerate(response.image_contents, start=1):
            try:
                st.image(base64.b64decode(block.data), caption=str(index))
            except Exception as exc:  # 单张图片坏掉不影响文本结果
                logger.warning("Failed to render result image: %s", exc)


def _run_query(query: str, top_k: int, collection: str) -> Any:
    """调用 MCP 同款检索链路（异步接口 → 同步桥接）。

    ponytail: 直接复用 QueryKnowledgeHubTool，而不是在页面里重新拼装
    HybridSearch + Reranker；行为与 MCP 客户端完全一致。
    """
    from src.mcp_server.tools.query_knowledge_hub import get_tool_instance

    tool = get_tool_instance()
    return asyncio.run(
        tool.execute(query=query, top_k=top_k, collection=collection)
    )