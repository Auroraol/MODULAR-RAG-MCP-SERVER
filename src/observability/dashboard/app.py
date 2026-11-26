"""Modular RAG Dashboard – 多页 Streamlit 应用。

入口：``streamlit run src/observability/dashboard/app.py``

页面通过 ``st.navigation()`` 注册，并由 ``pages/`` 下各自的模块渲染。
界面文案由 :mod:`src.observability.dashboard.i18n` 统一管理，
可在侧边栏切换中文 / English。

快速启动: ``python scripts/start_dashboard.py``
"""

from __future__ import annotations

import streamlit as st

from src.observability.dashboard.i18n import init_language, language_selector, tr

# ── 页面定义 ────────────────────────────────────────────────────────

def _page_overview() -> None:
    from src.observability.dashboard.pages.overview import render
    render()


def _page_data_browser() -> None:
    from src.observability.dashboard.pages.data_browser import render
    render()


def _page_query_search() -> None:
    from src.observability.dashboard.pages.query_search import render
    render()


def _page_ingestion_manager() -> None:
    from src.observability.dashboard.pages.ingestion_manager import render
    render()


def _page_ingestion_traces() -> None:
    from src.observability.dashboard.pages.ingestion_traces import render
    render()


def _page_query_traces() -> None:
    from src.observability.dashboard.pages.query_traces import render
    render()


def _page_evaluation_panel() -> None:
    from src.observability.dashboard.pages.evaluation_panel import render
    render()


# ── 导航 ────────────────────────────────────────────────────────────

def build_pages() -> list:
    """构建导航页面列表（标题随当前语言变化）。"""
    return [
        st.Page(_page_overview, title=tr("nav.overview"), icon="📊", default=True),
        st.Page(_page_data_browser, title=tr("nav.data_browser"), icon="🔍"),
        st.Page(_page_query_search, title=tr("nav.query_search"), icon="💬"),
        st.Page(_page_ingestion_manager, title=tr("nav.ingestion_manager"), icon="📥"),
        st.Page(_page_ingestion_traces, title=tr("nav.ingestion_traces"), icon="🔬"),
        st.Page(_page_query_traces, title=tr("nav.query_traces"), icon="🔎"),
        st.Page(_page_evaluation_panel, title=tr("nav.evaluation_panel"), icon="📏"),
    ]


def main() -> None:
    """渲染整个 Dashboard。

    注意：本函数必须在 Streamlit 运行时内执行（``streamlit run`` 或
    :class:`streamlit.testing.v1.AppTest`）。直接用 ``python app.py``
    运行只会进入 bare mode，不会启动 Web 服务——请改用
    ``python scripts/start_dashboard.py``。
    """
    init_language()

    st.set_page_config(
        page_title=tr("app.title"),
        page_icon="📊",
        layout="wide",
    )

    language_selector()

    nav = st.navigation(build_pages())
    nav.run()

# IDE 配置启动:
#     模块 (Module)：streamlit
#     脚本形参 (Parameters)：填 run src/observability/dashboard/app.py --server.port 8601
#     工作目录 (Working directory)：填项目根目录（如 C:\Users\<用户名>\Documents\GitHub\MODULAR-RAG-MCP-SERVER）
#     环境变量 (Environment Variables)：填 config/settings_dev.yaml
# 脚本启动:
#     scripts/start_dashboard.py，就是为了绕过这种 IDE 配置的坑,
#     使用方法 python scripts/start_dashboard.py -c config/settings_dev.yaml
if __name__ == "__main__":
    main()
