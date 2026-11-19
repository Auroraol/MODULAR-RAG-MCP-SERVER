"""Overview 页面 – 系统配置与数据统计。

展示：
- 组件配置卡片（LLM、Embedding、VectorStore …）
- Collection 统计信息（文档数、chunk 数、图片数）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from src.observability.dashboard.i18n import tr
from src.observability.dashboard.services.config_service import ConfigService


def _safe_collection_stats() -> Dict[str, Any]:
    """尝试从 ChromaDB 加载 collection 统计信息。

    失败时返回空 dict，以确保页面仍可正常渲染。
    """
    try:
        from src.core.settings import load_settings, resolve_path
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        settings = load_settings()
        persist_dir = str(
            resolve_path(settings.vector_store.persist_directory)
        )
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        stats: Dict[str, Any] = {}
        for col in client.list_collections():
            name = col.name if hasattr(col, "name") else str(col)
            collection = client.get_collection(name)
            stats[name] = {"chunk_count": collection.count()}
        return stats
    except Exception:
        return {}


def render() -> None:
    """渲染 Overview 页面。"""
    st.header(tr("overview.header"))

    # ── 组件配置卡片 ──────────────────────────────────────────────
    st.subheader(tr("overview.component_configuration"))

    try:
        config_service = ConfigService()
        cards = config_service.get_component_cards()
    except Exception as exc:
        st.error(tr("overview.config_failed", exc=exc))
        return

    cols = st.columns(min(len(cards), 3))
    for idx, card in enumerate(cards):
        with cols[idx % len(cols)]:
            st.markdown(f"**{card.name}**")
            st.caption(tr("overview.card_caption", provider=card.provider, model=card.model))
            with st.expander(tr("overview.details")):
                for k, v in card.extra.items():
                    st.text(f"{k}: {v}")

    # ── Collection 统计信息 ───────────────────────────────────────
    st.subheader(tr("overview.collection_statistics"))

    stats = _safe_collection_stats()
    if stats:
        stat_cols = st.columns(min(len(stats), 4))
        for idx, (name, info) in enumerate(sorted(stats.items())):
            with stat_cols[idx % len(stat_cols)]:
                count = info.get("chunk_count", "?")
                st.metric(label=name, value=count)
                if count == 0 or count == "?":
                    st.caption(tr("overview.empty"))
    else:
        st.warning(tr("overview.no_collections"))

    # ── Trace 文件统计 ────────────────────────────────────────────
    st.subheader(tr("overview.trace_statistics"))

    from src.core.settings import resolve_path
    traces_path = resolve_path("logs/traces.jsonl")
    if traces_path.exists():
        line_count = sum(1 for _ in traces_path.open(encoding="utf-8"))
        if line_count > 0:
            st.metric(tr("overview.total_traces"), line_count)
        else:
            st.info(tr("overview.no_traces"))
    else:
        st.info(tr("overview.no_traces"))
