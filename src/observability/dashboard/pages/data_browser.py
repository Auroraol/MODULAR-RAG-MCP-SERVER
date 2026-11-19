"""Data Browser 页面 – 浏览已摄入的文档、chunk 与图片。

布局：
1. Collection 选择器（侧边栏）
2. 文档列表及 chunk 数量
3. 可展开的文档详情 → 包含文本与元数据的 chunk 卡片
4. 图片预览画廊
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.observability.dashboard.i18n import tr
from src.observability.dashboard.services.data_service import DataService


def render() -> None:
    """渲染 Data Browser 页面。"""
    st.header(tr("data_browser.header"))

    try:
        svc = DataService()
    except Exception as exc:
        st.error(tr("data_browser.init_failed", exc=exc))
        return

    # ── Collection 选择器 ─────────────────────────────────────────
    collections = svc.list_collections()
    if "default" not in collections:
        collections.insert(0, "default")
    collection = st.selectbox(
        tr("common.collection"),
        options=collections,
        index=0,
        key="db_collection_filter",
    )
    coll_arg = collection if collection else None

    # ── 危险区域：清空所有数据 ────────────────────────────────────
    st.divider()
    with st.expander(tr("data_browser.danger_zone"), expanded=False):
        st.warning(tr("data_browser.danger_warning"))
        col_btn, col_status = st.columns([1, 2])
        with col_btn:
            if st.button(tr("data_browser.clear_all"), type="primary", key="btn_clear_all"):
                st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.error(tr("data_browser.confirm_prompt"))
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                if st.button(tr("data_browser.confirm_yes"), key="btn_confirm_clear"):
                    result = svc.reset_all()
                    st.session_state["confirm_clear"] = False
                    if result["errors"]:
                        st.warning(
                            tr(
                                "data_browser.clear_errors",
                                count=len(result["errors"]),
                                errors="; ".join(result["errors"]),
                            )
                        )
                    else:
                        st.success(
                            tr(
                                "data_browser.clear_ok",
                                count=result["collections_deleted"],
                            )
                        )
                    st.rerun()
            with c2:
                if st.button(tr("common.cancel"), key="btn_cancel_clear"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

    st.divider()

    # ── 文档列表 ──────────────────────────────────────────────────
    try:
        docs = svc.list_documents(coll_arg)
    except Exception as exc:
        st.error(tr("data_browser.load_failed", exc=exc))
        return

    if not docs:
        st.info(tr("data_browser.no_docs"))
        return

    st.subheader(tr("data_browser.documents", count=len(docs)))

    for idx, doc in enumerate(docs):
        source_name = Path(doc["source_path"]).name
        label = tr(
            "data_browser.doc_label",
            name=source_name,
            chunks=doc["chunk_count"],
            images=doc["image_count"],
        )
        with st.expander(label, expanded=(len(docs) == 1)):
            # ── 文档元数据 ────────────────────────────────────────
            col_a, col_b, col_c = st.columns(3)
            col_a.metric(tr("common.chunks"), doc["chunk_count"])
            col_b.metric(tr("common.images"), doc["image_count"])
            col_c.metric(tr("common.collection_label"), doc.get("collection", "—"))
            st.caption(
                tr(
                    "data_browser.doc_caption",
                    source=doc["source_path"],
                    hash=doc["source_hash"][:16],
                    processed=doc.get("processed_at", "—"),
                )
            )

            st.divider()

            # ── Chunk 卡片 ────────────────────────────────────────
            chunks = svc.get_chunks(doc["source_hash"], coll_arg)
            if chunks:
                st.markdown(tr("data_browser.chunks_title", count=len(chunks)))
                for cidx, chunk in enumerate(chunks):
                    text = chunk.get("text", "")
                    meta = chunk.get("metadata", {})
                    chunk_id = chunk["id"]

                    # 标题取自元数据或首行
                    title = meta.get("title", "")
                    if not title:
                        title = text[:60].replace("\n", " ").strip()
                        if len(text) > 60:
                            title += "…"

                    with st.container(border=True):
                        st.markdown(
                            tr(
                                "data_browser.chunk_meta",
                                index=cidx + 1,
                                chunk_id=chunk_id[-16:],
                                chars=len(text),
                            )
                        )
                        # 显示实际的 chunk 文本（可滚动）
                        _height = max(120, min(len(text) // 2, 600))
                        st.text_area(
                            tr("data_browser.content"),
                            value=text,
                            height=_height,
                            disabled=True,
                            key=f"chunk_text_{idx}_{cidx}",
                            label_visibility="collapsed",
                        )
                        # 可展开的元数据
                        with st.expander(tr("common.metadata"), expanded=False):
                            st.json(meta)
            else:
                st.caption(tr("data_browser.no_chunks"))

            # ── 图片预览 ──────────────────────────────────────────
            images = svc.get_images(doc["source_hash"], coll_arg)
            if images:
                st.divider()
                st.markdown(tr("data_browser.images_title", count=len(images)))
                img_cols = st.columns(min(len(images), 4))
                for iidx, img in enumerate(images):
                    with img_cols[iidx % len(img_cols)]:
                        img_path = Path(img.get("file_path", ""))
                        if img_path.exists():
                            st.image(str(img_path), caption=img["image_id"], width=200)
                        else:
                            st.caption(
                                tr("data_browser.image_missing", image_id=img["image_id"])
                            )
