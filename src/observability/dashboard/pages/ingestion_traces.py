"""Ingestion Traces 页面 – 浏览摄入 trace 历史及各阶段详情。

布局：
1. Trace 列表（倒序排列，按 trace_type=="ingestion" 过滤）
2. Pipeline 概览：源文件、总耗时、阶段耗时瀑布图
3. 各阶段详情标签页：
   📄 Load    – 原始文档文本预览
   ✂️ Split   – chunk 列表及文本
   🔄 Transform – 前后对比 diff、enrichment 元数据
   🔢 Embed   – 向量统计
   💾 Upsert  – 已存储的 ID
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from src.observability.dashboard.i18n import tr
from src.observability.dashboard.services.trace_service import TraceService

logger = logging.getLogger(__name__)


def render() -> None:
    """渲染 Ingestion Traces 页面。"""
    st.header(tr("ingestion_traces.header"))

    svc = TraceService()
    traces = svc.list_traces(trace_type="ingestion")

    if not traces:
        st.info(tr("ingestion_traces.none"))
        return

    st.subheader(tr("ingestion_traces.history", count=len(traces)))

    for idx, trace in enumerate(traces):
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        total_ms = trace.get("elapsed_ms")
        total_label = f"{total_ms:.0f} ms" if total_ms is not None else "—"
        meta = trace.get("metadata", {})
        source_path = meta.get("source_path", "—")

        # 构建 expander 标题
        file_name = source_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if source_path != "—" else "—"
        expander_title = tr(
            "ingestion_traces.expander_title",
            file=file_name,
            elapsed=total_label,
            started=started[:19],
        )

        with st.expander(expander_title, expanded=(idx == 0)):
            timings = svc.get_stage_timings(trace)
            stages_by_name = {t["stage_name"]: t for t in timings}

            # ── 1. 概览指标 ───────────────────────────────────
            st.markdown(tr("ingestion_traces.pipeline_overview"))
            st.caption(tr("ingestion_traces.source_caption", source=source_path))

            load_d = stages_by_name.get("load", {}).get("data", {})
            split_d = stages_by_name.get("split", {}).get("data", {})
            transform_d = stages_by_name.get("transform", {}).get("data", {})
            embed_d = stages_by_name.get("embed", {}).get("data", {})
            upsert_d = stages_by_name.get("upsert", {}).get("data", {})

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric(
                    tr("ingestion_traces.metric_doc_length"),
                    tr("ingestion_traces.chars_value", count=load_d.get("text_length", 0)),
                )
            with c2:
                st.metric(tr("common.chunks"), split_d.get("chunk_count", 0))
            with c3:
                st.metric(tr("common.images"), load_d.get("image_count", 0))
            with c4:
                st.metric(
                    tr("ingestion_traces.metric_vectors"), upsert_d.get("vector_count", 0)
                )
            with c5:
                st.metric(tr("ingestion_traces.metric_total_time"), total_label)

            st.divider()

            # ── 2. 阶段耗时瀑布图 ──────────────────────────────
            # 仅保留主编排阶段（不含子阶段）
            main_stages = [
                t for t in timings
                if t["stage_name"] in ("load", "split", "transform", "embed", "upsert")
            ]
            if main_stages:
                st.markdown(tr("ingestion_traces.stage_timings"))
                chart_data = {t["stage_name"]: t["elapsed_ms"] for t in main_stages}
                st.bar_chart(chart_data, horizontal=True)
                st.table([
                    {
                        tr("common.stage"): t["stage_name"],
                        tr("common.elapsed_ms"): round(t["elapsed_ms"], 2),
                    }
                    for t in main_stages
                ])

            # ── 诊断信息 ───────────────────────────────────────
            _render_ingestion_diagnostics(stages_by_name, load_d, split_d, transform_d, embed_d, upsert_d)

            st.divider()

            # ── 3. 各阶段详情标签页 ────────────────────────────
            st.markdown(tr("ingestion_traces.stage_details"))

            tab_defs = []
            if "load" in stages_by_name:
                tab_defs.append((tr("ingestion_traces.tab_load"), "load"))
            if "split" in stages_by_name:
                tab_defs.append((tr("ingestion_traces.tab_split"), "split"))
            if "transform" in stages_by_name:
                tab_defs.append((tr("ingestion_traces.tab_transform"), "transform"))
            if "embed" in stages_by_name:
                tab_defs.append((tr("ingestion_traces.tab_embed"), "embed"))
            if "upsert" in stages_by_name:
                tab_defs.append((tr("ingestion_traces.tab_upsert"), "upsert"))

            if tab_defs:
                tabs = st.tabs([label for label, _ in tab_defs])
                for tab, (label, key) in zip(tabs, tab_defs):
                    with tab:
                        stage = stages_by_name[key]
                        data = stage.get("data", {})
                        elapsed = stage.get("elapsed_ms")
                        if elapsed is not None:
                            st.caption(f"⏱️ {elapsed:.1f} ms")

                        if key == "load":
                            _render_load_stage(data, trace_idx=idx)
                        elif key == "split":
                            _render_split_stage(data, trace_idx=idx)
                        elif key == "transform":
                            _render_transform_stage(data, trace_idx=idx)
                        elif key == "embed":
                            _render_embed_stage(data)
                        elif key == "upsert":
                            _render_upsert_stage(data)
            else:
                st.info(tr("ingestion_traces.no_stage_details"))


def _render_ingestion_diagnostics(
    stages_by_name: Dict[str, Any],
    load_d: Dict[str, Any],
    split_d: Dict[str, Any],
    transform_d: Dict[str, Any],
    embed_d: Dict[str, Any],
    upsert_d: Dict[str, Any],
) -> None:
    """渲染摄入 Pipeline 各阶段的诊断提示。"""
    expected = ["load", "split", "transform", "embed", "upsert"]
    present = [s for s in expected if s in stages_by_name]
    missing = [s for s in expected if s not in stages_by_name]

    if missing:
        missing_labels = {
            "load": tr("ingestion_traces.tab_load"),
            "split": tr("ingestion_traces.tab_split"),
            "transform": tr("ingestion_traces.tab_transform"),
            "embed": tr("ingestion_traces.tab_embed"),
            "upsert": tr("ingestion_traces.tab_upsert"),
        }
        names = ", ".join(missing_labels.get(m, m) for m in missing)
        if "load" in missing:
            st.error(tr("ingestion_traces.diag_missing_load", names=names))
        else:
            st.warning(tr("ingestion_traces.diag_missing_other", names=names))

    # 各阶段特有的诊断信息
    if "load" in stages_by_name and load_d.get("text_length", 0) == 0:
        st.warning(tr("ingestion_traces.diag_load_empty"))

    if "split" in stages_by_name and split_d.get("chunk_count", 0) == 0:
        st.warning(tr("ingestion_traces.diag_split_empty"))

    if "transform" in stages_by_name:
        refined_llm = transform_d.get("refined_by_llm", 0)
        refined_rule = transform_d.get("refined_by_rule", 0)
        if refined_llm == 0 and refined_rule == 0:
            st.info(tr("ingestion_traces.diag_transform_none"))

    if "embed" in stages_by_name and embed_d.get("dense_vector_count", 0) == 0:
        st.warning(tr("ingestion_traces.diag_embed_empty"))

    if "upsert" in stages_by_name:
        vec_count = upsert_d.get("vector_count", upsert_d.get("dense_store", {}).get("count", 0))
        if vec_count == 0:
            st.warning(tr("ingestion_traces.diag_upsert_empty"))

    # 检查任一阶段数据中的错误字段
    for stage_name in present:
        stage_data = stages_by_name[stage_name].get("data", {})
        err = stage_data.get("error", "")
        if err:
            label = stage_name.replace("_", " ").title()
            st.error(tr("ingestion_traces.stage_error", stage=label, error=err))


# ═══════════════════════════════════════════════════════════════
# 各阶段渲染器
# ═══════════════════════════════════════════════════════════════

def _render_load_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """渲染 Load 阶段：原始文档预览。"""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(tr("ingestion_traces.load_metric_doc_id"), data.get("doc_id", "—")[:16])
    with c2:
        st.metric(
            tr("ingestion_traces.load_metric_text_length"),
            f"{data.get('text_length', 0):,}",
        )
    with c3:
        st.metric(tr("common.images"), data.get("image_count", 0))

    preview = data.get("text_preview", "")
    if preview:
        st.markdown(tr("ingestion_traces.raw_text"))
        st.text_area(
            "raw_text",
            value=preview,
            height=max(120, min(len(preview) // 2, 600)),
            disabled=True,
            label_visibility="collapsed",
            key=f"load_raw_text_{trace_idx}",
        )
    else:
        st.info(tr("ingestion_traces.no_text_preview"))


def _render_split_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """渲染 Split 阶段：chunk 列表及文本。"""
    c1, c2 = st.columns(2)
    with c1:
        st.metric(tr("common.chunks"), data.get("chunk_count", 0))
    with c2:
        st.metric(
            tr("ingestion_traces.split_metric_avg_size"),
            tr("ingestion_traces.chars_suffix", count=data.get("avg_chunk_size", 0)),
        )

    chunks = data.get("chunks", [])
    if chunks:
        st.markdown(tr("ingestion_traces.chunks_after_split"))
        for i, chunk in enumerate(chunks):
            char_len = chunk.get("char_len", 0)
            chunk_id = chunk.get("chunk_id", "")
            text = chunk.get("text", "")
            header = tr(
                "ingestion_traces.split_chunk_header",
                index=i + 1,
                chunk_id=chunk_id[:20],
                chars=char_len,
            )
            with st.expander(header, expanded=(i < 2)):
                st.text_area(
                    f"split_{i}",
                    value=text,
                    height=max(100, min(len(text) // 2, 500)),
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"split_{trace_idx}_{i}",
                )
    else:
        st.info(tr("ingestion_traces.no_chunk_text"))


def _render_transform_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """渲染 Transform 阶段：精炼前后对比 + enrichment 元数据。"""
    # 汇总指标
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            tr("ingestion_traces.transform_refined"),
            f"{data.get('refined_by_llm', 0)} / {data.get('refined_by_rule', 0)}",
        )
    with c2:
        st.metric(
            tr("ingestion_traces.transform_enriched"),
            f"{data.get('enriched_by_llm', 0)} / {data.get('enriched_by_rule', 0)}",
        )
    with c3:
        st.metric(
            tr("ingestion_traces.transform_captioned"), data.get("captioned_chunks", 0)
        )

    chunks = data.get("chunks", [])
    if chunks:
        st.markdown(tr("ingestion_traces.per_chunk_transform"))
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", "")
            refined_by = chunk.get("refined_by", "")
            enriched_by = chunk.get("enriched_by", "")
            title = chunk.get("title", "")
            tags = chunk.get("tags", [])
            summary = chunk.get("summary", "")
            text_before = chunk.get("text_before", "")
            text_after = chunk.get("text_after", "")

            badge_parts = []
            if refined_by:
                badge_parts.append(f"refined:`{refined_by}`")
            if enriched_by:
                badge_parts.append(f"enriched:`{enriched_by}`")
            badges = " · ".join(badge_parts)

            header = tr(
                "ingestion_traces.transform_chunk_header",
                index=i + 1,
                chunk_id=chunk_id[:20],
                badges=badges,
            )
            with st.expander(header, expanded=(i == 0)):
                # 来自 enrichment 的元数据
                if title or tags or summary:
                    st.markdown(tr("ingestion_traces.enriched_metadata"))
                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        st.markdown(
                            tr("ingestion_traces.title_label", title=title)
                            if title
                            else tr("ingestion_traces.no_title")
                        )
                    with meta_cols[1]:
                        if tags:
                            st.markdown(
                                tr(
                                    "ingestion_traces.tags_label",
                                    tags=", ".join(f"`{t}`" for t in tags),
                                )
                            )
                        else:
                            st.markdown(tr("ingestion_traces.no_tags"))
                    with meta_cols[2]:
                        if summary:
                            st.markdown(
                                tr("ingestion_traces.summary_label", summary=summary)
                            )

                # 前后文本对比
                if text_before or text_after:
                    st.markdown(tr("ingestion_traces.text_comparison"))
                    # 计算统一高度，使两侧对齐
                    _max_len = max(len(text_before or ""), len(text_after or ""))
                    _h = max(150, min(_max_len // 2, 600))
                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.markdown(tr("ingestion_traces.before_refinement"))
                        st.text_area(
                            f"before_{i}",
                            value=text_before if text_before else tr("ingestion_traces.empty_text"),
                            height=_h,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"transform_before_{trace_idx}_{i}",
                        )
                    with col_after:
                        st.markdown(tr("ingestion_traces.after_refinement"))
                        st.text_area(
                            f"after_{i}",
                            value=text_after if text_after else tr("ingestion_traces.empty_text"),
                            height=_h,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"transform_after_{trace_idx}_{i}",
                        )
    else:
        st.info(tr("ingestion_traces.no_transform_data"))


def _render_embed_stage(data: Dict[str, Any]) -> None:
    """渲染 Embed 阶段：Dense + Sparse 双路径编码详情。"""
    # ── 概览指标 ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(tr("ingestion_traces.embed_metric_dense"), data.get("dense_vector_count", 0))
    with c2:
        st.metric(tr("ingestion_traces.embed_metric_dimension"), data.get("dense_dimension", 0))
    with c3:
        st.metric(tr("ingestion_traces.embed_metric_sparse"), data.get("sparse_doc_count", 0))
    with c4:
        st.metric(tr("common.method"), data.get("method", "—"))

    chunks = data.get("chunks", [])
    if not chunks:
        st.info(tr("ingestion_traces.no_encoding_data"))
        return

    # ── 双路径逐 chunk 表格 ──
    st.markdown("---")
    dense_tab, sparse_tab = st.tabs(
        [tr("ingestion_traces.dense_tab"), tr("ingestion_traces.sparse_tab")]
    )

    with dense_tab:
        st.markdown(tr("ingestion_traces.dense_explain"))
        dense_rows = []
        for i, chunk in enumerate(chunks):
            char_len = chunk.get("char_len", 0)
            dense_rows.append({
                tr("ingestion_traces.col_index"): i + 1,
                tr("common.chunk_id"): chunk.get("chunk_id", ""),
                tr("common.chars"): char_len,
                tr("ingestion_traces.col_est_tokens"): max(1, char_len // 3),
                tr("ingestion_traces.col_dense_dim"): chunk.get(
                    "dense_dim", data.get("dense_dimension", "—")
                ),
            })
        st.table(dense_rows)

    with sparse_tab:
        st.markdown(tr("ingestion_traces.sparse_explain"))
        sparse_rows = []
        for i, chunk in enumerate(chunks):
            sparse_rows.append({
                tr("ingestion_traces.col_index"): i + 1,
                tr("common.chunk_id"): chunk.get("chunk_id", ""),
                tr("ingestion_traces.col_doc_length"): chunk.get("doc_length", "—"),
                tr("ingestion_traces.col_unique_terms"): chunk.get("unique_terms", "—"),
            })
        st.table(sparse_rows)

        # 每个 chunk 的高频词项
        for i, chunk in enumerate(chunks):
            top_terms = chunk.get("top_terms", [])
            if top_terms:
                with st.expander(
                    tr("ingestion_traces.top_terms_title", index=i + 1), expanded=False
                ):
                    term_rows = [
                        {
                            tr("ingestion_traces.col_term"): t["term"],
                            tr("ingestion_traces.col_freq"): t["freq"],
                        }
                        for t in top_terms
                    ]
                    st.table(term_rows)


def _render_upsert_stage(data: Dict[str, Any]) -> None:
    """渲染 Upsert 阶段：各存储的详情及 chunk 映射。"""
    dense_store = data.get("dense_store", {})
    sparse_store = data.get("sparse_store", {})
    image_store = data.get("image_store", {})

    # ── 概览指标 ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            tr("ingestion_traces.embed_metric_dense"),
            dense_store.get("count", data.get("vector_count", 0)),
        )
    with c2:
        st.metric(
            tr("ingestion_traces.upsert_metric_sparse"),
            sparse_store.get("count", data.get("bm25_docs", 0)),
        )
    with c3:
        st.metric(tr("common.images"), image_store.get("count", data.get("images_indexed", 0)))

    # ── Dense 存储详情 ──
    if dense_store:
        with st.expander(tr("ingestion_traces.dense_store_title"), expanded=True):
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(
                    f"**{tr('common.backend')}:** `{dense_store.get('backend', '—')}`"
                )
                st.markdown(
                    f"**{tr('common.collection_label')}:** `{dense_store.get('collection', '—')}`"
                )
            with dc2:
                st.markdown(f"**{tr('common.path')}:** `{dense_store.get('path', '—')}`")
                st.markdown(
                    f"**{tr('ingestion_traces.col_vectors')}:** {dense_store.get('count', 0)}"
                )

    # ── Sparse 存储详情 ──
    if sparse_store:
        with st.expander(tr("ingestion_traces.sparse_store_title"), expanded=True):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(
                    f"**{tr('common.backend')}:** `{sparse_store.get('backend', '—')}`"
                )
                st.markdown(
                    f"**{tr('common.collection_label')}:** `{sparse_store.get('collection', '—')}`"
                )
            with sc2:
                st.markdown(f"**{tr('common.path')}:** `{sparse_store.get('path', '—')}`")
                st.markdown(
                    f"**{tr('ingestion_traces.col_documents')}:** {sparse_store.get('count', 0)}"
                )

    # ── 图片存储详情 ──
    if image_store and image_store.get("count", 0) > 0:
        with st.expander(
            tr("ingestion_traces.image_store_title", count=image_store.get("count", 0)),
            expanded=True,
        ):
            st.markdown(f"**{tr('common.backend')}:** `{image_store.get('backend', '—')}`")
            imgs = image_store.get("images", [])
            if imgs:
                img_rows = [
                    {
                        tr("ingestion_traces.col_image_id"): img.get("image_id", ""),
                        tr("ingestion_traces.col_page"): img.get("page", 0),
                        tr("ingestion_traces.col_file"): img.get("file_path", ""),
                        tr("ingestion_traces.col_doc_hash"): img.get("doc_hash", "")[:16] + "…",
                    }
                    for img in imgs
                ]
                st.table(img_rows)

    # ── Chunk → 向量 ID 映射 ──
    chunk_mapping = data.get("chunk_mapping", [])
    if chunk_mapping:
        with st.expander(
            tr("ingestion_traces.chunk_mapping_title", count=len(chunk_mapping)),
            expanded=False,
        ):
            mapping_rows = [
                {
                    tr("ingestion_traces.col_index"): i + 1,
                    tr("common.chunk_id"): m.get("chunk_id", ""),
                    tr("ingestion_traces.col_vector_id"): m.get("vector_id", ""),
                    tr("ingestion_traces.col_store"): m.get("store", ""),
                    tr("common.collection_label"): m.get("collection", ""),
                }
                for i, m in enumerate(chunk_mapping)
            ]
            st.table(mapping_rows)

    # ── 兜底：仅含 vector_ids 的旧格式 ──
    if not chunk_mapping and not dense_store:
        vector_ids = data.get("vector_ids", [])
        if vector_ids:
            with st.expander(tr("ingestion_traces.vector_ids"), expanded=False):
                for vid in vector_ids:
                    st.code(vid, language=None)
