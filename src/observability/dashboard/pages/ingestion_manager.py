"""Ingestion Manager 页面 – 上传文件、触发摄入、删除文档。

布局：
1. 文件上传器 + collection 选择器
2. 摄入按钮 → 进度条（使用 on_progress 回调）
3. 文档列表及删除按钮
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.observability.dashboard.i18n import tr
from src.observability.dashboard.services.data_service import DataService


def _run_ingestion(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
) -> None:
    """将上传的文件保存到临时位置并运行 Pipeline。"""
    from src.core.settings import load_settings
    from src.core.trace import TraceContext, TraceCollector
    from src.ingestion.pipeline import IngestionPipeline

    settings = load_settings()

    # 将上传的文件写入临时位置
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    _STAGE_LABELS = {
        "integrity": tr("ingestion_manager.stage_integrity"),
        "load": tr("ingestion_manager.stage_load"),
        "split": tr("ingestion_manager.stage_split"),
        "transform": tr("ingestion_manager.stage_transform"),
        "embed": tr("ingestion_manager.stage_embed"),
        "upsert": tr("ingestion_manager.stage_upsert"),
    }

    def on_progress(stage: str, current: int, total: int) -> None:
        frac = (current - 1) / total  # 阶段刚开始，显示部分进度
        label = _STAGE_LABELS.get(stage, stage)
        progress_bar.progress(
            frac, text=tr("ingestion_manager.progress", current=current, total=total, label=label)
        )
        status_text.caption(label)

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["source_path"] = uploaded_file.name
    trace.metadata["collection"] = collection
    trace.metadata["source"] = "dashboard"

    try:
        # 开启离线数据摄入(Ingestion Pipeline)
        # Document （文档加载）
        # chunking （切分
        # Transform (增强)
        # Embedding （向量化）
        # storage （存储）)
        pipeline = IngestionPipeline(settings, collection=collection)
        pipeline.run(
            file_path=tmp_path,
            trace=trace,
            on_progress=on_progress,
        )
        progress_bar.progress(1.0, text=tr("ingestion_manager.complete"))
        status_text.success(
            tr("ingestion_manager.success", file=uploaded_file.name, collection=collection)
        )
    except Exception as exc:
        status_text.error(tr("ingestion_manager.failed", exc=exc))
    finally:
        TraceCollector().collect(trace)
        # 清理临时文件
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def render() -> None:
    """渲染 Ingestion Manager 页面。"""
    st.header(tr("ingestion_manager.header"))

    # ── 上传区域 ──────────────────────────────────────────────────
    st.subheader(tr("ingestion_manager.upload_section"))

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader(
            tr("ingestion_manager.file_label"),
            type=["pdf", "txt", "md", "docx"],
            key="ingest_uploader",
        )
    with col2:
        collection = st.text_input(
            tr("common.collection"), value="default", key="ingest_collection"
        )

    if uploaded is not None:
        if st.button(tr("ingestion_manager.start"), key="btn_ingest"):
            progress_bar = st.progress(0, text=tr("ingestion_manager.preparing"))
            status_text = st.empty()
            _run_ingestion(uploaded, collection.strip() or "default", progress_bar, status_text)

    st.divider()

    # ── 文档管理区域 ──────────────────────────────────────────────
    st.subheader(tr("ingestion_manager.manage"))

    try:
        svc = DataService()
        docs = svc.list_documents()
    except Exception as exc:
        st.error(tr("ingestion_manager.load_failed", exc=exc))
        return

    if not docs:
        st.info(tr("ingestion_manager.no_docs"))
        return

    for idx, doc in enumerate(docs):
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.markdown(
                tr(
                    "ingestion_manager.doc_line",
                    path=doc["source_path"],
                    collection=doc.get("collection", "—"),
                    chunks=doc["chunk_count"],
                    images=doc["image_count"],
                )
            )
        with col_btn:
            if st.button(tr("ingestion_manager.delete"), key=f"del_{idx}"):
                try:
                    result = svc.delete_document(
                        source_path=doc["source_path"],
                        collection=doc.get("collection", "default"),
                        source_hash=doc.get("source_hash"),
                    )
                    if result.success:
                        st.success(
                            tr(
                                "ingestion_manager.deleted",
                                chunks=result.chunks_deleted,
                                images=result.images_deleted,
                            )
                        )
                        st.rerun()
                    else:
                        st.warning(tr("ingestion_manager.partial", errors=result.errors))
                except Exception as exc:
                    st.error(tr("ingestion_manager.delete_failed", exc=exc))
