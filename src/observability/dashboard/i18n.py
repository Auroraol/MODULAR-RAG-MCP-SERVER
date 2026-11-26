"""Dashboard 国际化（i18n）。

提供中英文界面文案的集中管理：所有 Dashboard 页面通过 ``tr(key)`` 取文案，
语言状态保存在 ``st.session_state["lang"]`` 中，由 :func:`language_selector`
在侧边栏切换。

设计要点：

- ``_EN`` 中的文案与原硬编码字符串逐字一致，保证英文模式外观不变。
- ``_ZH`` 与 ``_EN`` 的键集合必须完全相同
  （``tests/unit/test_dashboard_i18n.py`` 会校验）。
- 含占位符的文案使用 ``str.format`` 风格，调用时传入关键字参数；
  不含占位符的文案原样返回，避免正文中的花括号被误解析。
"""

from __future__ import annotations

from typing import Any

import streamlit as st

DEFAULT_LANGUAGE = "zh"

#: 语言代码 → 显示名
LANGUAGE_NAMES: dict[str, str] = {
    "zh": "中文",
    "en": "English",
}

_SESSION_KEY = "lang"


# ═══════════════════════════════════════════════════════════════
# 英文文案（与原硬编码逐字一致）
# ═══════════════════════════════════════════════════════════════

_EN: dict[str, str] = {
    # ── 通用 ──────────────────────────────────────────────────
    "common.language": "Language",
    "common.collection": "Collection",
    "common.chunks": "Chunks",
    "common.images": "Images",
    "common.metadata": "📋 Metadata",
    "common.cancel": "❌ Cancel",
    "common.stage": "Stage",
    "common.elapsed_ms": "Elapsed (ms)",
    "common.method": "Method",
    "common.provider": "Provider",
    "common.collection_label": "Collection",
    "common.backend": "Backend",
    "common.path": "Path",
    "common.chunk_id": "Chunk ID",
    "common.chars": "Chars",
    # ── 应用框架 ──────────────────────────────────────────────
    "app.title": "Modular RAG Dashboard",
    "nav.overview": "Overview",
    "nav.data_browser": "Data Browser",
    "nav.query_search": "Query Search",
    "nav.ingestion_manager": "Ingestion Manager",
    "nav.ingestion_traces": "Ingestion Traces",
    "nav.query_traces": "Query Traces",
    "nav.evaluation_panel": "Evaluation Panel",
    # ── Overview ──────────────────────────────────────────────
    "overview.header": "📊 System Overview",
    "overview.component_configuration": "🔧 Component Configuration",
    "overview.details": "Details",
    "overview.collection_statistics": "📁 Collection Statistics",
    "overview.empty": "⚠️ Empty",
    "overview.no_collections": (
        "**No collections found or ChromaDB unavailable.** "
        "Go to the Ingestion Manager page to upload and ingest documents."
    ),
    "overview.trace_statistics": "📈 Trace Statistics",
    "overview.total_traces": "Total traces",
    "overview.no_traces": "No traces recorded yet. Run a query or ingestion first.",
    "overview.config_failed": "Failed to load configuration: {exc}",
    "overview.card_caption": "Provider: `{provider}`  \nModel: `{model}`",
    # ── Data Browser ──────────────────────────────────────────
    "data_browser.header": "🔍 Data Browser",
    "data_browser.init_failed": "Failed to initialise DataService: {exc}",
    "data_browser.danger_zone": "⚠️ Danger Zone",
    "data_browser.danger_warning": (
        "This will **permanently delete** all data: "
        "ChromaDB collections, BM25 indexes, images, ingestion history, and trace logs."
    ),
    "data_browser.clear_all": "🗑️ Clear All Data",
    "data_browser.confirm_prompt": "Are you sure? This action cannot be undone!",
    "data_browser.confirm_yes": "✅ Yes, delete everything",
    "data_browser.clear_errors": "Cleared with {count} error(s): {errors}",
    "data_browser.clear_ok": "All data cleared! {count} collection(s) deleted.",
    "data_browser.load_failed": "Failed to load documents: {exc}",
    "data_browser.no_docs": (
        "**No documents found in this collection.** "
        "Use the Ingestion Manager page to upload and ingest files, "
        "or select a different collection from the dropdown above."
    ),
    "data_browser.documents": "📄 Documents ({count})",
    "data_browser.doc_label": "📑 {name}  —  {chunks} chunks · {images} images",
    "data_browser.doc_caption": (
        "**Source:** {source}  ·  **Hash:** `{hash}…`  ·  **Processed:** {processed}"
    ),
    "data_browser.chunks_title": "### 📦 Chunks ({count})",
    "data_browser.chunk_meta": "**Chunk {index}** · `{chunk_id}` · {chars} chars",
    "data_browser.content": "Content",
    "data_browser.no_chunks": "No chunks found in vector store for this document.",
    "data_browser.images_title": "### 🖼️ Images ({count})",
    "data_browser.image_missing": "{image_id} (file missing)",
    # ── Query Search ──────────────────────────────────────────
    "query_search.header": "🔎 Query Search",
    "query_search.query_label": "Query",
    "query_search.placeholder": "Ask a question or enter keywords…",
    "query_search.top_k": "Top K",
    "query_search.search": "🔍 Search",
    "query_search.searching": "Searching…",
    "query_search.empty_query": "Please enter a query.",
    "query_search.failed": "Search failed: {exc}",
    "query_search.results_caption": "Collection `{collection}` · {count} result(s)",
    "query_search.images": "🖼️ Related Images",
    # ── Ingestion Manager ─────────────────────────────────────
    "ingestion_manager.header": "📥 Ingestion Manager",
    "ingestion_manager.upload_section": "📤 Upload & Ingest",
    "ingestion_manager.file_label": "Select a file to ingest",
    "ingestion_manager.start": "🚀 Start Ingestion",
    "ingestion_manager.preparing": "Preparing…",
    "ingestion_manager.stage_integrity": "🔍 Checking file integrity…",
    "ingestion_manager.stage_load": "📄 Loading document…",
    "ingestion_manager.stage_split": "✂️ Chunking document…",
    "ingestion_manager.stage_transform": "🔄 Transforming chunks (LLM refine + enrich)…",
    "ingestion_manager.stage_embed": "🔢 Encoding vectors…",
    "ingestion_manager.stage_upsert": "💾 Storing to database…",
    "ingestion_manager.progress": "[{current}/{total}] {label}",
    "ingestion_manager.complete": "✅ Complete",
    "ingestion_manager.success": (
        "Successfully ingested **{file}** into collection **{collection}**."
    ),
    "ingestion_manager.failed": "Ingestion failed: {exc}",
    "ingestion_manager.manage": "🗑️ Manage Documents",
    "ingestion_manager.load_failed": "Failed to load documents: {exc}",
    "ingestion_manager.no_docs": (
        "**No documents ingested yet.** "
        'Upload a PDF, TXT, MD, or DOCX file above and click "Start Ingestion" to begin.'
    ),
    "ingestion_manager.doc_line": (
        "**{path}** — collection: `{collection}` | chunks: {chunks} | images: {images}"
    ),
    "ingestion_manager.delete": "🗑️ Delete",
    "ingestion_manager.deleted": "Deleted: {chunks} chunks, {images} images removed.",
    "ingestion_manager.partial": "Partial delete. Errors: {errors}",
    "ingestion_manager.delete_failed": "Delete failed: {exc}",
    # ── Ingestion Traces ──────────────────────────────────────
    "ingestion_traces.header": "🔬 Ingestion Traces",
    "ingestion_traces.none": "No ingestion traces recorded yet. Run an ingestion first!",
    "ingestion_traces.history": "📋 Trace History ({count})",
    "ingestion_traces.expander_title": "📄 **{file}** · {elapsed} · {started}",
    "ingestion_traces.pipeline_overview": "#### 📊 Pipeline Overview",
    "ingestion_traces.source_caption": "Source: `{source}`",
    "ingestion_traces.metric_doc_length": "Doc Length",
    "ingestion_traces.metric_vectors": "Vectors",
    "ingestion_traces.metric_total_time": "Total Time",
    "ingestion_traces.chars_value": "{count:,} chars",
    "ingestion_traces.stage_timings": "#### ⏱️ Stage Timings",
    "ingestion_traces.stage_details": "#### 🔍 Stage Details",
    "ingestion_traces.no_stage_details": "No stage details available.",
    "ingestion_traces.tab_load": "📄 Load",
    "ingestion_traces.tab_split": "✂️ Split",
    "ingestion_traces.tab_transform": "🔄 Transform",
    "ingestion_traces.tab_embed": "🔢 Embed",
    "ingestion_traces.tab_upsert": "💾 Upsert",
    "ingestion_traces.diag_missing_load": (
        "**Pipeline incomplete — missing stages: {names}.** "
        "The Load stage failed or was skipped. The document may be corrupted or unsupported."
    ),
    "ingestion_traces.diag_missing_other": (
        "**Pipeline incomplete — missing stages: {names}.** "
        "An error may have occurred during processing. Check the logs for details."
    ),
    "ingestion_traces.diag_load_empty": (
        "**Load stage produced empty text.** "
        "The document may be image-only or in an unsupported format."
    ),
    "ingestion_traces.diag_split_empty": (
        "**Split stage produced 0 chunks.** The document text may be too short or empty."
    ),
    "ingestion_traces.diag_transform_none": (
        "**Transform:** No chunks were refined. "
        "LLM refinement may be disabled or skipped for short chunks."
    ),
    "ingestion_traces.diag_embed_empty": (
        "**Embed stage produced 0 vectors.** "
        "Embedding API may have failed. Check API key and endpoint."
    ),
    "ingestion_traces.diag_upsert_empty": (
        "**Upsert stage stored 0 vectors.** Database write may have failed."
    ),
    "ingestion_traces.stage_error": "**{stage} stage error:** {error}",
    "ingestion_traces.load_metric_doc_id": "Doc ID",
    "ingestion_traces.load_metric_text_length": "Text Length",
    "ingestion_traces.raw_text": "**Raw Document Text**",
    "ingestion_traces.no_text_preview": "No text preview recorded in this trace.",
    "ingestion_traces.split_metric_avg_size": "Avg Size",
    "ingestion_traces.chars_suffix": "{count} chars",
    "ingestion_traces.chunks_after_split": "**Chunks after splitting**",
    "ingestion_traces.split_chunk_header": (
        "📝 **Chunk #{index}** — `{chunk_id}` — {chars} chars"
    ),
    "ingestion_traces.no_chunk_text": (
        "No chunk text recorded. Re-run ingestion to generate new traces."
    ),
    "ingestion_traces.transform_refined": "Refined (LLM / Rule)",
    "ingestion_traces.transform_enriched": "Enriched (LLM / Rule)",
    "ingestion_traces.transform_captioned": "Captioned",
    "ingestion_traces.per_chunk_transform": "**Per-chunk transform results**",
    "ingestion_traces.transform_chunk_header": "🔄 **Chunk #{index}** — `{chunk_id}` — {badges}",
    "ingestion_traces.enriched_metadata": "**Enriched Metadata**",
    "ingestion_traces.title_label": "**Title:** {title}",
    "ingestion_traces.no_title": "_No title_",
    "ingestion_traces.tags_label": "**Tags:** {tags}",
    "ingestion_traces.no_tags": "_No tags_",
    "ingestion_traces.summary_label": "**Summary:** {summary}",
    "ingestion_traces.text_comparison": "**Text Comparison**",
    "ingestion_traces.before_refinement": "*Before refinement:*",
    "ingestion_traces.after_refinement": "*After refinement + enrichment:*",
    "ingestion_traces.empty_text": "(empty)",
    "ingestion_traces.no_transform_data": (
        "No per-chunk transform data recorded. Re-run ingestion for new traces."
    ),
    "ingestion_traces.embed_metric_dense": "Dense Vectors",
    "ingestion_traces.embed_metric_dimension": "Dimension",
    "ingestion_traces.embed_metric_sparse": "Sparse Docs",
    "ingestion_traces.no_encoding_data": "No chunk encoding data recorded.",
    "ingestion_traces.dense_tab": "🟦 Dense Encoding",
    "ingestion_traces.sparse_tab": "🟨 Sparse Encoding (BM25)",
    "ingestion_traces.dense_explain": (
        "Each chunk → **float vector** via embedding model (e.g. `text-embedding-ada-002`)"
    ),
    "ingestion_traces.sparse_explain": (
        "Each chunk → **term frequency stats** for BM25 indexing"
    ),
    "ingestion_traces.col_index": "#",
    "ingestion_traces.col_est_tokens": "Est. Tokens",
    "ingestion_traces.col_dense_dim": "Dense Dim",
    "ingestion_traces.col_doc_length": "Doc Length (terms)",
    "ingestion_traces.col_unique_terms": "Unique Terms",
    "ingestion_traces.top_terms_title": "🔤 Chunk {index} — Top Terms",
    "ingestion_traces.col_term": "Term",
    "ingestion_traces.col_freq": "Freq",
    "ingestion_traces.upsert_metric_sparse": "Sparse (BM25)",
    "ingestion_traces.dense_store_title": "🟦 Dense Vector Store (ChromaDB)",
    "ingestion_traces.sparse_store_title": "🟨 Sparse Index (BM25)",
    "ingestion_traces.image_store_title": "🖼️ Image Storage ({count} images)",
    "ingestion_traces.col_vectors": "Vectors",
    "ingestion_traces.col_documents": "Documents",
    "ingestion_traces.col_image_id": "Image ID",
    "ingestion_traces.col_page": "Page",
    "ingestion_traces.col_file": "File",
    "ingestion_traces.col_doc_hash": "Doc Hash",
    "ingestion_traces.chunk_mapping_title": "🔗 Chunk → Vector Mapping ({count} entries)",
    "ingestion_traces.col_vector_id": "Vector ID",
    "ingestion_traces.col_store": "Store",
    "ingestion_traces.vector_ids": "Vector IDs",
    # ── Query Traces ──────────────────────────────────────────
    "query_traces.header": "🔎 Query Traces",
    "query_traces.none": "No query traces recorded yet. Run a query first!",
    "query_traces.keyword_label": "Search by query keyword",
    "query_traces.history": "📋 Query History ({count})",
    "query_traces.expander_title": '🔍 "{query}"  ·  {elapsed}  ·  {started}',
    "query_traces.query_section": "#### 💬 Query",
    "query_traces.source_label": "**Source:** {emoji} `{source}`",
    "query_traces.top_k_label": "**Top-K:** `{top_k}`",
    "query_traces.collection_label": "**Collection:** `{collection}`",
    "query_traces.metric_dense_hits": "Dense Hits",
    "query_traces.metric_sparse_hits": "Sparse Hits",
    "query_traces.metric_fused": "Fused",
    "query_traces.metric_after_rerank": "After Rerank",
    "query_traces.metric_total_time": "Total Time",
    "query_traces.stage_timings": "#### ⏱️ Stage Timings",
    "query_traces.stage_details": "#### 🔍 Stage Details",
    "query_traces.no_stage_details": "No stage details available.",
    "query_traces.tab_query_processing": "🔤 Query Processing",
    "query_traces.tab_dense": "🟦 Dense Retrieval",
    "query_traces.tab_sparse": "🟨 Sparse Retrieval",
    "query_traces.tab_fusion": "🟩 Fusion (RRF)",
    "query_traces.tab_rerank": "🟪 Rerank",
    "query_traces.diag_dense_failed": "**Dense Retrieval failed:** {error}",
    "query_traces.diag_dense_empty": (
        "Dense Retrieval returned **0 results**. Check if the collection has indexed data."
    ),
    "query_traces.diag_sparse_failed": "**Sparse Retrieval failed:** {error}",
    "query_traces.diag_sparse_empty": (
        "Sparse (BM25) Retrieval returned **0 results**. "
        "BM25 index may be empty or not yet built for this collection."
    ),
    "query_traces.diag_fusion_missing_both": (
        "Fusion stage was not recorded even though both retrievers returned results."
    ),
    "query_traces.diag_fusion_skipped": (
        "**Fusion (RRF) skipped:** only {source} retrieval returned results. "
        "Fusion requires both Dense and Sparse results to merge."
    ),
    "query_traces.diag_rerank_skipped": (
        "**Rerank skipped:** reranker is not enabled or not configured. "
        "Enable `reranker` in settings.yaml to apply LLM-based reranking."
    ),
    "query_traces.diag_no_results": (
        "**No results found.** The collection may be empty, or the query "
        "doesn't match any indexed content. Try ingesting data first."
    ),
    "query_traces.ragas_section": "#### 📏 Ragas Evaluation",
    "query_traces.answer_label": "✏️ Generated Answer",
    "query_traces.answer_placeholder": "Paste the generated answer, or the LLM's actual output…",
    "query_traces.answer_help": (
        "Ragas uses an LLM-as-Judge. faithfulness measures whether the answer "
        "stays true to the retrieved context; answer_relevancy measures how "
        "relevant the answer is to the question. Without an answer, the scores "
        "are not meaningful."
    ),
    "query_traces.ragas_button": "📏 Ragas Evaluate",
    "query_traces.ragas_button_help": "Re-run this query and score with Ragas (LLM-as-Judge)",
    "query_traces.ragas_answer_required": "⚠️ Please enter an answer above before running Ragas.",
    "query_traces.running_ragas": "Running Ragas evaluation…",
    "query_traces.eval_failed": "❌ Evaluation failed: {error}",
    "query_traces.no_metrics": "No metrics returned.",
    "query_traces.ragas_scores": "**📏 Ragas Scores**",
    "query_traces.no_chunks_retrieved": "No chunks retrieved — is data indexed?",
    "query_traces.ragas_not_installed": "Ragas not installed: {error}",
    "query_traces.original_query": "**Original Query**",
    "query_traces.extracted_keywords": "**Extracted Keywords**",
    "query_traces.no_keywords": "No keywords extracted.",
    "query_traces.metric_keywords": "Keywords",
    "query_traces.metric_results": "Results",
    "query_traces.top_k_requested": "**Top-K requested:** `{top_k}`",
    "query_traces.no_results_returned": "No {label} results returned.",
    "query_traces.metric_input_lists": "Input Lists",
    "query_traces.metric_fused_results": "Fused Results",
    "query_traces.top_k_short": "**Top-K:** `{top_k}`",
    "query_traces.no_fusion": "No fusion results.",
    "query_traces.metric_input": "Input",
    "query_traces.metric_output": "Output",
    "query_traces.no_reranked": "No reranked results.",
    "query_traces.chunk_header": "{bar} **#{index}** — Score: `{score}`",
    "query_traces.chunk_id_caption": "Chunk ID: `{chunk_id}`",
    "query_traces.source_caption": "Source: `{source}`",
    "query_traces.no_text": "_No text available_",
    # ── Evaluation Panel ──────────────────────────────────────
    "evaluation_panel.header": "📏 Evaluation Panel",
    "evaluation_panel.intro": (
        "Run evaluation against a **golden test set** to measure retrieval "
        "and generation quality. Results include per-query details and "
        "aggregate metrics."
    ),
    "evaluation_panel.configuration": "⚙️ Configuration",
    "evaluation_panel.backend_label": "Evaluator Backend",
    "evaluation_panel.backend_help": "Select which evaluator backend to use.",
    "evaluation_panel.custom_hint": (
        "ℹ️ **Custom Evaluator** is still a placeholder: it needs "
        "`expected_chunk_ids` in the golden test set as ground truth before it "
        "can compute hit_rate / MRR. Use the **ragas** backend for now."
    ),
    "evaluation_panel.top_k_label": "Top-K",
    "evaluation_panel.top_k_help": "Number of chunks to retrieve per query.",
    "evaluation_panel.collection_label": "Collection (optional)",
    "evaluation_panel.collection_help": "Limit retrieval to a specific collection.",
    "evaluation_panel.golden_label": "Golden Test Set Path",
    "evaluation_panel.golden_help": "Path to the golden_test_set.json file.",
    "evaluation_panel.golden_missing": (
        "⚠️ **Golden test set not found:** `{path}`. "
        "Create a JSON file with test queries and expected results. "
        "See `tests/fixtures/golden_test_set.json` for the format."
    ),
    "evaluation_panel.answers_section": "✏️ Provide Answers",
    "evaluation_panel.answers_caption": (
        "**RAGAS needs Query + Context + Answer.** The traces only contain the "
        "query and the retrieved context. Fill in the actual system answer for "
        "each test case to get meaningful faithfulness and answer_relevancy scores."
    ),
    "evaluation_panel.answer_placeholder": "Enter the system answer for this question…",
    "evaluation_panel.answer_help": (
        "Query: {query}\n\n"
        "Provide the LLM output or the expected answer. Ragas scores "
        "faithfulness and answer_relevancy against it."
    ),
    "evaluation_panel.filled_warning": (
        "⚠️ {filled}/{total} answers filled. Unfilled cases will fall back to "
        "concatenated chunks as the answer (scores may be inaccurate)."
    ),
    "evaluation_panel.filled_ok": "✅ All {total} answers filled.",
    "evaluation_panel.load_cases_failed": "Could not load test case preview: {exc}",
    "evaluation_panel.run_button": "▶️  Run Evaluation",
    "evaluation_panel.loading": "Loading evaluator and running evaluation…",
    "evaluation_panel.failed": "❌ Evaluation failed: {exc}",
    "evaluation_panel.complete": "✅ Evaluation complete!",
    "evaluation_panel.aggregate": "📊 Aggregate Metrics",
    "evaluation_panel.no_aggregate": "No aggregate metrics available.",
    "evaluation_panel.aggregate_caption": (
        "Evaluator: **{evaluator}** · Queries: **{queries}** · "
        "Total time: **{elapsed:.0f} ms**"
    ),
    "evaluation_panel.per_query": "🔍 Per-Query Details",
    "evaluation_panel.no_per_query": "No per-query results available.",
    "evaluation_panel.query_summary_none": "no metrics",
    "evaluation_panel.query_expander": "**Q{index}**: {query} — {elapsed:.0f} ms — {metrics}",
    "evaluation_panel.retrieved_chunks": "**Retrieved Chunks** ({count}):",
    "evaluation_panel.generated_answer": "**Generated Answer:**",
    "evaluation_panel.history": "📈 Evaluation History",
    "evaluation_panel.no_history": (
        "**No evaluation history yet.** "
        'Configure the evaluator above and click "Run Evaluation" to start. '
        "Results will be saved here for comparison across runs."
    ),
    "evaluation_panel.col_timestamp": "Timestamp",
    "evaluation_panel.col_evaluator": "Evaluator",
    "evaluation_panel.col_queries": "Queries",
    "evaluation_panel.col_time_ms": "Time (ms)",
}


# ═══════════════════════════════════════════════════════════════
# 中文文案
# ═══════════════════════════════════════════════════════════════

_ZH: dict[str, str] = {
    # ── 通用 ──────────────────────────────────────────────────
    "common.language": "语言",
    "common.collection": "Collection",
    "common.chunks": "Chunks",
    "common.images": "图片",
    "common.metadata": "📋 元数据",
    "common.cancel": "❌ 取消",
    "common.stage": "阶段",
    "common.elapsed_ms": "耗时 (ms)",
    "common.method": "方法",
    "common.provider": "Provider",
    "common.collection_label": "Collection",
    "common.backend": "后端",
    "common.path": "路径",
    "common.chunk_id": "Chunk ID",
    "common.chars": "字符数",
    # ── 应用框架 ──────────────────────────────────────────────
    "app.title": "Modular RAG Dashboard",
    "nav.overview": "系统总览",
    "nav.data_browser": "数据浏览",
    "nav.query_search": "在线检索",
    "nav.ingestion_manager": "摄取管理",
    "nav.ingestion_traces": "摄取 Trace",
    "nav.query_traces": "查询 Trace",
    "nav.evaluation_panel": "评估面板",
    # ── Overview ──────────────────────────────────────────────
    "overview.header": "📊 系统总览",
    "overview.component_configuration": "🔧 组件配置",
    "overview.details": "详情",
    "overview.collection_statistics": "📁 Collection 统计",
    "overview.empty": "⚠️ 空",
    "overview.no_collections": (
        "**未找到 Collection，或 ChromaDB 不可用。** "
        "请前往「摄取管理」页面上传并摄取文档。"
    ),
    "overview.trace_statistics": "📈 Trace 统计",
    "overview.total_traces": "Trace 总数",
    "overview.no_traces": "暂无 Trace 记录，请先运行一次查询或摄取。",
    "overview.config_failed": "配置加载失败：{exc}",
    "overview.card_caption": "Provider：`{provider}`  \n模型：`{model}`",
    # ── Data Browser ──────────────────────────────────────────
    "data_browser.header": "🔍 数据浏览",
    "data_browser.init_failed": "DataService 初始化失败：{exc}",
    "data_browser.danger_zone": "⚠️ 危险区域",
    "data_browser.danger_warning": (
        "此操作将**永久删除**所有数据："
        "ChromaDB collections、BM25 索引、图片、摄取历史以及 Trace 日志。"
    ),
    "data_browser.clear_all": "🗑️ 清空所有数据",
    "data_browser.confirm_prompt": "确定吗？此操作无法撤销！",
    "data_browser.confirm_yes": "✅ 确定，全部删除",
    "data_browser.clear_errors": "已清空，但有 {count} 个错误：{errors}",
    "data_browser.clear_ok": "所有数据已清空！共删除 {count} 个 collection。",
    "data_browser.load_failed": "文档加载失败：{exc}",
    "data_browser.no_docs": (
        "**该 collection 中未找到文档。** "
        "请使用「摄取管理」页面上传并摄取文件，或在上方下拉框中选择其他 collection。"
    ),
    "data_browser.documents": "📄 文档（{count}）",
    "data_browser.doc_label": "📑 {name}  —  {chunks} 个 chunk · {images} 张图片",
    "data_browser.doc_caption": (
        "**来源：** {source}  ·  **哈希：** `{hash}…`  ·  **处理时间：** {processed}"
    ),
    "data_browser.chunks_title": "### 📦 Chunks（{count}）",
    "data_browser.chunk_meta": "**Chunk {index}** · `{chunk_id}` · {chars} 字符",
    "data_browser.content": "内容",
    "data_browser.no_chunks": "该文档在向量库中没有对应的 chunk。",
    "data_browser.images_title": "### 🖼️ 图片（{count}）",
    "data_browser.image_missing": "{image_id}（文件缺失）",
    # ── Query Search ──────────────────────────────────────────
    "query_search.header": "🔎 在线检索",
    "query_search.query_label": "查询",
    "query_search.placeholder": "输入问题或关键词…",
    "query_search.top_k": "返回数量",
    "query_search.search": "🔍 检索",
    "query_search.searching": "检索中…",
    "query_search.empty_query": "请输入查询内容。",
    "query_search.failed": "检索失败：{exc}",
    "query_search.results_caption": "集合 `{collection}` · {count} 条结果",
    "query_search.images": "🖼️ 相关图片",
    # ── Ingestion Manager ─────────────────────────────────────
    "ingestion_manager.header": "📥 摄取管理",
    "ingestion_manager.upload_section": "📤 上传并摄取",
    "ingestion_manager.file_label": "选择要摄取的文件",
    "ingestion_manager.start": "🚀 开始摄取",
    "ingestion_manager.preparing": "准备中…",
    "ingestion_manager.stage_integrity": "🔍 校验文件完整性…",
    "ingestion_manager.stage_load": "📄 加载文档…",
    "ingestion_manager.stage_split": "✂️ 切分文档…",
    "ingestion_manager.stage_transform": "🔄 转换 chunk（LLM 精炼 + 元数据增强）…",
    "ingestion_manager.stage_embed": "🔢 向量编码…",
    "ingestion_manager.stage_upsert": "💾 写入数据库…",
    "ingestion_manager.progress": "[{current}/{total}] {label}",
    "ingestion_manager.complete": "✅ 完成",
    "ingestion_manager.success": "已成功将 **{file}** 摄取到 collection **{collection}**。",
    "ingestion_manager.failed": "摄取失败：{exc}",
    "ingestion_manager.manage": "🗑️ 管理文档",
    "ingestion_manager.load_failed": "文档加载失败：{exc}",
    "ingestion_manager.no_docs": (
        "**尚未摄取任何文档。** "
        "请在上方上传 PDF、TXT、MD 或 DOCX 文件，然后点击「开始摄取」。"
    ),
    "ingestion_manager.doc_line": (
        "**{path}** — collection：`{collection}` | chunks：{chunks} | images：{images}"
    ),
    "ingestion_manager.delete": "🗑️ 删除",
    "ingestion_manager.deleted": "已删除：{chunks} 个 chunk，{images} 张图片。",
    "ingestion_manager.partial": "部分删除失败。错误：{errors}",
    "ingestion_manager.delete_failed": "删除失败：{exc}",
    # ── Ingestion Traces ──────────────────────────────────────
    "ingestion_traces.header": "🔬 摄取 Trace",
    "ingestion_traces.none": "暂无摄取 Trace 记录，请先运行一次摄取！",
    "ingestion_traces.history": "📋 Trace 历史（{count}）",
    "ingestion_traces.expander_title": "📄 **{file}** · {elapsed} · {started}",
    "ingestion_traces.pipeline_overview": "#### 📊 Pipeline 概览",
    "ingestion_traces.source_caption": "来源：`{source}`",
    "ingestion_traces.metric_doc_length": "文档长度",
    "ingestion_traces.metric_vectors": "向量数",
    "ingestion_traces.metric_total_time": "总耗时",
    "ingestion_traces.chars_value": "{count:,} 字符",
    "ingestion_traces.stage_timings": "#### ⏱️ 阶段耗时",
    "ingestion_traces.stage_details": "#### 🔍 阶段详情",
    "ingestion_traces.no_stage_details": "无阶段详情可显示。",
    "ingestion_traces.tab_load": "📄 加载",
    "ingestion_traces.tab_split": "✂️ 切分",
    "ingestion_traces.tab_transform": "🔄 转换",
    "ingestion_traces.tab_embed": "🔢 编码",
    "ingestion_traces.tab_upsert": "💾 入库",
    "ingestion_traces.diag_missing_load": (
        "**Pipeline 不完整 —— 缺失阶段：{names}。** "
        "Load 阶段失败或被跳过，文档可能已损坏或格式不受支持。"
    ),
    "ingestion_traces.diag_missing_other": (
        "**Pipeline 不完整 —— 缺失阶段：{names}。** "
        "处理过程中可能发生错误，请查看日志了解详情。"
    ),
    "ingestion_traces.diag_load_empty": (
        "**Load 阶段产出为空文本。** 文档可能只有图片，或格式不受支持。"
    ),
    "ingestion_traces.diag_split_empty": (
        "**Split 阶段产出 0 个 chunk。** 文档文本可能过短或为空。"
    ),
    "ingestion_traces.diag_transform_none": (
        "**Transform：** 没有 chunk 被精炼。可能是 LLM 精炼被关闭，或 chunk 太短被跳过。"
    ),
    "ingestion_traces.diag_embed_empty": (
        "**Embed 阶段产出 0 个向量。** Embedding API 可能失败，请检查 API Key 与端点。"
    ),
    "ingestion_traces.diag_upsert_empty": (
        "**Upsert 阶段写入 0 个向量。** 数据库写入可能失败。"
    ),
    "ingestion_traces.stage_error": "**{stage} 阶段错误：** {error}",
    "ingestion_traces.load_metric_doc_id": "文档 ID",
    "ingestion_traces.load_metric_text_length": "文本长度",
    "ingestion_traces.raw_text": "**原始文档文本**",
    "ingestion_traces.no_text_preview": "该 Trace 未记录文本预览。",
    "ingestion_traces.split_metric_avg_size": "平均大小",
    "ingestion_traces.chars_suffix": "{count} 字符",
    "ingestion_traces.chunks_after_split": "**切分后的 chunk**",
    "ingestion_traces.split_chunk_header": (
        "📝 **Chunk #{index}** — `{chunk_id}` — {chars} 字符"
    ),
    "ingestion_traces.no_chunk_text": "未记录 chunk 文本，请重新运行摄取以生成新 Trace。",
    "ingestion_traces.transform_refined": "精炼（LLM / 规则）",
    "ingestion_traces.transform_enriched": "增强（LLM / 规则）",
    "ingestion_traces.transform_captioned": "图片描述",
    "ingestion_traces.per_chunk_transform": "**逐 chunk 转换结果**",
    "ingestion_traces.transform_chunk_header": "🔄 **Chunk #{index}** — `{chunk_id}` — {badges}",
    "ingestion_traces.enriched_metadata": "**增强后的元数据**",
    "ingestion_traces.title_label": "**标题：** {title}",
    "ingestion_traces.no_title": "_无标题_",
    "ingestion_traces.tags_label": "**标签：** {tags}",
    "ingestion_traces.no_tags": "_无标签_",
    "ingestion_traces.summary_label": "**摘要：** {summary}",
    "ingestion_traces.text_comparison": "**文本对比**",
    "ingestion_traces.before_refinement": "*精炼前：*",
    "ingestion_traces.after_refinement": "*精炼 + 增强后：*",
    "ingestion_traces.empty_text": "（空）",
    "ingestion_traces.no_transform_data": (
        "未记录逐 chunk 转换数据，请重新运行摄取以生成新 Trace。"
    ),
    "ingestion_traces.embed_metric_dense": "Dense 向量数",
    "ingestion_traces.embed_metric_dimension": "维度",
    "ingestion_traces.embed_metric_sparse": "Sparse 文档数",
    "ingestion_traces.no_encoding_data": "未记录 chunk 编码数据。",
    "ingestion_traces.dense_tab": "🟦 Dense 编码",
    "ingestion_traces.sparse_tab": "🟨 Sparse 编码（BM25）",
    "ingestion_traces.dense_explain": (
        "每个 chunk → 经 embedding 模型转换为 **float 向量**（如 `text-embedding-ada-002`）"
    ),
    "ingestion_traces.sparse_explain": "每个 chunk → **词频统计**，用于 BM25 索引",
    "ingestion_traces.col_index": "#",
    "ingestion_traces.col_est_tokens": "估算 Token 数",
    "ingestion_traces.col_dense_dim": "Dense 维度",
    "ingestion_traces.col_doc_length": "文档长度（词项数）",
    "ingestion_traces.col_unique_terms": "唯一词项数",
    "ingestion_traces.top_terms_title": "🔤 Chunk {index} — 高频词项",
    "ingestion_traces.col_term": "词项",
    "ingestion_traces.col_freq": "频次",
    "ingestion_traces.upsert_metric_sparse": "Sparse（BM25）",
    "ingestion_traces.dense_store_title": "🟦 Dense 向量库（ChromaDB）",
    "ingestion_traces.sparse_store_title": "🟨 Sparse 索引（BM25）",
    "ingestion_traces.image_store_title": "🖼️ 图片存储（{count} 张）",
    "ingestion_traces.col_vectors": "向量数",
    "ingestion_traces.col_documents": "文档数",
    "ingestion_traces.col_image_id": "图片 ID",
    "ingestion_traces.col_page": "页码",
    "ingestion_traces.col_file": "文件",
    "ingestion_traces.col_doc_hash": "文档哈希",
    "ingestion_traces.chunk_mapping_title": "🔗 Chunk → 向量映射（{count} 条）",
    "ingestion_traces.col_vector_id": "向量 ID",
    "ingestion_traces.col_store": "存储",
    "ingestion_traces.vector_ids": "向量 ID",
    # ── Query Traces ──────────────────────────────────────────
    "query_traces.header": "🔎 查询 Trace",
    "query_traces.none": "暂无查询 Trace 记录，请先运行一次查询！",
    "query_traces.keyword_label": "按查询关键词搜索",
    "query_traces.history": "📋 查询历史（{count}）",
    "query_traces.expander_title": '🔍 "{query}"  ·  {elapsed}  ·  {started}',
    "query_traces.query_section": "#### 💬 查询",
    "query_traces.source_label": "**来源：** {emoji} `{source}`",
    "query_traces.top_k_label": "**Top-K：** `{top_k}`",
    "query_traces.collection_label": "**Collection：** `{collection}`",
    "query_traces.metric_dense_hits": "Dense 命中",
    "query_traces.metric_sparse_hits": "Sparse 命中",
    "query_traces.metric_fused": "融合后",
    "query_traces.metric_after_rerank": "重排后",
    "query_traces.metric_total_time": "总耗时",
    "query_traces.stage_timings": "#### ⏱️ 阶段耗时",
    "query_traces.stage_details": "#### 🔍 阶段详情",
    "query_traces.no_stage_details": "无阶段详情可显示。",
    "query_traces.tab_query_processing": "🔤 查询处理",
    "query_traces.tab_dense": "🟦 Dense 检索",
    "query_traces.tab_sparse": "🟨 Sparse 检索",
    "query_traces.tab_fusion": "🟩 融合（RRF）",
    "query_traces.tab_rerank": "🟪 重排序",
    "query_traces.diag_dense_failed": "**Dense 检索失败：** {error}",
    "query_traces.diag_dense_empty": (
        "Dense 检索返回 **0 条结果**。请检查该 collection 是否已索引数据。"
    ),
    "query_traces.diag_sparse_failed": "**Sparse 检索失败：** {error}",
    "query_traces.diag_sparse_empty": (
        "Sparse（BM25）检索返回 **0 条结果**。"
        "BM25 索引可能为空，或该 collection 尚未建立索引。"
    ),
    "query_traces.diag_fusion_missing_both": (
        "两路检索都有结果，但未记录 Fusion 阶段。"
    ),
    "query_traces.diag_fusion_skipped": (
        "**已跳过 Fusion（RRF）：** 只有 {source} 检索返回了结果。"
        "融合需要 Dense 与 Sparse 两路结果才能合并。"
    ),
    "query_traces.diag_rerank_skipped": (
        "**已跳过重排：** reranker 未启用或未配置。"
        "在 settings.yaml 中启用 `reranker` 即可应用 LLM 重排。"
    ),
    "query_traces.diag_no_results": (
        "**未找到任何结果。** collection 可能为空，或查询与已索引内容不匹配。"
        "请先摄取数据。"
    ),
    "query_traces.ragas_section": "#### 📏 Ragas 评估",
    "query_traces.answer_label": "✏️ 生成的回答",
    "query_traces.answer_placeholder": "请粘贴系统生成的回答，或 LLM 的实际输出…",
    "query_traces.answer_help": (
        "Ragas 使用 LLM-as-Judge 评估回答质量。"
        "faithfulness 衡量回答是否忠于检索到的上下文；"
        "answer_relevancy 衡量回答与问题的相关性。"
        "不填写回答，评估结果没有意义。"
    ),
    "query_traces.ragas_button": "📏 Ragas 评估",
    "query_traces.ragas_button_help": "重新运行该查询并用 Ragas 打分（LLM-as-Judge）",
    "query_traces.ragas_answer_required": "⚠️ 请先在上方输入回答内容，再运行 Ragas 评估。",
    "query_traces.running_ragas": "正在运行 Ragas 评估…",
    "query_traces.eval_failed": "❌ 评估失败：{error}",
    "query_traces.no_metrics": "未返回任何指标。",
    "query_traces.ragas_scores": "**📏 Ragas 评分**",
    "query_traces.no_chunks_retrieved": "未检索到 chunk —— 数据是否已索引？",
    "query_traces.ragas_not_installed": "Ragas 未安装：{error}",
    "query_traces.original_query": "**原始查询**",
    "query_traces.extracted_keywords": "**提取的关键词**",
    "query_traces.no_keywords": "未提取到关键词。",
    "query_traces.metric_keywords": "关键词数",
    "query_traces.metric_results": "结果数",
    "query_traces.top_k_requested": "**请求的 Top-K：** `{top_k}`",
    "query_traces.no_results_returned": "{label} 检索未返回结果。",
    "query_traces.metric_input_lists": "输入列表数",
    "query_traces.metric_fused_results": "融合结果数",
    "query_traces.top_k_short": "**Top-K：** `{top_k}`",
    "query_traces.no_fusion": "无融合结果。",
    "query_traces.metric_input": "输入数",
    "query_traces.metric_output": "输出数",
    "query_traces.no_reranked": "无重排结果。",
    "query_traces.chunk_header": "{bar} **#{index}** — 分数：`{score}`",
    "query_traces.chunk_id_caption": "Chunk ID：`{chunk_id}`",
    "query_traces.source_caption": "来源：`{source}`",
    "query_traces.no_text": "_无文本_",
    # ── Evaluation Panel ──────────────────────────────────────
    "evaluation_panel.header": "📏 评估面板",
    "evaluation_panel.intro": (
        "基于 **golden 测试集** 运行评估，衡量检索与生成质量。"
        "结果包含逐查询明细与聚合指标。"
    ),
    "evaluation_panel.configuration": "⚙️ 配置",
    "evaluation_panel.backend_label": "评估器后端",
    "evaluation_panel.backend_help": "选择要使用的评估器后端。",
    "evaluation_panel.custom_hint": (
        "ℹ️ **Custom Evaluator** 目前只是预留接口：需要在 golden 测试集中提供 "
        "`expected_chunk_ids` 作为 ground truth，才能计算 hit_rate / MRR。"
        "目前建议使用 **ragas** 后端。"
    ),
    "evaluation_panel.top_k_label": "Top-K",
    "evaluation_panel.top_k_help": "每个查询检索的 chunk 数量。",
    "evaluation_panel.collection_label": "Collection（可选）",
    "evaluation_panel.collection_help": "将检索限制在指定 collection 内。",
    "evaluation_panel.golden_label": "Golden 测试集路径",
    "evaluation_panel.golden_help": "golden_test_set.json 文件路径。",
    "evaluation_panel.golden_missing": (
        "⚠️ **未找到 golden 测试集：** `{path}`。"
        "请创建一个包含测试查询与预期结果的 JSON 文件，"
        "格式参考 `tests/fixtures/golden_test_set.json`。"
    ),
    "evaluation_panel.answers_section": "✏️ 填写回答",
    "evaluation_panel.answers_caption": (
        "**RAGAS 需要 Query + Context + Answer 三要素。** "
        "Trace 里只有查询和检索到的上下文，"
        "请为每个测试用例填写实际的系统回答，"
        "才能得到有意义的 faithfulness 与 answer_relevancy 评分。"
    ),
    "evaluation_panel.answer_placeholder": "请输入该问题对应的系统回答…",
    "evaluation_panel.answer_help": (
        "Query: {query}\n\n"
        "填写 LLM 生成的回答或期望的回答文本。"
        "Ragas 会据此评估 faithfulness（忠实度）与 answer_relevancy（相关性）。"
    ),
    "evaluation_panel.filled_warning": (
        "⚠️ 已填写 {filled}/{total} 个回答。"
        "未填写的用例会用检索片段拼接作为回答（评估结果可能不准确）。"
    ),
    "evaluation_panel.filled_ok": "✅ 所有 {total} 个回答已填写。",
    "evaluation_panel.load_cases_failed": "无法加载测试用例预览：{exc}",
    "evaluation_panel.run_button": "▶️  运行评估",
    "evaluation_panel.loading": "正在加载评估器并运行评估…",
    "evaluation_panel.failed": "❌ 评估失败：{exc}",
    "evaluation_panel.complete": "✅ 评估完成！",
    "evaluation_panel.aggregate": "📊 聚合指标",
    "evaluation_panel.no_aggregate": "暂无聚合指标。",
    "evaluation_panel.aggregate_caption": (
        "评估器：**{evaluator}** · 查询数：**{queries}** · 总耗时：**{elapsed:.0f} ms**"
    ),
    "evaluation_panel.per_query": "🔍 逐查询明细",
    "evaluation_panel.no_per_query": "暂无逐查询结果。",
    "evaluation_panel.query_summary_none": "无指标",
    "evaluation_panel.query_expander": "**Q{index}**：{query} — {elapsed:.0f} ms — {metrics}",
    "evaluation_panel.retrieved_chunks": "**检索到的 Chunk**（{count}）：",
    "evaluation_panel.generated_answer": "**生成的回答：**",
    "evaluation_panel.history": "📈 评估历史",
    "evaluation_panel.no_history": (
        "**暂无评估历史。** "
        "在上方配置好评估器后点击「运行评估」，结果会保存在这里，便于多次运行对比。"
    ),
    "evaluation_panel.col_timestamp": "时间",
    "evaluation_panel.col_evaluator": "评估器",
    "evaluation_panel.col_queries": "查询数",
    "evaluation_panel.col_time_ms": "耗时 (ms)",
}


_TRANSLATIONS: dict[str, dict[str, str]] = {"zh": _ZH, "en": _EN}


def get_language() -> str:
    """返回当前语言代码（默认 :data:`DEFAULT_LANGUAGE`）。"""
    return st.session_state.get(_SESSION_KEY, DEFAULT_LANGUAGE)


def tr(key: str, **kwargs: Any) -> str:
    """按当前语言取文案。

    参数:
        key: 文案键，如 ``"overview.header"``。
        **kwargs: 文案模板中的占位符取值。仅在传入时执行 ``str.format``。

    返回:
        当前语言的文案；键缺失时回退到英文，再回退到键本身。
    """
    language = get_language()
    text = _TRANSLATIONS.get(language, {}).get(key)
    if text is None:
        text = _EN.get(key, key)
    return text.format(**kwargs) if kwargs else text


def init_language() -> None:
    """确保 ``st.session_state`` 中存在语言设置（幂等）。"""
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = DEFAULT_LANGUAGE


def language_selector() -> None:
    """在侧边栏渲染语言切换器。"""
    init_language()
    st.sidebar.selectbox(
        tr("common.language"),
        options=list(LANGUAGE_NAMES),
        format_func=lambda code: LANGUAGE_NAMES[code],
        key=_SESSION_KEY,
    )
