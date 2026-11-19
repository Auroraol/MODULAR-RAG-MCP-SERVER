"""Evaluation Panel 页面 – 运行评估并查看指标。

布局：
1. 配置区域：选择评估器后端、golden 测试集、top_k
2. 带进度指示的运行按钮
3. 结果区域：聚合指标、逐查询明细表
4. 可选：历史评估结果对比
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.observability.dashboard.i18n import tr

logger = logging.getLogger(__name__)

# golden 测试集的默认位置
DEFAULT_GOLDEN_SET = Path("tests/fixtures/golden_test_set.json")
# 评估结果历史文件
EVAL_HISTORY_PATH = Path("logs/eval_history.jsonl")


def render() -> None:
    """渲染 Evaluation Panel 页面。"""
    st.header(tr("evaluation_panel.header"))
    st.markdown(tr("evaluation_panel.intro"))

    # ── 配置区域 ──────────────────────────────────────────────────
    st.subheader(tr("evaluation_panel.configuration"))

    col1, col2, col3 = st.columns(3)

    with col1:
        backend = st.selectbox(
            tr("evaluation_panel.backend_label"),
            options=["custom", "ragas", "composite"],
            index=0,
            key="eval_backend",
            help=tr("evaluation_panel.backend_help"),
        )

    # 根据所选后端显示 info/warning 提示
    if backend in ("custom", "composite"):
        st.info(tr("evaluation_panel.custom_hint"), icon="🚧")

    with col2:
        top_k = st.number_input(
            tr("evaluation_panel.top_k_label"),
            min_value=1,
            max_value=50,
            value=10,
            key="eval_top_k",
            help=tr("evaluation_panel.top_k_help"),
        )

    with col3:
        collection = st.text_input(
            tr("evaluation_panel.collection_label"),
            value="",
            key="eval_collection",
            help=tr("evaluation_panel.collection_help"),
        )

    # golden 测试集文件选择
    golden_path_str = st.text_input(
        tr("evaluation_panel.golden_label"),
        value=str(DEFAULT_GOLDEN_SET),
        key="eval_golden_path",
        help=tr("evaluation_panel.golden_help"),
    )
    golden_path = Path(golden_path_str)

    # 校验 golden 测试集是否存在
    if not golden_path.exists():
        st.warning(tr("evaluation_panel.golden_missing", path=golden_path))

    # ── 回答输入区域（用于 Ragas） ─────────────────────────────────
    user_answers: Dict[int, str] = {}
    if backend == "ragas" and golden_path.exists():
        st.divider()
        st.subheader(tr("evaluation_panel.answers_section"))
        st.caption(tr("evaluation_panel.answers_caption"))
        try:
            _test_cases = _load_golden_queries(golden_path)
            for tc_idx, tc in enumerate(_test_cases):
                ans_key = f"eval_answer_tc_{tc_idx}"
                default_val = tc.get("reference_answer", "")
                q_preview = tc["query"][:60] + ("…" if len(tc["query"]) > 60 else "")
                user_ans = st.text_area(
                    f"Q{tc_idx + 1}: {q_preview}",
                    value=st.session_state.get(ans_key, default_val),
                    height=80,
                    key=ans_key,
                    placeholder=tr("evaluation_panel.answer_placeholder"),
                    help=tr("evaluation_panel.answer_help", query=tc["query"]),
                )
                if user_ans.strip():
                    user_answers[tc_idx] = user_ans.strip()

            # 显示填写状态
            filled = len(user_answers)
            total = len(_test_cases)
            if filled < total:
                st.warning(
                    tr("evaluation_panel.filled_warning", filled=filled, total=total)
                )
            else:
                st.success(tr("evaluation_panel.filled_ok", total=total))
        except Exception as exc:
            st.warning(tr("evaluation_panel.load_cases_failed", exc=exc))

    # ── 运行评估 ──────────────────────────────────────────────────
    st.divider()

    run_clicked = st.button(
        tr("evaluation_panel.run_button"),
        type="primary",
        key="eval_run_btn",
        disabled=not golden_path.exists(),
    )

    if run_clicked:
        _run_evaluation(
            backend=backend,
            golden_path=golden_path,
            top_k=int(top_k),
            collection=collection.strip() or None,
            user_answers=user_answers if user_answers else None,
        )

    # ── 历史结果 ──────────────────────────────────────────────────
    st.divider()
    _render_history()


def _run_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: Optional[str],
    user_answers: Optional[Dict[int, str]] = None,
) -> None:
    """执行一次评估运行并展示结果。

    尝试加载评估器、运行 golden 测试集，并展示聚合指标与逐查询
    指标。失败时回退为友好的错误提示。
    """
    with st.spinner(tr("evaluation_panel.loading")):
        try:
            report_dict = _execute_evaluation(
                backend=backend,
                golden_path=golden_path,
                top_k=top_k,
                collection=collection,
                user_answers=user_answers,
            )
        except Exception as exc:
            st.error(tr("evaluation_panel.failed", exc=exc))
            logger.exception("Evaluation failed")
            return

    # ── 展示结果 ──────────────────────────────────────────────────
    st.success(tr("evaluation_panel.complete"))

    _render_aggregate_metrics(report_dict)
    _render_query_details(report_dict)

    # 保存到历史记录
    _save_to_history(report_dict)


def _execute_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: Optional[str],
    user_answers: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """运行评估流水线并返回报告 dict。

    该函数延迟导入重量级依赖，以便在页面未使用时保持
    Dashboard 的响应速度。
    """
    from dataclasses import replace as dc_replace

    from src.core.settings import load_settings
    from src.libs.evaluator.evaluator_factory import EvaluatorFactory
    from src.observability.evaluation.eval_runner import EvalRunner

    settings = load_settings()

    # 使用 UI 选择覆盖评估器 provider —— 构造一个新的完整
    # Settings 对象，以便 RagasEvaluator 仍能访问 .llm / .embedding。
    eval_settings = settings.evaluation
    overridden_eval = type(eval_settings)(
        enabled=True,
        provider=backend,
        metrics=eval_settings.metrics if hasattr(eval_settings, "metrics") else [],
    )
    # 仅替换完整 settings 中的 evaluation 子配置
    settings_with_override = dc_replace(settings, evaluation=overridden_eval)

    evaluator = EvaluatorFactory.create(settings_with_override)

    # 尝试创建 HybridSearch（可选 —— 未配置时也可正常运行）
    target_collection = collection or "default"
    hybrid_search = _try_create_hybrid_search(settings, target_collection)

    # 若已启用则创建 reranker
    reranker = None
    try:
        from src.core.query_engine.reranker import create_core_reranker
        reranker = create_core_reranker(settings=settings)
        if not reranker.is_enabled:
            reranker = None
    except Exception as exc:
        logger.warning("Could not create reranker: %s", exc)

    # 构建 answer_override 映射：索引 → 用户提供的回答文本
    # EvalRunner 将使用这些内容，而不是从 chunk 自动生成回答。
    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
        answer_overrides=user_answers,
        reranker=reranker,
    )

    report = runner.run(
        test_set_path=golden_path,
        top_k=top_k,
        collection=collection,
    )

    return report.to_dict()


def _try_create_hybrid_search(settings: Any, collection: str = "default") -> Any:
    """尝试创建 HybridSearch 实例。

    若所需依赖不可用（例如没有已索引的数据），则返回 None。
    """
    try:
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.retriever.dense_retriever import create_dense_retriever
        from src.core.query_engine.retriever.sparse_retriever import create_sparse_retriever
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        vector_store = VectorStoreFactory.create(
            settings, collection_name=collection,
        )
        embedding_client = EmbeddingFactory.create(settings)
        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection

        query_processor = QueryProcessor()
        return create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )
    except Exception as exc:
        logger.warning("Could not create HybridSearch: %s", exc)
        return None


def _render_aggregate_metrics(report: Dict[str, Any]) -> None:
    """以指标卡片形式展示聚合指标。"""
    st.subheader(tr("evaluation_panel.aggregate"))

    agg = report.get("aggregate_metrics", {})

    if not agg:
        st.info(tr("evaluation_panel.no_aggregate"))
        return

    cols = st.columns(min(len(agg), 4))
    for idx, (name, value) in enumerate(sorted(agg.items())):
        with cols[idx % len(cols)]:
            st.metric(
                label=name.replace("_", " ").title(),
                value=f"{value:.4f}",
            )

    st.caption(
        tr(
            "evaluation_panel.aggregate_caption",
            evaluator=report.get("evaluator_name", "—"),
            queries=report.get("query_count", 0),
            elapsed=report.get("total_elapsed_ms", 0),
        )
    )


def _render_query_details(report: Dict[str, Any]) -> None:
    """以可展开表格展示逐查询的评估结果。"""
    st.subheader(tr("evaluation_panel.per_query"))

    query_results = report.get("query_results", [])
    if not query_results:
        st.info(tr("evaluation_panel.no_per_query"))
        return

    for idx, qr in enumerate(query_results):
        query = qr.get("query", "—")
        elapsed = qr.get("elapsed_ms", 0)
        metrics = qr.get("metrics", {})

        # 为 expander 标签构建指标摘要
        metric_summary = " · ".join(
            f"{k}: {v:.3f}" for k, v in sorted(metrics.items())
        )
        if not metric_summary:
            metric_summary = tr("evaluation_panel.query_summary_none")

        with st.expander(
            tr(
                "evaluation_panel.query_expander",
                index=idx + 1,
                query=query[:80],
                elapsed=elapsed,
                metrics=metric_summary,
            ),
            expanded=False,
        ):
            # 指标
            if metrics:
                mcols = st.columns(min(len(metrics), 4))
                for midx, (mname, mval) in enumerate(sorted(metrics.items())):
                    with mcols[midx % len(mcols)]:
                        st.metric(mname, f"{mval:.4f}")

            # 检索到的 chunk
            chunks = qr.get("retrieved_chunk_ids", [])
            if chunks:
                st.markdown(tr("evaluation_panel.retrieved_chunks", count=len(chunks)))
                st.code(", ".join(chunks[:20]), language=None)

            # 生成的回答
            answer = qr.get("generated_answer")
            if answer:
                st.markdown(tr("evaluation_panel.generated_answer"))
                st.text(answer[:500])


def _render_history() -> None:
    """展示历史评估结果以便对比。"""
    st.subheader(tr("evaluation_panel.history"))

    history = _load_history()
    if not history:
        st.info(tr("evaluation_panel.no_history"))
        return

    # 以表格形式展示最近的运行记录
    rows = []
    for entry in history[-10:]:  # 最近 10 次运行
        rows.append(
            {
                tr("evaluation_panel.col_timestamp"): entry.get("timestamp", "—"),
                tr("evaluation_panel.col_evaluator"): entry.get("evaluator_name", "—"),
                tr("evaluation_panel.col_queries"): entry.get("query_count", 0),
                tr("evaluation_panel.col_time_ms"): round(entry.get("total_elapsed_ms", 0)),
                **{
                    k: round(v, 4)
                    for k, v in entry.get("aggregate_metrics", {}).items()
                },
            }
        )

    st.dataframe(rows, use_container_width=True)


def _save_to_history(report: Dict[str, Any]) -> None:
    """将评估报告追加到历史文件。"""
    try:
        EVAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **report,
        }
        with EVAL_HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to save evaluation history: %s", exc)


def _load_history() -> List[Dict[str, Any]]:
    """从 JSONL 文件加载评估历史。"""
    if not EVAL_HISTORY_PATH.exists():
        return []

    entries: List[Dict[str, Any]] = []
    try:
        with EVAL_HISTORY_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        logger.warning("Failed to load evaluation history: %s", exc)

    return entries


def _load_golden_queries(golden_path: Path) -> List[Dict[str, Any]]:
    """从 golden 测试集加载测试用例以供 UI 展示。

    返回 dict 列表，每项至少包含 'query' 键，并可选包含
    'reference_answer' 键。
    """
    with golden_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("test_cases", [])
