"""F5 —— Pipeline on_progress 回调的测试。

验证 IngestionPipeline.run() 在每个 pipeline 阶段
以 (stage_name, current, total) 触发可选的 on_progress 回调。
"""

from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

from src.core.trace.trace_context import TraceContext
from src.core.types import Document, Chunk
from src.ingestion.pipeline import IngestionPipeline


# 辅助函数
# ─────────────────────────────────────────────────────────────────────


def _make_fake_pipeline() -> object:
    """构建无需真实配置的伪 IngestionPipeline。"""

    class FP:
        collection = "test"
        force = False

    fp = FP()

    # 阶段 1：完整性校验
    fp.integrity_checker = MagicMock()
    fp.integrity_checker.compute_sha256.return_value = "hash123"
    fp.integrity_checker.should_skip.return_value = False

    # 阶段 2：加载器
    fp.loader = MagicMock()
    fp.loader.load.return_value = Document(
        id="doc1", text="Hello world. " * 50, metadata={"source_path": "test.pdf", "images": []}
    )

    # 阶段 3：分块器
    chunks = [
        Chunk(id=f"c{i}", text=f"Chunk {i} text. " * 5, metadata={"source_path": "test.pdf"})
        for i in range(3)
    ]
    fp.chunker = MagicMock()
    fp.chunker.split_document.return_value = chunks

    # 阶段 4：转换
    fp.chunk_refiner = MagicMock()
    fp.chunk_refiner.transform.return_value = chunks
    fp.metadata_enricher = MagicMock()
    fp.metadata_enricher.transform.return_value = chunks
    fp.image_captioner = MagicMock()
    fp.image_captioner.transform.return_value = chunks

    # 阶段 5：编码
    batch_result = MagicMock()
    batch_result.dense_vectors = [[0.1, 0.2]] * 3
    batch_result.sparse_stats = [{"doc_id": f"c{i}"} for i in range(3)]
    fp.batch_processor = MagicMock()
    fp.batch_processor.process.return_value = batch_result

    # 阶段 6：存储
    fp.vector_upserter = MagicMock()
    fp.vector_upserter.upsert.return_value = ["v0", "v1", "v2"]
    fp.bm25_indexer = MagicMock()
    fp.image_storage = MagicMock()

    return fp


def _collect_progress(fp) -> List[Tuple[str, int, int]]:
    """以回调运行 pipeline 并返回收集到的调用。"""
    calls: List[Tuple[str, int, int]] = []

    def on_progress(stage: str, current: int, total: int) -> None:
        calls.append((stage, current, total))

    IngestionPipeline.run(fp, "test.pdf", on_progress=on_progress)
    return calls


# 测试
# ────────────────────────────────────────────────────────────────────


class TestPipelineProgressCallback:
    """校验 on_progress 被正确调用。"""

    def test_callback_called_for_all_stages(self) -> None:
        fp = _make_fake_pipeline()
        calls = _collect_progress(fp)
        stage_names = [c[0] for c in calls]
        assert "integrity" in stage_names
        assert "load" in stage_names
        assert "split" in stage_names
        assert "transform" in stage_names
        assert "embed" in stage_names
        assert "upsert" in stage_names

    def test_total_is_six(self) -> None:
        fp = _make_fake_pipeline()
        calls = _collect_progress(fp)
        for _, _, total in calls:
            assert total == 6

    def test_current_is_monotonically_increasing(self) -> None:
        fp = _make_fake_pipeline()
        calls = _collect_progress(fp)
        currents = [c[1] for c in calls]
        assert currents == sorted(currents)
        assert currents == list(range(1, 7))

    def test_no_callback_no_crash(self) -> None:
        """on_progress=None 不应破坏任何逻辑。"""
        fp = _make_fake_pipeline()
        result = IngestionPipeline.run(fp, "test.pdf", on_progress=None)
        assert result.success

    def test_callback_with_trace(self) -> None:
        """on_progress 与 trace 可同时工作。"""
        fp = _make_fake_pipeline()
        calls: List[Tuple[str, int, int]] = []
        trace = TraceContext(trace_type="ingestion")

        def on_progress(stage: str, current: int, total: int) -> None:
            calls.append((stage, current, total))

        IngestionPipeline.run(fp, "test.pdf", trace=trace, on_progress=on_progress)
        assert len(calls) == 6
        assert len(trace.stages) >= 5  # Trace 从 F4 开始记录

    def test_ordering(self) -> None:
        fp = _make_fake_pipeline()
        calls = _collect_progress(fp)
        stage_names = [c[0] for c in calls]
        expected_order = ["integrity", "load", "split", "transform", "embed", "upsert"]
        assert stage_names == expected_order
