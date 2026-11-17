"""F4 —— 摄入 pipeline Trace 埋点的集成测试。

校验 IngestionPipeline.run() 会用预期的阶段（load/split/transform/embed/upsert）
和耗时数据填充 TraceContext。

通过 monkey-patch 替换重型外部组件，使测试无需真实的
Azure / ChromaDB / LLM 依赖即可运行。
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.core.trace.trace_context import TraceContext
from src.core.types import Document, Chunk


# ── 伪造的重型组件 ────────────────────────────────────────────

def _fake_document(path: str = "test.pdf") -> Document:
    return Document(
        id="doc_hash_abc123",
        text="Hello world. " * 50,
        metadata={"source_path": path, "images": []},
    )


def _fake_chunks(n: int = 3) -> List[Chunk]:
    return [
        Chunk(
            id=f"chunk_{i}",
            text=f"Chunk text {i}. " * 10,
            metadata={"source_doc_id": "doc_hash_abc123", "source_path": "test.pdf"},
        )
        for i in range(n)
    ]


@dataclass
class FakeBatchResult:
    dense_vectors: List[List[float]] = field(default_factory=lambda: [[0.1, 0.2]] * 3)
    sparse_stats: List[Dict[str, Any]] = field(default_factory=lambda: [{"doc_id": f"chunk_{i}"} for i in range(3)])


class FakePipeline:
    """模仿 IngestionPipeline，但所有重型组件均被替换。"""

    def __init__(self):
        self.collection = "test_collection"
        self.force = False

        # 对每个组件进行 mock
        self.integrity_checker = MagicMock()
        self.integrity_checker.compute_sha256.return_value = "abc123"
        self.integrity_checker.should_skip.return_value = False

        self.loader = MagicMock()
        self.loader.load.return_value = _fake_document()

        self.chunker = MagicMock()
        chunks = _fake_chunks()
        self.chunker.split_document.return_value = chunks

        self.chunk_refiner = MagicMock()
        self.chunk_refiner.transform.return_value = chunks

        self.metadata_enricher = MagicMock()
        self.metadata_enricher.transform.return_value = chunks

        self.image_captioner = MagicMock()
        self.image_captioner.transform.return_value = chunks

        self.batch_processor = MagicMock()
        self.batch_processor.process.return_value = FakeBatchResult()

        self.vector_upserter = MagicMock()
        self.vector_upserter.upsert.return_value = ["vid_0", "vid_1", "vid_2"]

        self.bm25_indexer = MagicMock()
        self.image_storage = MagicMock()


def _run_fake_pipeline(trace: Optional[TraceContext] = None):
    """导入真实的 run() 逻辑，但将其绑定到 FakePipeline。"""
    from src.ingestion.pipeline import IngestionPipeline

    fp = FakePipeline()
    # 借用真实的 `run` 方法，但绑定到伪造实例上
    return IngestionPipeline.run(fp, "test.pdf", trace=trace)


# ── 测试 ────────────────────────────────────────────────────────────


class TestIngestionPipelineTrace:
    """校验 IngestionPipeline.run() 记录 5 个必需阶段。"""

    def test_records_load_stage(self) -> None:
        trace = TraceContext(trace_type="ingestion")
        _run_fake_pipeline(trace)
        stage_names = [s["stage"] for s in trace.stages]
        assert "load" in stage_names

    def test_records_split_stage(self) -> None:
        trace = TraceContext(trace_type="ingestion")
        _run_fake_pipeline(trace)
        stage_names = [s["stage"] for s in trace.stages]
        assert "split" in stage_names

    def test_records_transform_stage(self) -> None:
        trace = TraceContext(trace_type="ingestion")
        _run_fake_pipeline(trace)
        stage_names = [s["stage"] for s in trace.stages]
        assert "transform" in stage_names

    def test_records_embed_stage(self) -> None:
        trace = TraceContext(trace_type="ingestion")
        _run_fake_pipeline(trace)
        stage_names = [s["stage"] for s in trace.stages]
        assert "embed" in stage_names

    def test_records_upsert_stage(self) -> None:
        trace = TraceContext(trace_type="ingestion")
        _run_fake_pipeline(trace)
        stage_names = [s["stage"] for s in trace.stages]
        assert "upsert" in stage_names

    def test_all_stages_have_elapsed_ms(self) -> None:
        trace = TraceContext(trace_type="ingestion")
        _run_fake_pipeline(trace)
        for entry in trace.stages:
            assert "elapsed_ms" in entry, f"stage '{entry['stage']}' missing elapsed_ms"
            assert entry["elapsed_ms"] >= 0

    def test_all_stages_have_method_field(self) -> None:
        trace = TraceContext(trace_type="ingestion")
        _run_fake_pipeline(trace)
        for entry in trace.stages:
            assert "method" in entry["data"], f"stage '{entry['stage']}' missing method"

    def test_trace_type_is_ingestion(self) -> None:
        trace = TraceContext(trace_type="ingestion")
        _run_fake_pipeline(trace)
        assert trace.trace_type == "ingestion"

    def test_to_dict_contains_all_stages(self) -> None:
        trace = TraceContext(trace_type="ingestion")
        _run_fake_pipeline(trace)
        trace.finish()
        d = trace.to_dict()
        assert d["trace_type"] == "ingestion"
        assert len(d["stages"]) >= 5  # load、split、transform、embed、upsert

    def test_no_trace_no_crash(self) -> None:
        """trace 为 None 时 run() 不得抛出异常。"""
        result = _run_fake_pipeline(trace=None)
        assert result.success

    def test_stage_ordering(self) -> None:
        """各阶段应按 pipeline 顺序出现。"""
        trace = TraceContext(trace_type="ingestion")
        _run_fake_pipeline(trace)
        stage_names = [s["stage"] for s in trace.stages]
        expected_order = ["load", "split", "transform", "embed", "upsert"]
        # 所有预期阶段均已出现且顺序正确
        positions = [stage_names.index(s) for s in expected_order]
        assert positions == sorted(positions)
