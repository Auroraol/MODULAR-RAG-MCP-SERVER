"""FixedLength / Markdown / Semantic / Structure / Propositional 切分器的单元测试。"""

from unittest.mock import MagicMock

import pytest

from src.libs.llm.base_llm import ChatResponse
from src.libs.splitter.fixed_length_splitter import FixedLengthSplitter
from src.libs.splitter.markdown_splitter import MarkdownSplitter
from src.libs.splitter.propositional_splitter import PropositionalSplitter
from src.libs.splitter.semantic_splitter import SemanticSplitter
from src.libs.splitter.structure_splitter import StructureSplitter


def _settings(chunk_size: int = 10, chunk_overlap: int = 2) -> MagicMock:
    settings = MagicMock()
    settings.ingestion.chunk_size = chunk_size
    settings.ingestion.chunk_overlap = chunk_overlap
    return settings


class FakeEmbedding:
    """按句子首字母给出两组正交向量，模拟“主题 A / 主题 B”。"""

    def embed(self, texts, trace=None, **kwargs):
        return [[1.0, 0.0] if t.startswith("A") else [0.0, 1.0] for t in texts]

    def get_dimension(self):
        return 2


class BrokenEmbedding:
    def embed(self, texts, trace=None, **kwargs):
        raise RuntimeError("api down")

    def get_dimension(self):
        return 2


class TestFixedLengthSplitter:
    def test_chunks_respect_size_and_overlap(self):
        splitter = FixedLengthSplitter(_settings(10, 2))
        chunks = splitter.split_text("a" * 25)
        assert [len(c) for c in chunks] == [10, 10, 9]
        # 相邻块重叠 2 个字符
        assert chunks[1].startswith(chunks[0][-2:])

    def test_snaps_to_whitespace(self):
        splitter = FixedLengthSplitter(_settings(13, 0))
        chunks = splitter.split_text("hello world foo bar")
        assert chunks == ["hello world ", "foo bar"]

    def test_invalid_overlap(self):
        with pytest.raises(ValueError, match="must be less than"):
            FixedLengthSplitter(_settings(10, 10))


class TestMarkdownSplitter:
    def test_small_markdown_is_one_chunk(self):
        splitter = MarkdownSplitter(_settings(200, 20))
        text = "# Title\n\n```python\nprint('hi')\n```\n\n## Section\n\nSome text."
        assert splitter.split_text(text) == [text]


class TestSemanticSplitter:
    def test_splits_on_topic_change(self):
        text = " ".join(
            [f"A{i}." for i in range(10)] + [f"B{i}." for i in range(10)] + [f"A{i}." for i in range(10)]
        )
        splitter = SemanticSplitter(_settings(1000, 0), embedding=FakeEmbedding())
        chunks = splitter.split_text(text)
        assert len(chunks) == 3
        assert chunks[1].startswith("B0.")

    def test_short_text_returned_as_is(self):
        splitter = SemanticSplitter(_settings(1000, 0), embedding=FakeEmbedding())
        assert splitter.split_text("A1. A2.") == ["A1. A2."]

    def test_over_size_limit_enforced(self):
        text = " ".join(f"A{i}." for i in range(10))
        splitter = SemanticSplitter(_settings(20, 0), embedding=FakeEmbedding())
        assert all(len(c) <= 20 for c in splitter.split_text(text))

    def test_falls_back_to_recursive_on_embedding_error(self):
        text = " ".join(f"A{i}." for i in range(10))
        splitter = SemanticSplitter(
            _settings(1000, 0), embedding=BrokenEmbedding(), breakpoint_percentile=95
        )
        chunks = splitter.split_text(text)
        assert chunks and all(c.strip() for c in chunks)


class TestStructureSplitter:
    def test_splits_by_heading(self):
        splitter = StructureSplitter(_settings(15, 0))
        chunks = splitter.split_text("# A\n\nAAAA\n\n# B\n\nBBBB")
        assert len(chunks) == 2
        assert chunks[0].startswith("# A")
        assert chunks[1].startswith("# B")

    def test_oversized_section_keeps_heading_path(self):
        splitter = StructureSplitter(_settings(50, 0))
        text = "# Root\n\n" + " ".join(["word"] * 60)
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        assert all(c.startswith("# Root") for c in chunks)

    def test_plain_text_single_chunk(self):
        splitter = StructureSplitter(_settings(1000, 0))
        assert splitter.split_text("plain text") == ["plain text"]


class FakeLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def chat(self, messages, trace=None, **kwargs):
        self.calls.append(messages)
        return ChatResponse(content=self.content, model="fake")


class BrokenLLM:
    def chat(self, messages, trace=None, **kwargs):
        raise RuntimeError("llm down")


class TestPropositionalSplitter:
    def test_extracts_and_packs_propositions(self):
        llm = FakeLLM("- P1\n- P2")
        splitter = PropositionalSplitter(_settings(1000, 0), llm=llm)
        assert splitter.split_text("some text") == ["P1\nP2"]
        assert len(llm.calls) == 1

    def test_packs_within_chunk_size(self):
        llm = FakeLLM("- " + "A" * 6 + "\n- " + "B" * 6)
        splitter = PropositionalSplitter(_settings(10, 0), llm=llm)
        assert splitter.split_text("text") == ["AAAAAA", "BBBBBB"]

    def test_falls_back_to_original_on_llm_error(self):
        splitter = PropositionalSplitter(_settings(1000, 0), llm=BrokenLLM())
        assert splitter.split_text("some text") == ["some text"]