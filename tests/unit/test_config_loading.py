"""配置加载与校验的测试。"""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from src.core.settings import SettingsError, load_settings


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


_VALID_CONFIG = """
    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
      max_tokens: 1024
    embedding:
      provider: openai
      model: text-embedding-3-small
      dimensions: 1536
    vector_store:
      provider: chroma
      persist_directory: ./data/db/chroma
      collection_name: knowledge_hub
    retrieval:
      dense_top_k: 20
      sparse_top_k: 20
      fusion_top_k: 10
      rrf_k: 60
    rerank:
      enabled: false
      provider: none
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k: 5
    evaluation:
      enabled: false
      provider: custom
      metrics:
        - hit_rate
        - mrr
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    ingestion:
      chunk_size: 1000
      chunk_overlap: 200
      splitter: recursive
      batch_size: 100
    """


def test_load_settings_success(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, _VALID_CONFIG)

    settings = load_settings(settings_path)

    assert settings.llm.provider == "openai"
    assert settings.embedding.dimensions == 1536
    assert settings.vector_store.collection_name == "knowledge_hub"
    assert settings.retrieval.rrf_k == 60
    # 未显式配置时，检索开关默认全开（兼容旧配置文件）
    assert settings.retrieval.enable_dense is True
    assert settings.retrieval.enable_sparse is True
    assert settings.retrieval.parallel_retrieval is True
    assert settings.retrieval.metadata_filter_post is True
    assert settings.rerank.provider == "none"
    assert settings.evaluation.metrics == ["hit_rate", "mrr"]
    assert settings.observability.log_level == "INFO"
    assert settings.ingestion is not None


def test_retrieval_switches_read_from_config(tmp_path: Path) -> None:
    config = _VALID_CONFIG.replace(
        "rrf_k: 60",
        "rrf_k: 60\n"
        "      enable_dense: false\n"
        "      metadata_filter_post: false\n"
        "      parallel_retrieval: false",
    )
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    settings = load_settings(settings_path)

    assert settings.retrieval.enable_dense is False
    assert settings.retrieval.metadata_filter_post is False
    assert settings.retrieval.parallel_retrieval is False
    assert settings.retrieval.enable_sparse is True  # 未写 -> 默认 True


def test_missing_required_field_raises_error(tmp_path: Path) -> None:
    config = """
    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
      max_tokens: 1024
    embedding:
      model: text-embedding-3-small
      dimensions: 1536
    vector_store:
      provider: chroma
      persist_directory: ./data/db/chroma
      collection_name: knowledge_hub
    retrieval:
      dense_top_k: 20
      sparse_top_k: 20
      fusion_top_k: 10
      rrf_k: 60
    rerank:
      enabled: false
      provider: none
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k: 5
    evaluation:
      enabled: false
      provider: custom
      metrics:
        - hit_rate
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    with pytest.raises(SettingsError, match="embedding.provider"):
        load_settings(settings_path)