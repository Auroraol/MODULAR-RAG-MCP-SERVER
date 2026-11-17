"""G1 测试 —— Dashboard ConfigService 与页面导入。

涵盖：
- ConfigService.get_component_cards() 返回预期组件
- app.py 与 pages/overview.py 可无错误导入
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.observability.dashboard.services.config_service import (
    ComponentInfo,
    ConfigService,
)


# ── 假配置 ────────────────────────────────────────────────────

def _fake_settings():
    """构建字段值贴近真实的 mock Settings 对象。"""
    s = MagicMock()
    s.llm.provider = "azure_openai"
    s.llm.model = "gpt-4o"
    s.llm.temperature = 0.0
    s.llm.max_tokens = 4096

    s.embedding.provider = "azure_openai"
    s.embedding.model = "text-embedding-ada-002"
    s.embedding.dimensions = 1536

    s.vector_store.provider = "chroma"
    s.vector_store.collection_name = "default"
    s.vector_store.persist_directory = "data/db/chroma"

    s.retrieval.dense_top_k = 20
    s.retrieval.sparse_top_k = 20
    s.retrieval.fusion_top_k = 10

    s.rerank.enabled = True
    s.rerank.provider = "llm"
    s.rerank.model = "gpt-4o"
    s.rerank.top_k = 5

    s.vision_llm.enabled = True
    s.vision_llm.provider = "azure_openai"
    s.vision_llm.model = "gpt-4o"
    s.vision_llm.max_image_size = 2048

    s.ingestion.splitter = "recursive"
    s.ingestion.chunk_size = 1000
    s.ingestion.chunk_overlap = 200
    s.ingestion.batch_size = 100

    return s


# ── 测试 ────────────────────────────────────────────────────────────


class TestConfigService:
    """验证 ConfigService 生成组件卡片。"""

    @patch("src.observability.dashboard.services.config_service.load_settings")
    def test_get_component_cards_returns_list(self, mock_load) -> None:
        mock_load.return_value = _fake_settings()
        svc = ConfigService("config/settings.yaml")
        cards = svc.get_component_cards()
        assert isinstance(cards, list)
        assert len(cards) >= 5  # LLM、Embedding、VectorStore、Retrieval、Reranker

    @patch("src.observability.dashboard.services.config_service.load_settings")
    def test_llm_card(self, mock_load) -> None:
        mock_load.return_value = _fake_settings()
        svc = ConfigService()
        cards = svc.get_component_cards()
        llm = next(c for c in cards if c.name == "LLM")
        assert llm.provider == "azure_openai"
        assert llm.model == "gpt-4o"

    @patch("src.observability.dashboard.services.config_service.load_settings")
    def test_embedding_card(self, mock_load) -> None:
        mock_load.return_value = _fake_settings()
        svc = ConfigService()
        cards = svc.get_component_cards()
        emb = next(c for c in cards if c.name == "Embedding")
        assert emb.extra["dimensions"] == 1536

    @patch("src.observability.dashboard.services.config_service.load_settings")
    def test_rerank_disabled(self, mock_load) -> None:
        settings = _fake_settings()
        settings.rerank.enabled = False
        mock_load.return_value = settings
        svc = ConfigService()
        cards = svc.get_component_cards()
        reranker = next(c for c in cards if c.name == "Reranker")
        assert reranker.provider == "disabled"

    @patch("src.observability.dashboard.services.config_service.load_settings")
    def test_vision_llm_card_present(self, mock_load) -> None:
        mock_load.return_value = _fake_settings()
        svc = ConfigService()
        cards = svc.get_component_cards()
        vision = [c for c in cards if c.name == "Vision LLM"]
        assert len(vision) == 1

    @patch("src.observability.dashboard.services.config_service.load_settings")
    def test_ingestion_card_present(self, mock_load) -> None:
        mock_load.return_value = _fake_settings()
        svc = ConfigService()
        cards = svc.get_component_cards()
        ingestion = next(c for c in cards if c.name == "Ingestion")
        assert ingestion.extra["chunk_size"] == 1000

    @patch("src.observability.dashboard.services.config_service.load_settings")
    def test_reload_clears_cache(self, mock_load) -> None:
        mock_load.return_value = _fake_settings()
        svc = ConfigService()
        _ = svc.get_component_cards()
        svc.reload()
        assert svc._settings is None


class TestDashboardImports:
    """验证主应用模块可导入。"""

    def test_config_service_importable(self) -> None:
        from src.observability.dashboard.services.config_service import ConfigService
        assert ConfigService is not None

    def test_overview_importable(self) -> None:
        from src.observability.dashboard.pages import overview
        assert hasattr(overview, "render")

    def test_app_file_exists(self) -> None:
        app_path = Path("src/observability/dashboard/app.py")
        assert app_path.exists()

    def test_start_script_exists(self) -> None:
        script_path = Path("scripts/start_dashboard.py")
        assert script_path.exists()
