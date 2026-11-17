"""Ollama Embedding provider 实现的单元测试。

本测试套件通过 mock 的 HTTP 响应验证 Ollama Embedding 实现，
以确保测试可靠、快速且可离线运行。
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.libs.embedding.ollama_embedding import OllamaEmbedding, OllamaEmbeddingError
from src.libs.embedding.embedding_factory import EmbeddingFactory


# =============================================================================
# 测试夹具
# =============================================================================

@pytest.fixture
def mock_settings_ollama() -> Any:
    """创建用于 Ollama embedding 的 mock 配置。"""
    settings = Mock()
    settings.embedding = Mock()
    settings.embedding.provider = "ollama"
    settings.embedding.model = "nomic-embed-text"
    settings.embedding.dimensions = 768
    return settings


@pytest.fixture
def mock_ollama_response() -> dict[str, Any]:
    """创建 mock 的 Ollama embeddings 响应。"""
    return {
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]  # 为测试截断
    }


# =============================================================================
# Ollama Embedding 测试
# =============================================================================

class TestOllamaEmbedding:
    """OllamaEmbedding 实现的测试套件。"""

    def test_initialization_default(self, mock_settings_ollama: Any) -> None:
        """测试使用默认 base_url 成功初始化。"""
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        assert embedding.model == "nomic-embed-text"
        assert embedding.dimension == 768
        assert embedding.base_url == OllamaEmbedding.DEFAULT_BASE_URL
        assert embedding.timeout == OllamaEmbedding.DEFAULT_TIMEOUT
    
    def test_initialization_with_custom_base_url(self, mock_settings_ollama: Any) -> None:
        """测试使用自定义 base URL 初始化。"""
        custom_url = "http://custom-ollama:11434"
        embedding = OllamaEmbedding(mock_settings_ollama, base_url=custom_url)
        
        assert embedding.base_url == custom_url
    
    def test_initialization_with_env_var(
        self, mock_settings_ollama: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """测试使用环境变量中的 base URL 初始化。"""
        env_url = "http://env-ollama:11434"
        monkeypatch.setenv("OLLAMA_BASE_URL", env_url)
        
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        assert embedding.base_url == env_url
    
    def test_initialization_with_custom_timeout(self, mock_settings_ollama: Any) -> None:
        """测试使用自定义超时初始化。"""
        custom_timeout = 60.0
        embedding = OllamaEmbedding(mock_settings_ollama, timeout=custom_timeout)
        
        assert embedding.timeout == custom_timeout
    
    def test_initialization_without_dimensions_setting(self, mock_settings_ollama: Any) -> None:
        """测试配置中无 dimensions 时初始化（使用默认值）。"""
        delattr(mock_settings_ollama.embedding, 'dimensions')
        
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        assert embedding.dimension == OllamaEmbedding.DEFAULT_DIMENSION
    
    @patch('httpx.Client')
    def test_embed_single_text(
        self, 
        mock_client_class: Mock,
        mock_settings_ollama: Any,
        mock_ollama_response: dict[str, Any],
    ) -> None:
        """测试嵌入单个文本。"""
        # 设置 mock HTTP 响应
        mock_response = Mock()
        mock_response.json.return_value = mock_ollama_response
        mock_response.raise_for_status.return_value = None

        # 设置 mock 客户端
        mock_client_class.return_value.__enter__.return_value.post.return_value = mock_response

        # 执行
        embedding = OllamaEmbedding(mock_settings_ollama)
        result = embedding.embed(["hello world"])

        # 断言
        assert len(result) == 1
        assert result[0] == mock_ollama_response["embedding"]

        # 校验 API 调用
        mock_client_class.return_value.__enter__.return_value.post.assert_called_once()
        call_args = mock_client_class.return_value.__enter__.return_value.post.call_args
        assert call_args[0][0] == f"{embedding.base_url}/api/embeddings"
        assert call_args[1]["json"]["model"] == "nomic-embed-text"
        assert call_args[1]["json"]["prompt"] == "hello world"
    
    @patch('httpx.Client')
    def test_embed_multiple_texts(
        self, 
        mock_client_class: Mock,
        mock_settings_ollama: Any,
    ) -> None:
        """测试嵌入多个文本（批量处理）。"""
        # 设置 mock HTTP 客户端，为每个文本返回不同的 embedding
        def create_response(url, json) -> Mock:
            response = Mock()
            # 根据 prompt 返回不同 embedding
            if "hello" in json["prompt"]:
                response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
            else:
                response.json.return_value = {"embedding": [0.4, 0.5, 0.6]}
            response.raise_for_status.return_value = None
            return response

        mock_client_class.return_value.__enter__.return_value.post.side_effect = create_response

        # 执行
        embedding = OllamaEmbedding(mock_settings_ollama)
        result = embedding.embed(["hello world", "test"])

        # 断言
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

        # 校验 API 被调用两次（每个文本一次）
        assert mock_client_class.return_value.__enter__.return_value.post.call_count == 2

    def test_embed_empty_list(self, mock_settings_ollama: Any) -> None:
        """测试嵌入空列表抛出 ValueError。"""
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            embedding.embed([])
    
    def test_embed_with_empty_string(self, mock_settings_ollama: Any) -> None:
        """测试嵌入空字符串抛出 ValueError。"""
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            embedding.embed([""])
    
    def test_embed_with_non_string(self, mock_settings_ollama: Any) -> None:
        """测试嵌入非字符串抛出 ValueError。"""
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        with pytest.raises(ValueError, match="not a string"):
            embedding.embed([123])  # type: ignore
    
    @patch('httpx.Client')
    def test_embed_http_status_error(
        self,
        mock_client_class: Mock,
        mock_settings_ollama: Any,
    ) -> None:
        """测试 HTTP 状态错误（4xx、5xx）的处理。"""
        # 设置 mock 以抛出 HTTPStatusError
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = __import__('httpx').HTTPStatusError(
            "Not Found", request=Mock(), response=mock_response
        )

        mock_client_class.return_value.__enter__.return_value.post.return_value = mock_response

        # 执行并断言
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        with pytest.raises(OllamaEmbeddingError, match="Ollama API request failed with status 404"):
            embedding.embed(["test"])
    
    @patch('httpx.Client')
    def test_embed_connection_error(
        self,
        mock_client_class: Mock,
        mock_settings_ollama: Any,
    ) -> None:
        """测试连接错误（服务器不可达）的处理。"""
        # 设置 mock 以抛出 ConnectError
        mock_client_class.return_value.__enter__.return_value.post.side_effect = __import__('httpx').ConnectError(
            "Connection refused"
        )

        # 执行并断言
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        with pytest.raises(OllamaEmbeddingError, match="Failed to connect to Ollama server"):
            embedding.embed(["test"])
    
    @patch('httpx.Client')
    def test_embed_timeout_error(
        self,
        mock_client_class: Mock,
        mock_settings_ollama: Any,
    ) -> None:
        """测试超时错误的处理。"""
        # 设置 mock 以抛出 TimeoutException
        mock_client_class.return_value.__enter__.return_value.post.side_effect = __import__('httpx').TimeoutException(
            "Request timed out"
        )

        # 执行并断言
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        with pytest.raises(OllamaEmbeddingError, match="Ollama API request timed out"):
            embedding.embed(["test"])
    
    @patch('httpx.Client')
    def test_embed_missing_embedding_field(
        self,
        mock_client_class: Mock,
        mock_settings_ollama: Any,
    ) -> None:
        """测试响应缺少 'embedding' 字段的处理。"""
        # 设置无效响应格式的 mock
        mock_response = Mock()
        mock_response.json.return_value = {"wrong_field": "data"}
        mock_response.raise_for_status.return_value = None

        mock_client_class.return_value.__enter__.return_value.post.return_value = mock_response

        # 执行并断言
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        with pytest.raises(OllamaEmbeddingError, match="Unexpected response format"):
            embedding.embed(["test"])
    
    @patch('httpx.Client')
    def test_embed_json_parse_error(
        self,
        mock_client_class: Mock,
        mock_settings_ollama: Any,
    ) -> None:
        """测试 JSON 解析错误的处理。"""
        # 设置 JSON 解析失败的响应 mock
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status.return_value = None

        mock_client_class.return_value.__enter__.return_value.post.return_value = mock_response

        # 执行并断言
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        with pytest.raises(OllamaEmbeddingError, match="Failed to parse Ollama API response"):
            embedding.embed(["test"])
    
    @patch.dict('sys.modules', {'httpx': None})
    def test_embed_missing_httpx_dependency(
        self,
        mock_settings_ollama: Any,
    ) -> None:
        """测试缺少 httpx 库时抛出清晰错误。"""
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        with pytest.raises(OllamaEmbeddingError, match="httpx library is required"):
            embedding.embed(["test"])
    
    def test_get_dimension(self, mock_settings_ollama: Any) -> None:
        """测试 get_dimension 方法返回配置的维度。"""
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        assert embedding.get_dimension() == 768
    
    def test_get_dimension_default(self, mock_settings_ollama: Any) -> None:
        """测试未配置时 get_dimension 返回默认值。"""
        delattr(mock_settings_ollama.embedding, 'dimensions')
        
        embedding = OllamaEmbedding(mock_settings_ollama)
        
        assert embedding.get_dimension() == OllamaEmbedding.DEFAULT_DIMENSION


# =============================================================================
# 工厂集成测试
# =============================================================================

class TestOllamaEmbeddingFactoryIntegration:
    """Ollama Embedding 工厂集成的测试套件。"""

    def test_factory_creates_ollama_embedding(self, mock_settings_ollama: Any) -> None:
        """测试工厂正确创建 Ollama embedding 实例。"""
        # 注册 Ollama provider
        EmbeddingFactory.register_provider("ollama", OllamaEmbedding)

        # 通过工厂创建
        embedding = EmbeddingFactory.create(mock_settings_ollama)

        # 断言正确类型
        assert isinstance(embedding, OllamaEmbedding)
        assert embedding.model == "nomic-embed-text"

    def test_factory_with_override_kwargs(self, mock_settings_ollama: Any) -> None:
        """测试带参数覆盖的工厂。"""
        EmbeddingFactory.register_provider("ollama", OllamaEmbedding)

        # 使用覆盖参数创建
        embedding = EmbeddingFactory.create(
            mock_settings_ollama,
            base_url="http://override:11434",
            timeout=30.0,
        )
        
        assert isinstance(embedding, OllamaEmbedding)
        assert embedding.base_url == "http://override:11434"
        assert embedding.timeout == 30.0
