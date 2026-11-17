"""LLM provider 实现的冒烟测试。

本模块使用 mock 的 HTTP 响应测试 OpenAI、Azure 和 DeepSeek LLM provider，
以避免真实的 API 调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.libs.llm import (
    AzureLLM,
    AzureLLMError,
    BaseLLM,
    DeepSeekLLM,
    DeepSeekLLMError,
    LLMFactory,
    Message,
    OpenAILLM,
    OpenAILLMError,
)


# -----------------------------------------------------------------------------
# 模块级设置：确保 provider 已注册
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def ensure_providers_registered():
    """确保每个测试前所有 LLM provider 均已注册。

    这是必需的，因为其他测试（例如 test_llm_factory.py）可能
    在其 setup 中清空 provider 注册表。
    """
    # 若 provider 已被清空则重新注册
    if "openai" not in LLMFactory._PROVIDERS:
        LLMFactory.register_provider("openai", OpenAILLM)
    if "azure" not in LLMFactory._PROVIDERS:
        LLMFactory.register_provider("azure", AzureLLM)
    if "deepseek" not in LLMFactory._PROVIDERS:
        LLMFactory.register_provider("deepseek", DeepSeekLLM)
    yield


# -----------------------------------------------------------------------------
# 测试夹具
# -----------------------------------------------------------------------------


@dataclass
class MockLLMSettings:
    """用于 LLM 测试的 mock 配置。"""
    
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class MockSettings:
    """mock 的应用配置。"""
    
    llm: MockLLMSettings = None
    
    def __post_init__(self):
        if self.llm is None:
            self.llm = MockLLMSettings()


def make_mock_response(
    content: str = "Hello! How can I help you?",
    model: str = "gpt-4o-mini",
    status_code: int = 200,
) -> MagicMock:
    """创建 mock 的 HTTP 响应。"""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1704067200,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }
    return response


def make_error_response(
    status_code: int = 400,
    error_message: str = "Invalid request",
) -> MagicMock:
    """创建 mock 的错误 HTTP 响应。"""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {
        "error": {
            "message": error_message,
            "type": "invalid_request_error",
        }
    }
    response.text = f"Error: {error_message}"
    return response


# -----------------------------------------------------------------------------
# 工厂注册测试
# -----------------------------------------------------------------------------


class TestLLMFactoryRegistration:
    """LLM 工厂 provider 注册的测试。"""

    def test_openai_registered(self):
        """OpenAI provider 应已注册。"""
        assert "openai" in LLMFactory.list_providers()
    
    def test_azure_registered(self):
        """Azure provider 应已注册。"""
        assert "azure" in LLMFactory.list_providers()
    
    def test_deepseek_registered(self):
        """DeepSeek provider 应已注册。"""
        assert "deepseek" in LLMFactory.list_providers()
    
    def test_factory_creates_openai(self):
        """provider=openai 时工厂应创建 OpenAI 实例。"""
        settings = MockSettings(llm=MockLLMSettings(provider="openai"))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = LLMFactory.create(settings)
            assert isinstance(llm, OpenAILLM)
    
    def test_factory_creates_azure(self):
        """provider=azure 时工厂应创建 Azure 实例。"""
        settings = MockSettings(llm=MockLLMSettings(provider="azure"))
        env_vars = {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env_vars):
            llm = LLMFactory.create(settings)
            assert isinstance(llm, AzureLLM)
    
    def test_factory_creates_deepseek(self):
        """provider=deepseek 时工厂应创建 DeepSeek 实例。"""
        settings = MockSettings(llm=MockLLMSettings(provider="deepseek"))
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            llm = LLMFactory.create(settings)
            assert isinstance(llm, DeepSeekLLM)
    
    def test_factory_unknown_provider_error(self):
        """未知 provider 时工厂应抛出错误。"""
        settings = MockSettings(llm=MockLLMSettings(provider="unknown"))
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMFactory.create(settings)


# -----------------------------------------------------------------------------
# OpenAI LLM 测试
# -----------------------------------------------------------------------------


class TestOpenAILLM:
    """OpenAI LLM 实现的测试。"""

    def test_init_with_api_key(self):
        """应使用传入的 API key 初始化。"""
        settings = MockSettings()
        llm = OpenAILLM(settings, api_key="test-key")
        assert llm.api_key == "test-key"
        assert llm.model == "gpt-4o-mini"
    
    def test_init_with_env_var(self):
        """应使用环境变量中的 API key 初始化。"""
        settings = MockSettings()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            llm = OpenAILLM(settings)
            assert llm.api_key == "env-key"
    
    def test_init_missing_api_key(self):
        """API key 缺失时应抛出错误。"""
        settings = MockSettings()
        with patch.dict("os.environ", {}, clear=True):
            # 确保环境中不存在 OPENAI_API_KEY
            import os
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            with pytest.raises(ValueError, match="API key not provided"):
                OpenAILLM(settings)
    
    def test_custom_base_url(self):
        """提供时应使用自定义 base URL。"""
        settings = MockSettings()
        llm = OpenAILLM(settings, api_key="test-key", base_url="https://custom.api.com")
        assert llm.base_url == "https://custom.api.com"
    
    def test_chat_success(self):
        """API 调用成功时应返回 ChatResponse。"""
        settings = MockSettings()
        llm = OpenAILLM(settings, api_key="test-key")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_mock_response("Test response", "gpt-4o-mini")
            )
            
            response = llm.chat([Message(role="user", content="Hello")])
            
            assert response.content == "Test response"
            assert response.model == "gpt-4o-mini"
            assert response.usage["total_tokens"] == 30
    
    def test_chat_empty_messages_error(self):
        """空消息列表应抛出 ValueError。"""
        settings = MockSettings()
        llm = OpenAILLM(settings, api_key="test-key")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            llm.chat([])
    
    def test_chat_invalid_role_error(self):
        """无效消息角色应抛出 ValueError。"""
        settings = MockSettings()
        llm = OpenAILLM(settings, api_key="test-key")
        
        with pytest.raises(ValueError, match="invalid role"):
            llm.chat([Message(role="invalid", content="Hello")])
    
    def test_chat_api_error(self):
        """API 出错时应抛出 OpenAILLMError。"""
        settings = MockSettings()
        llm = OpenAILLM(settings, api_key="test-key")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_error_response(400, "Bad request")
            )
            
            with pytest.raises(OpenAILLMError, match="API error"):
                llm.chat([Message(role="user", content="Hello")])


# -----------------------------------------------------------------------------
# Azure LLM 测试
# -----------------------------------------------------------------------------


class TestAzureLLM:
    """Azure OpenAI LLM 实现的测试。"""

    def test_init_with_credentials(self):
        """应使用传入的凭证初始化。"""
        settings = MockSettings()
        llm = AzureLLM(
            settings,
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )
        assert llm.api_key == "test-key"
        assert llm.endpoint == "https://test.openai.azure.com"
    
    def test_init_with_env_vars(self):
        """应使用环境变量中的凭证初始化。"""
        settings = MockSettings()
        env_vars = {
            "AZURE_OPENAI_API_KEY": "env-key",
            "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com",
        }
        with patch.dict("os.environ", env_vars):
            llm = AzureLLM(settings)
            assert llm.api_key == "env-key"
            assert llm.endpoint == "https://env.openai.azure.com"
    
    def test_init_missing_api_key(self):
        """API key 缺失时应抛出错误。"""
        settings = MockSettings()
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.com"}):
            with pytest.raises(ValueError, match="API key not provided"):
                AzureLLM(settings)
    
    def test_init_missing_endpoint(self):
        """endpoint 缺失时应抛出错误。"""
        settings = MockSettings()
        with patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="endpoint not provided"):
                AzureLLM(settings)
    
    def test_chat_success(self):
        """API 调用成功时应返回 ChatResponse。"""
        settings = MockSettings()
        llm = AzureLLM(
            settings,
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_mock_response("Azure response", "gpt-4o-mini")
            )
            
            response = llm.chat([Message(role="user", content="Hello")])
            
            assert response.content == "Azure response"
            assert response.usage["total_tokens"] == 30
    
    def test_chat_api_error(self):
        """API 出错时应抛出 AzureLLMError。"""
        settings = MockSettings()
        llm = AzureLLM(
            settings,
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_error_response(401, "Unauthorized")
            )
            
            with pytest.raises(AzureLLMError, match="API error"):
                llm.chat([Message(role="user", content="Hello")])


# -----------------------------------------------------------------------------
# DeepSeek LLM 测试
# -----------------------------------------------------------------------------


class TestDeepSeekLLM:
    """DeepSeek LLM 实现的测试。"""

    def test_init_with_api_key(self):
        """应使用传入的 API key 初始化。"""
        settings = MockSettings()
        llm = DeepSeekLLM(settings, api_key="test-key")
        assert llm.api_key == "test-key"
        assert llm.base_url == "https://api.deepseek.com"
    
    def test_init_with_env_var(self):
        """应使用环境变量中的 API key 初始化。"""
        settings = MockSettings()
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "env-key"}):
            llm = DeepSeekLLM(settings)
            assert llm.api_key == "env-key"
    
    def test_init_missing_api_key(self):
        """API key 缺失时应抛出错误。"""
        settings = MockSettings()
        with patch.dict("os.environ", {}, clear=True):
            import os
            if "DEEPSEEK_API_KEY" in os.environ:
                del os.environ["DEEPSEEK_API_KEY"]
            with pytest.raises(ValueError, match="API key not provided"):
                DeepSeekLLM(settings)
    
    def test_custom_base_url(self):
        """提供时应使用自定义 base URL。"""
        settings = MockSettings()
        llm = DeepSeekLLM(settings, api_key="test-key", base_url="https://custom.deepseek.com")
        assert llm.base_url == "https://custom.deepseek.com"
    
    def test_chat_success(self):
        """API 调用成功时应返回 ChatResponse。"""
        settings = MockSettings()
        llm = DeepSeekLLM(settings, api_key="test-key")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_mock_response("DeepSeek response", "deepseek-chat")
            )
            
            response = llm.chat([Message(role="user", content="Hello")])
            
            assert response.content == "DeepSeek response"
            assert response.model == "deepseek-chat"
    
    def test_chat_api_error(self):
        """API 出错时应抛出 DeepSeekLLMError。"""
        settings = MockSettings()
        llm = DeepSeekLLM(settings, api_key="test-key")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_error_response(500, "Internal server error")
            )
            
            with pytest.raises(DeepSeekLLMError, match="API error"):
                llm.chat([Message(role="user", content="Hello")])


# -----------------------------------------------------------------------------
# 消息校验测试
# -----------------------------------------------------------------------------


class TestMessageValidation:
    """所有 provider 的消息校验测试。"""

    @pytest.mark.parametrize("llm_class,api_key_env", [
        (OpenAILLM, "OPENAI_API_KEY"),
        (DeepSeekLLM, "DEEPSEEK_API_KEY"),
    ])
    def test_empty_content_validation(self, llm_class, api_key_env):
        """应拒绝内容为空的消息。"""
        settings = MockSettings()
        with patch.dict("os.environ", {api_key_env: "test-key"}):
            llm = llm_class(settings)
            with pytest.raises(ValueError, match="empty content"):
                llm.chat([Message(role="user", content="")])
    
    @pytest.mark.parametrize("llm_class,api_key_env", [
        (OpenAILLM, "OPENAI_API_KEY"),
        (DeepSeekLLM, "DEEPSEEK_API_KEY"),
    ])
    def test_valid_roles_accepted(self, llm_class, api_key_env):
        """应接受有效角色：system、user、assistant。"""
        settings = MockSettings()
        with patch.dict("os.environ", {api_key_env: "test-key"}):
            llm = llm_class(settings)
            
            # 这些不应触发校验错误
            messages = [
                Message(role="system", content="You are helpful"),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
            ]
            
            with patch("httpx.Client") as mock_client:
                mock_client.return_value.__enter__.return_value.post.return_value = (
                    make_mock_response()
                )
                # 不应抛出异常
                llm.chat(messages)


# -----------------------------------------------------------------------------
# 集成风格测试（仍然使用 mock）
# -----------------------------------------------------------------------------


class TestLLMIntegration:
    """使用工厂模式的集成风格测试。"""

    def test_factory_to_chat_flow_openai(self):
        """测试 OpenAI 的完整流程：factory -> create -> chat。"""
        settings = MockSettings(llm=MockLLMSettings(provider="openai"))
        
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            llm = LLMFactory.create(settings)
            
            with patch("httpx.Client") as mock_client:
                mock_client.return_value.__enter__.return_value.post.return_value = (
                    make_mock_response("Integration test response")
                )
                
                response = llm.chat([Message(role="user", content="Test")])
                assert response.content == "Integration test response"
    
    def test_factory_to_chat_flow_deepseek(self):
        """测试 DeepSeek 的完整流程：factory -> create -> chat。"""
        settings = MockSettings(llm=MockLLMSettings(provider="deepseek"))
        
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            llm = LLMFactory.create(settings)
            
            with patch("httpx.Client") as mock_client:
                mock_client.return_value.__enter__.return_value.post.return_value = (
                    make_mock_response("DeepSeek integration response")
                )
                
                response = llm.chat([Message(role="user", content="Test")])
                assert response.content == "DeepSeek integration response"
