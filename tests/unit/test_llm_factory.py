"""LLM 工厂的单元测试。

测试工厂模式的实现、provider 路由逻辑，
以及 LLM 抽象层的错误处理。
"""

import pytest

from src.core.settings import load_settings
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from src.libs.llm.llm_factory import LLMFactory


def create_test_config(provider="fake", **overrides):
    """创建完整测试配置的辅助函数。"""
    base_config = {
        "llm": {
            "provider": provider,
            "model": "fake-model",
            "temperature": 0.0,
            "max_tokens": 1000,
        },
        "embedding": {
            "provider": "fake",
            "model": "fake-emb",
            "dimensions": 768,
        },
        "vector_store": {
            "provider": "fake",
            "persist_directory": "./data/db",
            "collection_name": "test",
        },
        "retrieval": {
            "dense_top_k": 10,
            "sparse_top_k": 10,
            "fusion_top_k": 5,
            "rrf_k": 60,
        },
        "rerank": {
            "enabled": False,
            "provider": "none",
            "model": "test",
            "top_k": 5,
        },
        "evaluation": {
            "enabled": False,
            "provider": "custom",
            "metrics": ["hit_rate"],
        },
        "observability": {
            "log_level": "INFO",
            "trace_enabled": False,
            "trace_file": "./logs/traces.jsonl",
            "structured_logging": False,
        },
        "ingestion": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "splitter": "recursive",
            "batch_size": 100,
        },
    }
    
    # 应用覆盖项
    for key, value in overrides.items():
        if "." in key:
            section, field = key.split(".", 1)
            if section in base_config:
                base_config[section][field] = value
        else:
            base_config[key] = value
    
    # 转换为 YAML 字符串
    import yaml
    return yaml.dump(base_config)


class FakeLLM(BaseLLM):
    """用于测试的 mock LLM 实现。"""
    
    def __init__(self, settings, **kwargs):
        self.settings = settings
        self.kwargs = kwargs
    
    def chat(self, messages, trace=None, **kwargs):
        """返回确定性的测试响应。"""
        self.validate_messages(messages)
        return ChatResponse(
            content="fake response",
            model="fake-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


class TestLLMFactory:
    """LLMFactory 的测试套件。"""

    def setup_method(self):
        """每个测试前清空 provider 注册表。"""
        LLMFactory._PROVIDERS.clear()
    
    def test_register_provider_success(self):
        """测试 provider 注册成功。"""
        LLMFactory.register_provider("fake", FakeLLM)
        assert "fake" in LLMFactory._PROVIDERS
        assert LLMFactory._PROVIDERS["fake"] == FakeLLM
    
    def test_register_provider_case_insensitive(self):
        """测试 provider 注册会统一转为小写。"""
        LLMFactory.register_provider("FAKE", FakeLLM)
        assert "fake" in LLMFactory._PROVIDERS
        assert "FAKE" not in LLMFactory._PROVIDERS
    
    def test_register_provider_invalid_class(self):
        """测试注册会拒绝非 BaseLLM 类。"""
        class NotAnLLM:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseLLM"):
            LLMFactory.register_provider("invalid", NotAnLLM)
    
    def test_create_success(self, tmp_path):
        """测试从 settings 成功创建 LLM。"""
        # 注册测试 provider
        LLMFactory.register_provider("fake", FakeLLM)

        # 创建最小配置
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(create_test_config(provider="fake"))
        
        settings = load_settings(str(config_file))
        llm = LLMFactory.create(settings)
        
        assert isinstance(llm, FakeLLM)
        assert llm.settings == settings
    
    def test_create_provider_case_insensitive(self, tmp_path):
        """测试工厂以大小写不敏感方式处理 provider 名称。"""
        LLMFactory.register_provider("fake", FakeLLM)
        
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(create_test_config(provider="FAKE"))
        
        settings = load_settings(str(config_file))
        llm = LLMFactory.create(settings)
        
        assert isinstance(llm, FakeLLM)
    
    def test_create_unknown_provider(self, tmp_path):
        """测试未知 provider 时工厂抛出清晰错误。"""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(create_test_config(provider="unknown_provider"))
        
        settings = load_settings(str(config_file))
        
        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create(settings)
        
        error_msg = str(exc_info.value)
        assert "unknown_provider" in error_msg
        assert "Unsupported LLM provider" in error_msg
        assert "Available providers" in error_msg
    
    def test_create_missing_provider_config(self, tmp_path):
        """测试 provider 配置缺失时工厂抛出错误。"""
        config_file = tmp_path / "settings.yaml"
        # 创建不含 provider 字段的配置
        config_text = create_test_config(provider="fake")
        # 移除 provider 行
        config_text = "\n".join(
            line for line in config_text.split("\n")
            if "provider:" not in line or "embedding:" in line or "vector_store:" in line
            or "rerank:" in line or "evaluation:" in line
        )
        config_file.write_text(config_text)
        
        with pytest.raises(Exception):  # SettingsError 或 ValueError
            settings = load_settings(str(config_file))
            LLMFactory.create(settings)
    
    def test_list_providers_empty(self):
        """测试无 provider 注册时的列表查询。"""
        assert LLMFactory.list_providers() == []
    
    def test_list_providers_sorted(self):
        """测试 provider 按排序顺序列出。"""
        LLMFactory.register_provider("zebra", FakeLLM)
        LLMFactory.register_provider("alpha", FakeLLM)
        LLMFactory.register_provider("beta", FakeLLM)
        
        providers = LLMFactory.list_providers()
        assert providers == ["alpha", "beta", "zebra"]


class TestBaseLLM:
    """BaseLLM 校验逻辑的测试套件。"""

    def test_validate_messages_success(self):
        """测试合法消息通过校验。"""
        llm = FakeLLM(settings=None)
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        
        # 不应抛出异常
        llm.validate_messages(messages)

    def test_validate_messages_empty_list(self):
        """测试校验拒绝空消息列表。"""
        llm = FakeLLM(settings=None)
        
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            llm.validate_messages([])
    
    def test_validate_messages_invalid_role(self):
        """测试校验拒绝无效角色。"""
        llm = FakeLLM(settings=None)
        messages = [Message(role="invalid_role", content="test")]
        
        with pytest.raises(ValueError, match="invalid role 'invalid_role'"):
            llm.validate_messages(messages)
    
    def test_validate_messages_empty_content(self):
        """测试校验拒绝空内容。"""
        llm = FakeLLM(settings=None)
        messages = [Message(role="user", content="   ")]
        
        with pytest.raises(ValueError, match="empty content"):
            llm.validate_messages(messages)
    
    def test_validate_messages_not_message_instance(self):
        """测试校验拒绝非 Message 对象。"""
        llm = FakeLLM(settings=None)
        messages = [{"role": "user", "content": "test"}]
        
        with pytest.raises(ValueError, match="not a Message instance"):
            llm.validate_messages(messages)


class TestFakeLLMIntegration:
    """FakeLLM mock 的集成测试。"""

    def test_chat_returns_expected_response(self):
        """测试 FakeLLM 的 chat 返回正确的 ChatResponse。"""
        llm = FakeLLM(settings=None)
        messages = [Message(role="user", content="test")]
        
        response = llm.chat(messages)
        
        assert isinstance(response, ChatResponse)
        assert response.content == "fake response"
        assert response.model == "fake-model"
        assert response.usage["total_tokens"] == 15
    
    def test_chat_validates_messages(self):
        """测试 FakeLLM 在处理前校验消息。"""
        llm = FakeLLM(settings=None)
        
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            llm.chat([])
