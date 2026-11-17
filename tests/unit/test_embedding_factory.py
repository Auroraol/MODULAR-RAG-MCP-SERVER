"""Embedding 工厂与 Base Embedding 的单元测试。

测试覆盖：
- 工厂模式：provider 注册、创建与路由
- 配置驱动的实例化
- 未知/缺失 provider 的错误处理
- BaseEmbedding 中的校验逻辑
"""

from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    """用于测试的假 embedding provider。

    返回确定性的假向量以便测试可重复。
    """

    def __init__(self, settings: Any = None, dimension: int = 384, **kwargs: Any):
        """初始化假 embedding provider。

        参数:
            settings: 可选配置（假实现中未使用）。
            dimension: 要返回的向量维度。
            **kwargs: 额外参数（未使用）。
        """
        self.settings = settings
        self.dimension = dimension
        self.call_count = 0

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """生成假 embedding。"""
        self.validate_texts(texts)
        self.call_count += 1

        # 返回确定性的假向量
        return [[float(i + j) for j in range(self.dimension)] for i in range(len(texts))]

    def get_dimension(self) -> int:
        """返回配置的维度。"""
        return self.dimension


class TestBaseEmbedding:
    """BaseEmbedding 抽象类的测试。"""

    def test_validate_texts_success(self):
        """有效文本列表应通过校验。"""
        embedding = FakeEmbedding()
        # 不应抛出异常
        embedding.validate_texts(["hello", "world"])

    def test_validate_texts_empty_list(self):
        """空列表应抛出 ValueError。"""
        embedding = FakeEmbedding()
        with pytest.raises(ValueError, match="cannot be empty"):
            embedding.validate_texts([])
    
    def test_validate_texts_non_string(self):
        """非字符串条目应抛出 ValueError。"""
        embedding = FakeEmbedding()
        with pytest.raises(ValueError, match="not a string"):
            embedding.validate_texts(["valid", 123, "text"])  # type: ignore

    def test_validate_texts_empty_string(self):
        """空字符串或仅含空白的字符串应抛出 ValueError。"""
        embedding = FakeEmbedding()
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            embedding.validate_texts(["valid", "   ", "text"])
    
    def test_get_dimension_implemented(self):
        """FakeEmbedding 应返回配置的维度。"""
        embedding = FakeEmbedding(dimension=512)
        assert embedding.get_dimension() == 512
    
    def test_get_dimension_not_implemented(self):
        """未重写 get_dimension 的 BaseEmbedding 应抛出 NotImplementedError。"""

        class IncompleteEmbedding(BaseEmbedding):
            def embed(self, texts: List[str], trace: Optional[Any] = None, **kwargs: Any) -> List[List[float]]:
                return [[0.0]]
        
        incomplete = IncompleteEmbedding()
        with pytest.raises(NotImplementedError, match="must implement get_dimension"):
            incomplete.get_dimension()


class TestFakeEmbedding:
    """FakeEmbedding provider 实现的测试。"""

    def test_embed_single_text(self):
        """对单个文本进行 embedding 应返回一个向量。"""
        embedding = FakeEmbedding(dimension=3)
        result = embedding.embed(["hello"])

        assert len(result) == 1
        assert len(result[0]) == 3
        assert result[0] == [0.0, 1.0, 2.0]

    def test_embed_multiple_texts(self):
        """对多个文本进行 embedding 应返回数量匹配的向量。"""
        embedding = FakeEmbedding(dimension=2)
        result = embedding.embed(["hello", "world", "test"])
        
        assert len(result) == 3
        assert result[0] == [0.0, 1.0]
        assert result[1] == [1.0, 2.0]
        assert result[2] == [2.0, 3.0]
    
    def test_embed_increments_call_count(self):
        """每次 embed 调用应递增计数器。"""
        embedding = FakeEmbedding()
        assert embedding.call_count == 0
        
        embedding.embed(["test1"])
        assert embedding.call_count == 1
        
        embedding.embed(["test2", "test3"])
        assert embedding.call_count == 2
    
    def test_embed_validates_input(self):
        """embed() 应调用 validate_texts 并在输入无效时抛出异常。"""
        embedding = FakeEmbedding()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            embedding.embed([])
        
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            embedding.embed(["  "])


class TestEmbeddingFactory:
    """EmbeddingFactory 的测试。"""

    def setup_method(self):
        """每个测试前重置工厂注册表。"""
        EmbeddingFactory._PROVIDERS.clear()
    
    def test_register_provider_success(self):
        """注册有效的 provider 应成功。"""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        assert "fake" in EmbeddingFactory._PROVIDERS
        assert EmbeddingFactory._PROVIDERS["fake"] == FakeEmbedding
    
    def test_register_provider_case_insensitive(self):
        """provider 名称应被规范为小写。"""
        EmbeddingFactory.register_provider("OpenAI", FakeEmbedding)
        assert "openai" in EmbeddingFactory._PROVIDERS
    
    def test_register_provider_invalid_class(self):
        """注册非 BaseEmbedding 子类应抛出 ValueError。"""

        class NotAnEmbedding:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseEmbedding"):
            EmbeddingFactory.register_provider("invalid", NotAnEmbedding)  # type: ignore
    
    def test_list_providers_empty(self):
        """未注册任何 provider 时 list_providers 应返回空列表。"""
        assert EmbeddingFactory.list_providers() == []
    
    def test_list_providers_sorted(self):
        """list_providers 应返回排序后的 provider 名称。"""
        EmbeddingFactory.register_provider("zebra", FakeEmbedding)
        EmbeddingFactory.register_provider("alpha", FakeEmbedding)
        EmbeddingFactory.register_provider("beta", FakeEmbedding)
        
        providers = EmbeddingFactory.list_providers()
        assert providers == ["alpha", "beta", "zebra"]
    
    def test_create_success(self):
        """创建已注册的 provider 应成功。"""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        
        settings = MagicMock()
        settings.embedding.provider = "fake"
        
        embedding = EmbeddingFactory.create(settings)
        
        assert isinstance(embedding, FakeEmbedding)
        assert embedding.settings == settings
    
    def test_create_case_insensitive(self):
        """provider 查找应大小写不敏感。"""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        
        settings = MagicMock()
        settings.embedding.provider = "FAKE"
        
        embedding = EmbeddingFactory.create(settings)
        assert isinstance(embedding, FakeEmbedding)
    
    def test_create_with_overrides(self):
        """工厂应将覆盖 kwargs 传给 provider 构造函数。"""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        
        settings = MagicMock()
        settings.embedding.provider = "fake"
        
        embedding = EmbeddingFactory.create(settings, dimension=1024)
        assert embedding.dimension == 1024
    
    def test_create_unknown_provider(self):
        """创建未注册的 provider 应抛出明确错误。"""
        EmbeddingFactory.register_provider("fake", FakeEmbedding)
        
        settings = MagicMock()
        settings.embedding.provider = "unknown"
        
        with pytest.raises(ValueError) as exc_info:
            EmbeddingFactory.create(settings)
        
        error_message = str(exc_info.value)
        assert "Unsupported Embedding provider: 'unknown'" in error_message
        assert "Available providers:" in error_message
    
    def test_create_missing_provider_config(self):
        """配置中缺少 provider 应抛出明确错误。"""
        settings = MagicMock()
        del settings.embedding  # 模拟配置缺失
        
        with pytest.raises(ValueError) as exc_info:
            EmbeddingFactory.create(settings)
        
        error_message = str(exc_info.value)
        assert "Missing required configuration" in error_message
        assert "settings.embedding.provider" in error_message
        assert "settings.yaml" in error_message
    
    def test_create_provider_instantiation_failure(self):
        """provider 构造函数错误应被包装为 RuntimeError。"""

        class BrokenEmbedding(BaseEmbedding):
            def __init__(self, settings: Any, **kwargs: Any):
                raise ValueError("Intentional init error")

            def embed(self, texts: List[str], trace: Optional[Any] = None, **kwargs: Any) -> List[List[float]]:
                return [[0.0]]

        EmbeddingFactory.register_provider("broken", BrokenEmbedding)

        settings = MagicMock()
        settings.embedding.provider = "broken"

        with pytest.raises(RuntimeError) as exc_info:
            EmbeddingFactory.create(settings)

        error_message = str(exc_info.value)
        assert "Failed to instantiate Embedding provider 'broken'" in error_message
        assert "Intentional init error" in error_message

    def test_create_no_providers_registered(self):
        """注册表为空时创建 provider 应给出有用的提示信息。"""
        settings = MagicMock()
        settings.embedding.provider = "openai"
        
        with pytest.raises(ValueError) as exc_info:
            EmbeddingFactory.create(settings)
        
        error_message = str(exc_info.value)
        assert "Unsupported Embedding provider: 'openai'" in error_message
        assert "Available providers: none" in error_message
