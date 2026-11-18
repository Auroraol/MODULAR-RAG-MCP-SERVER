"""用于创建 Embedding Provider 实例的工厂。

本模块实现工厂模式，根据配置实例化相应的 Embedding Provider，
从而无需修改代码即可通过配置驱动选择不同后端。

工厂注册
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from src.core.settings import Settings


class EmbeddingFactory:
    """用于创建 Embedding Provider 实例的工厂。

    本工厂从 settings 读取 Provider 配置并实例化相应的 Embedding 实现。
    当前支持的 Provider 有 OpenAI、Azure，后续任务还会添加更多。

    应用的设计原则：
    - 工厂模式：集中对象创建逻辑。
    - 配置驱动：根据 settings.yaml 选择 Provider。
    - 快速失败：对未知 Provider 抛出明确的错误。
    """

    # 受支持 Provider 的注册表
    _PROVIDERS: dict[str, type[BaseEmbedding]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseEmbedding]) -> None:
        """注册一个新的 Embedding Provider 实现。

        该方法允许各 Provider 实现向工厂自行注册，以支持可扩展性。

        参数:
            name: Provider 标识符（例如 'openai'、'azure'、'local'）。
            provider_class: 实现该 Provider 的 BaseEmbedding 子类。

        异常:
            ValueError: 当 provider_class 未继承 BaseEmbedding 时抛出。
        """
        if not issubclass(provider_class, BaseEmbedding):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseEmbedding"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEmbedding:
        """根据配置创建 Embedding 实例。

        参数:
            settings: 包含 Embedding 配置的应用配置。
            **override_kwargs: 用于覆盖配置值的可选参数。

        返回:
            配置的 Embedding Provider 实例。

        异常:
            ValueError: 当配置的 Provider 不受支持时抛出。
            AttributeError: 当缺少必需的配置字段时抛出。

        示例:
            >>> settings = Settings.load('config/settings.yaml')
            >>> embedding = EmbeddingFactory.create(settings)
            >>> vectors = embedding.embed(["hello world", "test"])
        """
        # 从 settings 中提取 Provider 名称
        try:
            provider_name = settings.embedding.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.embedding.provider. "
                "Please ensure 'embedding.provider' is specified in settings.yaml"
            ) from e
        
        # 在注册表中查找 Provider 类
        provider_class = cls._PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Embedding provider: '{provider_name}'. "
                f"Available providers: {available}"
            )
        
        # 实例化 Provider
        # Provider 类应接受 settings 和可选的 kwargs
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Embedding provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的 Provider 名称。

        返回:
            可用 Provider 标识符的排序列表。
        """
        return sorted(cls._PROVIDERS.keys())


# 在模块导入时自动注册 Provider
def _register_builtin_providers() -> None:
    """向工厂注册内置的 Embedding Provider。"""
    try:
        from src.libs.embedding.openai_embedding import OpenAIEmbedding
        EmbeddingFactory.register_provider("openai", OpenAIEmbedding)
    except ImportError:
        pass  # OpenAI Provider 不可用

    try:
        from src.libs.embedding.azure_embedding import AzureEmbedding
        EmbeddingFactory.register_provider("azure", AzureEmbedding)
    except ImportError:
        pass  # Azure Provider 不可用

    try:
        from src.libs.embedding.ollama_embedding import OllamaEmbedding
        EmbeddingFactory.register_provider("ollama", OllamaEmbedding)
    except ImportError:
        pass  # Ollama Provider 不可用


# 模块被导入时注册 Provider
_register_builtin_providers()
