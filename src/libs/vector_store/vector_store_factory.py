"""用于创建 VectorStore provider 实例的工厂。

本模块实现工厂模式，根据配置实例化相应的 VectorStore provider，
从而无需修改代码即可配置驱动地选择不同后端。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings


class VectorStoreFactory:
    """用于创建 VectorStore provider 实例的工厂。

    此工厂从 settings 读取 provider 配置，并实例化相应的 VectorStore 实现。
    受支持的 provider 将在后续任务（B7.6 及之后）中添加。

    应用的设计原则：
    - 工厂模式：集中管理对象创建逻辑。
    - 配置驱动：基于 settings.yaml 选择 provider。
    - 快速失败：对未知 provider 抛出明确的错误。
    """

    # 受支持的 provider 注册表（将在 B7.x 任务中填充）
    _PROVIDERS: dict[str, type[BaseVectorStore]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseVectorStore]) -> None:
        """注册新的 VectorStore provider 实现。

        此方法允许 provider 实现向工厂自行注册，
        以支持可扩展性。

        参数:
            name: provider 标识符（如 'chroma'、'qdrant'、'milvus'）。
            provider_class: 实现该 provider 的 BaseVectorStore 子类。

        异常:
            ValueError: 当 provider_class 未继承 BaseVectorStore 时。
        """
        if not issubclass(provider_class, BaseVectorStore):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseVectorStore"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseVectorStore:
        """根据配置创建 VectorStore 实例。

        参数:
            settings: 包含 VectorStore 配置的应用设置。
            **override_kwargs: 用于覆盖配置值的可选参数。

        返回:
            配置的 VectorStore provider 实例。

        异常:
            ValueError: 当配置的 provider 不受支持时。
            AttributeError: 当缺少必需的配置字段时。

        示例:
            >>> settings = Settings.load('config/settings.yaml')
            >>> vector_store = VectorStoreFactory.create(settings)
            >>> vector_store.upsert([{'id': 'doc1', 'vector': [0.1, 0.2]}])
        """
        # 从 settings 中提取 provider 名称
        try:
            provider_name = settings.vector_store.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vector_store.provider. "
                "Please ensure 'vector_store.provider' is specified in settings.yaml"
            ) from e

        # 在注册表中查找 provider 类
        provider_class = cls._PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported VectorStore provider: '{provider_name}'. "
                f"Available providers: {available}. "
                f"Provider implementations will be added in task B7.6 and beyond."
            )
        
        # 实例化 provider
        # provider 类应接受 settings 与可选的 kwargs
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate VectorStore provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的 provider 名称。

        返回:
            provider 名称的排序列表。

        示例:
            >>> VectorStoreFactory.list_providers()
            ['chroma', 'milvus', 'qdrant']
        """
        return sorted(cls._PROVIDERS.keys())
