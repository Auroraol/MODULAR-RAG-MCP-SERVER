"""用于创建 LLM Provider 实例的工厂。

本模块实现工厂模式，根据配置实例化合适的 LLM Provider，
从而实现无需改动代码、由配置驱动地选择不同后端。

同时支持纯文本 LLM 与 Vision LLM（多模态）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.base_vision_llm import BaseVisionLLM

if TYPE_CHECKING:
    from src.core.settings import Settings


# 在模块加载时导入并注册 Vision LLM Provider
def _register_vision_providers() -> None:
    """注册所有 Vision LLM Provider 实现。

    该函数在模块导入时调用，用于填充
    Vision LLM Provider 注册表。新 Provider 实现后在此处添加。
    """
    try:
        from src.libs.llm.azure_vision_llm import AzureVisionLLM
        from src.libs.llm.llm_factory import LLMFactory
        LLMFactory.register_vision_provider("azure", AzureVisionLLM)
    except ImportError:
        # Provider 尚未实现，跳过注册
        pass
    
    try:
        from src.libs.llm.openai_vision_llm import OpenAIVisionLLM
        from src.libs.llm.llm_factory import LLMFactory
        LLMFactory.register_vision_provider("openai", OpenAIVisionLLM)
    except ImportError:
        pass


class LLMFactory:
    """用于创建 LLM Provider 实例的工厂。

    该工厂从 settings 读取 Provider 配置并实例化对应的 LLM 实现。
    同时支持纯文本 LLM 与 Vision LLM（多模态）。

    遵循的设计原则:
    - 工厂模式: 集中管理对象创建逻辑。
    - 配置驱动: 根据 settings.yaml 选择 Provider。
    - 快速失败: 对未知 Provider 抛出明确错误。
    - 分离: 文本与 Vision LLM 注册表彼此独立。
    """

    # 支持的纯文本 LLM Provider 注册表（将在 B7.x 任务中填充）
    _PROVIDERS: dict[str, type[BaseLLM]] = {}

    # 支持的 Vision LLM Provider 注册表（将在 B9+ 任务中填充）
    _VISION_PROVIDERS: dict[str, type[BaseVisionLLM]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLM]) -> None:
        """注册一个新的 LLM Provider 实现。

        该方法允许各 Provider 实现向工厂注册自身，
        从而支持可扩展性。

        参数:
            name: Provider 标识符（例如 'openai'、'azure'、'ollama'）。
            provider_class: 实现该 Provider 的 BaseLLM 子类。

        异常:
            ValueError: provider_class 未继承 BaseLLM 时抛出。
        """
        if not issubclass(provider_class, BaseLLM):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseLLM"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseLLM:
        """根据配置创建 LLM 实例。

        参数:
            settings: 包含 LLM 配置的应用设置。
            **override_kwargs: 用于覆盖配置值的可选参数。

        返回:
            已配置的 LLM Provider 实例。

        异常:
            ValueError: 配置的 Provider 不被支持时抛出。
            AttributeError: 缺少必需的配置字段时抛出。

        示例:
            >>> settings = Settings.load('config/settings.yaml')
            >>> llm = LLMFactory.create(settings)
            >>> response = llm.chat([Message(role='user', content='Hello')])
        """
        # 从 settings 中提取 Provider 名称
        try:
            provider_name = settings.llm.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.llm.provider. "
                "Please ensure 'llm.provider' is specified in settings.yaml"
            ) from e
        
        # 在注册表中查找 Provider 类
        provider_class = cls._PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported LLM provider: '{provider_name}'. "
                f"Available providers: {available}. "
                f"Provider implementations will be added in tasks B7.1-B7.2."
            )
        
        # 实例化 Provider
        # Provider 类应接受 settings 以及可选的 kwargs
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate LLM provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的 Provider 名称。

        返回:
            可用 Provider 标识符的排序列表。
        """
        return sorted(cls._PROVIDERS.keys())
    
    @classmethod
    def register_vision_provider(
        cls,
        name: str,
        provider_class: type[BaseVisionLLM]
    ) -> None:
        """注册一个新的 Vision LLM Provider 实现。

        该方法允许各 Vision LLM Provider 实现向工厂注册自身，
        从而支持可扩展性。

        参数:
            name: Provider 标识符（例如 'azure'、'ollama'）。
            provider_class: 实现该 Provider 的 BaseVisionLLM 子类。

        异常:
            ValueError: provider_class 未继承 BaseVisionLLM 时抛出。
        """
        if not issubclass(provider_class, BaseVisionLLM):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseVisionLLM"
            )
        cls._VISION_PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create_vision_llm(
        cls,
        settings: Settings,
        **override_kwargs: Any
    ) -> BaseVisionLLM:
        """根据配置创建 Vision LLM 实例。

        Vision LLM 支持多模态输入（文本 + 图像），用于
        图像描述、视觉问答以及含嵌入图像的文档理解等任务。

        参数:
            settings: 包含 Vision LLM 配置的应用设置。
            **override_kwargs: 用于覆盖配置值的可选参数。

        返回:
            已配置的 Vision LLM Provider 实例。

        异常:
            ValueError: 配置的 Provider 不被支持或缺少配置时抛出。
            RuntimeError: Provider 实例化失败时抛出。

        示例:
            >>> settings = Settings.load('config/settings.yaml')
            >>> vision_llm = LLMFactory.create_vision_llm(settings)
            >>> image = ImageInput(path="diagram.png")
            >>> response = vision_llm.chat_with_image("Describe this", image)
        """
        # 从 settings 中提取 Provider 名称
        # Vision LLM 配置可能嵌套在 settings.vision_llm 或 settings.llm 下
        try:
            # 优先尝试 vision_llm 配置段
            if hasattr(settings, 'vision_llm') and hasattr(settings.vision_llm, 'provider'):
                provider_name = settings.vision_llm.provider.lower()
            # 回退到 llm.provider（部分 Provider 同时支持文本与视觉）
            elif hasattr(settings, 'llm') and hasattr(settings.llm, 'provider'):
                provider_name = settings.llm.provider.lower()
            else:
                raise AttributeError("No vision_llm or llm provider configuration found")
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vision_llm.provider or settings.llm.provider. "
                "Please ensure 'vision_llm.provider' or 'llm.provider' is specified in settings.yaml"
            ) from e
        
        # 在 vision 注册表中查找 Provider 类
        provider_class = cls._VISION_PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._VISION_PROVIDERS.keys())) if cls._VISION_PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Vision LLM provider: '{provider_name}'. "
                f"Available Vision LLM providers: {available}. "
                f"Vision LLM implementations will be added in tasks B9+."
            )
        
        # 实例化 Provider
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Vision LLM provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_vision_providers(cls) -> list[str]:
        """列出所有已注册的 Vision LLM Provider 名称。

        返回:
            可用 Vision LLM Provider 标识符的排序列表。
        """
        return sorted(cls._VISION_PROVIDERS.keys())


# 在模块加载时注册 Vision LLM Provider
_register_vision_providers()
