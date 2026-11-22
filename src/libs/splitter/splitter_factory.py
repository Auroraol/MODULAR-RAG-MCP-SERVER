"""用于创建 Splitter 实例的工厂。

本模块实现工厂模式，根据配置实例化相应的 Splitter provider，
从而无需修改代码即可配置驱动地选择不同切分策略。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.splitter.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from src.core.settings import Settings


def _register_builtin_providers() -> None:
    """注册内置的切分器 provider。

    模块被导入时会自动调用此函数。
    它向工厂注册所有可用的切分器实现。
    """
    # 在此导入以避免循环导入，并优雅处理缺失的依赖
    try:
        from src.libs.splitter.recursive_splitter import RecursiveSplitter
        # 注册递归字符文本切分器, TODO 后续新增策略时请在此处添加
        SplitterFactory.register_provider("recursive", RecursiveSplitter)
    except ImportError:
        pass  # RecursiveSplitter 不可用（缺少 langchain 依赖）


class SplitterFactory:
    """用于创建 Splitter provider 实例的工厂。

    此工厂从 settings 读取 splitter 配置，并实例化相应的 Splitter 实现。
    受支持的 provider 将在后续任务（B7.5）中添加。

    应用的设计原则：
    - 工厂模式：集中管理对象创建逻辑。
    - 配置驱动：基于 settings.yaml 选择 provider。
    - 快速失败：对未知 provider 抛出明确的错误。
    """
    
    _PROVIDERS: dict[str, type[BaseSplitter]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseSplitter]) -> None:
        """注册新的 Splitter provider 实现。

        参数:
            name: provider 标识符（如 'recursive'、'semantic'、'fixed'）。
            provider_class: 实现该 provider 的 BaseSplitter 子类。

        异常:
            ValueError: 当 provider_class 未继承 BaseSplitter 时。
        """
        if not issubclass(provider_class, BaseSplitter):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseSplitter"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseSplitter:
        """根据配置创建 Splitter 实例。

        参数:
            settings: 包含 ingestion 配置的应用设置。
            **override_kwargs: 用于覆盖配置值的可选参数。

        返回:
            配置的 Splitter provider 实例。

        异常:
            ValueError: 当配置的 provider 不受支持或缺失时。
        """
        try:
            ingestion_settings = settings.ingestion
            if ingestion_settings is None:
                raise AttributeError("settings.ingestion is None")
            provider_name = ingestion_settings.splitter.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.ingestion.splitter. "
                "Please ensure 'ingestion.splitter' is specified in settings.yaml"
            ) from e
        
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Splitter provider: '{provider_name}'. "
                f"Available providers: {available}. "
                "Provider implementations will be added in task B7.5."
            )
        
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Splitter provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的 provider 名称。

        返回:
            可用 provider 标识符的排序列表。
        """
        return sorted(cls._PROVIDERS.keys())


# 模块导入时自动注册内置 provider
_register_builtin_providers()