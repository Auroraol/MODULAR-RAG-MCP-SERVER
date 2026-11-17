"""用于创建 Reranker provider 实例的工厂。

本模块实现工厂模式，根据配置实例化相应的 Reranker provider，
从而无需修改代码即可配置驱动地选择不同后端。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker

if TYPE_CHECKING:
    from src.core.settings import Settings


def _lazy_import_llm_reranker():
    """延迟导入以避免循环依赖。"""
    from src.libs.reranker.llm_reranker import LLMReranker
    return LLMReranker


def _lazy_import_cross_encoder_reranker():
    """延迟导入以避免循环依赖。"""
    from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
    return CrossEncoderReranker


class RerankerFactory:
    """用于创建 Reranker provider 实例的工厂。

    此工厂从 settings 读取 rerank 配置，并实例化相应的 Reranker 实现。
    Provider 实现将在后续任务（B7.7、B7.8）中添加。

    应用的设计原则：
    - 工厂模式：集中管理对象创建逻辑。
    - 配置驱动：基于 settings.yaml 选择 provider。
    - 降级：禁用或 provider 为 'none' 时返回 NoneReranker。
    - 快速失败：对未知 provider 抛出明确的错误。
    """
    
    _PROVIDERS: dict[str, type[BaseReranker]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseReranker]) -> None:
        """注册新的 Reranker provider 实现。

        参数:
            name: provider 标识符（如 'cross_encoder'、'llm'）。
            provider_class: 实现该 provider 的 BaseReranker 子类。

        异常:
            ValueError: 当 provider_class 未继承 BaseReranker 时。
        """
        if not issubclass(provider_class, BaseReranker):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseReranker"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseReranker:
        """根据配置创建 Reranker 实例。

        参数:
            settings: 包含 rerank 配置的应用设置。
            **override_kwargs: 用于覆盖配置值的可选参数。

        返回:
            配置的 Reranker provider 实例。

        异常:
            ValueError: 当配置的 provider 不受支持或缺失时。
            RuntimeError: 当 provider 初始化失败时。
        """
        # 若尚未注册，则延迟注册 LLM reranker
        if "llm" not in cls._PROVIDERS:
            LLMReranker = _lazy_import_llm_reranker()
            cls.register_provider("llm", LLMReranker)

        # 若尚未注册，则延迟注册 Cross-Encoder reranker
        if "cross_encoder" not in cls._PROVIDERS:
            CrossEncoderReranker = _lazy_import_cross_encoder_reranker()
            cls.register_provider("cross_encoder", CrossEncoderReranker)
        
        try:
            rerank_settings = settings.rerank
            if rerank_settings is None:
                raise AttributeError("settings.rerank is None")
            provider_name = rerank_settings.provider.lower()
            enabled = bool(rerank_settings.enabled)
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.rerank.provider. "
                "Please ensure 'rerank.provider' is specified in settings.yaml"
            ) from e
        
        if not enabled or provider_name == "none":
            return NoneReranker(settings=settings, **override_kwargs)
        
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Reranker provider: '{provider_name}'. "
                f"Available providers: {available}."
            )
        
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Reranker provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的 provider 名称。

        返回:
            可用 provider 标识符的排序列表。
        """
        return sorted(cls._PROVIDERS.keys())
