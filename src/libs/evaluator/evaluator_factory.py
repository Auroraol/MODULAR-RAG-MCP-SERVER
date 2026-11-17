"""用于创建 Evaluator Provider 实例的工厂。

本模块实现工厂模式，根据配置实例化相应的 Evaluator Provider。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.evaluator.base_evaluator import BaseEvaluator, NoneEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator

if TYPE_CHECKING:
    from src.core.settings import Settings


def _get_ragas_evaluator() -> type[BaseEvaluator]:
    """延迟导入 RagasEvaluator，以避免对 ragas 的硬依赖。"""
    from src.observability.evaluation.ragas_evaluator import RagasEvaluator
    return RagasEvaluator


def _get_composite_evaluator() -> type[BaseEvaluator]:
    """延迟导入 CompositeEvaluator。"""
    from src.observability.evaluation.composite_evaluator import CompositeEvaluator
    return CompositeEvaluator


class EvaluatorFactory:
    """用于创建 Evaluator Provider 实例的工厂。

    应用的设计原则：
    - 工厂模式：集中对象创建逻辑。
    - 配置驱动：根据 settings.yaml 选择 Provider。
    - 回退：评估被禁用时返回 NoneEvaluator。
    - 快速失败：对未知 Provider 抛出明确的错误。
    """

    _PROVIDERS: dict[str, type[BaseEvaluator]] = {
        "custom": CustomEvaluator,
    }

    # 延迟加载的 Provider（按需导入，以避免硬依赖）
    _LAZY_PROVIDERS: dict[str, Any] = {
        "ragas": _get_ragas_evaluator,
        "composite": _get_composite_evaluator,
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseEvaluator]) -> None:
        """注册一个新的 Evaluator Provider 实现。

        参数:
            name: Provider 标识符（例如 'ragas'、'custom'）。
            provider_class: 实现该 Provider 的 BaseEvaluator 子类。

        异常:
            ValueError: 当 provider_class 未继承 BaseEvaluator 时抛出。
        """
        if not issubclass(provider_class, BaseEvaluator):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseEvaluator"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEvaluator:
        """根据配置创建 Evaluator 实例。

        参数:
            settings: 包含评估配置的应用配置。
            **override_kwargs: 用于覆盖配置值的可选参数。

        返回:
            配置的 Evaluator Provider 实例。

        异常:
            ValueError: 当配置的 Provider 不受支持或缺失时抛出。
            RuntimeError: 当 Provider 初始化失败时抛出。
        """
        try:
            # 接受完整的 Settings（带 .evaluation 属性），
            # 或直接接受裸的 EvaluationSettings 对象。
            if hasattr(settings, "evaluation"):
                evaluation_settings = settings.evaluation
            elif hasattr(settings, "provider") and hasattr(settings, "enabled"):
                evaluation_settings = settings
            else:
                raise AttributeError("settings has no 'evaluation' attribute")
            if evaluation_settings is None:
                raise AttributeError("settings.evaluation is None")
            provider_name = evaluation_settings.provider.lower()
            enabled = bool(evaluation_settings.enabled)
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.evaluation.provider. "
                "Please ensure 'evaluation.provider' is specified in settings.yaml"
            ) from e

        if not enabled or provider_name in {"none", "disabled"}:
            return NoneEvaluator(settings=settings, **override_kwargs)

        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None and provider_name in cls._LAZY_PROVIDERS:
            try:
                provider_class = cls._LAZY_PROVIDERS[provider_name]()
                cls._PROVIDERS[provider_name] = provider_class  # 缓存以供下次调用
            except ImportError as e:
                raise ValueError(
                    f"Provider '{provider_name}' requires additional dependencies: {e}"
                ) from e
        if provider_class is None:
            all_providers = sorted(set(cls._PROVIDERS.keys()) | set(cls._LAZY_PROVIDERS.keys()))
            available = ", ".join(all_providers) if all_providers else "none"
            raise ValueError(
                f"Unsupported Evaluator provider: '{provider_name}'. "
                f"Available providers: {available}."
            )

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Evaluator provider '{provider_name}': {e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的 Provider 名称。

        返回:
            可用 Provider 标识符的排序列表。
        """
        return sorted(cls._PROVIDERS.keys())