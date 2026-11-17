"""OpenAI Embedding 实现。

本模块提供配合标准 OpenAI Embeddings API 使用的 OpenAI Embedding 实现。
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class OpenAIEmbeddingError(RuntimeError):
    """当 OpenAI Embeddings API 调用失败时抛出。"""


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding Provider 实现。

    本类为 OpenAI 的 Embeddings API 实现了 BaseEmbedding 接口。
    支持 text-embedding-3-small、text-embedding-3-large 以及
    text-embedding-ada-002 等较旧的模型。

    属性:
        api_key: 用于认证的 API key。
        model: 使用的模型标识符。
        dimensions: 可选的维度缩减（仅适用于 text-embedding-3-*）。
        base_url: API 的基础 URL（默认：OpenAI 的 endpoint）。

    示例:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = OpenAIEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """
    
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 OpenAI Embedding Provider。

        参数:
            settings: 包含 Embedding 配置的应用配置。
            api_key: 可选的 API key 覆盖（回退到 settings.embedding.api_key 或环境变量）。
            base_url: 可选的 base URL 覆盖。
            **kwargs: 其他配置覆盖。

        异常:
            ValueError: 当未提供 API key 且环境中也未找到时抛出。

        注意:
            当 settings 中存在 azure_endpoint 时，Provider 会自动
            构造兼容 Azure 的 OpenAI URL，并使用 api-key 认证。
        """
        self.model = settings.embedding.model

        # 提取可选的 dimensions 配置
        self.dimensions = getattr(settings.embedding, 'dimensions', None)

        # API key 优先级：显式参数 > settings > 环境变量
        self.api_key = (
            api_key
            or getattr(settings.embedding, 'api_key', None)
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set in settings.yaml (embedding.api_key), "
                "OPENAI_API_KEY environment variable, or pass api_key parameter."
            )
        
        # Azure 兼容模式检测
        azure_endpoint = getattr(settings.embedding, 'azure_endpoint', None)
        self.api_version = getattr(settings.embedding, 'api_version', None)
        self._use_azure_auth = False
        
        if base_url:
            self.base_url = base_url
        elif azure_endpoint:
            # Azure 兼容模式：构造基于部署的 URL
            deployment = getattr(settings.embedding, 'deployment_name', None) or self.model
            self.base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment}"
            self._use_azure_auth = True
            if not self.api_version:
                self.api_version = "2024-02-15-preview"
        else:
            settings_base_url = getattr(settings.embedding, 'base_url', None)
            self.base_url = settings_base_url if settings_base_url else self.DEFAULT_BASE_URL
        
        # 保存其余 kwargs 以供将来使用
        self._extra_config = kwargs
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """使用 OpenAI API 为一组文本生成 Embedding。

        参数:
            texts: 待 Embedding 的文本字符串列表，不能为空。
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F）。
            **kwargs: 覆盖参数（dimensions 等）。

        返回:
            Embedding 向量列表，其中每个向量是一个浮点数列表。
            外层列表长度与 len(texts) 一致。

        异常:
            ValueError: 当 texts 列表为空或包含无效条目时抛出。
            OpenAIEmbeddingError: 当 API 调用失败时抛出。
        """
        # 校验输入
        self.validate_texts(texts)

        # 导入 OpenAI 客户端（延迟导入，避免模块级依赖）
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI Python package not installed. "
                "Install with: pip install openai"
            ) from e
        
        # 初始化 OpenAI 客户端
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        # Azure 兼容模式：添加 api-version 查询参数和 api-key 请求头
        if self._use_azure_auth and self.api_version:
            client_kwargs["default_query"] = {"api-version": self.api_version}
            client_kwargs["default_headers"] = {"api-key": self.api_key}
        
        client = OpenAI(**client_kwargs)
        
        # 准备 API 调用参数
        api_params = {
            "input": texts,
            "model": self.model,
        }
        
        # 若指定了 dimensions 则添加（仅适用于 text-embedding-3-* 模型）
        # text-embedding-ada-002 不支持 dimensions 参数
        dimensions = kwargs.get("dimensions", self.dimensions)
        if dimensions is not None and self.model.startswith("text-embedding-3"):
            api_params["dimensions"] = dimensions
        
        # 调用 OpenAI API
        try:
            response = client.embeddings.create(**api_params)
        except Exception as e:
            raise OpenAIEmbeddingError(
                f"OpenAI Embeddings API call failed: {e}"
            ) from e
        
        # 从响应中提取 Embedding
        # 响应格式：response.data 是带有 .embedding 属性的对象列表
        try:
            embeddings = [item.embedding for item in response.data]
        except (AttributeError, KeyError) as e:
            raise OpenAIEmbeddingError(
                f"Failed to parse OpenAI Embeddings API response: {e}"
            ) from e
        
        # 校验输出数量与输入数量是否一致
        if len(embeddings) != len(texts):
            raise OpenAIEmbeddingError(
                f"Output length mismatch: expected {len(texts)}, got {len(embeddings)}"
            )
        
        return embeddings
    
    def get_dimension(self) -> Optional[int]:
        """获取当前配置模型的 Embedding 维度。

        返回:
            Embedding 维度；若无法确定则返回 None。

        注意:
            对于带自定义维度的 text-embedding-3-* 模型，返回配置的维度。
            对于其他模型，返回其默认维度。
        """
        # 若显式配置了 dimensions，直接返回
        if self.dimensions is not None:
            return self.dimensions
        
        # 各模型的默认维度
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        return model_dimensions.get(self.model)
