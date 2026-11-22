"""用于本地 Embedding 模型的 Ollama Embedding 实现。

本模块提供可配合本地运行的 Ollama 实例使用的 Ollama Embedding 实现。
Ollama 支持在本地硬件上运行 nomic-embed-text、mxbai-embed-large 等 Embedding 模型。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class OllamaEmbeddingError(RuntimeError):
    """当 Ollama Embeddings API 调用失败时抛出。

    该异常提供清晰的错误信息，同时不会暴露内部 URL 等敏感配置细节。
    """


class OllamaEmbedding(BaseEmbedding):
    """用于本地 Embedding 的 Ollama Embedding Provider 实现。

    本类为 Ollama 的 embeddings API 实现了 BaseEmbedding 接口，
    从而无需云依赖即可在本地生成 Embedding。

    属性:
        base_url: Ollama 服务器的基础 URL（默认：http://localhost:11434）。
        model: 使用的模型标识符（例如 'nomic-embed-text'、'mxbai-embed-large'）。
        timeout: 请求超时时间（秒）。
        dimension: 该模型产出的 Embedding 维度。

    示例:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = OllamaEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0  # 本地推理使用更长的超时时间
    DEFAULT_DIMENSION = 768  # 本地 Embedding 模型的常见维度
    
    def __init__(
        self,
        settings: Any,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Ollama Embedding Provider。

        参数:
            settings: 包含 Embedding 配置的应用配置。
            base_url: 可选的 base URL 覆盖（回退到环境变量 OLLAMA_BASE_URL）。
            timeout: 可选的请求超时覆盖。
            **kwargs: 其他配置覆盖。

        异常:
            ValueError: 当缺少必需配置时抛出。
        """
        self.model = settings.embedding.model

        # Base URL 优先级：显式参数 > settings > 环境变量 > 默认值
        settings_base_url = getattr(settings.embedding, "base_url", None)
        if not isinstance(settings_base_url, str) or not settings_base_url:
            settings_base_url = None
        self.base_url = (
            base_url
            or settings_base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        
        # 超时优先级：显式参数 > 默认值
        self.timeout = timeout or self.DEFAULT_TIMEOUT

        # 维度优先级：settings > 默认值
        self.dimension = getattr(settings.embedding, 'dimensions', self.DEFAULT_DIMENSION)

        # 保存其余 kwargs 以供将来使用
        self._extra_config = kwargs
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """使用 Ollama API 为一组文本生成 Embedding。

        参数:
            texts: 待 Embedding 的文本字符串列表，不能为空。
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F）。
            **kwargs: 其他参数（当前未使用，预留给将来）。

        返回:
            Embedding 向量列表，其中每个向量是一个浮点数列表。

        异常:
            ValueError: 当 texts 列表为空或包含无效条目时抛出。
            OllamaEmbeddingError: 当 API 调用失败时抛出。

        示例:
            >>> embeddings = embedding.embed(["hello", "world"])
            >>> len(embeddings)  # 2 vectors
            >>> len(embeddings[0])  # dimension (e.g., 768)
        """
        # 校验输入
        self.validate_texts(texts)
        
        try:
            import httpx
        except ImportError as e:
            raise OllamaEmbeddingError(
                "httpx library is required for Ollama Embedding. "
                "Install with: pip install httpx"
            ) from e
        
        # 准备 API 请求
        url = f"{self.base_url}/api/embeddings"
        
        embeddings: List[List[float]] = []
        
        # 逐条处理文本（Ollama API 只接受单个 prompt）
        for text in texts:
            payload = {
                "model": self.model,
                "prompt": text,
            }
            
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # 从响应中提取 Embedding
                    if "embedding" not in result:
                        raise OllamaEmbeddingError(
                            f"Unexpected response format from Ollama API. "
                            f"Expected 'embedding' field but got: {list(result.keys())}"
                        )
                    
                    embeddings.append(result["embedding"])
                    
            except httpx.HTTPStatusError as e:
                # HTTP 错误（4xx、5xx）
                raise OllamaEmbeddingError(
                    f"Ollama API request failed with status {e.response.status_code}. "
                    f"Ensure Ollama is running and model '{self.model}' is available."
                ) from e
            except httpx.ConnectError as e:
                # 连接错误（服务器不可达）
                raise OllamaEmbeddingError(
                    f"Failed to connect to Ollama server at {self.base_url}. "
                    f"Ensure Ollama is running (try: ollama serve)"
                ) from e
            except httpx.TimeoutException as e:
                # 请求超时
                raise OllamaEmbeddingError(
                    f"Ollama API request timed out after {self.timeout}s. "
                    f"The model may be loading or the request is too large."
                ) from e
            except httpx.RequestError as e:
                # 其他请求错误
                raise OllamaEmbeddingError(
                    f"Ollama API request failed: {str(e)}"
                ) from e
            except (KeyError, ValueError, TypeError) as e:
                # JSON 解析或数据提取错误
                raise OllamaEmbeddingError(
                    f"Failed to parse Ollama API response: {str(e)}"
                ) from e
        
        return embeddings
    
    def get_dimension(self) -> int:
        """获取该 Provider 产出的 Embedding 维度。

        返回:
            该实例配置的向量维度。

        注意:
            实际维度可能随模型而异。常见维度：
            - nomic-embed-text: 768
            - mxbai-embed-large: 1024
            可通过 settings.embedding.dimensions 配置，默认值为 768。
        """
        return self.dimension
