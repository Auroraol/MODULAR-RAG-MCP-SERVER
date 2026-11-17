"""Azure OpenAI LLM 实现。

本模块提供基于 Azure 的 OpenAI Service API 的 Azure OpenAI LLM 实现。
它处理 Azure 特有的认证与端点配置。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class AzureLLMError(RuntimeError):
    """Azure OpenAI API 调用失败时抛出。"""


class AzureLLM(BaseLLM):
    """Azure OpenAI LLM Provider 实现。

    本类为 Azure 的 OpenAI Service 实现 BaseLLM 接口。
    与标准 OpenAI 相比，Azure 使用不同的认证方式（请求头中的 API key）
    以及不同的端点结构。

    属性:
        api_key: 用于认证的 Azure API key。
        endpoint: Azure OpenAI 端点 URL。
        deployment_name: 模型对应的部署名称。
        api_version: 使用的 API 版本。
        default_temperature: 生成的默认 temperature。
        default_max_tokens: 生成的默认最大 Token 数。

    示例:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = AzureLLM(settings, endpoint='https://my-resource.openai.azure.com')
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """
    
    DEFAULT_API_VERSION = "2024-02-15-preview"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Azure OpenAI LLM Provider。

        参数:
            settings: 包含 LLM 配置的应用设置。
            api_key: 可选的 API key 覆盖值（回退到 settings.llm.api_key 或环境变量）。
            endpoint: 可选的端点覆盖值（回退到 settings.llm.azure_endpoint 或环境变量）。
            deployment_name: 可选的部署名称（默认为 settings.llm.deployment_name 或 model）。
            api_version: 可选的 API 版本覆盖值。
            **kwargs: 额外的配置覆盖项。

        异常:
            ValueError: 缺少必需配置时抛出。
        """
        # 部署名称：显式传入 > settings.deployment_name > settings.model
        self.deployment_name = (
            deployment_name 
            or getattr(settings.llm, 'deployment_name', None) 
            or settings.llm.model
        )
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
        
        # API key：显式传入 > settings > 环境变量
        self.api_key = (
            api_key 
            or getattr(settings.llm, 'api_key', None) 
            or os.environ.get("AZURE_OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not provided. Set in settings.yaml (llm.api_key), "
                "AZURE_OPENAI_API_KEY environment variable, or pass api_key parameter."
            )
        
        # 端点：显式传入 > settings > 环境变量
        self.endpoint = (
            endpoint 
            or getattr(settings.llm, 'azure_endpoint', None) 
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        if not self.endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not provided. Set in settings.yaml (llm.azure_endpoint), "
                "AZURE_OPENAI_ENDPOINT environment variable, or pass endpoint parameter."
            )
        
        # API 版本：显式传入 > settings > 默认值
        self.api_version = (
            api_version 
            or getattr(settings.llm, 'api_version', None) 
            or self.DEFAULT_API_VERSION
        )
        
        # 保存额外的 kwargs 以备后用
        self._extra_config = kwargs
    
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """使用 Azure OpenAI API 生成对话补全。

        参数:
            messages: 对话消息列表。
            trace: 可选、用于可观测性的 TraceContext（预留给 Stage F）。
            **kwargs: 覆盖参数（temperature、max_tokens 等）。

        返回:
            包含生成内容与元数据的 ChatResponse。

        异常:
            ValueError: 消息非法时抛出。
            AzureLLMError: API 调用失败时抛出。
        """
        # 校验输入
        self.validate_messages(messages)
        
        # 准备请求参数
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        deployment = kwargs.get("deployment_name", self.deployment_name)
        
        # 将消息转换为 API 格式
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        # 发起 API 调用
        try:
            response_data = self._call_api(
                messages=api_messages,
                deployment=deployment,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # 解析响应
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage")
            
            return ChatResponse(
                content=content,
                model=response_data.get("model", deployment),
                usage=usage,
                raw_response=response_data,
            )
        except KeyError as e:
            raise AzureLLMError(
                f"[Azure] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, AzureLLMError):
                raise
            raise AzureLLMError(
                f"[Azure] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        deployment: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """向 Azure OpenAI 发起实际的 API 调用。

        该方法被单独拆分出来，便于在测试中轻松 Mock。

        参数:
            messages: API 格式的消息。
            deployment: 部署名称。
            temperature: 生成 temperature。
            max_tokens: 生成的最大 Token 数。

        返回:
            以字典形式返回的原始 API 响应。

        异常:
            AzureLLMError: API 调用失败时抛出。
        """
        import httpx

        # Azure 端点格式：
        # {endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}
        url = (
            f"{self.endpoint.rstrip('/')}/openai/deployments/{deployment}/"
            f"chat/completions?api-version={self.api_version}"
        )
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise AzureLLMError(
                        f"[Azure] API error (HTTP {response.status_code}): {error_detail}"
                    )
                
                return response.json()
        except httpx.TimeoutException as e:
            raise AzureLLMError(
                f"[Azure] Request timed out after 60 seconds"
            ) from e
        except httpx.RequestError as e:
            raise AzureLLMError(
                f"[Azure] Connection failed: {type(e).__name__}: {e}"
            ) from e
    
    def _parse_error_response(self, response: Any) -> str:
        """从 API 响应中解析错误详情。

        参数:
            response: HTTP 响应对象。

        返回:
            可读的错误消息。
        """
        try:
            error_data = response.json()
            if "error" in error_data:
                error = error_data["error"]
                if isinstance(error, dict):
                    return error.get("message", str(error))
                return str(error)
            return response.text
        except Exception:
            return response.text or "Unknown error"
