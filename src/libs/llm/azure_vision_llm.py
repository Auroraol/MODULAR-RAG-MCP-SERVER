"""Azure OpenAI Vision LLM 实现。

本模块提供用于多模态交互（文本 + 图像）的 Azure OpenAI Vision LLM 实现。
支持 GPT-4o 与 GPT-4-Vision-Preview 模型，用于图像描述、
视觉问答以及文档分析等图像理解任务。
"""

from __future__ import annotations

import base64
import io
import os
import requests
from pathlib import Path
from typing import Any, Optional

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput


class AzureVisionLLMError(RuntimeError):
    """Azure Vision API 调用失败时抛出。"""


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI Vision LLM Provider 实现。

    本类为 Azure 的 OpenAI Vision Service 实现 BaseVisionLLM 接口，
    支持 GPT-4o 与 GPT-4-Vision-Preview 模型。它处理 Azure 特有的认证、
    端点配置以及图像预处理。

    遵循的设计原则:
    - 可插拔: 实现 BaseVisionLLM 以支持 Provider 无缝切换
    - 配置驱动: 所有配置均来自 settings.yaml
    - 可观测: 接受 TraceContext 参数（预留给 Stage F）
    - 优雅报错: 提供清晰、可操作的错误消息
    - 图像预处理: 对超过 max_image_size 的图像自动压缩

    属性:
        api_key: 用于认证的 Azure API key。
        endpoint: Azure OpenAI 端点 URL。
        deployment_name: Vision 模型对应的部署名称。
        api_version: 使用的 API 版本。
        max_image_size: 图像最大边长（像素，默认 2048）。
        default_temperature: 生成的默认 temperature。
        default_max_tokens: 生成的默认最大 Token 数。

    示例:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> vision_llm = AzureVisionLLM(
        ...     settings,
        ...     endpoint='https://my-resource.openai.azure.com',
        ...     deployment_name='gpt-4o'
        ... )
        >>> image = ImageInput(path="diagram.png")
        >>> response = vision_llm.chat_with_image(
        ...     text="Describe this diagram",
        ...     image=image
        ... )
    """
    
    DEFAULT_API_VERSION = "2024-02-15-preview"
    DEFAULT_MAX_IMAGE_SIZE = 2048  # 像素
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        max_image_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Azure OpenAI Vision LLM Provider。

        参数:
            settings: 包含 LLM 配置的应用设置。
            api_key: 可选的 API key 覆盖值（回退到环境变量）。
            endpoint: 可选的端点覆盖值（回退到环境变量）。
            deployment_name: 可选的部署名称（默认为 settings.llm.model）。
            api_version: 可选的 API 版本覆盖值。
            max_image_size: 自动压缩所用的图像最大边长（像素）。
            **kwargs: 额外的配置覆盖项。

        异常:
            ValueError: 缺少必需配置时抛出。
        """
        self.deployment_name = deployment_name or settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
        
        # 检查 vision 配置来源
        vision_settings = getattr(settings, "vision_llm", None)

        # 解析部署名称
        # 优先级：参数 > vision_settings.deployment > vision_settings.model > settings.llm.model
        vision_dep = getattr(vision_settings, "deployment_name", None)
        vision_model = getattr(vision_settings, "model", None)
        self.deployment_name = deployment_name or vision_dep or vision_model or settings.llm.model
        
        # 解析最大图像尺寸
        # 优先级：参数 > vision_settings.max_image_size > 默认值
        vision_max_size = getattr(vision_settings, "max_image_size", None)
        self.max_image_size = max_image_size or vision_max_size or self.DEFAULT_MAX_IMAGE_SIZE

            
        # API Key 解析顺序：
        # 1. 构造函数参数
        # 2. Vision 配置（settings.vision_llm.api_key）
        # 3. 环境变量（AZURE_OPENAI_API_KEY）
        # 4. LLM 配置回退？（settings 对象中通常不存放 API key，但需确认）
        
        self.api_key = api_key 
        if not self.api_key and vision_settings and vision_settings.api_key:
             self.api_key = vision_settings.api_key
        if not self.api_key:
             self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
             
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY "
                "environment variable or pass api_key parameter."
            )
            
        # 端点解析顺序
        self.endpoint = endpoint
        if not self.endpoint and vision_settings and vision_settings.azure_endpoint:
             self.endpoint = vision_settings.azure_endpoint
        if not self.endpoint:
             self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or getattr(settings.llm, "azure_endpoint", None)
             
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint not provided.")
            
        # API 版本解析顺序
        self.api_version = api_version
        if not self.api_version and vision_settings and vision_settings.api_version:
             self.api_version = vision_settings.api_version
        if not self.api_version:
             self.api_version = getattr(settings.llm, "api_version", None) or self.DEFAULT_API_VERSION
        
        # 校验客户端是否已初始化
        # 此处不初始化客户端对象，因为每次调用都使用 request/httpx 或自定义逻辑
        # 但需确保各项 key 已存在。

        
        # 保存额外的 kwargs 以备后用
        self._extra_config = kwargs
    
    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: Optional[list[Message]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """根据文本提示与图像输入生成响应。

        该方法将文本与图像发送给 Azure OpenAI Vision API
        （GPT-4o 或 GPT-4-Vision-Preview），并返回生成的响应。

        参数:
            text: 针对图像的文本提示或问题。
            image: 图像输入（路径、字节或 base64）。
            messages: 可选的对话历史上下文。
            trace: 可选、用于可观测性的 TraceContext（预留给 Stage F）。
            **kwargs: 覆盖参数（temperature、max_tokens 等）。

        返回:
            包含生成文本与元数据的 ChatResponse。

        异常:
            ValueError: text 或图像输入非法时抛出。
            AzureVisionLLMError: API 调用失败时抛出。

        示例:
            >>> image = ImageInput(path="chart.png")
            >>> response = vision_llm.chat_with_image(
            ...     text="What does this chart show?",
            ...     image=image
            ... )
        """
        # 校验输入
        self.validate_text(text)
        self.validate_image(image)
        
        # 预处理图像（按需压缩）
        processed_image = self.preprocess_image(
            image,
            max_size=(self.max_image_size, self.max_image_size)
        )
        
        # 按需将图像转换为 base64
        image_base64 = self._get_image_base64(processed_image)
        
        # 准备请求参数
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        deployment = kwargs.get("deployment_name", self.deployment_name)
        
        # 构建消息列表
        api_messages = []
        if messages:
            # 添加对话历史
            api_messages.extend([{"role": m.role, "content": m.content} for m in messages])
        
        # 添加当前文本 + 图像消息
        current_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{processed_image.mime_type};base64,{image_base64}"
                    }
                }
            ]
        }
        api_messages.append(current_message)
        
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
            raise AzureVisionLLMError(
                f"[Azure Vision] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, AzureVisionLLMError):
                raise
            raise AzureVisionLLMError(
                f"[Azure Vision] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def preprocess_image(
        self,
        image: ImageInput,
        max_size: Optional[tuple[int, int]] = None,
    ) -> ImageInput:
        """在发送给 Azure Vision API 之前预处理图像。

        若图像超过 max_size，则进行压缩，以减小 API 负载体积
        并满足 Azure 的尺寸限制。使用 PIL 进行图像处理。

        该操作是幂等的——对同一输入多次调用
        会产生相同输出。

        参数:
            image: 待预处理的输入图像。
            max_size: 最大尺寸（宽、高），单位为像素。

        返回:
            预处理后的 ImageInput；如有需要则包含压缩后的数据。

        注意:
            若图像已在尺寸限制之内，则原样返回。
        """
        if not max_size:
            return image
        
        try:
            from PIL import Image
        except ImportError:
        # 若 PIL 不可用，则跳过预处理
            return image
        
        # 获取图像字节
        if image.data:
            image_bytes = image.data
        elif image.path:
            image_bytes = Path(image.path).read_bytes()
        elif image.base64:
            # 已编码，跳过预处理
            return image
        else:
            return image
        
        # 加载图像并检查尺寸
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        
        # 检查是否需要压缩
        max_width, max_height = max_size
        if width <= max_width and height <= max_height:
            # 无需压缩
            return image
        
        # 计算保持宽高比的新尺寸
        ratio = min(max_width / width, max_height / height)
        new_size = (int(width * ratio), int(height * ratio))
        
        # 缩放图像
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # 转换为字节
        buffer = io.BytesIO()
        # 尽可能保留原始格式
        img_format = img.format or "PNG"
        img_resized.save(buffer, format=img_format)
        compressed_bytes = buffer.getvalue()
        
        # 返回包含压缩数据的新 ImageInput
        return ImageInput(
            data=compressed_bytes,
            mime_type=image.mime_type
        )
    
    def _get_image_base64(self, image: ImageInput) -> str:
        """将 ImageInput 转换为 base64 字符串。

        参数:
            image: 待转换的图像。

        返回:
            Base64 编码的图像字符串。

        异常:
            AzureVisionLLMError: 图像无法编码时抛出。
        """
        try:
            if image.base64:
                return image.base64
            elif image.data:
                return base64.b64encode(image.data).decode("utf-8")
            elif image.path:
                image_bytes = Path(image.path).read_bytes()
                return base64.b64encode(image_bytes).decode("utf-8")
            else:
                raise ValueError("ImageInput has no valid data source")
        except Exception as e:
            raise AzureVisionLLMError(
                f"[Azure Vision] Failed to encode image: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: list[dict],
        deployment: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """向 Azure OpenAI Vision API 发起 HTTP 请求。

        参数:
            messages: API 格式的消息列表。
            deployment: 使用的部署名称。
            temperature: 生成 temperature。
            max_tokens: 生成的最大 Token 数。

        返回:
            以字典形式返回的 API 响应。

        异常:
            AzureVisionLLMError: API 调用失败时抛出。
        """
        # 清理端点末尾的斜杠
        endpoint = self.endpoint.rstrip("/")

        # 构建 API URL
        url = (
            f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )
        
        # 准备请求负载
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
             error_details = e.response.text if e.response else str(e)
             raise AzureVisionLLMError(
                 f"[Azure Vision] HTTP Error: {e} - Response: {error_details}"
             ) from e
        except requests.exceptions.RequestException as e:
             raise AzureVisionLLMError(
                 f"[Azure Vision] Request failed: {e}"
             ) from e
