"""Vision LLM Provider 抽象基类。

本模块定义 Vision 语言模型 Provider 的可插拔接口，
通过配置实现多模态交互（文本 + 图像）以及不同后端（Azure Vision、Ollama Vision 等）之间的无缝切换。

Vision LLM 在标准 LLM 的基础上扩展，可在文本提示之外接受图像输入，
从而支持图像描述、视觉问答以及含嵌入图像的文档理解等任务。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from src.libs.llm.base_llm import ChatResponse, Message


@dataclass
class ImageInput:
    """表示 Vision LLM 的图像输入。

    支持多种输入格式:
    - 文件路径: 待读取并编码的本地图像文件
    - 字节: 原始图像字节（已加载）
    - Base64: 已编码的图像字符串

    属性:
        path: 图像文件路径（若从磁盘加载）。
        data: 原始图像字节（若已加载）。
        base64: Base64 编码的图像字符串（若已编码）。
        mime_type: 图像的 MIME 类型（例如 'image/png'、'image/jpeg'）。
    """
    path: Optional[Union[str, Path]] = None
    data: Optional[bytes] = None
    base64: Optional[str] = None
    mime_type: str = "image/png"
    
    def __post_init__(self) -> None:
        """校验只提供了一种输入格式。"""
        provided_inputs = sum([
            self.path is not None,
            self.data is not None,
            self.base64 is not None,
        ])
        if provided_inputs == 0:
            raise ValueError("Must provide one of: path, data, or base64")
        if provided_inputs > 1:
            raise ValueError("Must provide exactly one of: path, data, or base64")


class BaseVisionLLM(ABC):
    """Vision LLM Provider 的抽象基类。

    Vision LLM 同时接受文本与图像输入，支持多模态理解任务，
    例如图像描述、视觉问答以及含嵌入图像的文档分析。

    所有 Vision LLM 实现都必须继承本类并实现 chat_with_image() 方法。
    以此保证不同 Provider（Azure Vision、Ollama Vision 等）之间接口一致。

    遵循的设计原则:
    - 可插拔: 子类可在不修改上游代码的情况下替换。
    - 可观测: 接受可选的 TraceContext 以集成可观测性。
    - 配置驱动: 实例由工厂根据 settings 创建。
    - 接口隔离: 接口精简，聚焦于多模态输入。
    - 扩展点: 可在子类中添加图像预处理（压缩、格式转换）而无需改动基类接口。
    """
    
    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: Optional[list[Message]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """根据文本提示与图像输入生成响应。

        该方法支持多模态交互：模型可以“看到”图像，
        并对其回答问题或生成描述。

        参数:
            text: 针对图像的文本提示或问题。
            image: 图像输入（路径、字节或 base64）。
            messages: 可选的对话历史上下文。若提供，
                文本 + 图像将作为最新一条 user 消息追加。
            trace: 可选、用于可观测性的 TraceContext（预留给 Stage F）。
            **kwargs: Provider 专属参数（temperature、max_tokens 等）。

        返回:
            包含生成文本与元数据的 ChatResponse。

        异常:
            ValueError: text 为空或图像输入非法时抛出。
            RuntimeError: Vision LLM Provider 调用失败时抛出。

        示例:
            >>> image = ImageInput(path="diagram.png")
            >>> response = vision_llm.chat_with_image(
            ...     text="Describe this diagram",
            ...     image=image
            ... )
            >>> print(response.content)
            "This diagram shows a system architecture with..."
        """
        pass
    
    def validate_text(self, text: str) -> None:
        """校验文本提示。

        参数:
            text: 待校验的文本提示。

        异常:
            ValueError: text 为空或不是字符串时抛出。
        """
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, got {type(text).__name__}")
        if not text or not text.strip():
            raise ValueError("Text prompt cannot be empty")
    
    def validate_image(self, image: ImageInput) -> None:
        """校验图像输入。

        参数:
            image: 待校验的图像输入。

        异常:
            ValueError: image 不是 ImageInput 实例时抛出。
        """
        if not isinstance(image, ImageInput):
            raise ValueError(
                f"Image must be an ImageInput instance, got {type(image).__name__}"
            )
    
    def preprocess_image(
        self,
        image: ImageInput,
        max_size: Optional[tuple[int, int]] = None,
    ) -> ImageInput:
        """在发送给 Vision LLM 之前预处理图像。

        该方法提供图像预处理的扩展点，例如:
        - 缩放以满足 Provider 的尺寸限制
        - 格式转换（例如 PNG 转 JPEG）
        - 压缩以减小负载体积

        默认实现原样返回图像。子类可重写以添加
        Provider 专属的预处理逻辑。

        参数:
            image: 待预处理的输入图像。
            max_size: 可选的最大尺寸（宽、高），单位为像素。

        返回:
            预处理后的 ImageInput（若无须改动则可能返回同一实例）。

        注意:
            预处理应保持幂等——对同一输入多次调用应产生相同输出。
        """
        # 默认：不做预处理
        return image
