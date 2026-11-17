"""LLM Provider 抽象基类。

本模块定义语言模型 Provider 的可插拔接口，
通过配置驱动的实例化，实现不同后端（OpenAI、Azure、Ollama 等）之间的无缝切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """表示对话中的单条消息。

    属性:
        role: 消息发送者的角色（'system'、'user' 或 'assistant'）。
        content: 消息的文本内容。
    """
    role: str
    content: str


@dataclass
class ChatResponse:
    """LLM 对话补全的响应。

    属性:
        content: 生成的文本响应。
        model: 生成该响应的模型标识符。
        usage: 可选的 Token 用量统计（prompt_tokens、completion_tokens、total_tokens）。
        raw_response: 可选、供调试使用的 Provider 原始响应。
    """
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class BaseLLM(ABC):
    """LLM Provider 的抽象基类。

    所有 LLM 实现都必须继承本类并实现 chat() 方法。
    以此保证不同 Provider（OpenAI、Azure、DeepSeek、Ollama 等）之间接口一致。

    遵循的设计原则:
    - 可插拔: 子类可在不修改上游代码的情况下替换。
    - 可观测: 接受可选的 TraceContext 以集成可观测性。
    - 配置驱动: 实例由工厂根据 settings 创建。
    """
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """生成对话补全响应。

        参数:
            messages: 对话消息列表（角色 + 内容）。
            trace: 可选、用于可观测性的 TraceContext（预留给 Stage F）。
            **kwargs: Provider 专属参数（temperature、max_tokens 等）。

        返回:
            包含生成文本与元数据的 ChatResponse。

        异常:
            ValueError: messages 列表为空或格式非法时抛出。
            RuntimeError: LLM Provider 调用失败时抛出。
        """
        pass
    
    def validate_messages(self, messages: List[Message]) -> None:
        """校验消息列表结构。

        参数:
            messages: 待校验的消息列表。

        异常:
            ValueError: 消息列表为空或包含非法角色时抛出。
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        valid_roles = {"system", "user", "assistant"}
        for i, msg in enumerate(messages):
            if not isinstance(msg, Message):
                raise ValueError(f"Message at index {i} is not a Message instance")
            if msg.role not in valid_roles:
                raise ValueError(
                    f"Message at index {i} has invalid role '{msg.role}'. "
                    f"Must be one of: {valid_roles}"
                )
            if not msg.content or not msg.content.strip():
                raise ValueError(f"Message at index {i} has empty content")
