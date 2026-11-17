"""
LLM 模块。

本包包含 LLM 客户端的抽象与实现：
- Base LLM 类（纯文本）
- Base Vision LLM 类（多模态：文本 + 图像）
- LLM 工厂
- 各 Provider 实现（OpenAI、Azure、Ollama、DeepSeek）
"""

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.openai_llm import OpenAILLM, OpenAILLMError
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM, OpenAIVisionLLMError
from src.libs.llm.azure_llm import AzureLLM, AzureLLMError
from src.libs.llm.deepseek_llm import DeepSeekLLM, DeepSeekLLMError
from src.libs.llm.ollama_llm import OllamaLLM, OllamaLLMError

# 向工厂注册纯文本 LLM Provider
LLMFactory.register_provider("openai", OpenAILLM)
LLMFactory.register_provider("azure", AzureLLM)
LLMFactory.register_provider("deepseek", DeepSeekLLM)
LLMFactory.register_provider("ollama", OllamaLLM)

# 注意：Vision LLM Provider 将在任务 B9+ 中注册

__all__ = [
    # 基类
    "BaseLLM",
    "BaseVisionLLM",
    # 数据类型
    "ChatResponse",
    "Message",
    "ImageInput",
    # 工厂
    "LLMFactory",
    # 纯文本 LLM 实现
    "OpenAILLM",
    "OpenAILLMError",
    "AzureLLM",
    "AzureLLMError",
    "DeepSeekLLM",
    "DeepSeekLLMError",
    "OllamaLLM",
    "OllamaLLMError",
    # Vision LLM 实现
    "OpenAIVisionLLM",
    "OpenAIVisionLLMError",
]
