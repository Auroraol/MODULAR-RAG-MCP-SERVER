# 新 Provider 脚手架指南

当用户选择了尚未实现的 provider 时，按照本流程自动脚手架生成代码。本项目采用插件架构 — 新增 provider 是一个机械性过程。

## 快速检查：Provider 是否已内置？

内置 LLM provider：`openai`、`azure`、`deepseek`、`ollama`
内置 Embedding provider：`openai`、`azure`、`ollama`
内置 Vision provider：`openai`、`azure`

如果 provider 不在上述列表中，请继续执行下方的脚手架流程。

## 关键洞察：OpenAI 兼容的 Provider

许多现代 provider（Qwen/DashScope、Gemini、Groq、Mistral、Together AI 等）都提供 **OpenAI 兼容 API**。对于这些 provider，实现非常简单 — 继承现有的 `OpenAILLM` / `OpenAIEmbedding`，并覆盖 base URL + 认证即可。

## 步骤 1：创建 LLM Provider

创建 `src/libs/llm/{name}_llm.py`：

```python
"""<ProviderName> LLM implementation (OpenAI-compatible)."""
from __future__ import annotations
import os
from typing import Any, Optional
from src.libs.llm.openai_llm import OpenAILLM


class <ProviderName>LLMError(RuntimeError):
    """Raised when <ProviderName> API call fails."""


class <ProviderName>LLM(OpenAILLM):
    """<ProviderName> LLM provider — OpenAI-compatible endpoint.

    Inherits all chat logic from OpenAILLM; overrides base URL and
    API key resolution.
    """

    DEFAULT_BASE_URL = "<PROVIDER_BASE_URL>"

    def __init__(self, settings: Any, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, **kwargs: Any) -> None:
        # Allow base_url from settings.llm.base_url
        resolved_base = (
            base_url
            or getattr(settings.llm, 'base_url', None)
            or self.DEFAULT_BASE_URL
        )
        # Allow api_key from settings or env
        resolved_key = (
            api_key
            or getattr(settings.llm, 'api_key', None)
            or os.environ.get("<ENV_VAR_NAME>")
        )
        if not resolved_key:
            raise ValueError(
                "<ProviderName> API key not provided. Set in settings.yaml "
                "(llm.api_key) or <ENV_VAR_NAME> environment variable."
            )
        super().__init__(settings, api_key=resolved_key, base_url=resolved_base, **kwargs)
```

对于不兼容 OpenAI 的 provider，直接继承 `BaseLLM` 并实现 `chat()` — 可参考 `deepseek_llm.py` 或 `ollama_llm.py` 作为示例。

## 步骤 2：创建 Embedding Provider（如需要）

创建 `src/libs/embedding/{name}_embedding.py`：

```python
"""<ProviderName> Embedding implementation (OpenAI-compatible)."""
from __future__ import annotations
import os
from typing import Any, Optional
from src.libs.embedding.openai_embedding import OpenAIEmbedding


class <ProviderName>EmbeddingError(RuntimeError):
    """Raised when <ProviderName> Embedding API call fails."""


class <ProviderName>Embedding(OpenAIEmbedding):
    """<ProviderName> Embedding provider — OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "<PROVIDER_BASE_URL>"

    def __init__(self, settings: Any, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, **kwargs: Any) -> None:
        resolved_base = (
            base_url
            or getattr(settings.embedding, 'base_url', None)
            or self.DEFAULT_BASE_URL
        )
        resolved_key = (
            api_key
            or getattr(settings.embedding, 'api_key', None)
            or os.environ.get("<ENV_VAR_NAME>")
        )
        if not resolved_key:
            raise ValueError(
                "<ProviderName> API key not provided. Set in settings.yaml "
                "(embedding.api_key) or <ENV_VAR_NAME> environment variable."
            )
        super().__init__(settings, api_key=resolved_key, base_url=resolved_base, **kwargs)
```

## 步骤 3：创建 Vision LLM Provider（如需要）

创建 `src/libs/llm/{name}_vision_llm.py` — 继承 `OpenAIVisionLLM`，模式与步骤 1 相同。

## 步骤 4：注册 Provider

### LLM — 追加到 `src/libs/llm/__init__.py`：

```python
from src.libs.llm.{name}_llm import <ProviderName>LLM, <ProviderName>LLMError
LLMFactory.register_provider("{name}", <ProviderName>LLM)
```

### Embedding — 追加到 `src/libs/embedding/embedding_factory.py` 中的 `_register_builtin_providers()`：

```python
try:
    from src.libs.embedding.{name}_embedding import <ProviderName>Embedding
    EmbeddingFactory.register_provider("{name}", <ProviderName>Embedding)
except ImportError:
    pass
```

同时更新 `src/libs/embedding/__init__.py` 中的导入和 `__all__`。

### Vision — 追加到 `src/libs/llm/llm_factory.py` 中的 `_register_vision_providers()`：

```python
try:
    from src.libs.llm.{name}_vision_llm import <ProviderName>VisionLLM
    LLMFactory.register_vision_provider("{name}", <ProviderName>VisionLLM)
except ImportError:
    pass
```

## 步骤 5：确认 Settings 中存在 `base_url` 字段

检查 `src/core/settings.py` — `LLMSettings` 和 `EmbeddingSettings` 已包含 `base_url: Optional[str] = None`。无需修改。

## 步骤 6：安装 SDK（如有）

大多数 OpenAI 兼容 provider 只需 `uv add openai`（已在 `pyproject.toml` 的 dev 依赖中）。
如果不兼容 OpenAI，则需要 provider 专用 SDK：
- Qwen：`uv add dashscope`（备选，但推荐使用 OpenAI 兼容模式）
- Gemini：`uv add google-generativeai`（仅在未使用 OpenAI 兼容模式时需要）

> 若只需临时装入当前 `.venv` 而不写回 `pyproject.toml`，用 `uv pip install <sdk>`。

## Provider 专用参考

| Provider | Base URL (LLM) | Base URL (Embedding) | API Key 环境变量 | 兼容 OpenAI |
|----------|----------------|----------------------|-----------------|---------------|
| Qwen     | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 同上 | `DASHSCOPE_API_KEY` | 是 |
| Gemini   | `https://generativelanguage.googleapis.com/v1beta/openai/` | 同上 | `GEMINI_API_KEY` | 是 |
| Groq     | `https://api.groq.com/openai/v1` | 不适用（无 embedding） | `GROQ_API_KEY` | 是 |
| Mistral  | `https://api.mistral.ai/v1` | 同上 | `MISTRAL_API_KEY` | 是 |
| Together | `https://api.together.xyz/v1` | 同上 | `TOGETHER_API_KEY` | 是 |

## 验证

完成脚手架后，运行快速验证：

```python
uv run python -c "
from src.libs.llm import LLMFactory
from src.libs.embedding import EmbeddingFactory
print('LLM providers:', LLMFactory.list_providers())
print('Embedding providers:', EmbeddingFactory.list_providers())
print('Vision providers:', LLMFactory.list_vision_providers())
"
```

新 provider 应出现在列表中。然后继续执行 setup 的步骤 3（生成配置）。
