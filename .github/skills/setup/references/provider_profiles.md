# Provider 配置文件参考

受支持 provider 配置的快速参考。

## LLM Provider 列表

### OpenAI
```yaml
llm:
  provider: "openai"
  model: "gpt-4o"           # 或 gpt-4o-mini、gpt-3.5-turbo
  api_key: "<OPENAI_API_KEY>"
  temperature: 0.0
  max_tokens: 4096
```
必填字段：`provider`、`model`、`api_key`
删除或留空：`azure_endpoint`、`deployment_name`、`api_version`

### Azure OpenAI
```yaml
llm:
  provider: "azure"
  model: "gpt-4o"
  deployment_name: "<YOUR_DEPLOYMENT>"
  azure_endpoint: "https://<RESOURCE>.openai.azure.com/"
  api_version: "2024-02-15-preview"
  api_key: "<AZURE_API_KEY>"
  temperature: 0.0
  max_tokens: 4096
```
必填字段：如上所示全部字段

### DeepSeek
```yaml
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  api_key: "<DEEPSEEK_API_KEY>"
  temperature: 0.0
  max_tokens: 4096
```

### Ollama（本地）
```yaml
llm:
  provider: "ollama"
  model: "llama3"            # 或任何通过 `ollama pull` 拉取的模型
  base_url: "http://localhost:11434"
  temperature: 0.0
  max_tokens: 4096
```
无需 API Key。

## Embedding Provider 列表

### OpenAI
```yaml
embedding:
  provider: "openai"
  model: "text-embedding-ada-002"   # 或 text-embedding-3-small
  dimensions: 1536                   # ada-002 为 1536，3-small 为 1536
  api_key: "<OPENAI_API_KEY>"
```

### Azure OpenAI
```yaml
embedding:
  provider: "azure"
  model: "text-embedding-ada-002"
  dimensions: 1536
  deployment_name: "<YOUR_EMBEDDING_DEPLOYMENT>"
  azure_endpoint: "https://<RESOURCE>.openai.azure.com/"
  api_version: "2024-02-15-preview"
  api_key: "<AZURE_API_KEY>"
```

### Ollama
```yaml
embedding:
  provider: "ollama"
  model: "nomic-embed-text"
  dimensions: 768
  base_url: "http://localhost:11434"
```

## 尚未实现的 Provider（选择时自动脚手架）

> 这些 provider 使用 OpenAI 兼容 API。被选中时，setup 技能
> 通过继承 `OpenAILLM` / `OpenAIEmbedding` 自动脚手架生成 provider 代码。

### Qwen（阿里云 DashScope）
```yaml
llm:
  provider: "qwen"
  model: "qwen-turbo"         # 或 qwen-plus、qwen-max
  api_key: "<DASHSCOPE_API_KEY>"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  temperature: 0.0
  max_tokens: 4096
```
Embedding：
```yaml
embedding:
  provider: "qwen"
  model: "text-embedding-v3"
  dimensions: 1024
  api_key: "<DASHSCOPE_API_KEY>"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```
Vision 模型：`qwen-vl-max`
SDK：`pip install openai`（使用 OpenAI 兼容协议）

### Gemini（Google AI Studio）
```yaml
llm:
  provider: "gemini"
  model: "gemini-2.0-flash"   # 或 gemini-1.5-pro、gemini-2.0-flash-lite
  api_key: "<GEMINI_API_KEY>"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  temperature: 0.0
  max_tokens: 4096
```
Embedding：
```yaml
embedding:
  provider: "gemini"
  model: "text-embedding-004"
  dimensions: 768
  api_key: "<GEMINI_API_KEY>"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
```
Vision 模型：`gemini-2.0-flash`
SDK：`pip install openai`（使用 OpenAI 兼容协议）

## 模型 → 维度对照

| 模型                            | 维度       |
|---------------------------------|------------|
| text-embedding-ada-002          | 1536       |
| text-embedding-3-small          | 1536       |
| text-embedding-3-large          | 3072       |
| nomic-embed-text (Ollama)       | 768        |
| mxbai-embed-large (Ollama)      | 1024       |
| text-embedding-v3 (Qwen)        | 1024       |
| text-embedding-004 (Gemini)     | 768        |

## Vision LLM Provider 列表

Vision 使用**独立的配置段**（`vision_llm`），拥有自己的 `provider`、`api_key` 等。

### Vision 模型参考表

| Provider | 模型 | 质量 | 速度 | 成本 | 说明 |
|----------|-------|---------|-------|------|-------|
| OpenAI | `gpt-4o` | ⭐⭐⭐⭐⭐ | 快 | $$ | 推荐，综合最优 |
| OpenAI | `gpt-4o-mini` | ⭐⭐⭐⭐ | 很快 | $ | 性价比高，适合简单图片 |
| OpenAI | `gpt-4-turbo` | ⭐⭐⭐⭐⭐ | 中 | $$$ | 老牌高质量，成本较高 |
| Azure | `gpt-4o` | ⭐⭐⭐⭐⭐ | 快 | $$ | 同 OpenAI，需 Azure 部署 |
| Azure | `gpt-4o-mini` | ⭐⭐⭐⭐ | 很快 | $ | 同上，轻量版 |
| Ollama | `llava` (7B) | ⭐⭐⭐ | 中 | 免费 | 本地部署，无需 API Key |
| Ollama | `llava:13b` | ⭐⭐⭐⭐ | 慢 | 免费 | 质量更好，需更多显存 |
| Ollama | `llava:34b` | ⭐⭐⭐⭐⭐ | 很慢 | 免费 | 本地最高质量，需 24GB+ 显存 |
| Ollama | `llava-llama3` | ⭐⭐⭐⭐ | 中 | 免费 | 基于 LLaMA3，综合较好 |
| Ollama | `bakllava` | ⭐⭐⭐ | 中 | 免费 | BakLLaVA，轻量替代 |
| Ollama | `moondream` | ⭐⭐ | 很快 | 免费 | 最轻量（1.6B），资源占用极少 |
| Qwen | `qwen-vl-max` | ⭐⭐⭐⭐⭐ | 中 | $$ | 通义千问视觉旗舰 |
| Qwen | `qwen-vl-plus` | ⭐⭐⭐⭐ | 快 | $ | 性价比高 |
| Qwen | `qwen2.5-vl-72b-instruct` | ⭐⭐⭐⭐⭐ | 慢 | $$ | 最新高质量模型 |
| Qwen | `qwen2.5-vl-7b-instruct` | ⭐⭐⭐ | 快 | $ | 轻量版 |
| Gemini | `gemini-2.0-flash` | ⭐⭐⭐⭐⭐ | 很快 | $ | 推荐，速度快质量高 |
| Gemini | `gemini-1.5-pro` | ⭐⭐⭐⭐⭐ | 慢 | $$$ | 最高质量，适合复杂图片 |
| Gemini | `gemini-2.0-flash-lite` | ⭐⭐⭐ | 很快 | ¢ | 极低成本 |
| Gemini | `gemini-1.5-flash` | ⭐⭐⭐⭐ | 快 | $ | 均衡之选 |
| DeepSeek | — | — | — | — | ❌ 不支持 Vision，需选其他 provider |

### OpenAI Vision
```yaml
vision_llm:
  enabled: true
  provider: "openai"
  model: "gpt-4o"  # 或：gpt-4o-mini、gpt-4-turbo
  api_key: "<OPENAI_API_KEY>"
  max_image_size: 2048
```

### Azure Vision
```yaml
vision_llm:
  enabled: true
  provider: "azure"
  model: "gpt-4o"  # 或：gpt-4o-mini、gpt-4-turbo
  deployment_name: "<YOUR_VISION_DEPLOYMENT>"
  azure_endpoint: "https://<RESOURCE>.openai.azure.com/"
  api_version: "2024-02-15-preview"
  api_key: "<AZURE_API_KEY>"
  max_image_size: 2048
```

### Ollama Vision
```yaml
vision_llm:
  enabled: true
  provider: "ollama"
  model: "llava"  # 或：llava:13b、llava:34b、llava-llama3、bakllava、moondream
  base_url: "http://localhost:11434"
  max_image_size: 2048
```

### Qwen Vision
```yaml
vision_llm:
  enabled: true
  provider: "qwen"
  model: "qwen-vl-max"  # 或：qwen-vl-plus、qwen2.5-vl-72b-instruct、qwen2.5-vl-7b-instruct
  api_key: "<DASHSCOPE_API_KEY>"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  max_image_size: 2048
```

### Gemini Vision
```yaml
vision_llm:
  enabled: true
  provider: "gemini"
  model: "gemini-2.0-flash"  # 或：gemini-1.5-pro、gemini-2.0-flash-lite、gemini-1.5-flash
  api_key: "<GEMINI_API_KEY>"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  max_image_size: 2048
```

### 禁用 Vision
```yaml
vision_llm:
  enabled: false
  provider: "openai"
  model: "gpt-4o"
  max_image_size: 2048
```

## Rerank Provider 列表

### None（禁用）
```yaml
rerank:
  enabled: false
  provider: "none"
  model: ""
  top_k: 5
```

### Cross-Encoder

> ⚠️ **注意**：Cross-Encoder 仅完成了本地代码实现，尚未经过充分测试，可能存在兼容性问题。建议优先选择「No (disabled)」或「LLM-based」。

```yaml
rerank:
  enabled: true
  provider: "cross_encoder"
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k: 5
```

### 基于 LLM
```yaml
rerank:
  enabled: true
  provider: "llm"
  model: ""  # 使用已配置的 LLM
  top_k: 5
```
