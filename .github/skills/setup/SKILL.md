---
name: setup
description: "交互式项目初始化向导。从干净的代码库开始，引导用户完成 provider 选择（OpenAI/Azure/DeepSeek/Ollama/Qwen/Gemini 等）、API Key 配置、依赖安装、配置生成，并启动 Dashboard。若用户选择了尚未实现的 provider，则按插件架构自动脚手架生成 provider 代码。自动诊断并修复启动失败，最多 3 轮重试。当用户说 'setup'、'set up'、'configure'、'init project'、'初始化'、'环境配置'、'项目配置'、'first run'、'get started'、'quick start'，或希望从零配置并启动项目时使用。"
---

# Setup — 项目初始化

交互式向导：配置 provider → 安装依赖 → 生成配置 → 启动 dashboard → 自动修复问题。

---

## 流程

```
Preflight → Ask User → Generate Config → Install Deps → Validate → Launch → Usage Guide
```

> 自动修复循环：任一步骤失败时，诊断 → 修复 → 重试（≤3 轮）。

---

## 步骤 1：预检

在向用户提出任何问题之前，先验证前置条件：

### 1.1 检查 uv 与 Python 版本

本项目使用 **uv** 作为包与虚拟环境管理器（不直接使用 pip）。

```powershell
uv --version              # 需要 uv >= 0.5
uv python list --only-installed   # 查看 uv 可用的 Python
```

如果 uv 未安装，先引导用户安装：

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装后需重开终端（或 `source $HOME/.local/bin/env`）。若 `uv python list` 中没有任何 >= 3.10 的解释器，提示用户先执行 `uv python install 3.12`。

### 1.2 检查并创建虚拟环境

检查 `.venv` 目录是否已存在。若已存在，跳过创建，直接验证即可。若不存在，则用 uv 创建。

**重要：** 使用 `uv venv` 而不是 `python -m venv --without-pip` + `ensurepip` —— uv 自带 pip 兼容层，无需 bootstrap，在 Windows 上也不会卡住。

```powershell
# Step 1: 检查 .venv 是否存在
Test-Path ".venv"

# Step 2: 若不存在，用 uv 创建（默认取最新可用 Python）
uv venv .venv
# 需要固定版本时：
# uv venv .venv --python 3.12

# Step 3: 激活虚拟环境（可选 —— 后续统一用 `uv run` / `uv pip` 免激活）
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# Step 4: 验证 uv 能识别到该环境
uv pip list --python ".venv/Scripts/python.exe"    # 应为空（全新 venv）
```

如果 `.venv` 已存在，只需运行步骤 4（验证）。

> 💡 **uv 的工作方式**：在项目根目录执行 `uv pip install` / `uv run` 时，uv 会自动发现同级的 `.venv`，无需激活。下面所有命令都按这一约定编写。

> ⚠️ **已知坑：`VIRTUAL_ENV` 环境变量优先级最高**
> 如果当前终端里已存在 `VIRTUAL_ENV`（例如从 PyCharm 启动、或激活了别的项目的 venv），`uv pip` 会**忽略本地 `.venv` 而指向那个环境**，导致依赖装错地方。
> 预检时必须验证：
>
> ```powershell
> echo $env:VIRTUAL_ENV      # PowerShell；bash 用 echo $VIRTUAL_ENV
> ```
>
> 若它指向**本项目之外**的路径，先清空再继续：
>
> ```powershell
> # PowerShell
> Remove-Item Env:VIRTUAL_ENV
> # bash
> unset VIRTUAL_ENV
> ```
>
> 或者全程显式传 `--python`：`uv pip install -e ".[dev]" --python ".venv/Scripts/python.exe"`。

---

## 步骤 2：询问用户配置

使用 `ask_questions` 工具收集 provider 选择。分批提问（每次调用最多 4 个问题）。

### 批次 1：核心 Provider

一次性提出以下问题：

1. **LLM Provider** — 使用哪个 LLM provider？
   - 选项：`OpenAI`、`Azure OpenAI`、`DeepSeek`、`Ollama (local)`、`Qwen (Alibaba Cloud)`、`Gemini (Google)`
   - 推荐：`OpenAI`
   - 内置支持：OpenAI、Azure、DeepSeek、Ollama。其他 provider 需要自动脚手架（见步骤 2.5）。

2. **Embedding Provider** — 使用哪个 embedding provider？
   - 选项：`OpenAI`、`Azure OpenAI`、`Ollama (local)`、`Qwen (Alibaba Cloud)`、`Gemini (Google)`
   - 推荐：`OpenAI`（尽可能与 LLM provider 保持一致）
   - 内置支持：OpenAI、Azure、Ollama。其他 provider 需要自动脚手架（见步骤 2.5）。

3. **Vision** — 是否启用视觉/图片描述？
   - 选项：`Yes`、`No`
   - 推荐：`Yes`

4. **Rerank** — 是否启用重排序？
   - 选项：`No (fastest)`、`Cross-Encoder (local model)`、`LLM-based`
   - 推荐：`No (fastest)`
   - ⚠️ 注意：Cross-Encoder 仅完成了本地代码实现，尚未经过充分测试，可能存在兼容性问题。建议优先选择「不启用」或「LLM 重排序」。

### 批次 2：凭据（基于批次 1 的回答）

根据所选 provider 询问凭据。各 provider 的必填字段参见 [references/provider_profiles.md](references/provider_profiles.md)。

**如果选择 OpenAI：**
- 询问：OpenAI API Key
- 询问：LLM 模型（默认：`gpt-4o`）
- 询问：Embedding 模型（默认：`text-embedding-ada-002`）

**如果选择 Azure OpenAI：**
- 询问：Azure API Key
- 询问：Azure Endpoint URL
- 询问：LLM deployment 名称（默认：`gpt-4o`）
- 询问：Embedding deployment 名称（默认：`text-embedding-ada-002`）

**如果选择 DeepSeek：**
- 询问：DeepSeek API Key
- 单独询问：Embedding provider（DeepSeek 没有 embedding — 必须使用 OpenAI/Ollama）

**如果选择 Ollama：**
- 询问：Ollama base URL（默认：`http://localhost:11434`）
- 询问：LLM 模型名称（默认：`llama3`）
- 询问：Embedding 模型名称（默认：`nomic-embed-text`）
- 验证 Ollama 正在运行：`curl http://localhost:11434/api/tags` 或等价命令

**如果选择 Qwen：**
- 询问：Qwen API Key（DashScope）
- 询问：LLM 模型（默认：`qwen-turbo`）
- 询问：Embedding 模型（默认：`text-embedding-v3`）— 前提是 embedding 也选择 Qwen
- Base URL：`https://dashscope.aliyuncs.com/compatible-mode/v1`（OpenAI 兼容）

**如果选择 Gemini：**
- 询问：Gemini API Key（Google AI Studio）
- 询问：LLM 模型（默认：`gemini-2.0-flash`）
- 询问：Embedding 模型（默认：`text-embedding-004`）— 前提是 embedding 也选择 Gemini
- Base URL：`https://generativelanguage.googleapis.com/v1beta/openai/`（OpenAI 兼容）

### 批次 3：Vision 凭据（如果启用了 vision）

Vision LLM 有**自己独立的配置段**（`vision_llm`），包含单独的 `provider`、`api_key`、`azure_endpoint` 等。不要假设它与主 LLM 共享凭据。

最多询问 2 个问题：

1. **Vision Provider** — 使用哪个 provider 进行视觉/图片描述？
   - 选项：与 LLM provider 列表相同，但默认使用用户选择的 LLM
   - 内置 vision 支持：OpenAI、Azure。其他 provider 需要自动脚手架。

2. **Vision 凭据** — 视 provider 而定：
   - 如果 vision provider == LLM provider：询问“Vision 是否复用相同的 API key/endpoint？”（默认：是）
     - 若为是：将 LLM 凭据复制到 vision 配置
     - 若为否：询问单独的 vision API key / endpoint
   - 如果 vision provider != LLM provider：询问 vision 专用的 API key / endpoint / 模型

各 provider 的 Vision 模型（推荐模型列在最前）：

| Provider | 推荐 | 其他选项 | 说明 |
|----------|------------|---------------|-------|
| OpenAI | `gpt-4o` | `gpt-4o-mini`, `gpt-4-turbo` | gpt-4o-mini 更便宜，适合简单图片描述 |
| Azure | `gpt-4o` | `gpt-4o-mini`, `gpt-4-turbo` | 需要 deployment_name + azure_endpoint |
| Ollama | `llava` | `llava:13b`, `llava:34b`, `llava-llama3`, `bakllava`, `moondream` | moondream 最轻量，llava:34b 质量最高 |
| Qwen | `qwen-vl-max` | `qwen-vl-plus`, `qwen2.5-vl-72b-instruct`, `qwen2.5-vl-7b-instruct` | qwen-vl-plus 性价比高 |
| Gemini | `gemini-2.0-flash` | `gemini-1.5-pro`, `gemini-2.0-flash-lite`, `gemini-1.5-flash` | gemini-1.5-pro 质量最高但较慢 |
| DeepSeek | ❌ 无 Vision 模型 | — | 需选择其他 provider 作为 Vision LLM |

---

## 步骤 2.5：脚手架生成尚未实现的 Provider（如需要）

如果用户选择了尚未内置的 provider（例如 Qwen、Gemini 或任意自定义 provider），在进入配置生成之前，先自动脚手架生成实现。

完整的脚手架流程参见 [references/new_provider_guide.md](references/new_provider_guide.md)。

概要：
1. 在 `src/libs/llm/{name}_llm.py` 中创建 LLM 类（继承 `BaseLLM`）
2. 在 `src/libs/embedding/{name}_embedding.py` 中创建 Embedding 类（继承 `BaseEmbedding`）— 如需要
3. 在 `src/libs/llm/{name}_vision_llm.py` 中创建 Vision LLM 类（继承 `BaseVisionLLM`）— 如需要
4. 在 `src/libs/llm/__init__.py` 和 `src/libs/embedding/__init__.py` 中注册
5. 如果 provider 使用自定义 endpoint，向 `LLMSettings` / `EmbeddingSettings` 添加 `base_url` 字段
6. 如需要，安装 provider SDK：`uv add <sdk>`（若 `pyproject.toml` 已包含该依赖则用 `uv sync`）
7. 将 provider profile 添加到 `references/provider_profiles.md`

许多 provider（Qwen、Gemini、Groq、Mistral 等）都兼容 OpenAI — 只需继承 `OpenAILLM` / `OpenAIEmbedding`，并覆盖 `DEFAULT_BASE_URL` + 认证逻辑。

---

## 步骤 3：生成配置

从 [references/settings_template.yaml](references/settings_template.yaml) 读取模板，并根据用户回答填入值。

关键规则：
- 从 [references/provider_profiles.md](references/provider_profiles.md) 的模型→维度表中查找 `dimensions`
- 对于 Ollama：设置 `base_url`，将 `api_key`/`azure_endpoint`/`deployment_name` 留空
- 对于 OpenAI：将 `azure_endpoint`/`deployment_name`/`api_version` 留空
- 如果禁用了 vision：设置 `vision_llm.enabled: false`
- 对于 rerank：相应设置 `enabled`、`provider` 和 `model`

将生成的配置写入 `config/settings.yaml`。

同时确保所需目录存在：

```powershell
uv run python -c "from pathlib import Path; [Path(d).mkdir(parents=True, exist_ok=True) for d in ['data/db/chroma', 'data/images/default', 'logs', 'config/prompts']]"
```

---

## 步骤 4：安装依赖

**推荐方式（可复现）：** 若已提交 `uv.lock`，直接同步：

```powershell
uv sync --extra dev
```

**首次 / 无 lock 文件时：** 以可编辑模式安装项目及 dev 依赖：

```powershell
uv pip install -e ".[dev]"
```

> ⚠️ 若 `uv pip list` 显示出的是别的项目的包（见 1.2 的 `VIRTUAL_ENV` 坑），加上 `--python ".venv/Scripts/python.exe"` 强制指定目标环境。

> 说明：`uv pip install` 是 pip 的兼容子命令，行为与 `pip install` 一致但快得多；`uv add` 则会同时写入 `pyproject.toml` 并更新 `uv.lock`。修改依赖定义时优先用 `uv add`。

如果特定 provider 需要额外软件包，**添加到项目依赖**（推荐，会写回 `pyproject.toml`）：

```powershell
uv add sentence-transformers     # Cross-Encoder rerank
uv add streamlit                 # Streamlit dashboard
uv add openai                    # OpenAI SDK
```

若只需临时装入当前环境（不写回 `pyproject.toml`）：

```powershell
uv pip install sentence-transformers streamlit openai
```

验证关键导入：

```powershell
uv run python -c "import chromadb; import mcp; import yaml; print('Core deps OK')"
uv run python -c "import streamlit; print('Streamlit OK')"
uv run python -c "import openai; print('OpenAI SDK OK')"
```

---

## 步骤 5：验证配置

测试配置能否正确加载：

```powershell
uv run python -c "from src.core.settings import load_settings; s = load_settings(); print(f'Config OK: LLM={s.llm.provider}/{s.llm.model}, Embed={s.embedding.provider}/{s.embedding.model}')"
```

如果失败，进入**自动修复循环**：

### 自动修复循环（≤3 轮）

```
Round 0..2:
  Read error message
  Diagnose root cause (missing field, wrong type, bad provider name, etc.)
  Fix config/settings.yaml or install missing dependency
  Re-validate
  If pass → continue to Step 6
  If fail → next round
```

常见修复：
- `SettingsError: Missing required field` → 将该字段添加到 settings.yaml
- `ModuleNotFoundError` → `uv add <package>`（或 `uv sync` 拉齐依赖）
- `Connection refused`（Ollama）→ 告知用户启动 Ollama 服务
- `dimensions` 值错误 → 从 provider_profiles.md 查找正确值
- `No solution found` / 版本冲突 → 放宽 `pyproject.toml` 中的版本约束后重新 `uv sync`

如果 3 轮均失败，向用户报告问题并附上诊断，请求协助。

---

## 步骤 6：启动 Dashboard

```powershell
uv run python scripts/start_dashboard.py --port 8501
```

以**后台进程**运行。等待几秒，然后验证其可访问：

```powershell
uv run python -c "
import urllib.request
try:
    r = urllib.request.urlopen('http://localhost:8501/_stcore/health')
    print('Dashboard is running!' if r.status == 200 else f'Status: {r.status}')
except Exception as e:
    print(f'Dashboard not yet ready: {e}')
"
```

如果 dashboard 启动失败，进入自动修复循环：
- 读取后台终端的错误输出
- 常见问题：缺少 `streamlit`、端口已被占用、导入错误
- 修复并重试

---

## 步骤 7：使用指南

成功启动后，向用户展示以下内容：

```
🎉 Setup Complete!

Dashboard: http://localhost:8501

Quick Start:
  1. Ingest documents:  uv run python scripts/ingest.py <path-to-pdf-or-folder>
  2. Query:             uv run python scripts/query.py "your question here"
  3. Dashboard:         uv run python scripts/start_dashboard.py
  4. MCP Server:        uv run python main.py

Configuration: config/settings.yaml
Logs:          logs/traces.jsonl

Provider: {provider} / Model: {model}
```

根据用户选择的 provider 和语言调整消息内容（若用户使用中文交流，则用中文）。
