---
name: package
description: "清理并打包项目以便分发。移除 __pycache__、.venv、构建产物、数据缓存、日志、IDE 文件、覆盖率报告，并清理配置中的 API Key。生成最小化、可直接分享的代码库。当用户说 'package'、'clean project'、'clean up'、'打包'、'清理项目'、'清理缓存'、'prepare for distribution'、'remove caches'，或希望交付一份干净的代码副本时使用。"
---

# Package — 打包

一条命令即可清理项目以便分发：移除缓存、密钥和产物。

---

## 流程

```
Dry-run → Confirm → Execute → Verify
```

> **⚠️ 运行脚本前先激活 `.venv`。**
> - **Windows**：`.\.venv\Scripts\Activate.ps1`
> - **macOS/Linux**：`source .venv/bin/activate`

---

## 步骤 1：试运行

展示将要移除的内容，但不实际删除任何东西：

```powershell
python .github/skills/package/scripts/clean.py
```

与用户一起审阅输出。脚本会列出：
- 📁 要移除的目录（含大小）
- 📄 要移除的文件（含大小）
- 🔐 要脱敏的配置值（API Key → 占位符）

## 步骤 2：向用户确认

执行之前，汇总将要删除的内容并请用户确认。提供以下选项：
- `--keep-data` — 保留 `data/` 和 `logs/`（如果用户想保留已摄取的文档则很有用）
- `--no-sanitize` — 跳过 `config/settings.yaml` 中的 API Key 替换

## 步骤 3：执行

```powershell
# 完全清理（移除包括数据在内的一切）
python .github/skills/package/scripts/clean.py --execute

# 保留数据和日志
python .github/skills/package/scripts/clean.py --execute --keep-data

# 跳过密钥脱敏
python .github/skills/package/scripts/clean.py --execute --no-sanitize
```

## 步骤 4：验证

清理后，验证工作区是否干净：

```powershell
# Check no __pycache__ remains
python -c "import pathlib; found=list(pathlib.Path('.').rglob('__pycache__')); print(f'{len(found)} __pycache__ dirs remaining') if found else print('Clean')"

# Check no .venv remains
python -c "import pathlib; print('.venv still exists') if pathlib.Path('.venv').exists() else print('Clean')"

# Check settings.yaml has no real API keys
python -c "
import re, pathlib
text = pathlib.Path('config/settings.yaml').read_text()
keys = re.findall(r'api_key:\s*\"([^\"]+)\"', text)
real = [k for k in keys if k not in ('YOUR_API_KEY_HERE', '')]
print(f'{len(real)} real API key(s) found') if real else print('API keys sanitized')
"
```

向用户报告结果。

---

## 会被移除的内容

| 类别 | 模式 |
|----------|----------|
| Python 缓存 | `__pycache__/`、`*.pyc`、`*.pyo`、`.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/` |
| 虚拟环境 | `.venv/`、`venv/`、`env/` |
| 构建产物 | `build/`、`dist/`、`*.egg-info/`、`.eggs/`、`wheels/` |
| IDE 文件 | `.idea/`、`.vscode/`、`*.swp`、`*.swo` |
| 覆盖率 | `htmlcov/`、`.coverage`、`coverage.xml`、`.tox/`、`.nox/` |
| 数据与日志 | `data/`、`logs/`、`cache/`（使用 `--keep-data` 跳过） |
| 密钥 | `.env`、`.env.local`、`secrets.yaml`、`test_credentials.yaml` |
| 陈旧产物 | `nonexistent_traces.jsonl/` |
| 配置备份 | `settings.yaml.bak`、`settings.yaml.qa_backup` |
| 测试产物 | `.claude/skills/test-skill/`、`test_data/chroma/` |
| 技能缓存 | `.github/skills/auto-coder/.spec_hash`、`.claude/skills/auto-coder/.spec_hash` |

## 会被脱敏的内容（不删除）

- `config/settings.yaml`：`api_key` → `"YOUR_API_KEY_HERE"`，`azure_endpoint` → 占位符
