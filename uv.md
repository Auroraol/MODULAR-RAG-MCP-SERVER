## 初始化项目

`uv init` 命令会创建一个标准的项目结构，包含 `pyproject.toml`、`.python-version` 等文件，并自动初始化 Git 仓库。

```
uv init my-project
cd my-project
```

------

### 1️⃣ pip 命令

```bash
pip install package
pip list
pip freeze
pip uninstall package
```

------

### 2️⃣ uv 命令（语法完全一致）

```bash
uv pip install package
uv pip list
uv pip freeze
uv pip uninstall package
```

## 虚拟环境管理

### 1️⃣ 传统虚拟环境创建流程

```bash
python -m venv .venv      # 创建虚拟环境
source .venv/bin/activate
pip install -r requirements.txt
```

问题在于：

- 创建速度慢
- Python 版本选择需要人工保证
- 虚拟环境体积偏大

------

### 2️⃣ 使用 uv 的方式

```bash
uv venv                   # 创建虚拟环境（更快）
source .venv/bin/activate # Linux激活（完全相同）
.venv\Scripts\activate.bat #  Windows
uv pip install -r requirements.txt
```

