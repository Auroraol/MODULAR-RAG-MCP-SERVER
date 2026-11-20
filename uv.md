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

依赖的增删改查

```
bash
uv add streamlit              # 增
uv remove streamlit           # 删（同时改 toml + lock + 卸载）
uv add "streamlit>=1.35"      # 改版本约束（直接 add 会覆盖旧约束）
uv lock --upgrade-package streamlit   # 只升级某一个到最新
uv lock --upgrade             # 全部升级到允许范围内的最新
uv tree                       # 看依赖树，排查"谁引入了 onnxruntime"
uv pip list                   # 看已装的包
团队协作 / CI 场景
pyproject.toml 和 uv.lock 都要提交 git。别人拉代码后：

bash
uv sync        # 按 uv.lock 精确还原，保证版本一致
```



## 总结

```
场景	命令
加运行时依赖	uv add 包名
加开发依赖	uv add --dev 包名
删依赖	uv remove 包名
升级指定包	uv lock --upgrade-package 包名
同步环境	uv sync
看依赖树	uv tree
```


日常只用记住uv init、 uv add 和 uv sync 两个命令，其他都是按需查。