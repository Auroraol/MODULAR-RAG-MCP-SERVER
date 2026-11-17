# Dashboard 与 MCP 测试模式参考

> 执行 Dashboard（A–F）或 MCP（J）测试时阅读本文件。

---

## Dashboard 测试 — Streamlit AppTest

Streamlit `AppTest` 以无头模式（无浏览器）渲染页面。每个测试编写并执行一个内联 Python 脚本。

### 基础模板

```python
from streamlit.testing.v1 import AppTest

def page_script():
    from src.observability.dashboard.pages.<PAGE_MODULE> import render
    render()

at = AppTest.from_function(page_script, default_timeout=30)
at.run()
assert not at.exception, f"Exception: {at.exception}"

# Print elements for verification
for attr in ("header", "subheader", "info", "error", "warning", "markdown"):
    for el in getattr(at, attr, []):
        print(f"{attr}: {getattr(el, 'value', '')}")
for m in at.metric:
    print(f"metric: label={m.label}, value={m.value}")
for s in at.selectbox:
    print(f"selectbox: options={s.options}, value={s.value}")
print(f"button count: {len(at.button)}")
print(f"expander count: {len(at.expander)}")
```

页面模块：`overview`、`data_browser`、`ingestion_manager`、`ingestion_traces`、`query_traces`、`evaluation_panel`

### 交互模式

**选择下拉框 + 重新渲染：**
```python
at.selectbox[0].select("test_col")
at.run()
```

**点击按钮 + 重新渲染：**
```python
at.button[0].click()
at.run()
```

**文本输入：**
```python
at.text_input[0].input("search term")
at.run()
```

### 文件上传变通方案

`AppTest` 无法模拟 `st.file_uploader`。使用两步法：
1. 通过 CLI 摄取：`python scripts/ingest.py --path <file> [--collection <name>]`
2. 通过 AppTest 验证：渲染页面，检查文档数量 / chunk 列表

### 数据变更（清空全部 / 删除文档）

直接调用服务，然后通过 AppTest 或 CLI 验证：
```python
from src.observability.dashboard.services.data_service import DataService
svc = DataService()
svc.delete_document(source_path, collection, source_hash)  # or svc.reset_all()
```

### 参考冒烟测试

```
pytest tests/e2e/test_dashboard_smoke.py -v
```

---

## MCP 测试 — 子进程 JSON-RPC

### 方法 1：现有 pytest 测试（覆盖大多数 J-* 用例）

```
pytest tests/e2e/test_mcp_client.py -v         # 7 个 E2E 测试
pytest tests/integration/test_mcp_server.py -v  # 4 个协议测试
```

### 方法 2：内联 JSON-RPC 脚本

```python
import json, subprocess, sys, threading, time

proc = subprocess.Popen(
    [sys.executable, "-m", "src.mcp_server.server"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, encoding="utf-8",
)

messages = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize",
     "params": {"protocolVersion": "2025-06-18",
                "clientInfo": {"name": "qa-test", "version": "1.0"},
                "capabilities": {}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
     "params": {"name": "query_knowledge_hub",
                "arguments": {"query": "What is hybrid search?", "top_k": 5}}},
]

for msg in messages:
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()

responses = []
def reader():
    while True:
        line = proc.stdout.readline()
        if not line: break
        stripped = line.strip()
        if not stripped: continue
        try:
            data = json.loads(stripped)
            if "id" in data and ("result" in data or "error" in data):
                responses.append(data)
        except json.JSONDecodeError: pass

t = threading.Thread(target=reader, daemon=True)
t.start()
time.sleep(30)
proc.terminate()

for r in responses:
    print(json.dumps(r, indent=2, ensure_ascii=False)[:500])
```

### MCP 测试断言

| 测试 ID | 验证内容 | 关键断言 |
|---------|---------------|----------------|
| J-01 | 服务启动 | `initialize` 响应包含 `serverInfo`、`capabilities` |
| J-02 | tools/list | 3 个工具：`query_knowledge_hub`、`list_collections`、`get_document_summary` |
| J-03 | query_knowledge_hub | `content` 数组包含文本块 |
| J-04 | list_collections | 响应包含 collection 名称 |
| J-05 | get_document_summary | 响应包含 title/summary |
| J-06 | 多模态 | 响应包含 `type: "image"` 块 |
| J-07 | 错误 collection | 优雅报错，不崩溃 |
| J-08 | 无效参数 | JSON-RPC 错误或 `isError: true` |
| J-09 | 稳定性 | 5 次连续调用全部成功 |
| J-10 | 引用 | 响应包含 `source_file`、`page`、`score` |
