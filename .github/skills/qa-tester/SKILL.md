---
name: qa-tester
description: "面向 Modular RAG MCP Server 的全自动 QA 测试 agent。从 QA_TEST_PLAN.md 读取测试用例，无需人工干预即可自动执行全部测试类型 — CLI 命令、通过 Streamlit AppTest 无头渲染的 Dashboard UI、通过子进程 JSON-RPC 的 MCP 协议、provider 切换以及数据生命周期检查。诊断失败，最多进行 3 轮重试修复，并将结果记录到 QA_TEST_PROGRESS.md。当用户说 'run QA'、'QA test'、'QA 测试'、'执行测试'、'跑测试'、'test and fix'，或希望执行 QA 测试计划时使用。"
---

# QA 测试器

所有测试类型（CLI / Dashboard UI / MCP 协议）均**完全自动化** — 零人工干预。

可选修饰符：附加章节字母（`run QA G`）或测试 ID（`run QA G-01`）。

---

> ## ⛔ 铁律
>
> ### 规则 1：严格串行
> 选择一个测试 → 运行一条命令 → 等待输出 → 在 `.github/skills/qa-tester/QA_TEST_PROGRESS.md` 中记录一行 → 然后才选择下一个。
> 绝不在一条命令中运行两个测试。绝不在一次文件编辑中记录两行。绝不使用并行工具调用。绝不在记录当前测试之前规划下一个测试。
>
> ### 规则 2：PASS = 终端输出证据
> ✅ 意味着你在**本次会话中**运行了命令，且 Note 中包含从该输出中复制而来的**具体值**。
> 本次测试没有终端输出 → 标记为 ⬜。绝不标记 ✅。
>
> ### 规则 3：零交叉引用
> 绝不写 "Verified via X-YZ"、"Already verified in C-02"、"Same as…"、"Similar to…"。
> 即使 G-05 测试的是与 C-02 相同的 flag，也要独立运行 G-05 并粘贴它自己的输出。
>
> ### 规则 4：零推断
> **禁止的 Note 模式**（会被校验器捕获）：
> "Code uses…" / "Dataclass validates…" / "auto-creates…" → 阅读代码 ≠ 测试。
> "Should work…" / "Would raise…" / "Expected behavior…" → 推测 ≠ 测试。
> "Parameter accepted" / "Config controls behavior" → 含糊。没有输出就没有通过。
> 如果你没有为本次测试运行命令并看到输出，请标记为 ⬜。
>
> ### 规则 5：对抗性思维
> 寻找 bug，而不是确认。10 多次通过且零 bug → 重新审视你的严谨程度。
>
> ### 规则 6：章节末验证
> 完成一个章节后，运行 `python .github/skills/qa-tester/scripts/qa_validate_notes.py`。
> 在进入下一章节之前，重新执行任何被标记的测试。

---

## 流程（严格串行）

```
1. Pick ONE pending test (by ID order)
2. Set system state if needed
3. Run ONE command — WAIT for output
4. Verify ALL assertions from Expected Result vs ACTUAL output
5. Fix if needed (≤3 rounds)
6. ⛔ GATE: Edit `.github/skills/qa-tester/QA_TEST_PROGRESS.md` (row + counters) — ONE row per edit
7. Only NOW return to step 1
```

> 在任何 `python` 命令前先激活 `.venv`：`.\.venv\Scripts\Activate.ps1`

---

## 步骤 1：选择目标

1. 读取 `.github/skills/qa-tester/QA_TEST_PLAN.md` 获取测试步骤和预期结果。
2. 读取 `.github/skills/qa-tester/QA_TEST_PROGRESS.md` 获取当前状态。
3. 用户指定了章节/ID → 限定到该范围。否则 → 第一个 ⬜ 待测测试。
4. 如果存在任何 🔧 测试，先重测它们。
5. 按章节顺序（A→O）执行，章节内按 ID 顺序执行。

### 测试类别

| 章节 | 类型 | 执行方式 |
|----------|------|-----------------|
| A–F | Dashboard UI | AppTest 无头 — 参见 [references/test_patterns.md](references/test_patterns.md) |
| G、H、I | CLI | 终端命令，检查退出码 + stdout |
| J | MCP 协议 | JSON-RPC 子进程 — 参见 [references/test_patterns.md](references/test_patterns.md) |
| K、L | Provider 切换 | `qa_config.py apply <profile>` → 运行 CLI/Dashboard |
| M | 配置与容错 | 修改 settings → 运行 CLI → 验证错误处理 |
| N、O | 数据生命周期 | 使用 `qa_multistep.py <TEST_ID>` |

---

## 步骤 2：设置系统状态

```
Empty          → python .github/skills/qa-tester/scripts/qa_bootstrap.py clear
Baseline       → python .github/skills/qa-tester/scripts/qa_bootstrap.py baseline
DeepSeek       → python .github/skills/qa-tester/scripts/qa_config.py apply deepseek
Rerank_LLM     → python .github/skills/qa-tester/scripts/qa_config.py apply rerank_llm
NoVision       → python .github/skills/qa-tester/scripts/qa_config.py apply no_vision
InvalidKey     → python .github/skills/qa-tester/scripts/qa_config.py apply invalid_llm_key
InvalidEmbedKey→ python .github/skills/qa-tester/scripts/qa_config.py apply invalid_embed_key
Any            → no state change needed
```

配置档测试完成后 → `python .github/skills/qa-tester/scripts/qa_config.py restore`
检查状态 → `python .github/skills/qa-tester/scripts/qa_bootstrap.py status`

---

## 步骤 3：执行与验证

### CLI 测试（G、H、I，以及 K/L/M 的部分）

1. 从 `.github/skills/qa-tester/QA_TEST_PLAN.md` 读取测试的 **Steps** 列。
2. 在终端中运行完全一致的命令。
3. 将输出与**预期结果**对比：
   - **Ingest**：exit=0，输出包含各阶段名称（load/split/transform/embed/upsert）
   - **Query**：exit=0，结果包含 source_file 和 score
   - **Error**：exit≠0，stderr 包含描述性错误（而非原始堆栈跟踪）
   - **幂等性**：第二次运行显示 "skipped"，没有重复项

### Dashboard 测试（A–F）

阅读 [references/test_patterns.md](references/test_patterns.md) 了解 AppTest 模板、交互模式和文件上传的变通方案。

要点：
- 每个测试编写一个内联 Python 脚本，通过 `AppTest.from_function` 渲染
- 打印具体的元素值（metric 标签、selectbox 选项、expander 标签）
- 对于文件上传：先通过 CLI 摄取，再通过 AppTest 验证
- 对于数据变更：直接调用 `DataService`，通过 AppTest/CLI 验证

#### ⚠️ AppTest `st.tabs()` 限制 — 测试任何标签页内容前必读

AppTest 会**同时渲染所有标签页内容**为一个扁平的元素列表。标签页之间**没有空间隔离** — 你无法从 `at.info`、`at.metric`、`at.markdown` 等判断某个元素属于哪个标签页。

**后果：** 如果测试步骤说“点击标签页 X → 在标签页 X 内看到内容 Y”，AppTest **无法确认这种空间关系**。即使输出中存在内容 Y，它也可能来自不同的区域（例如标签页上方的诊断区域，或另一个标签页）。

**所有标签页内容测试的强制检查：**

1. **首先验证标签页标签是否存在：** 打印 `[t.label for t in at.tabs]`（如果 `at.tabs` 可用）或检查页面源码中是否存在该标签页标签字符串。如果预期的标签页标签**不存在** → 立即标记 ❌ — 该标签页不存在。
2. **然后验证内容元素存在** 于 AppTest 输出中。
3. **在 Note 中说明该限制：** 在 Note 后附加 `[AppTest: tab isolation unverifiable]`。

**示例 — 标签页不存在：**
```python
# For a test expecting "🟪 Rerank" tab with "skipped" message:
tab_labels = [el.value for el in at.markdown if '🟪' in el.value or 'Rerank' in el.value]
print('Rerank tab label found:', tab_labels)
# If empty → tab absent → mark ❌ with note "Rerank tab not rendered (stage absent)"
```

**禁止的模式：** 找到一个内容元素（例如 "Rerank skipped" 信息）就断定标签页测试通过 — 该元素可能来自诊断区域，而不是标签页内部。

### MCP 测试（J）

阅读 [references/test_patterns.md](references/test_patterns.md) 了解 JSON-RPC 脚本模板和断言矩阵。

主要方法：`pytest tests/e2e/test_mcp_client.py -v` 覆盖大多数 J-* 用例。

### 多步骤测试（N、O、M 配置、L-07）

包含 3 个及以上连续步骤的测试**必须**使用运行器脚本：

```
python .github/skills/qa-tester/scripts/qa_multistep.py <TEST_ID>
```

**支持：** `N-01`、`N-03`、`N-04`、`N-05`、`N-06`、`O-07`、`M-03`、`M-04`、`M-05`、`M-06`、`M-10`、`M-11`、`L-07`

脚本会执行每个子步骤，打印每一步的实际值，并输出 `VERDICT: PASS/FAIL`。将 VERDICT 和关键步骤值复制到 Note 中。

对于不在脚本中的测试，手动运行 `.github/skills/qa-tester/QA_TEST_PLAN.md` 中的命令并粘贴输出。

---

## 步骤 4：修复与重试（≤3 轮）

1. **诊断**：代码 bug / 配置问题 / 缺少数据 / 测试计划错误？
2. **修复**：仅做最小改动。在 Note 中记录文件/行号。
3. **重试**：重新运行同一命令。
4. 3 轮失败后 → 标记 ❌ 并附上详细说明。
5. 如果修复触及共享代码 → 重新运行同章节中之前通过的测试。

---

## 步骤 5：记录结果

**⛔ 门禁 — 在选择下一个测试之前完成此步骤。**

编辑 `.github/skills/qa-tester/QA_TEST_PROGRESS.md`：更新一行测试记录 + 汇总计数器。每次文件编辑只更新一行。

### ✅ 通过要求

全部满足才可：
1. **在本次会话中**运行了命令
2. 观察到**来自该命令**的实际输出
3. 验证了预期结果中的**每一项**断言
4. Note 包含**至少 2 个**来自终端输出的具体值

### Note 格式

```
<method>: <value_1>, <value_2>[, ...]
```

- **CLI**：`exit=0, stdout: 'Total chunks: 3', source_file=simple.pdf`
- **AppTest**：`at.metric[0].label='Total traces', at.metric[0].value=6`
- **多步骤**：`Step1: exit=0, chunks=3. Step2: sources=[simple.pdf]. Step3: deleted=1. Step4: sources=[]`
- **差的**（禁止）：`"Already verified in C-02"`、`"Code uses yaml.safe_load"`、`"Should work because..."`、`"Parameter accepted"`

### 状态图标

| 图标 | 含义 |
|------|---------|
| ✅ | 通过 — 所有断言均与实际输出核对 |
| ❌ | 失败 — 3 次修复尝试后仍失败 |
| ⏭️ | 跳过 — 缺少第三方 API Key（仅 K 系列） |
| 🔧 | 已修复 — 需要重测 |
| ⬜ | 待测 — 尚未测试 |

### 计数器

在同一次编辑中更新：`✅ Pass: X | ❌ Fail: Y | ⏭️ Skip: Z | 🔧 Fix: W | ⬜ Pending: P`（总和必须等于 Total）。

### 章节末门禁

每完成一个章节后：
```
python .github/skills/qa-tester/scripts/qa_validate_notes.py
```
重新执行任何被标记的测试。在标记数为 0 之前不得继续。

---

## 关键路径

| 文件 | 用途 |
|------|---------|
| `.github/skills/qa-tester/QA_TEST_PLAN.md` | 测试步骤和预期结果 |
| `.github/skills/qa-tester/QA_TEST_PROGRESS.md` | 执行状态和备注 |
| `config/settings.yaml` | 系统配置 |
| `scripts/ingest.py` / `query.py` / `evaluate.py` | CLI 命令 |
| `tests/e2e/test_mcp_client.py` | MCP E2E 测试 |
| `tests/e2e/test_dashboard_smoke.py` | Dashboard 冒烟测试 |
| `tests/fixtures/sample_documents/` | 测试 PDF |
| `tests/fixtures/golden_test_set.json` | 评估黄金集 |

## 测试文档

| 文件 | 语言 | 页数 | 图片 |
|------|----------|-------|--------|
| `simple.pdf` | 英文 | 1 | 0 |
| `with_images.pdf` | 英文 | 1 | 1 |
| `complex_technical_doc.pdf` | 英文 | ~8 | 3 |
| `chinese_technical_doc.pdf` | 中文 | ~8 | 0 |
| `chinese_table_chart_doc.pdf` | 中文 | ~6 | 3 |
| `chinese_long_doc.pdf` | 中文 | 30+ | 0 |
| `blogger_intro.pdf` | 中文 | ~4 | 2 |

全部位于 `tests/fixtures/sample_documents/`。
