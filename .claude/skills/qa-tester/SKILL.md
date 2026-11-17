---
name: qa-tester
description: "面向 Modular RAG MCP Server 的全自主 QA 测试 agent。从 QA_TEST_PLAN.md 读取测试用例，无需人工干预即可自动执行所有测试类型——CLI 命令、通过 Streamlit AppTest 无头渲染的 Dashboard UI、通过子进程 JSON-RPC 的 MCP 协议、provider 切换以及数据生命周期检查。诊断失败，应用修复并最多重试 3 轮，将结果记录到 QA_TEST_PROGRESS.md。当用户说 'run QA'、'QA test'、'QA 测试'、'执行测试'、'跑测试'、'test and fix'，或想要执行 QA 测试计划时使用。"
---

# QA 测试器

所有测试类型（CLI / Dashboard UI / MCP 协议）均**完全自动化**——零人工干预。

可选修饰符：追加章节字母（`run QA G`）或测试 ID（`run QA G-01`）。

---

> ## ⛔ 铁律
>
> ### 规则 1：严格串行
> 选择一个测试 → 运行一条命令 → 等待输出 → 在 `QA_TEST_PROGRESS.md` 中记录一行 → 然后才选择下一个。
> 绝不在一条命令中运行两个测试。绝不在一次文件编辑中记录两行。绝不使用并行工具调用。绝不在记录当前测试之前规划下一个测试。
>
> ### 规则 2：通过 = 终端输出证据
> ✅ 意味着你在**本次会话中**运行了命令，且 Note 中包含从该输出中复制的**具体值**。
> 本次测试没有终端输出 → 标记为 ⬜。绝不标记 ✅。
>
> ### 规则 3：零交叉引用
> 绝不写“已通过 X-YZ 验证”、“已在 C-02 中验证”、“与…相同”、“类似于…”。
> 即使 G-05 测试的 flag 与 C-02 相同，也要独立运行 G-05 并粘贴其自身的输出。
>
> ### 规则 4：零推断
> **禁止的 Note 模式**（会被验证器捕获）：
> “代码使用了…” / “Dataclass 验证…” / “自动创建…” → 阅读代码 ≠ 测试。
> “应该能工作…” / “会抛出…” / “预期行为…” → 推测 ≠ 测试。
> “参数已接受” / “配置控制行为” → 含糊。没有输出 = 没有通过。
> 如果你没有为本次测试运行命令并看到输出，则标记为 ⬜。
>
> ### 规则 5：对抗性思维
> 寻找 bug，而不是确认。连续 10+ 次通过且零 bug → 重新审视你的严谨性。
>
> ### 规则 6：章节结束验证
> 完成一个章节后，运行 `python .claude/skills/qa-tester/scripts/qa_validate_notes.py`。
> 在进入下一章节之前，重新执行任何被标记的测试。

---

## 流水线（严格串行）

```
1. Pick ONE pending test (by ID order)
2. Set system state if needed
3. Run ONE command — WAIT for output
4. Verify ALL assertions from Expected Result vs ACTUAL output
5. Fix if needed (≤3 rounds)
6. ⛔ GATE: Edit QA_TEST_PROGRESS.md (row + counters) — ONE row per edit
7. Only NOW return to step 1
```

> 在任何 `python` 命令之前激活 `.venv`：`.\.venv\Scripts\Activate.ps1`

---

## 步骤 1：选择目标

1. 阅读 `QA_TEST_PLAN.md` 获取测试步骤和预期结果。
2. 阅读 `QA_TEST_PROGRESS.md` 获取当前状态。
3. 用户指定的章节/ID → 限定到该范围。否则 → 第一个 ⬜ 待办测试。
4. 如果存在任何 🔧 测试，先重新测试这些。
5. 按章节顺序（A→O）执行，章节内按 ID 顺序。

### 测试类别

| 章节 | 类型 | 执行方法 |
|----------|------|-----------------|
| A–F | Dashboard UI | AppTest 无头——参见 [references/test_patterns.md](references/test_patterns.md) |
| G, H, I | CLI | 终端命令，检查退出码 + stdout |
| J | MCP 协议 | JSON-RPC 子进程——参见 [references/test_patterns.md](references/test_patterns.md) |
| K, L | Provider 切换 | `qa_config.py apply <profile>` → 运行 CLI/Dashboard |
| M | 配置与容错 | 修改 settings → 运行 CLI → 验证错误处理 |
| N, O | 数据生命周期 | 使用 `qa_multistep.py <TEST_ID>` |

---

## 步骤 2：设置系统状态

```
Empty          → python .claude/skills/qa-tester/scripts/qa_bootstrap.py clear
Baseline       → python .claude/skills/qa-tester/scripts/qa_bootstrap.py baseline
DeepSeek       → python .claude/skills/qa-tester/scripts/qa_config.py apply deepseek
Rerank_LLM     → python .claude/skills/qa-tester/scripts/qa_config.py apply rerank_llm
NoVision       → python .claude/skills/qa-tester/scripts/qa_config.py apply no_vision
InvalidKey     → python .claude/skills/qa-tester/scripts/qa_config.py apply invalid_llm_key
InvalidEmbedKey→ python .claude/skills/qa-tester/scripts/qa_config.py apply invalid_embed_key
Any            → no state change needed
```

配置档测试之后 → `python .claude/skills/qa-tester/scripts/qa_config.py restore`
检查状态 → `python .claude/skills/qa-tester/scripts/qa_bootstrap.py status`

---

## 步骤 3：执行与验证

### CLI 测试（G、H、I，部分 K/L/M）

1. 从 `QA_TEST_PLAN.md` 读取测试的 **Steps** 列。
2. 在终端中运行确切的命令。
3. 将输出与 **Expected Result** 比较：
   - **Ingest**：exit=0，输出包含阶段名称（load/split/transform/embed/upsert）
   - **Query**：exit=0，结果包含 source_file 和 score
   - **Error**：exit≠0，stderr 包含描述性错误（而非原始堆栈跟踪）
   - **Idempotency**：第二次运行显示“skipped”，无重复

### Dashboard 测试（A–F）

阅读 [references/test_patterns.md](references/test_patterns.md) 获取 AppTest 模板、交互模式和文件上传变通方案。

关键点：
- 每个测试编写一个内联 Python 脚本，通过 `AppTest.from_function` 渲染
- 打印特定元素值（metric 标签、selectbox 选项、expander 标签）
- 对于文件上传：先通过 CLI 摄取，再通过 AppTest 验证
- 对于数据变更：直接调用 `DataService`，通过 AppTest/CLI 验证

#### ⚠️ AppTest `st.tabs()` 限制——测试任何 tab 内容前必读

AppTest 将**所有 tab 内容同时**渲染到一个扁平元素列表中。tab 之间**没有空间隔离**——你无法从 `at.info`、`at.metric`、`at.markdown` 等判断哪个元素属于哪个 tab。

**后果：** 如果测试步骤说“点击 tab X → 在 tab X 内看到内容 Y”，AppTest **无法确认空间关系**。即使输出中存在内容 Y，它也可能来自不同的区域（例如 tab 上方的诊断区域，或不同的 tab）。

**所有 tab 内容测试的强制检查：**

1. **首先验证 tab 标签存在：** 打印 `[t.label for t in at.tabs]`（如果 `at.tabs` 可用），或检查页面源码中的 tab 标签字符串。如果预期的 tab 标签**缺失** → 立即标记 ❌——该 tab 不存在。
2. **然后验证内容元素存在**于 AppTest 输出中。
3. **在 Note 中承认这一限制：** 在 Note 中追加 `[AppTest: tab isolation unverifiable]`。

**示例——tab 不存在：**
```python
# For a test expecting "🟪 Rerank" tab with "skipped" message:
tab_labels = [el.value for el in at.markdown if '🟪' in el.value or 'Rerank' in el.value]
print('Rerank tab label found:', tab_labels)
# If empty → tab absent → mark ❌ with note "Rerank tab not rendered (stage absent)"
```

**禁止的模式：** 找到一个内容元素（例如“Rerank skipped”信息）就断定 tab 测试通过——该元素可能来自诊断区域，而非 tab 内部。

### MCP 测试（J）

阅读 [references/test_patterns.md](references/test_patterns.md) 获取 JSON-RPC 脚本模板和断言矩阵。

主要方法：`pytest tests/e2e/test_mcp_client.py -v` 覆盖大多数 J-* 用例。

### 多步骤测试（N、O、M 配置、L-07）

具有 3+ 个顺序步骤的测试**必须**使用运行器脚本：

```
python .claude/skills/qa-tester/scripts/qa_multistep.py <TEST_ID>
```

**支持：** `N-01`、`N-03`、`N-04`、`N-05`、`N-06`、`O-07`、`M-03`、`M-04`、`M-05`、`M-06`、`M-10`、`M-11`、`L-07`

该脚本执行每个子步骤，打印每一步的实际值，输出 `VERDICT: PASS/FAIL`。将 VERDICT 和关键步骤值复制到 Note 中。

对于不在脚本中的测试，手动运行 QA_TEST_PLAN.md 中的命令并粘贴输出。

---

## 步骤 4：修复与重试（≤3 轮）

1. **诊断**：代码 bug / 配置问题 / 数据缺失 / 测试计划错误？
2. **修复**：仅做最小改动。在 Note 中记录文件/行号。
3. **重试**：重新运行同一命令。
4. 3 轮失败后 → 标记 ❌ 并附上详细说明。
5. 如果修复涉及共享代码 → 重新运行同一章节中先前通过的测试。

---

## 步骤 5：记录结果

**⛔ 关卡——在选择下一个测试之前完成此操作。**

编辑 `QA_TEST_PROGRESS.md`：更新一个测试行 + 汇总计数器。每次文件编辑一行。

### ✅ 通过要求

所有条件必须为真：
1. **在本次会话中**运行了命令
2. 观察到了**来自该命令**的实际输出
3. 验证了 Expected Result 中的**每一条**断言
4. Note 包含**≥2 个**来自终端输出的具体值

### Note 格式

```
<method>: <value_1>, <value_2>[, ...]
```

- **CLI**：`exit=0, stdout: 'Total chunks: 3', source_file=simple.pdf`
- **AppTest**：`at.metric[0].label='Total traces', at.metric[0].value=6`
- **多步骤**：`Step1: exit=0, chunks=3. Step2: sources=[simple.pdf]. Step3: deleted=1. Step4: sources=[]`
- **不良**（禁止）：`"Already verified in C-02"`、`"Code uses yaml.safe_load"`、`"Should work because..."`、`"Parameter accepted"`

### 状态图标

| 图标 | 含义 |
|------|---------|
| ✅ | 通过——所有断言均已对照实际输出验证 |
| ❌ | 失败——3 次修复尝试后仍失败 |
| ⏭️ | 跳过——缺少第三方 API key（仅 K 系列） |
| 🔧 | 已应用修复——需要重新测试 |
| ⬜ | 待办——尚未测试 |

### 计数器

在同一次编辑中更新：`✅ Pass: X | ❌ Fail: Y | ⏭️ Skip: Z | 🔧 Fix: W | ⬜ Pending: P`（总和必须等于 Total）。

### 章节结束关卡

每个章节完成后：
```
python .claude/skills/qa-tester/scripts/qa_validate_notes.py
```
重新执行任何被标记的测试。在标记数为 0 之前不要继续。

---

## 关键路径

| 文件 | 用途 |
|------|---------|
| `QA_TEST_PLAN.md` | 测试步骤和预期结果 |
| `QA_TEST_PROGRESS.md` | 执行状态和说明 |
| `config/settings.yaml` | 系统配置 |
| `scripts/ingest.py` / `query.py` / `evaluate.py` | CLI 命令 |
| `tests/e2e/test_mcp_client.py` | MCP E2E 测试 |
| `tests/e2e/test_dashboard_smoke.py` | Dashboard 冒烟测试 |
| `tests/fixtures/sample_documents/` | 测试 PDF |
| `tests/fixtures/golden_test_set.json` | 评估黄金集 |

## 测试文档

| 文件 | 语言 | 页数 | 图片 |
|------|----------|-------|--------|
| `simple.pdf` | EN | 1 | 0 |
| `with_images.pdf` | EN | 1 | 1 |
| `complex_technical_doc.pdf` | EN | ~8 | 3 |
| `chinese_technical_doc.pdf` | ZH | ~8 | 0 |
| `chinese_table_chart_doc.pdf` | ZH | ~6 | 3 |
| `chinese_long_doc.pdf` | ZH | 30+ | 0 |
| `blogger_intro.pdf` | ZH | ~4 | 2 |

全部位于 `tests/fixtures/sample_documents/`。
