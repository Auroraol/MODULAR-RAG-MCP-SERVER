---
name: auto-coder
description: 自主的规范驱动开发 agent。将 DEV_SPEC.md 同步为按章节划分的参考文件，从进度表中识别下一个待办任务，按照规范架构和模式实现代码，运行测试并最多进行 3 轮自动修复，通过原子提交保存进度。当用户说 "auto code"、"自动开发"、"自动写代码"、"auto dev"、"一键开发"、"autopilot"，或想要全自动的从规范到代码的工作流时使用。
---

# 自动编码器

一次触发即可完成 **读取规范 → 查找任务 → 编码 → 测试 → 保存进度**。

可选修饰符：追加任务 ID（例如 `auto code B2`）以指定特定任务，或 `--no-commit` 以跳过 git 提交。

---

## 流水线

```
Sync Spec → Find Task → Implement → Test (≤3 fix rounds) → Persist
```

仅在最后暂停以确认提交。其他一切自主运行。

> **⚠️ 关键：在执行任何 `python`/`pytest` 命令之前先激活 `.venv`（幂等，不确定时重新运行）。**
> - **Windows**：`.\.venv\Scripts\Activate.ps1`
> - **macOS/Linux**：`source .venv/bin/activate`

## 参考文件索引

`.claude/skills/auto-coder/references/` 下的所有文件：

| 文件 | 内容 | 何时阅读 |
|------|---------|-------------|
| `01-overview.md` | 项目概述与目标 | 第一个任务，或需要项目上下文时 |
| `02-features.md` | 功能规范 | 实现功能相关任务时 |
| `03-tech-stack.md` | 技术栈与依赖 | 选择库或模式时 |
| `04-testing.md` | 测试约定 | 编写测试时 |
| `05-architecture.md` | 架构与模块设计 | 创建/修改模块时 |
| `06-schedule.md` | 任务进度与状态 | 每个周期（Sync Spec 步骤） |
| `07-future.md` | 未来路线图 | 规划或评估范围时 |

---

### 1. 同步规范

```powershell
python .claude/skills/auto-coder/scripts/sync_spec.py
```

然后读取进度文件以获取任务状态：
- 读取 `.claude/skills/auto-coder/references/06-schedule.md`

任务标记：

| 标记 | 状态 |
|--------|--------|
| `[ ]` / `⬜` | 未开始 |
| `[~]` / `🔶` / `(进行中)` | 进行中 |
| `[x]` / `✅` / `(已完成)` | 已完成 |

---

### 2. 查找任务

选择第一个 `IN_PROGRESS` 任务，然后是第一个 `NOT_STARTED` 任务。如果用户指定了任务 ID，则直接使用该任务。

快速检查前置产物是否存在（仅文件层面）。不匹配时记录警告并继续——仅当目标任务本身被阻塞时才停止。

---

### 3. 实现

1. **阅读相关规范**，来自 `.claude/skills/auto-coder/references/`：
   - 架构：`05-architecture.md`
   - 技术细节：`03-tech-stack.md`
   - 测试约定：`04-testing.md`

2. **提取**规范中的：输入/输出、设计原则（可插拔？配置驱动？工厂？）、文件列表、验收标准。

3. **规划**在编写任何代码之前要创建/修改的文件。

4. **编码**——项目特定规则：
   - 将规范视为唯一事实来源
   - 使用 `config/settings.yaml` 中的值，切勿硬编码
   - 匹配现有代码库的模式和风格

5. **与代码同步编写测试**：
   - 按照规范放置在 `tests/unit/` 或 `tests/integration/` 中
   - 在单元测试中 mock 外部依赖

6. **自审**，在运行测试之前：确认所有计划中的文件都存在，且测试能正确导入。

---

### 4. 测试与自动修复

```

Round 0..2:
  Run pytest on relevant test file
  If pass → go to step 5
  If fail → analyze error, apply fix, re-run

Round 3 still failing → STOP, show failure report to user
```

---

### 5. 保存进度

1. **更新 `DEV_SPEC.md`**（全局文件）：将任务标记从 `[ ]` 改为 `[x]`
2. **重新同步**：`python .claude/skills/auto-coder/scripts/sync_spec.py --force`
3. **显示摘要并询问**：

```
✅ [A3] 配置加载与校验 — done
   Files: src/core/settings.py, tests/unit/test_settings.py
   Tests: 8/8 passed
   Commit: feat(config): [A3] implement config loader

   "commit" → git add + commit
   "skip"   → end
   "next"   → commit + start next task
```

选择 "next" 时，回到步骤 1 并开始下一个任务。
