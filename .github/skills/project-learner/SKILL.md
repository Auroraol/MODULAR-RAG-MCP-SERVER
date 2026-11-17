---
name: project-learner
description: "通过面试式问答进行交互式项目学习辅导。读取代码库和文档，按知识域和子知识点动态生成面试问题，进行最多 4 轮追问，为回答评分，提供带代码/文档引用的学习指导，并持久化进度。10 个知识域 × 3-5 个子知识点 = 45 个知识点，实现全面的面试覆盖。当用户说 '学习项目'、'了解项目'、'检验项目'、'项目学习'、'面试准备'、'learn project'、'study project'、'review project'、'interview prep'、'knowledge check'，或希望通过引导式问答理解/掌握该项目时使用。"
---

# Project Learner（项目学习）

通过引导式问答帮助用户掌握本项目的交互式面试辅导。

所有面向用户的交互均使用**中文**。内部指令使用英文。

## 流程概览

```
Discovery → Check History → User Intent → Select Domain → Select Sub-topic
→ Generate Question → Interactive Q&A (≤4 follow-ups) → Evaluate
→ Learning Guide → Persist Progress → Continue or End
```

---

## 阶段 1：项目探索

自主建立项目理解。此时不要向用户提出任何问题。

1. 读取 `DEV_SPEC.md` — 项目目标、架构、技术栈、模块设计
2. 读取 `config/settings.yaml` — 配置系统
3. 列出 `src/` 目录树 — 模块结构（core/、ingestion/、libs/、mcp_server/、observability/）
4. 读取关键入口点：`main.py`、`scripts/ingest.py`、`scripts/query.py`
5. 列出 `tests/` — 测试策略概览

构建一个内部心智模型，覆盖以下 **10 个知识域**，每个知识域包含 **3-5 个子知识点**（知识点），共计 **45 个面试知识点**：

### 知识域与子知识点映射

| ID | 知识域 / 知识点 | 关键代码区域 |
|----|----------------|---------------|
| **D1** | **RAG Pipeline 整体架构** | |
| D1.1 | 端到端数据流：从文档上传到生成回答的完整链路 | `DEV_SPEC.md`, `main.py`, `scripts/` |
| D1.2 | 三层架构设计：core/ingestion/libs 各层职责与依赖方向 | `src/core/`, `src/ingestion/`, `src/libs/` |
| D1.3 | Pipeline 组装：配置驱动的组件组合机制 | `main.py`, `config/settings.yaml`, `src/core/settings.py` |
| D1.4 | 核心数据类型：Document、Chunk、QueryResult 等类型系统 | `src/core/types.py` |
| D1.5 | 入口脚本设计：CLI 脚本的职责划分与参数传递 | `scripts/ingest.py`, `scripts/query.py`, `scripts/evaluate.py` |
| **D2** | **数据摄取 Pipeline** | |
| D2.1 | Pipeline 整体流程：从文档加载到向量存储的阶段设计 | `src/ingestion/pipeline.py` |
| D2.2 | Chunking 策略：RecursiveSplitter 的分割逻辑与参数调优 | `src/ingestion/chunking/`, `src/libs/splitter/` |
| D2.3 | Transform 链：ChunkRefiner、MetadataEnricher 的职责与执行顺序 | `src/ingestion/transform/` |
| D2.4 | Embedding 编码：Dense/Sparse 双编码与 BatchProcessor 批处理 | `src/ingestion/embedding/` |
| D2.5 | 存储层：VectorUpserter、BM25Indexer、ImageStorage 三类存储协同 | `src/ingestion/storage/` |
| **D3** | **混合检索（Hybrid Search & Retrieval）** | |
| D3.1 | Dense Retrieval：向量检索原理与 DenseRetriever 实现 | `src/core/query_engine/dense_retriever.py` |
| D3.2 | Sparse Retrieval：BM25 稀疏检索与 SparseRetriever 实现 | `src/core/query_engine/sparse_retriever.py` |
| D3.3 | Hybrid Search 融合：RRF 算法与 Fusion 模块设计 | `src/core/query_engine/hybrid_search.py`, `fusion.py` |
| D3.4 | QueryProcessor：查询预处理与查询扩展机制 | `src/core/query_engine/query_processor.py` |
| D3.5 | Response 构建：ResponseBuilder、CitationGenerator、MultimodalAssembler | `src/core/response/` |
| **D4** | **Rerank 机制** | |
| D4.1 | Reranker 抽象与工厂模式：BaseReranker 与 RerankerFactory 设计 | `src/libs/reranker/base_reranker.py`, `reranker_factory.py` |
| D4.2 | CrossEncoder Reranker：模型原理与实现细节 | `src/libs/reranker/cross_encoder_reranker.py` |
| D4.3 | LLM Reranker：基于大语言模型的重排序方案与 Prompt 设计 | `src/libs/reranker/llm_reranker.py` |
| D4.4 | Rerank 在检索 Pipeline 中的集成位置与效果分析 | `src/core/query_engine/reranker.py` |
| **D5** | **MCP Server 协议** | |
| D5.1 | MCP 协议概述：JSON-RPC 交互模型与标准规范 | `src/mcp_server/server.py` |
| D5.2 | Tool 注册机制：三个工具的定义、参数与执行逻辑 | `src/mcp_server/tools/` |
| D5.3 | ProtocolHandler：请求路由、分发与能力协商 | `src/mcp_server/protocol_handler.py` |
| D5.4 | Server 生命周期管理与异常处理 | `src/mcp_server/server.py`, `protocol_handler.py` |
| **D6** | **可插拔架构 & 配置系统** | |
| D6.1 | 工厂模式全景：LLM/Embedding/Reranker/VectorStore/Evaluator 五大工厂 | `src/libs/*/factory*.py` |
| D6.2 | settings.yaml 配置结构与 Settings 类加载机制 | `config/settings.yaml`, `src/core/settings.py` |
| D6.3 | LLM Provider 多厂商支持：Azure/OpenAI/DeepSeek/Ollama 切换逻辑 | `src/libs/llm/` |
| D6.4 | Embedding Provider 抽象：多后端实现对比与选型策略 | `src/libs/embedding/` |
| D6.5 | Base 类设计哲学：接口抽象、继承层次与扩展点 | `src/libs/*/base_*.py` |
| **D7** | **多模态处理** | |
| D7.1 | PDF 解析：PDFLoader 实现与 FileIntegrity 文件校验 | `src/libs/loader/` |
| D7.2 | Vision LLM：Azure/OpenAI Vision 图片理解能力集成 | `src/libs/llm/azure_vision_llm.py`, `openai_vision_llm.py` |
| D7.3 | ImageCaptioner：图片描述生成流程与 Prompt 模板设计 | `src/ingestion/transform/image_captioner.py`, `config/prompts/` |
| D7.4 | 多模态 Chunk 存储与检索：ImageStorage 与 MultimodalAssembler 协同 | `src/ingestion/storage/image_storage.py`, `src/core/response/multimodal_assembler.py` |
| **D8** | **可观测性 & 评估体系** | |
| D8.1 | Trace 系统：TraceCollector 与 TraceContext 的采集与关联设计 | `src/core/trace/` |
| D8.2 | Dashboard 架构：Streamlit App 分页、Services 层数据流 | `src/observability/dashboard/` |
| D8.3 | 评估指标体系：Recall、Precision、MRR 等核心指标定义与计算 | `src/observability/evaluation/`, `scripts/evaluate.py` |
| D8.4 | 评估框架：CompositeEvaluator、CustomEvaluator、RAGAS 集成架构 | `src/libs/evaluator/`, `src/observability/evaluation/` |
| D8.5 | 日志系统：Logger 设计、日志分级与调试支持 | `src/observability/logger.py` |
| **D9** | **测试策略 & 工程化** | |
| D9.1 | 测试分层策略：Unit/Integration/E2E 各层覆盖范围与边界 | `tests/unit/`, `tests/integration/`, `tests/e2e/` |
| D9.2 | Test Fixtures 与 conftest.py：Mock 策略与测试数据管理 | `tests/conftest.py`, `tests/fixtures/` |
| D9.3 | pyproject.toml 工程配置：依赖管理、构建配置、工具链集成 | `pyproject.toml` |
| D9.4 | 脚本入口设计：四大脚本的职责边界与参数化设计 | `scripts/` |
| **D10** | **Document Manager & 幂等性** | |
| D10.1 | 文档去重：Hash 计算与重复检测机制 | `src/ingestion/document_manager.py`, `src/libs/loader/file_integrity.py` |
| D10.2 | 增量 Ingestion：幂等性保证与文档更新策略 | `src/ingestion/document_manager.py`, `pipeline.py` |
| D10.3 | Collection 管理：集合元数据关联与文档生命周期 | `src/ingestion/document_manager.py` |
| D10.4 | 文档状态追踪：已入库/待更新/已删除的状态流转 | `src/ingestion/document_manager.py` |

> **总计：10 个知识域 × 3-5 个子知识点 = 45 个知识点**
> 每个子知识点都可以用不同的问题多次学习，可提供 100+ 道可能的面试题。

---

## 阶段 2：检查学习历史

1. 尝试读取 `.github/skills/project-learner/references/LEARNING_PROGRESS.md`
2. **文件不存在** → 首次学习，进入阶段 3
3. **文件存在** → 解析**两个**表格：
   - **知识域汇总**：哪些知识域为 ⬜/🔴/🔶/✅
   - **子知识点进度**：哪些子知识点为 ⬜（未学习）、🔴（薄弱 ≤3）、🔶（学习中 4-6）、✅（已掌握 ≥7）
   - 统计：已掌握的子知识点总数 / 45
   - 找出得分最低的子知识点以推荐复习

---

## 阶段 3：用户意图

使用 `ask_questions`（中文）确定用户的意图：

**问题 1 — 学习模式**（单选）：

| 选项 | 说明 |
|--------|------------|
| 🆕 学习新知识点 | 从未学习/薄弱的子知识点中选择 |
| 📖 复习已学内容 | 复习之前学过的低分子知识点 |
| 📋 查看学习进度 | 显示进度表，然后结束 |
| 🎯 Agent 推荐 | 自动选择下一个最佳学习子知识点 |

如果用户选择 📋 → 显示 `LEARNING_PROGRESS.md` 中的完整进度表并停止。

如果用户选择 🎯 → Agent 自动选择最优子知识点（优先级：最弱知识域中的 ⬜ 未学习 → 🔴 薄弱 → 🔶 最低分）。跳过问题 2 和 3，直接进入阶段 4。

**问题 2 — 知识域选择**（单选，仅针对 🆕 或 📖）：

列出所有 10 个知识域及其当前状态 + 完成率。示例格式：
- `D1 RAG Pipeline 整体架构 [2/5 ✅] 🔶`
- `D2 数据摄取 Pipeline [0/5 ✅] ⬜`

对于 📖 模式：只显示有历史得分的知识域。对于 🆕 模式：优先显示 ⬜ 子知识点最多的知识域。

**问题 3 — 知识点选择**（单选，仅在问题 2 之后）：

列出所选知识域下的所有子知识点及其状态：
- `D2.1 Pipeline 整体流程 ⬜ 未学习`
- `D2.2 Chunking 策略 🔶 6/10`
- `D2.3 Transform 链 ✅ 8/10`

包含选项：
- 🎯 Agent 推荐 — 自动选择该知识域中最弱/未学习的子知识点

---

## 阶段 4：生成面试问题

基于选定的**子知识点**（而不仅是知识域）：

1. **深入阅读**该子知识点的具体源代码 — 阅读子知识点映射中列出的实际类定义、关键函数、配置段
2. **动态生成**一道基于该子知识点真实代码的面试主问题（中文）
3. **内部准备**最多 4 道递进式追问（此时不要展示）
4. **避免重复**之前会话中的问题 — 检查该子知识点的详细历史并生成不同的角度

### 问题设计原则

- 问题必须引用**本项目**的真实代码/架构，绝不使用泛泛而谈的表述
- 问题应针对子知识点，而不是整个知识域
- 追问的难度递进：
  - 追问 1：“为什么这样设计？”（设计理由）
  - 追问 2：“和替代方案对比有什么优劣？”（权衡取舍）
  - 追问 3：“边界条件/异常情况怎么处理？”（边界情况）
  - 追问 4：“如果让你重新设计，会怎么做？”（重新设计的思考）
- 根据用户的实际回答动态调整追问

### 问题角度多样性

每个子知识点都可以从多个角度提问。重新学习某个子知识点时，选择**不同的**角度：
- **What**：描述这个模块/机制做了什么
- **How**：具体实现细节，代码层面怎么做的
- **Why**：为什么选择这种设计方案
- **Compare**：和替代方案的对比
- **Debug**：如果出了问题怎么排查
- **Extend**：如果要扩展功能怎么做

### 问题格式

向用户展示：

```
## 🎯 面试问题

**知识域**: [Domain Name] > **知识点**: [Sub-topic Name]

**面试官问**: [Question text — specific to this sub-topic, referencing project components]

请回答：
```

---

## 阶段 5：交互式问答（≤4 轮追问）

```
Round 0: Main question → User answers
Round 1-4: Brief feedback on previous answer + follow-up question → User answers
Early exit: User says "结束"/"pass"/"跳过" OR answer is sufficiently comprehensive
```

### 每轮行为

1. **肯定**用户答对的部分（1-2 句，中文）
2. **提示**遗漏的内容，但不直接给出答案（1 句）
3. **提出追问**，根据其回答方向深入挖掘

### 追问输出格式

```
### 第 N 轮追问

✅ **答得好**: [What they got right]
💡 **提示**: [What they could explore further]

**追问**: [Follow-up question]
```

如果用户的回答已经涵盖计划中的追问内容，则跳到更难的问题或提前结束。

---

## 阶段 6：评估

问答结束后，输出结构化评估报告（中文）：

```markdown
## 📊 评价报告

**知识域**: [Domain] > **知识点**: [Sub-topic ID & Name] — [Question summary]
**追问轮数**: N/4

### ✅ 回答亮点
- [Strength 1 — specific to what they said]
- [Strength 2]

### ⚠️ 需要加强
- [Gap 1 — what was missed or inaccurate]
- [Gap 2]

### 📈 评分明细

| 维度 | 分数 | 说明 |
|------|------|------|
| 准确性 | X/10 | [Factual correctness of answers] |
| 深度 | X/10 | [How deep they went beyond surface] |
| 代码关联 | X/10 | [Did they reference actual code/config] |
| 设计思维 | X/10 | [Trade-off analysis, architecture reasoning] |

### 🏆 综合评分: X/10

### 📊 学习进度: [mastered count]/45 知识点已掌握
```

评分规则：
- 四个维度的平均值，四舍五入到最近的 0.5
- 9-10：专家级，能解释设计决策和权衡
- 7-8：扎实理解，知道如何做和为什么
- 4-6：基础理解，知道是什么但不够深入为什么
- 1-3：表面水平，需要大量学习

---

## 阶段 7：学习指南

评估后立即提供有针对性的学习资源（中文）：

```markdown
## 📚 学习指南

### 📂 相关代码
- [file_path](file_path#LX-LY) — 说明这段代码的作用和关键逻辑

### 📄 相关文档
- [DEV_SPEC.md 对应章节](DEV_SPEC.md) — 设计原理
- [config/settings.yaml](config/settings.yaml) — 相关配置项

### 🔗 参考资料
- [External concept name] — 1-sentence explanation of relevance

### 💡 建议学习路径
1. 先阅读 [file] 理解 [what]
2. 再看 [file] 掌握 [implementation detail]
3. 运行 `[command]` 实际体验效果
4. 尝试修改 [config/code] 观察变化
```

指南：
- 代码引用必须使用实际文件路径，并在相关处标注行号
- 只推荐阅读 3-5 个关键文件，而不是整个代码库
- 至少包含一条用户可以运行的实际命令
- 外部参考仅用于代码库中未解释的概念（例如 RRF 算法、BM25）

---

## 阶段 8：持久化进度

更新 `.github/skills/project-learner/references/LEARNING_PROGRESS.md`。

如果文件不存在，请从 [references/LEARNING_PROGRESS.md](references/LEARNING_PROGRESS.md) 中的模板创建。如果存在，则更新它。

### 更新规则

1. 向 `Detailed History` 表**追加**一行（包含子知识点 ID）
2. **更新**受影响子知识点的 `Sub-topic Progress` 表：
   - 已学 = 该子知识点的学习场次
   - 最高分 = 该子知识点所有场次中的最高分
   - 最近分 = 本次学习的得分
   - 状态：≥7 → ✅ 掌握，4-6 → 🔶 学习中，≤3 → 🔴 薄弱，0 场次 → ⬜ 未学习
3. **重新计算** `Domain Summary` 表：
   - 已掌握 = 该知识域中 ✅ 子知识点的数量 / 该知识域的子知识点总数
   - 已学习 = 非 ⬜ 子知识点的数量 / 子知识点总数
   - 平均分 = 该知识域中所有已学习子知识点的平均分
   - 知识域状态：所有子知识点 ✅ → ✅ 掌握；有已学习 → 🔶 学习中 或 🔴 薄弱（基于平均分）；无 → ⬜ 未学习
4. **更新** `Last updated` 时间戳
5. **更新**会话计数器 `#`（自动递增）
6. **更新**总体进度行：`总进度: X/45 知识点已掌握`

---

## 阶段 9：继续或结束

持久化后，询问用户（中文）：

| 选项 | 操作 |
|--------|--------|
| 🔄 继续学习下一个知识点 | 回到阶段 3 |
| 🎯 Agent 推荐下一个 | 自动选择最优的下一个子知识点，进入阶段 4 |
| 📋 查看当前学习进度 | 显示完整进度表 |
| 🏁 结束本次学习 | 显示本次总结，停止 |

### 本次总结（选择 🏁 结束时）

```markdown
## 📝 本次学习总结

- 完成知识点: N 个
- 平均得分: X/10
- 最强知识点: [sub-topic] (X/10)
- 需加强知识点: [sub-topic] (X/10)
- 总进度: X/45 知识点已掌握 (XX%)

继续加油！下次建议学习: [recommended sub-topic name]
```

---

## 关键路径

| 文件 | 用途 |
|------|---------|
| `.github/skills/project-learner/references/LEARNING_PROGRESS.md` | 持久化学习状态（45 个子知识点） |
| `DEV_SPEC.md` | 项目规范与架构 |
| `config/settings.yaml` | 配置参考 |
| `src/` | 所有源代码模块 |
| `tests/` | 用于理解测试策略的测试套件 |
| `scripts/` | CLI 入口点（ingest/query/evaluate） |
