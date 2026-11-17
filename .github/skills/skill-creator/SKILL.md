---
name: skill-creator
description: 创建高效 agent 技能的指南。当用户想要创建新技能（或更新现有技能），以专业知识、工作流或工具集成扩展 agent 的能力时，应使用本技能。当用户说 "create skill"、"new skill"、"build a skill"、"update skill"，或要求制作可复用的能力包时使用。
---

# Skill Creator（技能创建器）

本技能为创建高效技能提供指导。

## 关于技能

技能是模块化、自包含的包，通过提供专业知识、工作流和工具来扩展 agent 的能力。可以将它们视为特定领域或任务的“入职指南”——它们把通用 agent 转变为具备程序性知识的专业 agent，而这些知识任何模型都无法凭空完整具备。

### 技能提供什么

1. 专业工作流 — 针对特定领域的多步骤流程
2. 工具集成 — 使用特定文件格式或 API 的说明
3. 领域专长 — 公司特定知识、schema、业务逻辑
4. 打包资源 — 用于复杂和重复任务的脚本、参考文档和素材

## 核心原则

### 简洁是关键

上下文窗口是公共资源。技能与 agent 所需的其他一切共享上下文窗口：系统提示、对话历史、其他技能的元数据，以及实际的用户请求。

**默认假设：agent 已经非常聪明。** 只添加 agent 尚不拥有的上下文。对每条信息提出质疑：“agent 真的需要这个解释吗？”以及“这段文字值得它的 token 成本吗？”

优先使用简洁的示例，而不是冗长的解释。

### 设置适当的自由度

让具体程度与任务的脆弱性和可变性相匹配：

**高自由度（基于文本的指令）**：当存在多种有效方法、决策依赖上下文，或启发式规则指导方法时使用。

**中自由度（伪代码或带参数的脚本）**：当存在首选模式、允许一定变化，或配置会影响行为时使用。

**低自由度（具体脚本、少量参数）**：当操作脆弱且易出错、一致性至关重要，或必须遵循特定顺序时使用。

把 agent 想象成在探索一条路径：悬崖边的窄桥需要明确的护栏（低自由度），而开阔的田野允许许多路线（高自由度）。

### 技能的结构

每个技能都由必需的 SKILL.md 文件和可选的打包资源组成：

```
skill-name/
├── SKILL.md (required)
│   ├── YAML frontmatter metadata (required)
│   │   ├── name: (required)
│   │   ├── description: (required)
│   │   └── license: (optional)
│   └── Markdown instructions (required)
└── Bundled Resources (optional)
    ├── scripts/          - Executable code (Python/Bash/etc.)
    ├── references/       - Documentation intended to be loaded into context as needed
    └── assets/           - Files used in output (templates, icons, fonts, etc.)
```

#### SKILL.md（必需）

每个 SKILL.md 都包含：

- **Frontmatter**（YAML）：包含 `name` 和 `description` 字段（必需），以及 `license` 等可选字段。agent 只读取 `name` 和 `description` 来判断技能何时触发，因此要清晰、全面地说明技能是什么以及何时应使用它。
- **正文**（Markdown）：使用技能的说明和指导。仅在技能触发后才加载（如果会加载的话）。

#### 打包资源（可选）

##### 脚本（`scripts/`）

用于需要确定性可靠性或会被反复重写的任务的可执行代码（Python/Bash 等）。

- **何时包含**：当同一段代码被反复重写，或需要确定性可靠性时
- **示例**：用于 PDF 旋转任务的 `scripts/rotate_pdf.py`
- **优点**：token 高效、确定性，可以无需读入上下文即可执行
- **注意**：agent 仍可能需要读取脚本以进行打补丁或做环境特定的调整

##### 参考文档（`references/`）

按需加载到上下文中，用于为 agent 的流程和思考提供信息的文档与参考资料。

- **何时包含**：当 agent 在工作过程中应参考的文档
- **示例**：用于财务 schema 的 `references/finance.md`，用于 API 规范的 `references/api_docs.md`
- **使用场景**：数据库 schema、API 文档、领域知识、公司政策、详细工作流指南
- **优点**：让 SKILL.md 保持精简，仅在 agent 判断需要时加载
- **最佳实践**：如果文件较大（>1 万词），请在 SKILL.md 中包含 grep 搜索模式
- **避免重复**：信息应只存在于 SKILL.md 或 references 文件之一，而不是两者。除非信息确实是技能的核心，否则优先放在 references 文件中。

##### 素材（`assets/`）

不打算加载到上下文中，而是用于 agent 产出物中的文件。

- **何时包含**：当技能需要用于最终输出的文件时
- **示例**：用于品牌素材的 `assets/logo.png`，用于 PowerPoint 模板的 `assets/slides.pptx`，用于 HTML/React 样板代码的 `assets/frontend-template/`
- **使用场景**：模板、图片、图标、样板代码、字体、会被复制或修改的示例文档
- **优点**：将输出资源与文档分离，使 agent 无需将文件加载到上下文中即可使用

#### 技能中不应包含什么

技能应只包含直接支持其功能的必要文件。不要创建多余的文档或辅助文件，包括：

- README.md
- INSTALLATION_GUIDE.md
- QUICK_REFERENCE.md
- CHANGELOG.md
- 等

技能应只包含 AI agent 完成当前任务所需的信息。

### 渐进式披露设计原则

技能使用三级加载系统来高效管理上下文：

1. **元数据（name + description）** — 始终在上下文中（约 100 词）
2. **SKILL.md 正文** — 技能触发时加载（<5000 词）
3. **打包资源** — 由 agent 按需加载（无限制，因为脚本可以无需读入上下文窗口即可执行）

#### 渐进式披露模式

将 SKILL.md 正文保持在要点范围内，且不超过 500 行，以尽量减少上下文膨胀。接近此限制时，将内容拆分到单独文件中。将内容拆分到其他文件时，务必从 SKILL.md 中引用它们，并清楚说明何时读取。

**模式 1：带参考文档的高层指南**

```markdown
# PDF Processing

## Quick start

Extract text with pdfplumber:
[code example]

## Advanced features

- **Form filling**: See [FORMS.md](FORMS.md) for complete guide
- **API reference**: See [REFERENCE.md](REFERENCE.md) for all methods
```

**模式 2：按领域组织**

```
bigquery-skill/
├── SKILL.md (overview and navigation)
└── reference/
    ├── finance.md (revenue, billing metrics)
    ├── sales.md (opportunities, pipeline)
    └── product.md (API usage, features)
```

当用户询问销售指标时，agent 只读取 sales.md。

**模式 3：条件性细节**

```markdown
# DOCX Processing

## Creating documents

Use docx-js for new documents. See [DOCX-JS.md](DOCX-JS.md).

## Editing documents

For simple edits, modify the XML directly.

**For tracked changes**: See [REDLINING.md](REDLINING.md)
**For OOXML details**: See [OOXML.md](OOXML.md)
```

## 技能创建流程

技能创建包括以下步骤：

1. 通过具体示例理解技能
2. 规划可复用的技能内容（脚本、参考文档、素材）
3. 初始化技能（运行 init_skill.py）
4. 编辑技能（实现资源并编写 SKILL.md）
5. 根据实际使用情况迭代

按顺序执行这些步骤，仅在明确不适用时跳过。

### 步骤 1：通过具体示例理解技能

仅在技能的使用模式已经清晰明了时才跳过此步骤。即使处理现有技能，此步骤仍然有价值。

要创建高效的技能，需要清楚理解技能将如何被使用的具体示例。这种理解可以来自用户直接提供的示例，也可以来自经过用户反馈验证的生成示例。

例如，在构建图片编辑器技能时，相关问题包括：

- “image-editor 技能应支持哪些功能？编辑、旋转，还是其他？”
- “你能举例说明这个技能将如何被使用吗？”
- “用户会说什么话来触发这个技能？”

当对技能应支持的功能有清晰认识时，结束此步骤。

### 步骤 2：规划可复用的技能内容

要将具体示例转化为高效的技能，分析每个示例时：

1. 考虑如何从零开始执行该示例
2. 确定在重复执行这些工作流时，哪些脚本、参考文档和素材会有帮助

示例：在构建 `pdf-editor` 技能以处理“帮我旋转这个 PDF”这类查询时，分析表明：

1. 旋转 PDF 需要每次都重写相同的代码
2. 将 `scripts/rotate_pdf.py` 脚本存储在技能中会有帮助

示例：在设计用于“帮我构建一个待办事项应用”这类查询的 `frontend-webapp-builder` 技能时，分析表明：

1. 编写前端 webapp 每次都需要相同的 HTML/React 样板代码
2. 将包含样板 HTML/React 项目文件的 `assets/hello-world/` 模板存储在技能中会有帮助

要确定技能的内容，请分析每个具体示例，整理出要包含的可复用资源清单：脚本、参考文档和素材。

### 步骤 3：初始化技能

此时，该实际创建技能了。

仅在要开发的技能已经存在，且需要迭代或打包时跳过此步骤。

从零开始创建新技能时，始终运行 `init_skill.py` 脚本。该脚本会生成一个新的模板技能目录，自动包含技能所需的一切。

用法：

```bash
python .github/skills/skill-creator/scripts/init_skill.py <skill-name> --path .github/skills
```

该脚本会：

- 在指定路径创建技能目录
- 生成带有正确 frontmatter 和 TODO 占位符的 SKILL.md 模板
- 创建示例资源目录：`scripts/`、`references/` 和 `assets/`
- 在每个目录中添加可自定义或删除的示例文件

初始化后，按需自定义或删除生成的 SKILL.md 和示例文件。

### 步骤 4：编辑技能

编辑（新生成或现有的）技能时，请记住该技能是供另一个 agent 实例使用的。包含有益且非显而易见的信息。考虑哪些程序性知识、领域特定细节或可复用素材能帮助另一个 agent 实例更有效地执行这些任务。

#### 学习经过验证的设计模式

根据技能需求查阅这些有用的指南：

- **多步骤流程**：有关顺序工作流和条件逻辑，参见 [references/workflows.md](references/workflows.md)
- **特定输出格式或质量标准**：有关模板和示例模式，参见 [references/output-patterns.md](references/output-patterns.md)

这些文件包含经过验证的技能设计最佳实践。

#### 从可复用的技能内容开始

要开始实现，请从上面确定的可复用资源入手：`scripts/`、`references/` 和 `assets/` 文件。注意，此步骤可能需要用户输入。

添加的脚本必须通过实际运行来测试，以确保没有 bug 且输出符合预期。

技能不需要的任何示例文件和目录都应删除。

#### 更新 SKILL.md

**编写指南：** 始终使用祈使/不定式形式。

##### Frontmatter

编写包含 `name` 和 `description` 的 YAML frontmatter：

- `name`：技能名称（小写，用连字符代替空格）
- `description`：这是技能的主要触发机制。
  - 同时包含技能做什么，以及何时使用它的具体触发条件/上下文。
  - 在此处包含所有“何时使用”信息 — 不要放在正文中。正文仅在触发后加载，因此正文中的“何时使用此技能”章节没有帮助。
  - `docx` 技能的描述示例：“全面的文档创建、编辑和分析，支持修订、批注、格式保留和文本提取。当 agent 需要处理专业文档（.docx 文件）以完成以下任务时使用：(1) 创建新文档，(2) 修改或编辑内容，(3) 处理修订，(4) 添加批注，或任何其他文档任务”

除 `name` 和 `description` 外，不要在 YAML frontmatter 中包含任何其他必填字段。

##### 正文

编写使用技能及其打包资源的说明。

### 步骤 5：迭代

测试技能后，用户可能会请求改进。

**迭代工作流：**

1. 在真实任务中使用技能
2. 留意困难或低效之处
3. 确定应如何更新 SKILL.md 或打包资源
4. 实现更改并再次测试
