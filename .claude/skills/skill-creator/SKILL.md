---
name: skill-creator
description: 创建高效 agent 技能的指南。当用户想要创建新技能（或更新现有技能），以专业知识、工作流或工具集成扩展 agent 的能力时，应使用本技能。当用户说 "create skill"、"new skill"、"build a skill"、"update skill"，或要求制作可复用的能力包时使用。
---

# 技能创建器

本技能提供创建高效技能的指导。

## 关于技能

技能是模块化、自包含的包，通过提供专业知识、工作流和工具来扩展 agent 的能力。可以将其视为特定领域或任务的“入门指南”——它们将通用 agent 转变为配备程序化知识的专用 agent，而这些知识是任何模型都无法完全具备的。

### 技能提供什么

1. 专业化工作流 —— 针对特定领域的多步骤流程
2. 工具集成 —— 处理特定文件格式或 API 的说明
3. 领域专业知识 —— 公司特定知识、schema、业务逻辑
4. 捆绑资源 —— 用于复杂和重复任务的脚本、参考资料和资产

## 核心原则

### 简洁是关键

上下文窗口是一种公共资源。技能需与 agent 所需的其他一切共享上下文窗口：系统提示词、对话历史、其他技能的元数据以及实际用户请求。

**默认假设：agent 已经非常聪明。** 只添加 agent 尚不具备的上下文。对每条信息提出质疑：“agent 真的需要这个解释吗？”以及“这段文字是否值得其 token 成本？”

优先使用简洁的示例而非冗长的解释。

### 设置适当的自由度

根据任务的脆弱性和可变性匹配具体程度：

**高自由度（基于文本的指令）**：当存在多种有效方法、决策依赖上下文，或由启发式方法引导时使用。

**中等自由度（伪代码或带参数的脚本）**：当存在首选模式、可以接受一定变化，或配置会影响行为时使用。

**低自由度（特定脚本、少量参数）**：当操作脆弱且易出错、一致性至关重要，或必须遵循特定顺序时使用。

将 agent 想象成在探索一条路径：狭窄的悬崖桥需要具体的护栏（低自由度），而开阔的田野允许多条路线（高自由度）。

### 技能的组成结构

每个技能都由一个必需的 SKILL.md 文件和可选的捆绑资源组成：

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

- **Frontmatter**（YAML）：包含 `name` 和 `description` 字段（必需），以及 `license` 等可选字段。agent 只读取 `name` 和 `description` 来判断技能何时触发，因此要清晰、全面地说明技能是什么以及应在何时使用。
- **正文**（Markdown）：使用该技能的说明和指导。仅在技能触发后才加载（如果会加载的话）。

#### 捆绑资源（可选）

##### 脚本（`scripts/`）

用于需要确定性可靠性或会被反复重写的任务的可执行代码（Python/Bash 等）。

- **何时包含**：当同一段代码被反复重写，或需要确定性可靠性时
- **示例**：用于 PDF 旋转任务的 `scripts/rotate_pdf.py`
- **优势**：节省 token、确定性、可无需读入上下文即可执行
- **注意**：脚本可能仍需要 agent 读取，以便打补丁或做特定环境的调整

##### 参考资料（`references/`）

旨在按需加载到上下文中，为 agent 的流程和思考提供信息的文档与参考材料。

- **何时包含**：当 agent 在工作时应参考的文档
- **示例**：用于财务 schema 的 `references/finance.md`，用于 API 规范的 `references/api_docs.md`
- **使用场景**：数据库 schema、API 文档、领域知识、公司政策、详细的工作流指南
- **优势**：保持 SKILL.md 精简，仅在 agent 判断需要时加载
- **最佳实践**：如果文件较大（>10k 词），在 SKILL.md 中包含 grep 搜索模式
- **避免重复**：信息应只存在于 SKILL.md 或参考资料文件中，而非两者皆有。详细信息优先放在参考资料文件中，除非它确实是该技能的核心内容。

##### 资产（`assets/`）

不打算加载到上下文中，而是在 agent 产生的输出中使用的文件。

- **何时包含**：当技能需要用于最终输出的文件时
- **示例**：用于品牌资产的 `assets/logo.png`，用于 PowerPoint 模板的 `assets/slides.pptx`，用于 HTML/React 样板的 `assets/frontend-template/`
- **使用场景**：模板、图片、图标、样板代码、字体、会被复制或修改的示例文档
- **优势**：将输出资源与文档分离，使 agent 可以使用文件而无需将其加载到上下文中

#### 不应包含在技能中的内容

技能应只包含直接支持其功能所必需的文件。不要创建多余的文档或辅助文件，包括：

- README.md
- INSTALLATION_GUIDE.md
- QUICK_REFERENCE.md
- CHANGELOG.md
- 等

技能应只包含 AI agent 完成手头工作所需的信息。

### 渐进式披露设计原则

技能使用三级加载系统来高效管理上下文：

1. **元数据（name + description）** —— 始终在上下文中（约 100 词）
2. **SKILL.md 正文** —— 技能触发时（<5k 词）
3. **捆绑资源** —— 按 agent 需要（无限，因为脚本可以执行而无需读入上下文窗口）

#### 渐进式披露模式

将 SKILL.md 正文保持在要点范围并控制在 500 行以内，以尽量减少上下文膨胀。接近此限制时，将内容拆分到单独文件中。将内容拆分到其他文件时，务必从 SKILL.md 中引用它们，并清楚说明何时读取它们。

**模式 1：带参考资料的高层指南**

```markdown
# PDF Processing

## Quick start

Extract text with pdfplumber:
[code example]

## Advanced features

- **Form filling**: See [FORMS.md](FORMS.md) for complete guide
- **API reference**: See [REFERENCE.md](REFERENCE.md) for all methods
```

**模式 2：领域特定组织**

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
2. 规划可复用的技能内容（脚本、参考资料、资产）
3. 初始化技能（运行 init_skill.py）
4. 编辑技能（实现资源并编写 SKILL.md）
5. 根据实际使用情况迭代

按顺序执行这些步骤，仅在有明确理由说明不适用时才跳过。

### 步骤 1：通过具体示例理解技能

仅在已清楚理解技能的使用模式时才跳过此步骤。即使处理现有技能，这一步仍然很有价值。

要创建高效的技能，需清楚理解技能将如何被使用的具体示例。这种理解可以来自直接的用户示例，也可以来自通过用户反馈验证过的生成示例。

例如，在构建 image-editor 技能时，相关问题包括：

- “image-editor 技能应支持哪些功能？编辑、旋转，还是其他功能？”
- “能否举一些该技能如何使用的例子？”
- “用户会说什么话来触发这个技能？”

当对技能应支持的功能有清晰的认识时，即可结束此步骤。

### 步骤 2：规划可复用的技能内容

要将具体示例转化为高效的技能，通过以下方式分析每个示例：

1. 考虑如何从零开始执行该示例
2. 确定在重复执行这些工作流时哪些脚本、参考资料和资产会有帮助

示例：在构建 `pdf-editor` 技能以处理“帮我旋转这个 PDF”之类的请求时，分析表明：

1. 旋转 PDF 每次都需要重写相同的代码
2. 将 `scripts/rotate_pdf.py` 脚本存储在技能中会有帮助

示例：在设计 `frontend-webapp-builder` 技能以处理“给我构建一个 todo 应用”之类的请求时，分析表明：

1. 编写前端 webapp 每次都需相同的 HTML/React 样板
2. 将包含样板 HTML/React 项目文件的 `assets/hello-world/` 模板存储在技能中会有帮助

为了确定技能的内容，分析每个具体示例，列出要包含的可复用资源：脚本、参考资料和资产。

### 步骤 3：初始化技能

此时，可以实际创建技能了。

仅在所开发的技能已存在，且需要迭代或打包时才跳过此步骤。

从零开始创建新技能时，始终运行 `init_skill.py` 脚本。该脚本会生成一个新的模板技能目录，自动包含技能所需的一切。

用法：

```bash
python .claude/skills/skill-creator/scripts/init_skill.py <skill-name> --path .claude/skills
```

该脚本会：

- 在指定路径创建技能目录
- 生成带有正确 frontmatter 和 TODO 占位符的 SKILL.md 模板
- 创建示例资源目录：`scripts/`、`references/` 和 `assets/`
- 在每个目录中添加可自定义或删除的示例文件

初始化后，根据需要自定义或删除生成的 SKILL.md 和示例文件。

### 步骤 4：编辑技能

编辑（新生成或现有的）技能时，请记住技能是为 agent 的另一个实例使用而创建的。包含有益且不显而易见的信息。考虑哪些程序化知识、领域特定细节或可复用资产能帮助另一个 agent 实例更有效地执行这些任务。

#### 学习经过验证的设计模式

根据技能需求参考这些有用的指南：

- **多步骤流程**：有关顺序工作流和条件逻辑，请参阅 [references/workflows.md](references/workflows.md)
- **特定输出格式或质量标准**：有关模板和示例模式，请参阅 [references/output-patterns.md](references/output-patterns.md)

这些文件包含经过验证的有效技能设计最佳实践。

#### 从可复用的技能内容开始

要开始实现，请从上面确定的可复用资源开始：`scripts/`、`references/` 和 `assets/` 文件。注意，此步骤可能需要用户输入。

添加的脚本必须通过实际运行来测试，以确保没有 bug 且输出符合预期。

技能不需要的任何示例文件和目录都应删除。

#### 更新 SKILL.md

**写作指南：** 始终使用祈使/不定式形式。

##### Frontmatter

编写包含 `name` 和 `description` 的 YAML frontmatter：

- `name`：技能名称（小写，用连字符代替空格）
- `description`：这是技能的主要触发机制。
  - 同时包含技能做什么以及使用它的具体触发条件/上下文。
  - 在此处包含所有“何时使用”信息——不要放在正文中。正文仅在触发后加载，因此正文中的“何时使用本技能”章节没有帮助。
  - `docx` 技能的示例描述：“全面的文档创建、编辑和分析，支持修订跟踪、批注、格式保留和文本提取。当 agent 需要处理专业文档（.docx 文件）时使用：(1) 创建新文档，(2) 修改或编辑内容，(3) 处理修订跟踪，(4) 添加批注，或任何其他文档任务”

除 `name` 和 `description` 外，不要在 YAML frontmatter 中包含任何其他必需字段。

##### 正文

编写使用该技能及其捆绑资源的说明。

### 步骤 5：迭代

测试技能后，用户可能会要求改进。

**迭代工作流：**

1. 在实际任务中使用该技能
2. 注意困难或低效之处
3. 确定应如何更新 SKILL.md 或捆绑资源
4. 实施变更并再次测试
