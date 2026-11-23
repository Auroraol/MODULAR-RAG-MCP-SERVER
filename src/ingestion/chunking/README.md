# 文档切分（Chunking）

本目录负责把 `Document` 切分为 `Chunk`。`DocumentChunker` 是适配层，实际切分算法在 `src/libs/splitter`，当前提供 6 种 provider。

## 模块总流程

```mermaid
flowchart TD
    A["Document.text<br/>(Loader 已统一转成 Markdown)"] --> B["DocumentChunker.split_document()"]
    B --> C{"text 为空?"}
    C -- 是 --> X["ValueError"]
    C -- 否 --> D["self._splitter.split_text(text)<br/>实例由 __init__ 时的<br/>SplitterFactory.create(settings) 决定"]

    D --> P{"ingestion.splitter"}

    P --> R["recursive"]
    P --> F["fixed_length"]
    P --> M["markdown"]
    P --> S["structure"]
    P --> Sem["semantic"]
    P --> Prop["propositional"]

    R --> R1["分隔符按优先级递归降级<br/>\n\n → \n → '. ' → … → 字符<br/>合并至 chunk_size，保留 overlap"]
    F --> F1["定长滑动 + overlap<br/>切点尾部 20% 有空白则回退"]
    M --> M1["LangChain MarkdownTextSplitter<br/>标题 / 代码块为高优先级分隔符"]
    S --> S1["按 ATX 标题切 section<br/>小章节合并<br/>超大章节二次切 + 补标题路径"]
    Sem --> Sem1["切句 → embedding<br/>余弦距离 → 95 分位处断开"]
    Sem1 -- "embedding 调用失败" --> R1
    Prop --> Prop1["按窗口调 LLM 抽原子命题<br/>打包到 chunk_size"]
    Prop1 -- "LLM 调用失败" --> Prop2["保留原文窗口"]

    R1 --> Z["List[str] 文本片段"]
    F1 --> Z
    M1 --> Z
    S1 --> Z
    Sem1 --> Z
    Prop2 --> Z

    Z --> G["生成 Chunk ID<br/>doc_id + 序号 + 文本哈希前 8 位"]
    G --> H["继承 document.metadata<br/>+ chunk_index + source_ref"]
    H --> I["正则 \[IMAGE: xxx\]<br/>→ image_refs<br/>回填 chunk 级 images / page_num"]
    I --> J["List[Chunk] → ChunkRefiner → Enricher → Embedding"]
```

工厂在 `DocumentChunker.__init__` 时选一次 provider，`split_document` 每次调用都走同一个；换策略要改配置并重建 pipeline。

## `split_document` 的职责

`DocumentChunker` 本身**不切分**，只做业务对象转换（`document_chunker.py:74`）：

1. 委托 `self._splitter.split_text(document.text)` 得到纯文本片段
2. 生成确定性 Chunk ID：`{doc_id}_{index:04d}_{sha256(text)[:8]}`
3. 继承 document 元数据，补充 `chunk_index`、`source_ref`
4. 从文本中提取 `[IMAGE: xxx]` 占位符 → `image_refs`，并按 ID 回填 chunk 级 `images`
5. 尝试用首个引用图片的 `page` 填充 `page_num`

同一 Document 重复切分结果完全一致（ID 确定、无随机性）。

## 默认策略：Recursive

配置项：`config/settings.yaml`

```yaml
ingestion:
  splitter: "recursive"   # recursive / fixed_length / markdown / semantic / structure / propositional
  chunk_size: 1000
  chunk_overlap: 200
```

分隔符按优先级依次尝试（`src/libs/splitter/recursive_splitter.py:42`）：

```
"\n\n" → "\n" → ". " → "! " → "? " → "; " → ", " → " " → ""
```

短文段会被合并回 `chunk_size` 以内，相邻 Chunk 保留 `chunk_overlap` 字符重叠。超过上限的段落自动降级到下一级分隔符。

> 已注册的 provider：`recursive`（默认）、`fixed_length`、`markdown`、`semantic`、`structure`、`propositional`（`splitter_factory.py:_register_builtin_providers`）。`markdown` 依赖 `langchain-text-splitters`，缺失时自动不注册。

各策略适用场景：

| provider | 适用 |
|---|---|
| `recursive` | 默认，混合格式通用 |
| `fixed_length` | 无结构纯文本、需要严格等长块的场景 |
| `markdown` | Markdown 文档，保护代码块与标题 |
| `structure` | Loader 输出的 Markdown（PDF/Office 均会转成 Markdown），按标题切章节、每块带标题路径 |
| `semantic` | 主题跨度大的长文，愿付一次 embedding 成本 |
| `propositional` | 高价值条款/FAQ，每块是 LLM 抽出的自包含原子命题，最贵 |

## StructureSplitter 结构化切分器内部流程

代码：`src/libs/splitter/structure_splitter.py`

```mermaid
flowchart TD
    A["split_text(text)"] --> B["逐行扫描（splitlines keepends）"]
    B --> C{"行首是 # ~ ###### 标题?"}
    C -- 标题行 --> D["flush：当前正文 + 当前标题栈<br/>→ 一个 section"]
    D --> E["标题栈：弹出 level ≥ 新 level 的项<br/>压入 (level, 标题行)"]
    C -- 正文行 --> F["追加到 body_lines"]
    E --> B
    F --> B
    B -- 扫描结束 --> G["flush 最后一个 section"]
    G --> H["遍历 sections：<br/>piece = 标题路径 + 正文"]
    H --> I{"len(piece) ≤ chunk_size?"}
    I -- 是 --> J{"current + piece > chunk_size?"}
    J -- 是 --> K["输出 current<br/>current = piece"]
    J -- 否 --> L["current += piece"]
    I -- 否 --> M["先输出 current"]
    M --> N["_split_oversized(prefix, body)"]
    N --> O["budget = max(cs/2, cs - len(prefix))<br/>overlap = min(ov, budget - 1)"]
    O --> P{"recursive 可用?"}
    P -- 是 --> Q["RecursiveSplitter(body)<br/>每段前缀标题路径"]
    P -- 否 --> R["FixedLengthSplitter(prefix+body)"]
    K --> S["汇总 chunks"]
    L --> S
    Q --> S
    R --> S
    S --> T{"chunks 为空?"}
    T -- 是 --> U["[text]（如标题下无正文）"]
    T -- 否 --> V["返回 List[str]"]
    U --> V
```

关键点：

- **标题栈**决定"标题路径"：`# Root` + `## A` 下的正文，路径为 `# Root\n## A\n`，会原样拼到该 section 每个 Chunk 前面。
- **相邻小 section 会合并**（`current += piece`），避免每节只有一句话时产生大量碎片。
- **只有超大 section 才二次切分**；`budget` 给标题路径让位，路径比 chunk_size 一半还长时退回 `chunk_size//2`（`ponytail` 注释标记了这个兜底）。
- 无标题文本 = 一个 section（路径为空），本策略退化为"整段或按大小切"。

## SemanticSplitter 语义切分器内部流程

代码：`src/libs/splitter/semantic_splitter.py`

```mermaid
flowchart TD
    A["split_text(text)"] --> B["按句末标点切句<br/>。！？；!?; 以及 '. ' 空白"]
    B --> C{"句子数 < 3?"}
    C -- 是 --> D{"len(text) ≤ chunk_size?"}
    D -- 是 --> E["[text]（没有语义边界可言）"]
    D -- 否 --> F["_fallback: RecursiveSplitter"]
    C -- 否 --> G["_get_embedding().embed(sentences)"]
    G -- "异常（API/配置）" --> H["warning 日志"] --> F
    G --> I["distances[i] = 1 - cos(e[i], e[i+1])"]
    I --> J{"max(distances) - min(distances) < 1e-9?"}
    J -- 是 --> K["threshold = ∞<br/>不做语义断开，只按大小切"]
    J -- 否 --> L["threshold = distances 的<br/>breakpoint_percentile 分位值（默认 95）"]
    K --> M["逐句追加到 current"]
    L --> M
    M --> N{"distances[i-1] ≥ threshold<br/>或 current+当前句 > chunk_size?"}
    N -- 是 --> O["输出 current<br/>current = 当前句"]
    N -- 否 --> P["current += 当前句"]
    O --> Q{"还有句子?"}
    P --> Q
    Q -- 是 --> M
    Q -- 否 --> R["输出最后一个 current"]
    E --> S["返回 List[str]"]
    F --> S
    R --> S
```

关键点：

- 断点阈值用**分位数**而不是固定值：默认 95 分位，只在最高的约 5% 语义跳变处断开；`breakpoint_percentile` 调小切得更碎。
- 距离全部相同（例如重复句式）时视为"无语义跳变"，`threshold = ∞`，避免每句都成一块。
- 单块大小用 `_join(current + [下一句]).length` 判断，保证 Chunk 不会因拼接空格而超出 `chunk_size`。
- 拼接规则：中文标点后不补空格（`。！？；，、`），其余补一个空格，尽量还原原文。
- **不产生 overlap**（`chunk_overlap` 在本策略中不生效），靠语义边界保证连贯。
- 两句以下或 embedding 失败 → 降级 `RecursiveSplitter`；降级也失败（缺 langchain）会把 `ImportError` 抛给调用方。

## 常见分块策略对比

| 策略 | 依据 | 优点 | 缺点 |
|---|---|---|---|
| Fixed-length | 固定字符/token 数 + overlap | 最简单、最快 | 硬切句子，语义断裂 |
| **Recursive**（默认） | 分隔符优先级递归降级 | 尽量保段落/句子完整，通用 | 不理解语义 |
| Sentence / Paragraph | 标点、空行 | 边界自然 | 块大小不可控 |
| Structure-aware | Markdown / HTML / 表格结构 | 保留文档层级 | 依赖原文结构化 |
| Semantic | embedding 相似度突变点 | 主题边界准 | 摄入阶段即需算 embedding，慢且贵 |
| Parent-child | 小块检索、大块喂 LLM | 检索精度 + 上下文兼顾 | 索引结构复杂 |
| LLM / Agentic | LLM 判断边界 | 质量最高 | 极慢极贵，不适合批量 |

## 为什么选 Recursive

- **格式通用**：LoaderFactory 支持 pdf/txt/md/csv/json/docx/pptx/xlsx/html，recursive 不需要为每种格式单独适配。
- **零 LLM 成本**：摄入阶段不调用模型，结果确定、可复现。
- **下游有兜底**：Pipeline 后续还有 `ChunkRefiner`（LLM 精炼）和 `MetadataEnricher`，语义质量可在那里补齐。
- **可替换**：`BaseSplitter` + `SplitterFactory.register_provider` 已留好扩展口，加新策略不需要改上游代码。

## 已知问题

1. **中文切分不理想**：默认分隔符里的 `. ` `! ` `? ` 是带空格的英文句末标点，中文几乎不命中，长中文段落会一路降级到 `" "` 甚至按字符切。建议注册一个分隔符包含 `。！？；` 的版本。
2. **recursive 会切断代码块**：其分隔符列表里没有 ` ``` ` 和 `# `。需要保护代码块/章节时改用 `markdown` 或 `structure`。

## 新增策略的方式

```python
# 1. 实现 src/libs/splitter/<your>_splitter.py
class MySplitter(BaseSplitter):
    def split_text(self, text, trace=None, **kwargs) -> list[str]: ...

# 2. 在 splitter_factory._register_builtin_providers() 中注册
SplitterFactory.register_provider("my", MySplitter)

# 3. 改配置
ingestion:
  splitter: "my"
```

`DocumentChunker` 无需任何改动。