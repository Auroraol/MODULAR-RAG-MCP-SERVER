# 文档切分（Chunking）

本目录负责把 `Document` 切分为 `Chunk`。当前只有一个实现：`DocumentChunker`（适配层）+ `libs.splitter.recursive`（切分算法）。

## 调用链

```
Pipeline.run()
  └─ DocumentChunker.split_document(document)      # 本目录，只做适配
       └─ SplitterFactory.create(settings)          # src/libs/splitter
            └─ RecursiveSplitter.split_text(text)   # LangChain RecursiveCharacterTextSplitter
```

## `split_document` 的职责

`DocumentChunker` 本身**不切分**，只做业务对象转换（`document_chunker.py:74`）：

1. 委托 `self._splitter.split_text(document.text)` 得到纯文本片段
2. 生成确定性 Chunk ID：`{doc_id}_{index:04d}_{sha256(text)[:8]}`
3. 继承 document 元数据，补充 `chunk_index`、`source_ref`
4. 从文本中提取 `[IMAGE: xxx]` 占位符 → `image_refs`，并按 ID 回填 chunk 级 `images`
5. 尝试用首个引用图片的 `page` 填充 `page_num`

同一 Document 重复切分结果完全一致（ID 确定、无随机性）。

## 当前策略：Recursive

配置项：`config/settings.yaml`

```yaml
ingestion:
  splitter: "recursive"   # 可选项注释写着 recursive / semantic / fixed_length
  chunk_size: 1000
  chunk_overlap: 200
```

分隔符按优先级依次尝试（`src/libs/splitter/recursive_splitter.py:42`）：

```
"\n\n" → "\n" → ". " → "! " → "? " → "; " → ", " → " " → ""
```

短文段会被合并回 `chunk_size` 以内，相邻 Chunk 保留 `chunk_overlap` 字符重叠。超过上限的段落自动降级到下一级分隔符。

> 注意：settings 注释里列出了 `semantic` / `fixed_length`，但 **`_register_builtin_providers()` 目前只注册了 `recursive`**（`splitter_factory.py:25-28`），配置成其它值会直接抛 `Unsupported Splitter provider`。

## 常见分块策略对比

| 策略 | 依据 | 优点 | 缺点 |
|---|---|---|---|
| Fixed-length | 固定字符/token 数 + overlap | 最简单、最快 | 硬切句子，语义断裂 |
| **Recursive**（当前） | 分隔符优先级递归降级 | 尽量保段落/句子完整，通用 | 不理解语义 |
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
2. **代码块会被切断**：docstring 说"Markdown 感知"，但分隔符列表里没有 ` ``` ` 和 `# `。需要保留代码块可改用 LangChain 的 `MarkdownTextSplitter` 作为新 provider。

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