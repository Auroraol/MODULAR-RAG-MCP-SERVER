# 查询规划层（planner）

检索之前的编排层：**意图识别 → 意图路由 → 查询改写**。

一句话区分：

- `planner/`：决定「要不要检索、用哪条查询去检索」
- `query_processor.py`：决定「这条查询怎么切成关键词、滤掉什么」

两者是串行关系：`planner.plan()` 产出查询 → `HybridSearch.search()` 内部再调用 `QueryProcessor`。

---

## 数据流

```
用户查询
   │
   ▼
IntentClassifier ──► IntentResult{intent, confidence, metadata}
   │                  greeting / faq / follow_up / knowledge
   ▼
QueryRouter ──────► RouteDecision{route, rewrite_mode}
   │                  查 RoutingPolicy 策略表，不写 if/else
   │
   ├─ DIRECT   → 直答（问候/闲聊），不走向量检索，queries 为空
   ├─ FALLBACK → 兜底 / 转人工
   └─ RETRIEVE → 继续改写
                    │
                    ▼
             BaseRewriter ──► List[str] 待检索查询
                    │            NONE      原样返回（默认）
                    │            REWRITE   结合历史做指代消解
                    │            HYDE      生成假设文档
                    │            SUBQUERY  拆多个子查询
                    ▼
                QueryPlan
```

`QueryPlan` 是唯一出口：

| 字段 | 说明 |
|---|---|
| `query` | 原始查询 |
| `intent` / `confidence` | 意图与置信度 |
| `route` | `retrieve` / `direct` / `fallback` |
| `rewrite_mode` | 实际生效的改写模式 |
| `queries` | 待检索查询；`route != retrieve` 时为空 |

---

## 目录

```
planner/
├── __init__.py     # 统一导出
├── intent.py       # Intent 枚举、IntentResult、IntentClassifier（规则优先）
├── router.py       # Route / RewriteMode 枚举、RoutingPolicy 策略表、QueryRouter
├── rewriter.py     # BaseRewriter 接口；NoOpRewriter / LLMRewriter；create_rewriter
└── pipeline.py     # QueryPipeline：唯一入口 plan()
```

---

## 用法

### 默认（规则 + 不改写，行为与接入前一致）

```python
from src.core.query_engine.planner import QueryPipeline, Route

pipeline = QueryPipeline()
plan = pipeline.plan("BOM25 怎么接线")

plan.route        # Route.RETRIEVE
plan.queries      # ["BOM25 怎么接线"]
```

在检索入口处：

```python
plan = pipeline.plan(query, history=history, trace=trace)

if plan.route is Route.DIRECT:
    return direct_answer(plan.query)          # 问候：不检索

for q in plan.queries:                        # 通常只有一条
    results = hybrid_search.search(q, top_k=top_k, trace=trace)
```

传 `trace` 会记录一个 `query_planning` 阶段（意图、路由、改写结果、耗时），进 Dashboard 的 trace 视图。

### 接 LLM 做改写

`LLMRewriter` 接收任意 `BaseLLM` 实现（`src/libs/llm/`），失败自动回落原查询：

```python
from src.core.query_engine.planner import LLMRewriter, QueryPipeline, RewriteMode
from src.libs.llm.llm_factory import LLMFactory

rewriter = LLMRewriter(LLMFactory.create(settings), RewriteMode.REWRITE)
pipeline = QueryPipeline(rewriter=rewriter)

plan = pipeline.plan("它怎么接线", history=["BOM25 是什么"])
plan.queries      # ["BOM25 主板怎么接线"]
```

三种模式的取舍：

| 模式 | 产出 | 适用 |
|---|---|---|
| `REWRITE` | 一条独立查询 | 多轮追问、指代消解 |
| `HYDE` | 一段假设文档 | 只建议喂**稠密检索**；假设文档不适合 BM25 关键词匹配 |
| `SUBQUERY` | 多条子查询（按行解析） | 复合问题；需要调用方对多查询分别召回再融合 |

### 让 FAQ / LLM 参与意图识别

规则分类器只产 `greeting` / `follow_up` / `knowledge`。`faq` 需要业务知识（FAQ 库命中）或 LLM，通过 `custom` 回调注入，返回 `None` 表示「判不了，交回规则」：

```python
from src.core.query_engine.planner import Intent, IntentClassifier, IntentResult

def faq_lookup(query, history):
    hit = my_faq_store.search(query)
    if hit:
        return IntentResult(Intent.FAQ, hit.score, {"article_id": hit.id})
    return None

pipeline = QueryPipeline(classifier=IntentClassifier(custom=faq_lookup))
```

### 改路由策略

`RoutingPolicy` 是一张表，覆盖默认映射即可，不改流程代码：

```python
from src.core.query_engine.planner import Intent, QueryRouter, RewriteMode, RoutingPolicy

policy = RoutingPolicy(rewrites={Intent.KNOWLEDGE: RewriteMode.HYDE})
pipeline = QueryPipeline(router=QueryRouter(policy))
```

---

## 设计约定

1. **规则优先**：问候、追问用确定性规则，零模型调用、零延迟；LLM 是可选项。
2. **改写不阻断检索**：`LLMRewriter` 任何异常都回落原查询，并在日志中 warning。
3. **路由是数据不是控制流**：加意图 / 换策略只动 `RoutingPolicy`。
4. **本层不执行检索**：只产出计划，检索仍由 `HybridSearch` 负责。

## 测试

```bash
.venv/Scripts/python.exe -m pytest -q tests/unit/test_query_pipeline.py
```

## 尚未接入

- `query_knowledge_hub.execute()` 尚未调用 `QueryPipeline`（接入方式见上文代码片段）
- `SUBQUERY` 多查询并行召回 + RRF 二次融合（可复用 `fusion.RRFFusion`）
- 意图 / 改写参数写入 `config/settings.yaml`（当前用默认值 + 代码覆盖）