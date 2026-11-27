第 4 步：在线查询链路 —— src/core/query_engine/

  完整链路文档见同目录 README.md（入口、六步流程、降级矩阵、配置、Trace）。

  从 hybrid_search.py:203 的 search() 入口往下读，它按 6 个步骤走：

  步骤1 处理查询      → query_processor.py     （关键词提取 + 同义词 + 过滤器解析）
  步骤2 并行检索      → dense_retriever.py     （Embedding → Chroma 向量检索）
                      → sparse_retriever.py    （BM25 关键词检索）
  步骤3 降级处理      → 某路失败就用另一路
  步骤4 融合          → fusion.py              （RRF 倒数排名融合）
  步骤5 元数据过滤    → metadata filter
  步骤6 Top-K 截断

  （可选）reranker.py — CrossEncoder / LLM 精排，配置里 enabled: false 所以默认不启用。

  RRF 一定要看懂，代码就 fusion.py 里十几行：

  score = Σ 1 / (k + rank)    # k 默认 60

  检索之前的编排层看 planner/README.md —— 意图识别 / 路由 / 查询改写，
  决定"要不要检索、用哪条查询检索"，产出 QueryPlan 再交给上面的链路。
