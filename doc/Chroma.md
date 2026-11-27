```
CREATE EXTENSION vector;
   
CREATE TABLE chunks (
    id          BIGSERIAL PRIMARY KEY,
    embedding   VECTOR(1536) NOT NULL,   -- 向量列：类型就是 VECTOR(n)
    document    TEXT,                     -- 普通列
    doc_type    TEXT,                     -- 普通列
    source_path TEXT,
    page_num    INT,
    tags        TEXT,                     -- 想存数组可改用 TEXT[] 或 JSONB
    metadata    JSONB
);

-- 向量索引只建在向量列上，并声明距离度量
CREATE INDEX chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- 普通列的索引照旧
CREATE INDEX chunks_doc_type_idx ON chunks (doc_type);

区分的三个层次

1. 类型层：VECTOR(1536) 和 TEXT/INT/JSONB 是不同的列类型。引擎看到 VECTOR 才知道这列能算距离、能进近邻索引；其它列只能等值/范围/JSON
   条件。维度写死在类型里，插入时不匹配直接报错。
2. 索引层：USING hnsw（或 ivfflat）只接受向量列；普通列用 B-tree/GIN。所以"向量检索"实际上是"向量索引扫描"，WHERE 走的是另一套索引。
3. 存储层：向量列存进图索引/量化结构，普通列存进行存/列存。逻辑上是同一张表的一行，物理上是两套结构，这才让 WHERE 和 ORDER BY 距离 能组合。

查询时怎么用

SELECT id, document, doc_type,
       embedding <=> :q AS distance        -- 余弦距离，越小越近
FROM chunks
WHERE doc_type = 'pdf'                     -- 普通列过滤
ORDER BY embedding <=> :q                  -- 向量列排序，走 HNSW
LIMIT 10;

距离操作符是类型自带的：<-> 欧氏、<=> 余弦、<#> 负内积。它的返回值和普通列一样是个数值，所以你甚至能 SELECT 出来、能 ORDER BY，但不能 WHERE embedding = ... 做"等于"匹配。

过滤先还是近邻先

WHERE doc_type = 'pdf' ORDER BY embedding <=> :q LIMIT 10

两种执行计划：

- 先过滤再近邻：WHERE 筛出候选行 → 只在候选里做 ANN。结果必然凑够 10 条。
- 先近邻再过滤：ANN 取 Top-N → 丢掉不满足 WHERE 的。若过滤很苛刻，可能返回不足 10 条。

现代实现（如 pgvector 的 iterative index scan、Chroma/Milvus 的 filtered search）会识别这种组合，优先保证返回数量。这也解释了为什么"向量条件不能进
WHERE"不是限制缺陷——过滤和距离排序本就是两个不同索引各管一摊。

如果表里没声明 VECTOR 类型

那就退化成普通表：embedding 存成 TEXT 或 JSONB，你只能力气很大地在应用层算距离，SQL 层没有任何近邻能力——ORDER BY 全表扫、WHERE 只能精确匹配。是列类型赋予了过滤 +
距离排序组合的能力，不是数据库偷偷识别了"这看着像向量"。
```