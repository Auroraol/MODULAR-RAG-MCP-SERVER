# 字典

## 创建

Python 里没有“声明但未初始化”的字典，最推荐直接初始化：

```
# 1. 直接创建空字典（最推荐）
m2 = {}

# 2. 创建同时初始化（有确定键值时推荐）
m3 = {"Go": 100, "java": 90}

# 3. 用 dict() 创建（等价，可读性略低）
m1 = dict()
```

注意：Python 里 `{}` 是空字典，不是空集合；空集合要写 `set()`。

## 安全取值

Python 里 `m[k]` 键不存在会抛 `KeyError`，所以**读不存在的键要用 `.get`**：

```
# 不推荐：键不存在会 KeyError
value = m1[1]

# 推荐：键不存在返回 None
value = m1.get(1)

# 推荐：如果字典本身可能为 `None`，先兜底
value = (m1 or {}).get(1)

# 推荐：给默认值
value = m1.get(1, 0)

# 判断键是否存在 + 取值
if 1 in m1:
    value = m1[1]
```

## 遍历

```python
# 只遍历 key
for key in param_map:
    print(key, param_map[key])

# 遍历 key, value（最常用）
for key, value in param_map.items():
    print(key, value)

# 只遍历 value
for value in param_map.values():
    print(value)
```

注意：遍历时不要同时增删字典，否则可能 `RuntimeError: dictionary changed size during iteration`。需要删就先收集 key 再删：

```
for key in list(param_map.keys()):
    if should_delete(key):
        del param_map[key]
```



# `if` 

Python 中以下值在 `if` 里被视为 `False`：

```
None
False
0
0.0
""
[]
{}
()
set()
```

X if C else Y

```
这是 Python 的条件表达式，语法固定是：
X if C else Y
读的顺序：先中间的条件，再真分支，最后假分支。

new_chunk_ids[i] if i < len(new_chunk_ids) else "—"

1. 判断 i < len(new_chunk_ids)（越界检查）
2. 成立 → 取 new_chunk_ids[i]
3. 不成立 → 取 "—"

等价的 if 写法：

if i < len(new_chunk_ids):
vector_id = new_chunk_ids[i]
else:
vector_id = "—"

```

**布尔运算符**

| 运算符 | 含义             | 短路行为             |
| :----- | :--------------- | :------------------- |
| `and`  | 两边都为真才为真 | 左边为假，右边不执行 |
| `or`   | 一边为真就为真   | 左边为真，右边不执行 |
| `not`  | 取反             | 单目                 |

Python 布尔运算优先级从高到低：not  >  and  >  or

**常见复合形式**

```
# and：两个条件都满足
if img_id and img_id not in image_lookup:
    ...

# or：满足任意一个
if img_id is None or img_id == "":
    ...

# not + and
if not img_id or img_id in image_lookup:
    ...

# 混合，用括号明确优先级
if (img_id and img_id not in image_lookup) or force:
    ...
```





# 元组

```python
1. 数据形态决定：平行数组需要"配对"

sparse_stats: [stat0, stat1, stat2]     # 第 i 个 Chunk 的词项统计
chunk_ids:    [vid0,  vid1,  vid2]      # 第 i 个 Chunk 的向量 ID

含义上是"每个 Chunk 有两个属性"。Python 没有内置的 zip-projection，最小做法就是把 (stat_i, vid_i) 组成一条记录 —— 元组就是那条记录：

list(zip(a, b))  # [(stat0, vid0), (stat1, vid1), ...]

2. 元组的语义恰好匹配

- 定长：每对固定 2 个元素，不会多也不会少
- 异质、有角色：一个是 dict，一个是 str，位置有意义
- 不可变：这一对本身不该被改，改的是 dict 里的字段

这三条正是"用元组而不是列表"的判断标准。如果要存 10 个动态数量的同类元素，才是 list；如果要有名字的字段，才是 dataclass/dict。

3. 相比索引写法，元组解包更不容易错

# 索引写法：4 处下标准确性，改错一处就静默出错
for i in range(len(sparse_stats)):
    sparse_stats[i]["chunk_id"] = chunk_ids[i]

# 元组解包：没有下标，没有 len()，角色在循环头就写清楚
for stat, vid in zip(sparse_stats, chunk_ids):
    stat["chunk_id"] = vid

zip 还顺带带来两点：

- 惰性：不生成中间列表，逐条产出
- 通用：两个参数可以是任何可迭代物（生成器、dict 的 .items()、数据库游标），不用能 len() 的列表
```

