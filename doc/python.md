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