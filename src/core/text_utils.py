"""共享文本分词工具。

检索链路的**索引侧**（``SparseEncoder``）与**查询侧**（``QueryProcessor``）
必须使用完全一致的分词规则，否则 BM25 无法匹配。本模块提供该唯一实现，
两边共同调用。

分词规则：
1. 使用 jieba 切分中英文（保留中文词、英文单词、数字如 ``3.11``）；
2. 若 ``-`` 或 ``_`` 夹在两个字母数字词项之间，则与两侧合并为一个词项，
   使 ``machine-learning``、``deep_learning``、``GPT-4`` 保持完整；
3. 丢弃纯标点 / 纯空白词项。
"""

from __future__ import annotations

import re

import jieba

# 纯标点或纯空白的词项（不含任何字母、数字、汉字）
_PUNCT_ONLY = re.compile(r"[\s\W]+", re.UNICODE)

# 连接符：出现在两个字母数字词项之间时用于合并
_JOINERS = {"-", "_"}


def _is_word(token: str) -> bool:
    """判断词项是否含有实质内容（非纯标点/空白）。"""
    return bool(token) and not _PUNCT_ONLY.fullmatch(token)


def tokenize(text: str) -> list[str]:
    """将文本切分为词项列表。

    参数:
        text: 待分词的输入文本。

    返回:
        词项列表（保留原始大小写，调用方决定是否小写化）。

    示例:
        >>> tokenize("machine-learning GPT-4")
        ['machine-learning', 'GPT-4']
    """
    raw_tokens = jieba.lcut(text)

    tokens: list[str] = []
    index = 0
    total = len(raw_tokens)
    while index < total:
        token = raw_tokens[index].strip()
        index += 1

        if not _is_word(token):
            continue

        # 合并 a - b / a _ b 形式的连写词
        merged = token
        while index + 1 < total:
            joiner = raw_tokens[index]
            following = raw_tokens[index + 1].strip()
            if joiner in _JOINERS and _is_word(following):
                merged = f"{merged}{joiner}{following}"
                index += 2
            else:
                break

        tokens.append(merged)

    return tokens
