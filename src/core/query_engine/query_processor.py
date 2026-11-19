"""用于预处理用户查询的 Query Processor。

本模块提供查询预处理功能，包括：
- 基于规则的分词与关键词提取
- 中英文停用词过滤
- 从查询语法中解析过滤条件（例如 "collection:docs"）
- 查询归一化与清洗

设计原则：
- 规则优先：使用简单、确定的规则以保证可靠性
- 语言感知：同时支持中文与英文查询
- 可扩展：便于后续添加同义词扩展或基于 LLM 的处理
- 配置驱动：停用词与匹配模式可通过 settings 配置

关键词提取 + 同义词 + 过滤器解析
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Set

from src.core.text_utils import tokenize as shared_tokenize
from src.core.types import ProcessedQuery


# 中文默认停用词
CHINESE_STOPWORDS: Set[str] = {
    # 疑问词
    "如何", "怎么", "怎样", "什么", "哪个", "哪些", "为什么", "为何",
    "谁", "多少", "几", "是否", "能否", "可否",
    # 助词
    "的", "地", "得", "了", "着", "过", "吗", "呢", "吧", "啊", "呀",
    # 介词/连词
    "在", "于", "和", "与", "或", "及", "并", "而", "但", "但是",
    "因为", "所以", "如果", "那么", "虽然", "然而",
    # 代词
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那",
    "这个", "那个", "这些", "那些", "这里", "那里",
    # 副词
    "很", "非常", "特别", "更", "最", "都", "也", "还", "又", "再",
    "已", "已经", "正在", "将", "会", "能", "可以", "应该", "必须",
    # 动词(通用)
    "是", "有", "做", "进行", "使用", "通过",
    # 量词
    "个", "种", "类",
    # 标点等
    "？", "。", "！", "，", "、",
}

# 英文默认停用词
ENGLISH_STOPWORDS: Set[str] = {
    # 冠词
    "a", "an", "the",
    # 介词
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "through", "between", "after", "before",
    # 连词
    "and", "or", "but", "if", "then", "because", "while", "although",
    # 代词
    "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "these", "those", "what", "which", "who", "whom", "whose",
    # 助动词
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can",
    # 常用动词
    "get", "use", "make",
    # 疑问词
    "how", "why", "when", "where",
    # 其他
    "not", "no", "yes", "so", "very", "just", "also", "too",
}

# 合并后的默认停用词
DEFAULT_STOPWORDS: Set[str] = CHINESE_STOPWORDS | ENGLISH_STOPWORDS

# 过滤条件语法模式：key:value
FILTER_PATTERN: Pattern = re.compile(r'(\w+):([^\s]+)')


@dataclass
class QueryProcessorConfig:
    """QueryProcessor 的配置。

    属性:
        stopwords: 需要过滤的词语集合
        min_keyword_length: 关键词被纳入所需的最小长度
        max_keywords: 提取的最大关键词数量
        enable_filter_parsing: 是否从查询中解析过滤条件语法
    """
    stopwords: Set[str] = field(default_factory=lambda: DEFAULT_STOPWORDS.copy())
    min_keyword_length: int = 1
    max_keywords: int = 20
    enable_filter_parsing: bool = True


class QueryProcessor:
    """为用户查询的检索做预处理。

    提取关键词、过滤停用词并解析过滤条件语法，
    从而为稠密检索与稀疏检索准备查询。

    示例:
        >>> processor = QueryProcessor()
        >>> result = processor.process("如何配置 Azure OpenAI？")
        >>> print(result.keywords)
        ['配置', 'Azure', 'OpenAI']
    """

    def __init__(self, config: Optional[QueryProcessorConfig] = None):
        """初始化 QueryProcessor。

        参数:
            config: 可选配置。未提供时使用默认值。
        """
        self.config = config or QueryProcessorConfig()

    def process(self, query: str) -> ProcessedQuery:
        """将用户查询处理为结构化格式。

        参数:
            query: 原始用户查询字符串

        返回:
            包含提取的关键词与过滤条件的 ProcessedQuery
        """
        if not query or not query.strip():
            return ProcessedQuery(
                original_query=query or "",
                keywords=[],
                filters={}
            )

        # 归一化查询
        normalized = self._normalize(query)

        # 从查询中提取过滤条件（例如 "collection:docs"）
        filters, query_without_filters = self._extract_filters(normalized)

        # 分词并提取关键词
        tokens = self._tokenize(query_without_filters)

        # 过滤停用词并应用约束
        keywords = self._filter_keywords(tokens)

        return ProcessedQuery(
            original_query=query,
            keywords=keywords,
            filters=filters
        )

    def _normalize(self, query: str) -> str:
        """归一化查询字符串。

        - 去除首尾空白
        - 归一化 unicode
        - 转换为一致的格式

        参数:
            query: 原始查询字符串

        返回:
            归一化后的查询字符串
        """
        # 去除首尾空白并归一化空白字符
        normalized = " ".join(query.split())
        return normalized

    def _extract_filters(self, query: str) -> tuple[Dict[str, Any], str]:
        """从查询中提取过滤条件语法。

        支持形如 "collection:api-docs keyword1 keyword2" 的语法。

        参数:
            query: 归一化后的查询字符串

        返回:
            (过滤条件字典, 移除过滤语法后的查询) 元组
        """
        if not self.config.enable_filter_parsing:
            return {}, query

        filters: Dict[str, Any] = {}

        # 查找所有过滤条件模式
        matches = FILTER_PATTERN.findall(query)
        for key, value in matches:
            # 支持常见的过滤条件键
            key_lower = key.lower()
            if key_lower in ("collection", "col", "c"):
                filters["collection"] = value
            elif key_lower in ("type", "doc_type", "t"):
                filters["doc_type"] = value
            elif key_lower in ("source", "src", "s"):
                filters["source_path"] = value
            elif key_lower in ("tag", "tags"):
                # tags 可用逗号分隔
                if "tags" not in filters:
                    filters["tags"] = []
                filters["tags"].extend(value.split(","))
            else:
                # 通用过滤条件
                filters[key] = value

        # 从查询中移除过滤条件模式
        query_without_filters = FILTER_PATTERN.sub("", query).strip()
        query_without_filters = " ".join(query_without_filters.split())

        return filters, query_without_filters

    def _tokenize(self, text: str) -> List[str]:
        """将文本切分为词/词项。

        复用 ``src.core.text_utils.tokenize``，与索引侧的分词器（SparseEncoder）
        完全一致，从而保证 BM25 匹配有效，且带连字符/下划线的词
        （``machine-learning``、``GPT-4``）保持完整。

        参数:
            text: 待分词的文本

        返回:
            词项列表
        """
        return shared_tokenize(text)

    def _filter_keywords(self, tokens: List[str]) -> List[str]:
        """过滤词项以得到有意义的关键词。

        - 移除停用词
        - 应用最小长度约束
        - 保序去重
        - 应用数量上限

        参数:
            tokens: 词项列表

        返回:
            过滤后的关键词列表
        """
        seen: Set[str] = set()
        keywords: List[str] = []

        for token in tokens:
            # 为比较而归一化
            token_lower = token.lower()

            # 已出现过则跳过（大小写不敏感去重）
            if token_lower in seen:
                continue

            # 停用词则跳过（同时检查原始形式与小写形式）
            if token in self.config.stopwords or token_lower in self.config.stopwords:
                continue

            # 过短则跳过
            if len(token) < self.config.min_keyword_length:
                continue

            # 添加关键词（保留原始大小写）
            seen.add(token_lower)
            keywords.append(token)

            # 数量足够则停止
            if len(keywords) >= self.config.max_keywords:
                break

        return keywords

    def add_stopwords(self, words: Set[str]) -> None:
        """向停用词集合中添加词语。

        参数:
            words: 要添加的词语集合
        """
        self.config.stopwords.update(words)

    def remove_stopwords(self, words: Set[str]) -> None:
        """从停用词集合中移除词语。

        参数:
            words: 要移除的词语集合
        """
        self.config.stopwords -= words


def create_query_processor(
    stopwords: Optional[Set[str]] = None,
    min_keyword_length: int = 1,
    max_keywords: int = 20,
    enable_filter_parsing: bool = True
) -> QueryProcessor:
    """创建 QueryProcessor 的工厂函数。

    参数:
        stopwords: 自定义停用词集合。为 None 时使用默认值。
        min_keyword_length: 关键词最小长度
        max_keywords: 提取的最大关键词数量
        enable_filter_parsing: 是否解析过滤条件语法

    返回:
        配置好的 QueryProcessor 实例
    """
    config = QueryProcessorConfig(
        stopwords=stopwords if stopwords is not None else DEFAULT_STOPWORDS.copy(),
        min_keyword_length=min_keyword_length,
        max_keywords=max_keywords,
        enable_filter_parsing=enable_filter_parsing
    )
    return QueryProcessor(config)
