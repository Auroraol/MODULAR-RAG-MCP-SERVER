"""意图识别：把用户查询归类为问候 / FAQ / 追问 / 知识问答。

设计原则：
- 规则优先：问候、追问用确定性规则判定，零模型调用、零延迟
- 可插拔：custom 回调可接 LLM 或 FAQ 库匹配，返回 None 时回落到规则
- 只做分类，不做决策：分类结果交给 router 决定走向
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class Intent(str, Enum):
    """查询意图。继承 str 便于直接写入 trace / JSON。"""

    GREETING = "greeting"
    FAQ = "faq"
    FOLLOW_UP = "follow_up"
    KNOWLEDGE = "knowledge"


@dataclass
class IntentResult:
    """意图分类结果。

    属性:
        intent: 分类出的意图
        confidence: 置信度 0~1（规则给出固定值，LLM 可给出真实概率）
        metadata: 附加信息（如命中的 FAQ 条目 id），供 router / 上层使用
    """

    intent: Intent
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# 整句匹配才算问候，"你好，BOM25 怎么接线" 不会被误判
_GREETING_PATTERN = re.compile(
    r"^(你好|您好|hi|hello|hey|嗨|哈喽|早上好|下午好|晚上好|在吗|在不在"
    r"|谢谢|多谢|感谢|thanks|thank you)[!！。.~～\s]*$",
    re.IGNORECASE,
)

# 追问特征：代词 / 指代上文，且句子较短
_FOLLOW_UP_HINT = re.compile(
    r"(它|他|她|这个|那个|上述|刚才|前面|上面|继续|这些|那些"
    r"|\bit\b|\bthat\b|\bthey\b|\bthose\b|\bthis\b)",
    re.IGNORECASE,
)

# custom 回调签名：返回 None 表示"我判不了"，交回规则
ClassifierFn = Callable[[str, Optional[List[str]]], Optional[IntentResult]]


class IntentClassifier:
    """规则优先的意图分类器。

    判定顺序：
    1. custom 回调（可选）—— 可接 LLM 或 FAQ 索引，返回 None 则继续
    2. 整句问候 → GREETING
    3. 有对话历史 + 含指代 + 短句 → FOLLOW_UP
    4. 其余 → KNOWLEDGE

    FAQ 意图默认不会被规则产出，需要由 custom 回调（FAQ 库命中）给出。

    示例:
        >>> classifier = IntentClassifier()
        >>> classifier.classify("你好").intent
        <Intent.GREETING: 'greeting'>
        >>> classifier.classify("它怎么接线", history=["BOM25 是什么"]).intent
        <Intent.FOLLOW_UP: 'follow_up'>
    """

    def __init__(
        self,
        custom: Optional[ClassifierFn] = None,
        follow_up_max_len: int = 30,
    ) -> None:
        """初始化分类器。

        参数:
            custom: 可选的兜底前置分类器，适配 LLM / FAQ 库
            follow_up_max_len: 判定追问的最大字符长度
        """
        self.custom = custom
        self.follow_up_max_len = follow_up_max_len

    def classify(
        self,
        query: str,
        history: Optional[List[str]] = None,
    ) -> IntentResult:
        """对查询进行分类。

        参数:
            query: 用户查询
            history: 之前的对话轮次（用户消息或 Q/A 文本），为空表示首轮

        返回:
            IntentResult
        """
        q = (query or "").strip()

        if self.custom is not None:
            result = self.custom(q, history)
            if result is not None:
                return result

        if _GREETING_PATTERN.match(q):
            return IntentResult(Intent.GREETING, 0.95)

        if history and len(q) <= self.follow_up_max_len and _FOLLOW_UP_HINT.search(q):
            return IntentResult(Intent.FOLLOW_UP, 0.8)

        return IntentResult(Intent.KNOWLEDGE, 0.6)