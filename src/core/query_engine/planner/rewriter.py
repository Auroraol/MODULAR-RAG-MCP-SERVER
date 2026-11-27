"""查询改写：改写 / HyDE / 子查询扩展。

检索前的可选步骤。默认 NoOpRewriter（零改动、零风险）；
需要 LLM 时注入 BaseLLM 实例构造 LLMRewriter。任何 LLM 失败都回落
到原始查询 —— 改写是增强项，不允许拖垮检索链路。
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence

from src.core.query_engine.planner.router import RewriteMode
from src.libs.llm.base_llm import BaseLLM, Message

logger = logging.getLogger(__name__)

# 去掉 LLM 输出里的列表序号/项目符号："1. xxx" / "- xxx" / "• xxx"
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.、)]|[-*•])\s*")

_PROMPTS = {
    RewriteMode.REWRITE: (
        "Rewrite the user's question into one standalone, self-contained search query. "
        "Resolve pronouns and references using the conversation history. "
        "Keep the original language. Output only the rewritten query."
    ),
    RewriteMode.HYDE: (
        "Write a short hypothetical passage that would answer the user's question. "
        "Keep the original language. Output only the passage."
    ),
}


class BaseRewriter(ABC):
    """查询改写器的可插拔接口。"""

    @abstractmethod
    def rewrite(
        self,
        query: str,
        history: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """把原始查询改写为一组检索查询。

        参数:
            query: 原始用户查询
            history: 之前的对话轮次，用于指代消解

        返回:
            待检索的查询列表，至少包含一项
        """
        raise NotImplementedError


class NoOpRewriter(BaseRewriter):
    """不改写，原样返回。默认实现。"""

    def rewrite(
        self,
        query: str,
        history: Optional[Sequence[str]] = None,
    ) -> List[str]:
        return [query]


class LLMRewriter(BaseRewriter):
    """基于 LLM 的改写器，支持 REWRITE / HYDE / SUBQUERY 三种模式。

    NONE 模式建议直接用 NoOpRewriter，不要构造本类。

    注意：HyDE 生成的假设文档通常只适合喂稠密检索（向量相似度），
    不适合再交给 BM25 做关键词匹配 —— 由调用方按模式选择检索路径。

    示例:
        >>> rewriter = LLMRewriter(llm, RewriteMode.SUBQUERY, num_subqueries=3)
        >>> queries = rewriter.rewrite("BOM25 怎么接线和供电")
        >>> len(queries) >= 1
        True
    """

    def __init__(
        self,
        llm: BaseLLM,
        mode: RewriteMode,
        num_subqueries: int = 3,
    ) -> None:
        """初始化改写器。

        参数:
            llm: LLM Provider（BaseLLM 实现）
            mode: 改写模式，不能是 NONE
            num_subqueries: SUBQUERY 模式下的子查询数量上限
        """
        if mode == RewriteMode.NONE:
            raise ValueError("LLMRewriter does not support RewriteMode.NONE")
        self.llm = llm
        self.mode = mode
        self.num_subqueries = num_subqueries

    def rewrite(
        self,
        query: str,
        history: Optional[Sequence[str]] = None,
    ) -> List[str]:
        try:
            response = self.llm.chat(self._build_messages(query, history))
            queries = self._parse(response.content)
            if queries:
                return queries
        except Exception as e:  # 改写是增强项，失败不阻断检索
            logger.warning(f"query rewrite failed, using original query: {e}")
        return [query]

    def _build_messages(
        self,
        query: str,
        history: Optional[Sequence[str]],
    ) -> List[Message]:
        if self.mode == RewriteMode.SUBQUERY:
            system = (
                f"Break the user's question into up to {self.num_subqueries} independent "
                "search sub-queries, one per line. Keep the original language. "
                "Output only the sub-queries."
            )
        else:
            system = _PROMPTS[self.mode]

        if history:
            system += "\n\nConversation history:\n" + "\n".join(history[-5:])

        return [Message(role="system", content=system), Message(role="user", content=query)]

    def _parse(self, content: str) -> List[str]:
        text = (content or "").strip()
        if not text:
            return []

        if self.mode != RewriteMode.SUBQUERY:
            # 单条查询/假设文档：整段作为一个检索串
            return [text]

        lines = []
        for line in text.splitlines():
            line = _LIST_MARKER.sub("", line).strip()
            if line:
                lines.append(line)
        return lines[: self.num_subqueries]


def create_rewriter(
    mode: RewriteMode,
    llm: Any = None,
    num_subqueries: int = 3,
) -> BaseRewriter:
    """按模式创建改写器：NONE 或未提供 LLM 时退化为 NoOpRewriter。"""
    if mode == RewriteMode.NONE or llm is None:
        return NoOpRewriter()
    return LLMRewriter(llm, mode, num_subqueries=num_subqueries)