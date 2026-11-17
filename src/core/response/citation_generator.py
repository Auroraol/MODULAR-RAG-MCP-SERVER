"""用于生成结构化引用信息的 Citation Generator。

本模块从检索结果中生成引用信息，
使 MCP 工具能够返回格式规范的引用，
供 AI 助手进行来源标注。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult


@dataclass
class Citation:
    """表示单条引用/参考信息。

    属性:
        index: 引用序号（从 1 开始，用于显示为 [1]、[2] 等）
        chunk_id: 来源 Chunk 的唯一标识
        source: 来源文件路径或文档名称
        page: 来源文档中的页码（如适用）
        score: 检索得到的相关性分数
        text_snippet: 被引用内容的简短摘录
        metadata: 附加元数据（标题、章节等）
    """
    index: int
    chunk_id: str
    source: str
    score: float
    text_snippet: str
    page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典以便 JSON 序列化。"""
        result = {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "score": round(self.score, 4),
            "text_snippet": self.text_snippet,
        }
        if self.page is not None:
            result["page"] = self.page
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class CitationGenerator:
    """从检索结果生成引用信息。

    本类将 RetrievalResult 对象转换为带有正确编号
    与元数据提取的 Citation 对象。

    示例:
        >>> generator = CitationGenerator()
        >>> results = [RetrievalResult(chunk_id="doc1_001", score=0.95, ...)]
        >>> citations = generator.generate(results)
        >>> print(citations[0].index)  # 1
        >>> print(citations[0].source)  # "docs/guide.pdf"
    """

    def __init__(
        self,
        snippet_max_length: int = 200,
        include_metadata_fields: Optional[List[str]] = None,
    ) -> None:
        """初始化 CitationGenerator。

        参数:
            snippet_max_length: text_snippet 的最大字符数（默认：200）
            include_metadata_fields: 可选，要包含的元数据字段列表。
                为 None 时包含 'title'、'section'、'chunk_index'。
        """
        self.snippet_max_length = snippet_max_length
        self.include_metadata_fields = include_metadata_fields or [
            "title", "section", "chunk_index", "doc_type"
        ]

    def generate(self, results: List[RetrievalResult]) -> List[Citation]:
        """从检索结果生成引用。

        参数:
            results: 来自检索的 RetrievalResult 对象列表。

        返回:
            使用从 1 开始编号的 Citation 对象列表。
        """
        citations = []

        for idx, result in enumerate(results, start=1):
            citation = self._create_citation(idx, result)
            citations.append(citation)

        return citations

    def _create_citation(self, index: int, result: RetrievalResult) -> Citation:
        """从单个 RetrievalResult 创建 Citation。

        参数:
            index: 从 1 开始的引用序号。
            result: 待转换的 RetrievalResult。

        返回:
            包含所提取信息的 Citation 对象。
        """
        metadata = result.metadata or {}

        # 提取来源路径
        source = metadata.get("source_path", "unknown")

        # 提取页码（可能为 int 或字符串）
        page = metadata.get("page") or metadata.get("page_num")
        if page is not None:
            try:
                page = int(page)
            except (ValueError, TypeError):
                page = None

        # 生成文本摘录
        text_snippet = self._generate_snippet(result.text)

        # 提取选定的元数据字段
        extra_metadata = {}
        for field_name in self.include_metadata_fields:
            if field_name in metadata and field_name not in ("source_path", "page", "page_num"):
                extra_metadata[field_name] = metadata[field_name]

        return Citation(
            index=index,
            chunk_id=result.chunk_id,
            source=source,
            score=result.score,
            text_snippet=text_snippet,
            page=page,
            metadata=extra_metadata,
        )

    def _generate_snippet(self, text: str) -> str:
        """从文本生成截断的摘录。

        参数:
            text: 完整文本内容。

        返回:
            截断后的文本，必要时附加省略号。
        """
        if not text:
            return ""

        # 清理空白字符
        cleaned = " ".join(text.split())

        if len(cleaned) <= self.snippet_max_length:
            return cleaned

        # 截断并附加省略号
        truncated = cleaned[:self.snippet_max_length].rsplit(" ", 1)[0]
        return truncated + "..."

    def format_citation_marker(self, index: int) -> str:
        """格式化用于行内引用的标记。

        参数:
            index: 从 1 开始的引用序号。

        返回:
            形如 "[1]" 的格式化标记。
        """
        return f"[{index}]"
