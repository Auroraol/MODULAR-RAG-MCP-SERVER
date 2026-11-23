"""
Splitter 模块。

本包包含文本切分器抽象与实现：
- 切分器基类
- 切分器工厂
- 具体实现（Recursive(内部使用LangChain的切分器)、Semantic、FixedLength）
"""

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory
from src.libs.splitter.fixed_length_splitter import FixedLengthSplitter
from src.libs.splitter.semantic_splitter import SemanticSplitter
from src.libs.splitter.structure_splitter import StructureSplitter
from src.libs.splitter.propositional_splitter import PropositionalSplitter

# 导入具体实现（它们会自动向工厂注册）
try:
    from src.libs.splitter.recursive_splitter import RecursiveSplitter
except ImportError:
    RecursiveSplitter = None  # type: ignore[assignment, misc]

try:
    from src.libs.splitter.markdown_splitter import MarkdownSplitter
except ImportError:
    MarkdownSplitter = None  # type: ignore[assignment, misc]

__all__ = [
    "BaseSplitter",
    "SplitterFactory",
    "RecursiveSplitter",
    "FixedLengthSplitter",
    "SemanticSplitter",
    "MarkdownSplitter",
    "StructureSplitter",
    "PropositionalSplitter",
]
