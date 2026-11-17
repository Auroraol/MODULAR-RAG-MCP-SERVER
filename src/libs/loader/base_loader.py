"""文档加载器的抽象基类。

本模块定义文档加载器的可插拔接口，以统一的结构实现不同文档格式
（PDF、Markdown 等）的无缝加载。

设计原则：
- 单一职责：加载器只负责格式统一 + 结构提取
- 类型安全：返回 core.types 中标准化的 Document 类型
- 不切分：加载器不对文档分块，只做解析与规范化
- 优雅降级：可选功能（如图片提取）的失败不应阻塞文本解析
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.core.types import Document


class BaseLoader(ABC):
    """文档加载器的抽象基类。

    所有加载器都必须实现 load() 方法，用于解析文件并返回标准化的
    Document 对象，其中包含：
    - text: 规范化后的内容（建议使用 Markdown 格式）
    - metadata: 至少必须包含 'source_path'

    加载器应处理：
    - 格式特定的解析逻辑
    - 元数据提取（标题、页数等）
    - 结构规范化（尽可能转为 Markdown）
    - 可选：图片提取与占位符插入
    """
    
    @abstractmethod
    def load(self, file_path: str | Path) -> Document:
        """加载并解析文档文件。

        参数:
            file_path: 待加载文档文件的路径。

        返回:
            包含解析内容与元数据的 Document 对象。
            metadata 必须至少包含 'source_path'。

        异常:
            FileNotFoundError: 当文件不存在时抛出。
            ValueError: 当文件格式无效或不受支持时抛出。
            RuntimeError: 当解析严重失败时抛出。

        示例:
            >>> loader = PdfLoader()
            >>> doc = loader.load("data/documents/report.pdf")
            >>> assert "source_path" in doc.metadata
            >>> assert doc.text  # Non-empty text
        """
        pass
    
    @staticmethod
    def _validate_file(file_path: str | Path) -> Path:
        """校验文件存在且可读。

        参数:
            file_path: 待校验的路径。

        返回:
            解析后的 Path 对象。

        异常:
            FileNotFoundError: 当文件不存在时抛出。
            PermissionError: 当文件不可读时抛出。
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path
