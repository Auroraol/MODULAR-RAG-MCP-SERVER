"""
Loader 模块。

本包包含文档加载器组件：
- base_loader.py 基类
- pdf_loader.py pdf加载器
- txt_loader.py txt加载器
- file_integrity.py 文件检查器, 用于增量摄入的文件完整性检查器
"""

from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.text_loader import TextLoader
from src.libs.loader.office_loader import OfficeLoader
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker

__all__ = [
    "BaseLoader",
    "PdfLoader",
    "TextLoader",
    "OfficeLoader",
    "LoaderFactory",
    "FileIntegrityChecker",
    "SQLiteIntegrityChecker",
]
