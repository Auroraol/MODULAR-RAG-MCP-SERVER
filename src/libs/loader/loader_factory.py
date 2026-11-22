"""按文件扩展名分发到对应 Loader。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.office_loader import OfficeLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.text_loader import TextLoader


class LoaderFactory:
    """根据扩展名选择 Loader，并缓存实例。

    用法::

        factory = LoaderFactory(image_storage_dir="data/images/default")
        document = factory.load("docs/report.docx")
    """

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images",
    ) -> None:
        self._pdf_kwargs = {
            "extract_images": extract_images,
            "image_storage_dir": image_storage_dir,
        }
        self._loaders: Dict[type, BaseLoader] = {}

    @staticmethod
    def supported_extensions() -> set[str]:
        return {".pdf"} | TextLoader.SUPPORTED_EXTENSIONS | OfficeLoader.SUPPORTED_EXTENSIONS

    def get_loader(self, file_path: str | Path) -> BaseLoader:
        """返回处理该文件的 Loader 实例。

        异常:
            ValueError: 扩展名不受支持。
        """
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            loader_cls: type = PdfLoader
            kwargs = self._pdf_kwargs
        elif suffix in TextLoader.SUPPORTED_EXTENSIONS:
            loader_cls, kwargs = TextLoader, {}
        elif suffix in OfficeLoader.SUPPORTED_EXTENSIONS:
            loader_cls, kwargs = OfficeLoader, {}
        else:
            raise ValueError(
                f"Unsupported file type: {suffix or '(none)'}. "
                f"Supported: {', '.join(sorted(self.supported_extensions()))}"
            )

        if loader_cls not in self._loaders:
            self._loaders[loader_cls] = loader_cls(**kwargs)
        return self._loaders[loader_cls]

    def load(self, file_path: str | Path) -> Document:
        """选中 Loader 并加载文档。"""
        return self.get_loader(file_path).load(file_path)