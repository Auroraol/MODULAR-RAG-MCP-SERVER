"""基于 MarkItDown 的 PDF Loader 实现。

本模块实现支持图片提取的 PDF 解析，
将 PDF 转换为带图片占位符的标准 Markdown 格式。

功能：
- 通过 MarkItDown 进行文本提取与 Markdown 转换
- 图片提取与存储
- 插入图片占位符并跟踪元数据
- 图片提取失败时优雅降级
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from PIL import Image
import io

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):
    """使用 MarkItDown 进行文本提取与 Markdown 转换的 PDF Loader。

    本加载器：
    1. 从 PDF 提取文本并转换为 Markdown
    2. 提取图片并保存到 data/images/{doc_hash}/
    3. 按 [IMAGE: {image_id}] 格式插入图片占位符
    4. 在 Document.metadata.images 中记录图片元数据

    配置：
        extract_images: 启用/禁用图片提取（默认：True）
        image_storage_dir: 图片存储的基础目录（默认：data/images）

    优雅降级：
        若图片提取失败，则记录警告并继续以纯文本方式解析。
    """
    
    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images"
    ):
        """初始化 PDF Loader。

        参数:
            extract_images: 是否从 PDF 中提取图片。
            image_storage_dir: 存储提取图片的基础目录。
        """
        if not MARKITDOWN_AVAILABLE:
            raise ImportError(
                "MarkItDown is required for PdfLoader. "
                "Install with: pip install markitdown"
            )
        
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        self._markitdown = MarkItDown()
    
    def load(self, file_path: str | Path) -> Document:
        """加载并解析 PDF 文件。

        参数:
            file_path: PDF 文件路径。

        返回:
            包含 Markdown 文本与元数据的 Document。

        异常:
            FileNotFoundError: 当 PDF 文件不存在时抛出。
            ValueError: 当文件不是有效的 PDF 时抛出。
            RuntimeError: 当解析严重失败时抛出。
        """
        # 校验文件
        path = self._validate_file(file_path)
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {path}")

        # 计算文档哈希，用于唯一 ID 和图片目录
        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"

        # 使用 MarkItDown 解析 PDF
        try:
            result = self._markitdown.convert(str(path))
            text_content = result.text_content if hasattr(result, 'text_content') else str(result)
        except Exception as e:
            logger.error(f"Failed to parse PDF {path}: {e}")
            raise RuntimeError(f"PDF parsing failed: {e}") from e
        
        # 初始化元数据
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
        }
        
        # 若可用，从首个标题提取标题
        title = self._extract_title(text_content)
        if title:
            metadata["title"] = title

        # 处理图片提取（采用优雅降级）
        if self.extract_images:
            try:
                text_content, images_metadata = self._extract_and_process_images(
                    path, text_content, doc_hash
                )
                if images_metadata:
                    metadata["images"] = images_metadata
            except Exception as e:
                logger.warning(
                    f"Image extraction failed for {path}, continuing with text-only: {e}"
                )
        
        return Document(
            id=doc_id,
            text=text_content,
            metadata=metadata
        )
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件内容的 SHA256 哈希。

        参数:
            file_path: 文件路径。

        返回:
            SHA256 哈希的十六进制字符串。
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _extract_title(self, text: str) -> Optional[str]:
        """从首个 Markdown 标题或首个非空行提取标题。

        参数:
            text: Markdown 文本内容。

        返回:
            若找到标题则返回标题字符串，否则返回 None。
        """
        lines = text.split('\n')

        # 首先尝试查找 Markdown 标题
        for line in lines[:20]:  # 检查前 20 行
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()

        # 回退：使用首个非空行作为标题
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 0:
                return line
        
        return None
    
    def _extract_and_process_images(
        self,
        pdf_path: Path,
        text_content: str,
        doc_hash: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """从 PDF 中提取图片并插入占位符。

        使用 PyMuPDF 提取图片、保存到磁盘，并在文本内容中插入占位符。

        参数:
            pdf_path: PDF 文件路径。
            text_content: 已提取的文本内容。
            doc_hash: 用于图片目录的文档哈希。

        返回:
            (modified_text, images_metadata_list) 元组
        """
        if not self.extract_images:
            logger.debug(f"Image extraction disabled for {pdf_path}")
            return text_content, []
        
        if not PYMUPDF_AVAILABLE:
            logger.warning(f"PyMuPDF not available, skipping image extraction for {pdf_path}")
            return text_content, []
        
        images_metadata = []
        modified_text = text_content
        
        try:
            # 创建图片存储目录
            image_dir = self.image_storage_dir / doc_hash
            image_dir.mkdir(parents=True, exist_ok=True)

            # 使用 PyMuPDF 打开 PDF
            doc = fitz.open(pdf_path)

            # 跟踪用于插入占位符的文本偏移量
            text_offset = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        # 提取图片
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # 生成图片 ID 和文件名
                        image_id = self._generate_image_id(doc_hash, page_num + 1, img_index + 1)
                        image_filename = f"{image_id}.{image_ext}"
                        image_path = image_dir / image_filename

                        # 保存图片
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        # 获取图片尺寸
                        try:
                            img = Image.open(io.BytesIO(image_bytes))
                            width, height = img.size
                        except Exception:
                            width, height = 0, 0

                        # 创建占位符
                        placeholder = f"[IMAGE: {image_id}]"

                        # 在当前页内容末尾插入占位符
                        # （简化实现 —— 生产环境中应解析页面边界）
                        insert_position = len(modified_text)
                        modified_text += f"\n{placeholder}\n"

                        # 将路径转换为相对项目根目录的路径或绝对路径
                        try:
                            relative_path = image_path.relative_to(Path.cwd())
                        except ValueError:
                            # 若不在 cwd 下，则使用绝对路径
                            relative_path = image_path.absolute()

                        # 记录元数据
                        image_metadata = {
                            "id": image_id,
                            "path": str(relative_path),
                            "page": page_num + 1,
                            "text_offset": insert_position + 1,  # +1 表示换行符
                            "text_length": len(placeholder),
                            "position": {
                                "width": width,
                                "height": height,
                                "page": page_num + 1,
                                "index": img_index
                            }
                        }
                        images_metadata.append(image_metadata)
                        
                        logger.debug(f"Extracted image {image_id} from page {page_num + 1}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
            if images_metadata:
                logger.info(f"Extracted {len(images_metadata)} images from {pdf_path}")
            else:
                logger.debug(f"No images found in {pdf_path}")
            
            return modified_text, images_metadata
            
        except Exception as e:
            logger.warning(f"Image extraction failed for {pdf_path}: {e}")
            # 优雅降级：返回不含图片的原始文本
            return text_content, []
    
    @staticmethod
    def _generate_image_id(doc_hash: str, page: int, sequence: int) -> str:
        """生成唯一的图片 ID。

        参数:
            doc_hash: 文档哈希。
            page: 页码（从 0 开始）。
            sequence: 页面内的图片序号（从 0 开始）。

        返回:
            唯一的图片 ID 字符串。
        """
        return f"{doc_hash[:8]}_{page}_{sequence}"
