"""文档切分模块 - 为业务层适配 libs.splitter。

本模块是 libs.splitter（纯文本切分）与 Ingestion Pipeline（业务对象转换）
之间的适配层。它将 Document 对象转换为 Chunk 对象，并提供正确的
ID 生成、元数据继承和可追溯性。

相对 libs.splitter 的核心增值：
1. Chunk ID 生成：为每个 Chunk 生成确定且唯一的 ID
2. 元数据继承：将 Document 元数据传播到所有 Chunk
3. chunk_index：记录在文档中的顺序位置
4. source_ref：建立父子可追溯关系
5. 类型转换：str → Chunk 对象（core.types 契约）

设计原则：
- 适配器模式：衔接文本切分工具与业务对象
- 配置驱动：通过 SplitterFactory 进行基于配置的策略选择
- 确定性：同一 Document 重复切分产生相同的 Chunk ID
- 类型安全：遵循 core.types.Chunk 契约
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, List

from src.core.types import Chunk, Document
from src.libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from src.core.settings import Settings


class DocumentChunker:
    """将 Document 转换为带有业务级增强的 Chunk。

    该类封装 libs 层的文本切分器，并添加业务逻辑：
    - 生成稳定的 Chunk ID
    - 继承并扩展元数据
    - 维护文档可追溯性

    属性:
        _splitter: libs 层的底层文本切分器
        _settings: 切分行为的配置设置

    示例:
        >>> from src.core.settings import load_settings
        >>> from src.core.types import Document
        >>> settings = load_settings("config/settings.yaml")
        >>> chunker = DocumentChunker(settings)
        >>> document = Document(
        ...     id="doc_123",
        ...     text="Long document content...",
        ...     metadata={"source_path": "data/report.pdf"}
        ... )
        >>> chunks = chunker.split_document(document)
        >>> print(f"Generated {len(chunks)} chunks")
        >>> print(f"First chunk ID: {chunks[0].id}")
        >>> print(f"First chunk index: {chunks[0].metadata['chunk_index']}")
    """
    
    def __init__(self, settings: Settings):
        """使用配置初始化 DocumentChunker。

        参数:
            settings: 包含切分器配置的配置设置。
                     切分器配置应位于 settings.splitter.*

        异常:
            ValueError: 切分器配置无效或 provider 未知
        """
        self._settings = settings
        self._splitter = SplitterFactory.create(settings)
    
    def split_document(self, document: Document) -> List[Chunk]:
        """将 Document 切分为 Chunk，并完成完整的业务增强。

        这是编排转换过程的主入口：
        1. 使用底层切分器获取文本片段
        2. 为每个 Chunk 生成确定性 ID
        3. 从 document 继承并扩展元数据
        4. 创建符合 core.types 契约的 Chunk 对象

        参数:
            document: 待切分的源文档

        返回:
            Chunk 对象列表，包含：
            - 唯一且确定性的 ID
            - 继承的元数据 + chunk_index + source_ref
            - 正确的类型契约（core.types.Chunk）

        异常:
            ValueError: 文档没有文本或结构无效

        示例:
            >>> doc = Document(
            ...     id="doc_abc",
            ...     text="Section 1 content.\\n\\nSection 2 content.",
            ...     metadata={"source_path": "file.pdf", "title": "Report"}
            ... )
            >>> chunker = DocumentChunker(settings)
            >>> chunks = chunker.split_document(doc)
            >>> len(chunks) >= 1
            True
            >>> chunks[0].metadata["source_path"]
            'file.pdf'
            >>> chunks[0].metadata["chunk_index"]
            0
            >>> chunks[0].metadata["source_ref"]
            'doc_abc'
        """
        if not document.text or not document.text.strip():
            raise ValueError(f"Document {document.id} has no text content to split")
        
        # 步骤 1：使用底层切分器获取文本片段
        text_fragments = self._splitter.split_text(document.text)
        
        if not text_fragments:
            raise ValueError(
                f"Splitter returned no chunks for document {document.id}. "
                f"Text length: {len(document.text)}"
            )
        
        # 步骤 2：将文本切块片段转换为带有增强信息的 Chunk 对象
        chunks: List[Chunk] = []
        for index, text in enumerate(text_fragments):
            chunk_id = self._generate_chunk_id(document.id, index, text)
            chunk_metadata = self._inherit_metadata(document, index, text)
            
            chunk = Chunk(
                id=chunk_id,
                text=text,
                metadata=chunk_metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """生成唯一且确定性的 Chunk ID。

        ID 格式：{doc_id}_{index:04d}_{content_hash}
        - doc_id：父文档标识
        - index：顺序位置（零填充至 4 位）
        - content_hash：文本 SHA256 哈希的前 8 个字符

        这保证了：
        - 唯一性：doc_id + index + content_hash 的组合
        - 确定性：相同输入始终产生相同 ID
        - 可调试性：人类可读的结构

        参数:
            doc_id: 父文档 ID
            index: Chunk 的顺序位置（从 0 开始）
            text: Chunk 文本内容

        返回:
            唯一的 Chunk ID 字符串

        示例:
            >>> chunker._generate_chunk_id("doc_123", 0, "Hello world")
            'doc_123_0000_c0535e4b'
        """
        # 计算内容哈希以确保唯一性
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

        # 格式：{doc_id}_{index:04d}_{hash_8chars}
        return f"{doc_id}_{index:04d}_{content_hash}"
    
    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str = "") -> dict:
        """从 document 继承元数据并添加 Chunk 专属字段。

        创建一个新的元数据字典，包含：
        - document.metadata 的所有字段（复制而非引用）
        - chunk_index：顺序位置（从 0 开始）
        - source_ref：父文档 ID 的引用
        - image_refs：此 Chunk 中引用的 Image ID 列表（从占位符中提取）

        注意：文档级的 'images' 字段被有意从 Chunk 元数据中排除，
        因为它是冗余的。取而代之的是，根据 Chunk 文本中的
        [IMAGE: xxx] 占位符填充 Chunk 专属的 'image_refs'。

        参数:
            document: 要继承元数据的源文档
            chunk_index: 此 Chunk 的顺序位置
            chunk_text: 此 Chunk 的文本内容（用于提取 image_refs）

        返回:
            包含继承字段与 Chunk 专属字段的元数据字典

        示例:
            >>> doc = Document(
            ...     id="doc_123",
            ...     text="Content",
            ...     metadata={"source_path": "file.pdf", "title": "Report"}
            ... )
            >>> metadata = chunker._inherit_metadata(doc, 2, "See [IMAGE: img_001]")
            >>> metadata["source_path"]
            'file.pdf'
            >>> metadata["chunk_index"]
            2
            >>> metadata["source_ref"]
            'doc_123'
            >>> metadata["image_refs"]
            ['img_001']
        """
        import re
        
        # 复制所有文档元数据（基本类型使用浅拷贝即可）
        chunk_metadata = document.metadata.copy()

        # 获取文档级图片以便查找
        doc_images = document.metadata.get("images", [])

        # 移除文档级 'images' 字段 —— 下面将添加 Chunk 专属的 images
        chunk_metadata.pop("images", None)

        # 添加 Chunk 专属字段
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["source_ref"] = document.id

        # 通过查找 [IMAGE: xxx] 占位符从 Chunk 文本中提取 image_refs
        image_refs = []
        if chunk_text:
            # 匹配 [IMAGE: image_id] 占位符的模式
            pattern = r'\[IMAGE:\s*([^\]]+)\]'
            matches = re.findall(pattern, chunk_text)
            image_refs = [m.strip() for m in matches]

        chunk_metadata["image_refs"] = image_refs

        # 为引用的图片构建包含完整元数据的 Chunk 专属 'images' 列表
        # ImageCaptioner 需要它来访问图片路径以调用 Vision API
        chunk_images = []
        if image_refs and doc_images:
            image_lookup = {img.get("id"): img for img in doc_images}
            for img_id in image_refs:
                if img_id in image_lookup:
                    chunk_images.append(image_lookup[img_id])

        if chunk_images:
            chunk_metadata["images"] = chunk_images

        # 尝试从第一个引用的图片确定 page_num
        if chunk_images:
            chunk_metadata["page_num"] = chunk_images[0].get("page")
        
        return chunk_metadata
