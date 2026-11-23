"""元数据增强变换：基于规则 + 可选的 LLM 增强。"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from src.core.settings import Settings, resolve_path
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import BaseLLM, Message
from src.observability.logger import get_logger

logger = get_logger(__name__)

# LLM 调用的默认最大并行工作线程数
DEFAULT_MAX_WORKERS = 5


class MetadataEnricher(BaseTransform):
    """为 Chunk 元数据补充标题、摘要和标签。

    处理 Pipeline:
        1. 基于规则的增强：从内容中提取基础元数据
        2. （可选）LLM 增强：生成语义丰富的元数据
        3. LLM 失败时：优雅降级为基于规则的元数据

    输出元数据:
        - title: Chunk 的简短标题/标题
        - summary: 内容的简明摘要
        - tags: 相关关键词/主题列表
        - enriched_by: "rule" 或 "llm"

    配置（通过 settings.yaml）:
        - ingestion.metadata_enricher.use_llm: bool - 启用 LLM 增强
        - ingestion.metadata_enricher.prompt_path: str - 自定义提示词文件路径

    设计原则:
        - 优雅降级：LLM 错误不会阻塞摄入
        - 原子处理：每个 Chunk 独立处理
        - 可观测：在元数据中记录 enriched_by
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None
    ):
        """初始化 MetadataEnricher。

        参数:
            settings: 应用设置
            llm: 可选的 LLM 实例（用于测试；为 None 时自动创建）
            prompt_path: 可选的自定义提示词文件路径
        """
        self.settings = settings
        self._llm = llm
        self._prompt_template: Optional[str] = None
        self._prompt_path = prompt_path or str(resolve_path("config/prompts/metadata_enrichment.txt"))
        
        # 确定是否应使用 LLM
        enricher_config = {}
        if hasattr(settings, 'ingestion') and settings.ingestion is not None:
            ingestion_config = settings.ingestion
            # 检查 ingestion 是否具有 metadata_enricher 属性（dataclass）或为字典
            if hasattr(ingestion_config, 'metadata_enricher') and ingestion_config.metadata_enricher:
                enricher_config = ingestion_config.metadata_enricher
            elif isinstance(ingestion_config, dict):
                enricher_config = ingestion_config.get('metadata_enricher', {})
        
        self.use_llm = enricher_config.get('use_llm', False) if enricher_config else False
        
    @property
    def llm(self) -> Optional[BaseLLM]:
        """延迟加载 LLM 实例。"""
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(self.settings)
                logger.info("LLM initialized for metadata enrichment")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Falling back to rule-based only.")
                self.use_llm = False
        return self._llm
    
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """通过丰富 Chunk 元数据来变换 Chunk。

        参数:
            chunks: 要增强的 Chunk 列表
            trace: 可选的 trace 上下文

        返回:
            增强后的 Chunk 列表（长度与输入一致）
        """
        if not chunks:
            return []

        # 若启用 LLM，则并行处理 Chunk
        if self.use_llm and self.llm:
            return self._transform_parallel(chunks, trace)
        else:
            return self._transform_sequential(chunks, trace)
    
    def _enrich_single_chunk(
        self, 
        chunk: Chunk, 
        trace: Optional[TraceContext] = None
    ) -> Tuple[Chunk, str, Optional[str]]:
        """增强单个 Chunk。线程安全。

        参数:
            chunk: 要增强的 Chunk
            trace: 可选的 trace 上下文

        返回:
            (enriched_chunk, enriched_by, error_message) 元组
        """
        try:
            # 步骤 1: 基于规则的增强
            rule_metadata = self._rule_based_enrich(chunk.text)

            # 步骤 2: LLM 增强
            if self.use_llm and self.llm:
                llm_metadata = self._llm_enrich(chunk.text, trace)
                
                if llm_metadata:
                    enriched_metadata = llm_metadata
                    enriched_by = "llm"
                else:
                    enriched_metadata = rule_metadata
                    enriched_by = "rule"
                    enriched_metadata['enrich_fallback_reason'] = "llm_failed"
            else:
                enriched_metadata = rule_metadata
                enriched_by = "rule"
            
            final_metadata = {
                **(chunk.metadata or {}),
                **enriched_metadata,
                'enriched_by': enriched_by
            }
            
            enriched_chunk = Chunk(
                id=chunk.id,
                text=chunk.text,
                metadata=final_metadata,
                source_ref=chunk.source_ref
            )
            return (enriched_chunk, enriched_by, None)
            
        except Exception as e:
            logger.error(f"Failed to enrich chunk {chunk.id}: {e}")
            text_preview = ""
            if chunk.text:
                text_preview = chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text
            minimal_metadata = {
                **(chunk.metadata or {}),
                'title': 'Untitled',
                'summary': text_preview,
                'tags': [],
                'enriched_by': 'error',
                'enrich_error': str(e)
            }
            enriched_chunk = Chunk(
                id=chunk.id,
                text=chunk.text or "",
                metadata=minimal_metadata,
                source_ref=chunk.source_ref
            )
            return (enriched_chunk, "error", str(e))
    
    def _transform_parallel(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """使用 ThreadPoolExecutor 并行处理 Chunk。"""
        max_workers = min(DEFAULT_MAX_WORKERS, len(chunks))
        enriched_chunks = [None] * len(chunks)
        llm_enhanced_count = 0
        fallback_count = 0
        
        logger.debug(f"Processing {len(chunks)} chunks in parallel (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._enrich_single_chunk, chunk, trace): idx
                for idx, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    enriched_chunk, enriched_by, error = future.result()
                    enriched_chunks[idx] = enriched_chunk
                    
                    if enriched_by == "llm":
                        llm_enhanced_count += 1
                    elif enriched_by == "rule" and error is None:
                        fallback_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error in parallel enrichment: {e}")
                    enriched_chunks[idx] = chunks[idx]
        
        success_count = sum(1 for c in enriched_chunks if c is not None)
        
        if trace:
            trace.record_stage("metadata_enricher", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": True,
                "max_workers": max_workers
            })
        
        logger.info(
            f"Enriched {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, Fallback: {fallback_count})"
        )
        
        return enriched_chunks
    
    def _transform_sequential(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """顺序处理 Chunk（LLM 禁用时的回退方案）。"""
        enriched_chunks = []
        success_count = 0
        llm_enhanced_count = 0
        fallback_count = 0
        
        for chunk in chunks:
            try:
                # 步骤 1: 基于规则的增强（始终执行）
                rule_metadata = self._rule_based_enrich(chunk.text)

                # 步骤 2: 可选的 LLM 增强
                if self.use_llm and self.llm:
                    llm_metadata = self._llm_enrich(chunk.text, trace)

                    if llm_metadata:
                        # LLM 成功
                        enriched_metadata = llm_metadata
                        enriched_by = "llm"
                        llm_enhanced_count += 1
                    else:
                        # LLM 失败，回退到基于规则的结果
                        enriched_metadata = rule_metadata
                        enriched_by = "rule"
                        fallback_count += 1
                        enriched_metadata['enrich_fallback_reason'] = "llm_failed"
                else:
                    # LLM 已禁用，使用基于规则的结果
                    enriched_metadata = rule_metadata
                    enriched_by = "rule"

                # 将增强后的元数据与现有元数据合并, 字典解包合并的写法
                # 同名key覆盖: chunk.metadata < enriched_metadata < {'enriched_by': enriched_by}
                final_metadata = {
                    **(chunk.metadata or {}),
                    **enriched_metadata,
                    'enriched_by': enriched_by
                }

                # 创建增强后的 Chunk
                enriched_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=final_metadata,
                    source_ref=chunk.source_ref
                )
                enriched_chunks.append(enriched_chunk)
                success_count += 1
                
            except Exception as e:
                # 原子失败：记录日志并保留原始内容，使用最小元数据
                logger.error(f"Failed to enrich chunk {chunk.id}: {e}")
                # 处理文本为 None 的情况
                text_preview = ""
                if chunk.text:
                    text_preview = chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text
                minimal_metadata = {
                    **(chunk.metadata or {}),
                    'title': 'Untitled',
                    'summary': text_preview,
                    'tags': [],
                    'enriched_by': 'error',
                    'enrich_error': str(e)
                }
                enriched_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text or "",  # 确保 text 不为 None
                    metadata=minimal_metadata,
                    source_ref=chunk.source_ref
                )
                enriched_chunks.append(enriched_chunk)
        
        # 记录 trace
        if trace:
            trace.record_stage("metadata_enricher", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": False
            })
        
        logger.info(
            f"Enriched {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, Fallback: {fallback_count})"
        )
        
        return enriched_chunks
    
    def _rule_based_enrich(self, text: str) -> Dict[str, Any]:
        """使用基于规则的启发式方法提取元数据。

        参数:
            text: Chunk 文本内容

        返回:
            包含 title、summary、tags 的字典

        异常:
            TypeError: 当 text 为 None 时
        """
        if text is None:
            raise TypeError("Chunk text cannot be None")

        # 从第一个标题或第一行提取 title
        title = self._extract_title(text)

        # 根据前几句生成摘要
        summary = self._extract_summary(text)

        # 从常见模式中提取标签
        tags = self._extract_tags(text)
        
        return {
            'title': title,
            'summary': summary,
            'tags': tags
        }
    
    def _extract_title(self, text: str) -> str:
        """使用启发式方法从文本中提取标题。

        优先级:
            1. Markdown 标题（# Title）
            2. 首行（若足够短）
            3. 第一句
            4. 前 N 个字符
        """
        if not text:
            return "Untitled"

        # 检查 Markdown 标题
        heading_match = re.match(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()

        # 若首行较短且看起来像标题，则使用首行
        first_line = text.split('\n')[0].strip()
        if first_line and len(first_line) <= 100 and not first_line.endswith(('.', ',', ';')):
            return first_line

        # 使用第一句（移除末尾标点）
        sentences = re.split(r'[.!?]\s+', text)
        if sentences and sentences[0]:
            title = sentences[0].strip()
            # 若存在末尾标点则移除
            title = re.sub(r'[.!?]+$', '', title)
            if len(title) <= 150:
                return title
            return title[:147] + "..."

        # 回退方案：前 100 个字符
        return text[:100].strip() + ("..." if len(text) > 100 else "")

    def _extract_summary(self, text: str, max_sentences: int = 3) -> str:
        """使用前 N 句从文本中提取摘要。

        参数:
            text: 源文本
            max_sentences: 包含的最大句子数

        返回:
            摘要文本
        """
        if not text:
            return ""

        # 拆分为句子
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # 取前 N 句
        summary_sentences = sentences[:max_sentences]
        summary = ' '.join(summary_sentences).strip()

        # 限制长度
        if len(summary) > 500:
            summary = summary[:497] + "..."

        return summary

    def _extract_tags(self, text: str, max_tags: int = 10) -> List[str]:
        """使用关键词提取启发式方法提取标签。

        参数:
            text: 源文本
            max_tags: 提取的最大标签数

        返回:
            标签字符串列表
        """
        if not text:
            return []

        tags = set()

        # 提取首字母大写的单词（可能是专有名词）
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        tags.update(capitalized[:5])

        # 提取代码标识符（camelCase、snake_case）
        identifiers = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[a-z]+_[a-z_]+\b', text)
        tags.update(identifiers[:5])

        # 提取 Markdown 粗体/斜体词（可能是关键词）
        markdown_keywords = re.findall(r'\*\*(.+?)\*\*|\*(.+?)\*|__(.+?)__|_(.+?)_', text)
        for match in markdown_keywords[:5]:
            for group in match:
                if group:
                    tags.add(group.strip())

        # 转换为列表并限制数量
        tag_list = sorted(list(tags))[:max_tags]

        return tag_list

    def _llm_enrich(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> Optional[Dict[str, Any]]:
        """使用 LLM 增强元数据。

        参数:
            text: Chunk 文本内容
            trace: 可选的 trace 上下文

        返回:
            包含 title、summary、tags 的字典，失败时返回 None
        """
        if not self.llm:
            return None

        try:
            # 加载提示词模板
            prompt = self._load_prompt()

            # 用文本填充提示词
            formatted_prompt = prompt.replace("{chunk_text}", text[:2000])  # 限制文本长度

            # 调用 LLM
            messages = [Message(role="user", content=formatted_prompt)]
            response = self.llm.chat(messages)

            if not response:
                logger.warning("LLM returned empty response for metadata enrichment")
                return None

            # 从响应中提取文本（同时处理字符串和 ChatResponse 对象）
            response_text = response
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            elif not isinstance(response, str):
                response_text = str(response)

            # 解析 LLM 响应
            metadata = self._parse_llm_response(response_text)

            if trace:
                trace.record_stage("llm_enrich", {
                    "success": True,
                    "response_length": len(response_text)
                })

            return metadata
            
        except Exception as e:
            logger.warning(f"LLM enrichment failed: {e}")
            if trace:
                trace.record_stage("llm_enrich", {
                    "success": False,
                    "error": str(e)
                })
            return None
    
    def _load_prompt(self) -> str:
        """从文件加载提示词模板。

        返回:
            提示词模板字符串

        异常:
            FileNotFoundError: 当提示词文件不存在时
        """
        if self._prompt_template is not None:
            return self._prompt_template

        prompt_path = Path(self._prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self._prompt_path}")

        self._prompt_template = prompt_path.read_text(encoding='utf-8')
        logger.info(f"Loaded metadata enrichment prompt from {self._prompt_path}")

        return self._prompt_template

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """将 LLM 响应解析为结构化元数据。

        预期格式:
            Title: <title>
            Summary: <summary>
            Tags: <tag1>, <tag2>, <tag3>

        参数:
            response: LLM 响应文本

        返回:
            包含 title、summary、tags 的字典
        """
        metadata = {
            'title': '',
            'summary': '',
            'tags': []
        }

        # 提取 title
        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()

        # 提取 summary
        summary_match = re.search(r'Summary:\s*(.+?)(?:\n(?:Tags:|$))', response, re.IGNORECASE | re.DOTALL)
        if summary_match:
            metadata['summary'] = summary_match.group(1).strip()

        # 提取 tags
        tags_match = re.search(r'Tags:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if tags_match:
            tags_text = tags_match.group(1).strip()
            # 按逗号拆分并清理
            tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
            metadata['tags'] = tags

        # 校验：确保值非空
        if not metadata['title']:
            metadata['title'] = 'Untitled'
        if not metadata['summary']:
            metadata['summary'] = response[:500]  # 回退为原始响应

        return metadata
