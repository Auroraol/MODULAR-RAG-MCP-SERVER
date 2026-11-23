"""Chunk 精炼变换：基于规则的清洗 + 可选的 LLM 增强。"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

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


class ChunkRefiner(BaseTransform):
    """通过基于规则的清洗和可选的 LLM 增强来精炼 Chunk。

    处理 Pipeline:
        1. 基于规则的精炼：去除噪声（空白、页眉/页脚、HTML）
        2. （可选）LLM 精炼：智能内容改进
        3. LLM 失败时：优雅降级为基于规则的结果

    配置（通过 settings.yaml）:
        - ingestion.chunk_refiner.use_llm: bool - 启用 LLM 增强
        - ingestion.chunk_refiner.prompt_path: str - 自定义提示词文件路径

    设计原则:
        - 优雅降级：LLM 错误不会阻塞摄入
        - 原子处理：每个 Chunk 独立处理
        - 可观测：在元数据中记录 refined_by
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None
    ):
        """初始化 ChunkRefiner。

        参数:
            settings: 应用设置
            llm: 可选的 LLM 实例（用于测试；为 None 时自动创建）
            prompt_path: 可选的自定义提示词文件路径
        """
        self.settings = settings
        self._llm = llm
        self._prompt_template: Optional[str] = None
        # 提示词文件路径的默认值设置
        self._prompt_path = prompt_path or str(resolve_path("config/prompts/chunk_refinement.txt"))
        
        # 确定是否应使用 LLM
        self.use_llm = getattr(
            getattr(settings, 'ingestion', None), 
            'chunk_refiner', 
            {}
        ).get('use_llm', False) if hasattr(settings, 'ingestion') else False
        
    @property
    def llm(self) -> Optional[BaseLLM]:
        """延迟加载 LLM 实例。"""
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(self.settings)
                logger.info("LLM initialized for chunk refinement")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Falling back to rule-based only.")
                self.use_llm = False
        return self._llm
    
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """通过精炼 Pipeline 变换 Chunk。

        参数:
            chunks: 要精炼的 Chunk 列表
            trace: 可选的 trace 上下文

        返回:
            精炼后的 Chunk 列表（长度与输入一致）
        """
        if not chunks:
            return []

        # 若启用 LLM，则并行处理 Chunk
        if self.use_llm and self.llm:
            # 开 LLM 时瓶颈是有网络 I/O，5 个线程能成倍提速
            return self._transform_parallel(chunks, trace)
        else:
            return self._transform_sequential(chunks, trace)
    
    def _refine_single_chunk(
        self, 
        chunk: Chunk, 
        trace: Optional[TraceContext] = None
    ) -> Tuple[Chunk, str, Optional[str]]:
        """精炼单个 Chunk。线程安全。

        参数:
            chunk: 要精炼的 Chunk
            trace: 可选的 trace 上下文

        返回:
            (refined_chunk, refined_by, error_message) 元组
        """
        try:
            # 步骤 1: 基于规则的精炼
            rule_refined_text = self._rule_based_refine(chunk.text)

            # 步骤 2: LLM 增强
            if self.use_llm and self.llm:
                llm_refined_text = self._llm_refine(rule_refined_text, trace)
                
                if llm_refined_text:
                    refined_text = llm_refined_text
                    refined_by = "llm"
                else:
                    refined_text = rule_refined_text
                    refined_by = "rule"
            else:
                refined_text = rule_refined_text
                refined_by = "rule"
            
            refined_chunk = Chunk(
                id=chunk.id,
                text=refined_text,
                metadata={
                    **(chunk.metadata or {}),
                    'refined_by': refined_by
                },
                source_ref=chunk.source_ref
            )
            return (refined_chunk, refined_by, None)
            
        except Exception as e:
            logger.error(f"Failed to refine chunk {chunk.id}: {e}")
            return (chunk, "error", str(e))
    
    def _transform_parallel(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """使用 ThreadPoolExecutor 并行处理 Chunk。"""
        max_workers = min(DEFAULT_MAX_WORKERS, len(chunks))
        refined_chunks = [None] * len(chunks)
        llm_enhanced_count = 0
        fallback_count = 0
        
        logger.debug(f"Processing {len(chunks)} chunks in parallel (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(self._refine_single_chunk, chunk, trace): idx
                for idx, chunk in enumerate(chunks)
            }
            
            # 收集结果
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    refined_chunk, refined_by, error = future.result()
                    refined_chunks[idx] = refined_chunk
                    
                    if refined_by == "llm":
                        llm_enhanced_count += 1
                    elif refined_by == "rule" and error is None:
                        fallback_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error in parallel refinement: {e}")
                    refined_chunks[idx] = chunks[idx]
        
        success_count = sum(1 for c in refined_chunks if c is not None)
        
        if trace:
            trace.record_stage("chunk_refiner", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": True,
                "max_workers": max_workers
            })
        
        logger.info(
            f"Refined {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, fallback: {fallback_count})"
        )
        
        return refined_chunks
    
    def _transform_sequential(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """顺序处理 Chunk（LLM 禁用时的回退方案）。"""
        refined_chunks = []
        success_count = 0
        llm_enhanced_count = 0
        fallback_count = 0
        
        for chunk in chunks:
            try:
                # 步骤 1: 基于规则的精炼（始终执行）
                rule_refined_text = self._rule_based_refine(chunk.text)

                # 步骤 2: 可选的 LLM 增强
                if self.use_llm and self.llm:
                    llm_refined_text = self._llm_refine(rule_refined_text, trace)

                    if llm_refined_text:
                        # LLM 成功
                        refined_text = llm_refined_text
                        refined_by = "llm"
                        llm_enhanced_count += 1
                    else:
                        # LLM 失败，回退到基于规则的结果
                        refined_text = rule_refined_text
                        refined_by = "rule"
                        fallback_count += 1
                        if chunk.metadata:
                            chunk.metadata['refine_fallback_reason'] = "llm_failed"
                else:
                    # LLM 已禁用，使用基于规则的结果
                    refined_text = rule_refined_text
                    refined_by = "rule"

                # 创建精炼后的 Chunk
                refined_chunk = Chunk(
                    id=chunk.id,
                    text=refined_text,
                    metadata={
                        **(chunk.metadata or {}),
                        'refined_by': refined_by
                    },
                    source_ref=chunk.source_ref
                )
                refined_chunks.append(refined_chunk)
                success_count += 1
                
            except Exception as e:
                # 原子失败：记录日志并保留原始内容
                logger.error(f"Failed to refine chunk {chunk.id}: {e}")
                refined_chunks.append(chunk)
        
        # 记录 trace
        if trace:
            trace.record_stage("chunk_refiner", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": False
            })
        
        logger.info(
            f"Refined {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, fallback: {fallback_count})"
        )
        
        return refined_chunks
    
    def _rule_based_refine(self, text: str) -> str:
        """应用基于规则的文本清洗。

        清洗操作:
            1. 移除页眉/页脚（分隔线 + 元数据）
            2. 移除 HTML 注释
            3. 移除 HTML 标签（保留内容）
            4. 规范化多余的空白
            5. 保留代码块和 Markdown 格式

        参数:
            text: 原始 Chunk 文本

        返回:
            清洗后的文本
        """
        if not text:
            return text

        # 若仅含空白则提前返回
        if not text.strip():
            return ""

        # 保留代码块（先提取，稍后恢复）
        code_blocks = []
        code_block_pattern = r'```[\s\S]*?```'

        def extract_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"

        text = re.sub(code_block_pattern, extract_code_block, text)

        # 1. 移除带页码/页脚的分隔线
        # 模式: ────────────────
        # 后接: Page XX、Footer text 等
        text = re.sub(
            r'─{10,}.*?(?:Page \d+|Footer|Section \d+|©|Confidential).*?─{10,}',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        text = re.sub(r'─{10,}', '', text)  # 移除剩余的分隔线

        # 2. 移除 HTML 注释
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # 3. 移除 HTML 标签（但保留内容）
        text = re.sub(r'<[^>]+>', '', text)

        # 4. 规范化空白
        # - 将多个空格合并为单个空格
        text = re.sub(r' {2,}', ' ', text)

        # - 将 3 个及以上连续换行合并为 2 个（保留段落分隔）
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 5. 移除每行的首尾空白
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)

        # 6. 恢复代码块
        for i, code_block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", code_block)

        # 最终清理
        text = text.strip()

        return text

    def _llm_refine(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """应用基于 LLM 的智能精炼。

        参数:
            text: 经规则精炼后的文本
            trace: 可选的 trace 上下文

        返回:
            LLM 精炼后的文本，失败时返回 None
        """
        if not text or not text.strip():
            return text

        try:
            # 加载提示词模板
            prompt_template = self._load_prompt()
            if not prompt_template:
                logger.warning("Prompt template not found, skipping LLM refinement")
                return None

            # 填充提示词
            if '{text}' not in prompt_template:
                logger.error("Prompt template missing {text} placeholder")
                return None

            prompt = prompt_template.replace('{text}', text)

            # 使用 Message 对象调用 LLM
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages, trace=trace)

            # 从 ChatResponse 中提取文本
            if isinstance(response, str):
                refined_text = response
            else:
                # response 是 ChatResponse 对象
                refined_text = response.content
            
            if refined_text and refined_text.strip():
                return refined_text.strip()
            else:
                logger.warning("LLM returned empty result")
                return None
                
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}")
            return None
    
    def _load_prompt(self) -> Optional[str]:
        """从文件加载提示词模板。

        返回:
            提示词模板字符串，文件不存在时返回 None
        """
        if self._prompt_template is not None:
            return self._prompt_template
        
        try:
            prompt_path = Path(self._prompt_path)
            if not prompt_path.exists():
                logger.warning(f"Prompt file not found: {self._prompt_path}")
                return None
            
            self._prompt_template = prompt_path.read_text(encoding='utf-8')
            logger.debug(f"Loaded prompt template from {self._prompt_path}")
            return self._prompt_template
            
        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")
            return None
