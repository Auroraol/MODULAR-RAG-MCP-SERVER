"""ChunkRefiner 转换器的单元测试。"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.libs.llm.base_llm import BaseLLM


# 测试 Fixture

@pytest.fixture
def noisy_chunks_data():
    """从 fixture 中加载测试数据。"""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "noisy_chunks.json"
    with open(fixture_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def mock_settings():
    """创建未启用 LLM 的 mock 配置。"""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {'use_llm': False}
    return settings


@pytest.fixture
def mock_settings_with_llm():
    """创建启用 LLM 的 mock 配置。"""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {'use_llm': True}
    return settings


@pytest.fixture
def mock_llm():
    """创建 mock LLM。"""
    llm = Mock(spec=BaseLLM)
    llm.chat.return_value = "LLM refined text"
    return llm


@pytest.fixture
def sample_chunk():
    """创建一个示例 chunk。"""
    return Chunk(
        id="test_chunk_001",
        text="Sample text with  extra   spaces\n\n\n\nand newlines",
        metadata={"source": "test.pdf", "source_path": "test.pdf"},
        source_ref="test_doc"
    )


# 测试基于规则的清洗

class TestRuleBasedRefinement:
    """测试不依赖 LLM 的基于规则的清洗。"""

    def test_remove_excessive_whitespace(self, mock_settings, noisy_chunks_data):
        """测试移除过多空格与换行。"""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['excessive_whitespace']['input']
        expected = noisy_chunks_data['excessive_whitespace']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_remove_page_headers_footers(self, mock_settings, noisy_chunks_data):
        """测试移除带分隔线的页眉页脚。"""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['page_header_footer']['input']
        expected = noisy_chunks_data['page_header_footer']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_remove_html_tags_and_comments(self, mock_settings, noisy_chunks_data):
        """测试移除 HTML 标签与注释，同时保留 Markdown。"""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['format_markers']['input']
        expected = noisy_chunks_data['format_markers']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_preserve_code_blocks(self, mock_settings, noisy_chunks_data):
        """测试代码块内部格式保持不变。"""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['code_blocks']['input']
        expected = noisy_chunks_data['code_blocks']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_not_overclean_good_text(self, mock_settings, noisy_chunks_data):
        """测试干净文本不被过度处理。"""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['clean_text']['input']
        expected = noisy_chunks_data['clean_text']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_mixed_noise_scenario(self, mock_settings, noisy_chunks_data):
        """测试混合问题的综合噪声移除。"""
        refiner = ChunkRefiner(mock_settings)
        
        input_text = noisy_chunks_data['mixed_noise']['input']
        expected = noisy_chunks_data['mixed_noise']['expected_clean']
        
        result = refiner._rule_based_refine(input_text)
        assert result == expected
    
    def test_empty_string_handling(self, mock_settings):
        """测试空字符串处理。"""
        refiner = ChunkRefiner(mock_settings)

        assert refiner._rule_based_refine("") == ""
        # 仅含空白的字符串在 strip 后返回空串
        result = refiner._rule_based_refine("   \n\n  ")
        assert result.strip() == ""
    
    def test_none_handling(self, mock_settings):
        """测试 None 输入的处理。"""
        refiner = ChunkRefiner(mock_settings)
        result = refiner._rule_based_refine(None)
        assert result is None


# 测试转换流水线（仅规则模式）

class TestTransformPipelineRuleOnly:
    """测试不依赖 LLM 的完整转换流水线。"""

    def test_transform_single_chunk(self, mock_settings, sample_chunk):
        """测试转换单个 chunk。"""
        refiner = ChunkRefiner(mock_settings)
        
        result = refiner.transform([sample_chunk])
        
        assert len(result) == 1
        assert result[0].id == sample_chunk.id
        assert result[0].text == "Sample text with extra spaces\n\nand newlines"
        assert result[0].metadata['refined_by'] == 'rule'
    
    def test_transform_multiple_chunks(self, mock_settings):
        """测试转换多个 chunk。"""
        refiner = ChunkRefiner(mock_settings)
        
        chunks = [
            Chunk(id="c1", text="Text  with   spaces", metadata={"source_path": "test1.pdf"}),
            Chunk(id="c2", text="Another\n\n\nchunk", metadata={"source_path": "test2.pdf"}),
            Chunk(id="c3", text="Clean text", metadata={"source_path": "test3.pdf"})
        ]
        
        result = refiner.transform(chunks)
        
        assert len(result) == 3
        assert result[0].text == "Text with spaces"
        assert result[1].text == "Another\n\nchunk"
        assert result[2].text == "Clean text"
        assert all(r.metadata['refined_by'] == 'rule' for r in result)
    
    def test_transform_empty_list(self, mock_settings):
        """测试转换空的 chunk 列表。"""
        refiner = ChunkRefiner(mock_settings)
        result = refiner.transform([])
        assert result == []
    
    def test_metadata_preserved(self, mock_settings, sample_chunk):
        """测试原始元数据被保留。"""
        refiner = ChunkRefiner(mock_settings)
        
        result = refiner.transform([sample_chunk])
        
        assert result[0].metadata['source'] == 'test.pdf'
        assert 'refined_by' in result[0].metadata
    
    def test_trace_recording(self, mock_settings, sample_chunk):
        """测试 trace 上下文记录处理信息。"""
        refiner = ChunkRefiner(mock_settings)
        trace = TraceContext()
        
        refiner.transform([sample_chunk], trace=trace)
        
        stage_data = trace.get_stage_data('chunk_refiner')
        assert stage_data is not None
        assert stage_data['total_chunks'] == 1
        assert stage_data['success_count'] == 1
        assert stage_data['use_llm'] is False


# 测试 LLM 增强模式

class TestLLMEnhancement:
    """测试基于 LLM 的精炼。"""

    def test_llm_success_path(self, mock_settings_with_llm, mock_llm, sample_chunk):
        """测试 LLM 精炼成功。"""
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        mock_llm.chat.return_value = "Beautifully refined text by LLM"

        # mock prompt 加载
        refiner._prompt_template = "Refine this: {text}"
        
        result = refiner.transform([sample_chunk])
        
        assert len(result) == 1
        assert "Beautifully refined" in result[0].text
        assert result[0].metadata['refined_by'] == 'llm'
        mock_llm.chat.assert_called_once()
    
    def test_llm_fallback_on_failure(self, mock_settings_with_llm, mock_llm, sample_chunk):
        """测试 LLM 失败时回退到基于规则的方式。"""
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        mock_llm.chat.side_effect = Exception("LLM API error")
        
        result = refiner.transform([sample_chunk])
        
        assert len(result) == 1
        assert result[0].metadata['refined_by'] == 'rule'
    
    def test_llm_fallback_on_empty_response(self, mock_settings_with_llm, mock_llm, sample_chunk):
        """测试 LLM 返回空字符串时回退。"""
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        mock_llm.chat.return_value = ""
        refiner._prompt_template = "Refine: {text}"
        
        result = refiner.transform([sample_chunk])
        
        assert result[0].metadata['refined_by'] == 'rule'
    
    def test_llm_trace_recording(self, mock_settings_with_llm, mock_llm, sample_chunk):
        """测试 trace 记录 LLM 增强数量。"""
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        mock_llm.chat.return_value = "LLM result"
        refiner._prompt_template = "Refine: {text}"
        trace = TraceContext()
        
        refiner.transform([sample_chunk], trace=trace)
        
        stage_data = trace.get_stage_data('chunk_refiner')
        assert stage_data is not None
        assert stage_data['llm_enhanced_count'] == 1
        assert stage_data['fallback_count'] == 0


# 测试 Prompt 加载

class TestPromptLoading:
    """测试 prompt 模板加载。"""

    def test_load_existing_prompt(self, mock_settings):
        """测试加载已存在的 prompt 文件。"""
        refiner = ChunkRefiner(mock_settings)

        # 使用真实 prompt 文件
        prompt = refiner._load_prompt()
        
        assert prompt is not None
        assert '{text}' in prompt
    
    def test_prompt_cached_after_first_load(self, mock_settings):
        """测试首次加载后 prompt 被缓存。"""
        refiner = ChunkRefiner(mock_settings)
        
        prompt1 = refiner._load_prompt()
        prompt2 = refiner._load_prompt()
        
        assert prompt1 is prompt2  # 同一对象引用
    
    def test_missing_prompt_file(self, mock_settings):
        """测试缺少 prompt 文件时的处理。"""
        refiner = ChunkRefiner(mock_settings, prompt_path="nonexistent.txt")
        
        prompt = refiner._load_prompt()
        
        assert prompt is None
    
    def test_prompt_missing_placeholder(self, mock_settings, mock_llm, sample_chunk):
        """测试 prompt 缺少 {text} 时 LLM 精炼优雅失败。"""
        refiner = ChunkRefiner(mock_settings, llm=mock_llm)
        refiner._prompt_template = "This prompt has no placeholder"
        refiner.use_llm = True
        
        result = refiner._llm_refine("test text")
        
        assert result is None


# 测试原子处理

class TestAtomicProcessing:
    """测试单个 chunk 失败不影响其他 chunk。"""

    def test_exception_in_one_chunk_preserves_original(self, mock_settings):
        """测试处理某个 chunk 时异常会保留其原文。"""
        refiner = ChunkRefiner(mock_settings)

        # mock _rule_based_refine 使其在特定 chunk 上失败
        original_method = refiner._rule_based_refine
        
        def failing_refine(text):
            if "fail" in text:
                raise ValueError("Intentional test error")
            return original_method(text)
        
        refiner._rule_based_refine = failing_refine
        
        chunks = [
            Chunk(id="c1", text="Normal text", metadata={"source_path": "test1.pdf"}),
            Chunk(id="c2", text="This should fail processing", metadata={"source_path": "test2.pdf"}),
            Chunk(id="c3", text="Another normal text", metadata={"source_path": "test3.pdf"})
        ]
        
        result = refiner.transform(chunks)
        
        # 所有 chunk 均被返回
        assert len(result) == 3
        # 失败的 chunk 保持原样
        assert result[1].text == "This should fail processing"
        # 其他 chunk 正常处理
        assert result[0].metadata.get('refined_by') == 'rule'
        assert result[2].metadata.get('refined_by') == 'rule'


# 测试配置

class TestConfiguration:
    """测试配置处理。"""

    def test_use_llm_disabled_by_default(self):
        """测试配置缺失时 LLM 被禁用。"""
        settings = Mock(spec=Settings)
        settings.ingestion = None
        
        refiner = ChunkRefiner(settings)
        
        assert refiner.use_llm is False
    
    def test_lazy_llm_initialization(self, mock_settings_with_llm):
        """测试 LLM 仅在需要时初始化。"""
        with patch('src.ingestion.transform.chunk_refiner.LLMFactory.create') as mock_factory:
            refiner = ChunkRefiner(mock_settings_with_llm)

            # LLM 尚未初始化
            assert refiner._llm is None
            mock_factory.assert_not_called()

            # 访问 llm 属性触发初始化
            _ = refiner.llm
            mock_factory.assert_called_once()
    
    def test_llm_init_failure_disables_llm(self, mock_settings_with_llm):
        """测试 LLM 初始化失败会禁用 LLM 模式。"""
        with patch('src.ingestion.transform.chunk_refiner.LLMFactory.create', side_effect=Exception("Init failed")):
            refiner = ChunkRefiner(mock_settings_with_llm)

            # 尝试访问 LLM
            llm = refiner.llm

            # 失败后 LLM 模式被禁用
            assert llm is None
            assert refiner.use_llm is False
