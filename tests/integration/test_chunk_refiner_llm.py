"""ChunkRefiner 接入真实 LLM 提供方的集成测试。

这些测试需要真实的 API key，并会发起真实的 API 调用。
运行方式：pytest tests/integration/test_chunk_refiner_llm.py -v -s

所需环境变量:
    - OPENAI_API_KEY: 用于 OpenAI 测试
    - AZURE_OPENAI_API_KEY: 用于 Azure 测试
    - OLLAMA_BASE_URL: 用于 Ollama 测试（默认: http://localhost:11434）
"""

import os
import pytest
from unittest.mock import Mock

from src.core.settings import Settings, load_settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.chunk_refiner import ChunkRefiner


# 测试数据：来自 PDF 提取的真实含噪 chunk
NOISY_PDF_CHUNK = """
────────────────────────────
Page 42 | Technical Documentation
────────────────────────────


Chapter 5: System Architecture

The   microservices   architecture  consists  of  several  key  components.

<!-- Internal note: Update diagram -->

<div class="important">
Each service communicates via REST API or message queues.
</div>




The main   components   are:
- Gateway   Service  
- Authentication  Service  
- Data   Processing   Service


────────────────────────────
Footer: © 2024 Company | Confidential
────────────────────────────
"""

EXPECTED_CLEAN_RESULT_KEYWORDS = [
    "Chapter 5",
    "System Architecture",
    "microservices",
    "REST API",
    "message queues",
    "Gateway Service",
    "Authentication Service"
]


# Fixture 定义

@pytest.fixture
def sample_noisy_chunk():
    """创建用于测试的含噪 chunk 样本。"""
    return Chunk(
        id="test_pdf_chunk_001",
        text=NOISY_PDF_CHUNK,
        metadata={"source": "technical_doc.pdf", "source_path": "technical_doc.pdf", "page": 42},
        source_ref="doc_technical_2024"
    )


def create_settings_for_provider(provider: str) -> Settings:
    """为指定 provider 创建配置对象。

    对于 Azure provider，从 settings.yaml 加载实际配置。
    其他 provider 则使用环境变量。

    参数:
        provider: 'openai'、'azure'、'ollama' 之一

    返回:
        为对应 provider 配置的 Settings 对象
    """
    if provider == 'azure':
        # 对于 Azure，从 settings.yaml 加载真实配置
        try:
            import yaml
            with open("config/settings.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # 注入环境变量供 AzureLLM 使用
            if 'llm' in config and config['llm'].get('provider') == 'azure':
                os.environ["AZURE_OPENAI_API_KEY"] = config['llm'].get('api_key', '')
                os.environ["AZURE_OPENAI_ENDPOINT"] = config['llm'].get('azure_endpoint', '')
                os.environ["ENDPOINT"] = config['llm'].get('azure_endpoint', '')

                os.environ["AZURE_OPENAI_API_VERSION"] = config['llm'].get('api_version', '')
                os.environ["OPENAI_API_VERSION"] = config['llm'].get('api_version', '')

            real_settings = load_settings("config/settings.yaml")

            # 创建 Mock Settings 对象以便修改（真实配置是冻结的）
            settings = Mock(spec=Settings)
            # 复制必要的冻结配置
            settings.llm = real_settings.llm
            settings.ingestion = Mock()
            # 为 chunk refiner 启用 LLM
            settings.ingestion.chunk_refiner = {'use_llm': True}
            return settings
        except Exception as e:
            pytest.skip(f"Failed to load settings.yaml or configure Azure: {e}")

    # 非 Azure provider 使用环境变量
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {'use_llm': True}

    # LLM 配置
    settings.llm = Mock()
    settings.llm.provider = provider
    
    if provider == 'openai':
        settings.llm.model = "gpt-3.5-turbo"
        settings.llm.api_key = os.getenv('OPENAI_API_KEY')
        settings.llm.temperature = 0.3
        settings.llm.max_tokens = 1000
        
    elif provider == 'ollama':
        settings.llm.model = "llama2"
        settings.llm.base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        settings.llm.temperature = 0.3
        
    return settings


# 检查 provider 可用性的辅助函数

def is_provider_available(provider: str) -> tuple[bool, str]:
    """检查 provider 的凭据是否可用。

    返回:
        (is_available, env_var_name)
    """
    if provider == 'azure':
        # 对于 Azure，检查 settings.yaml 是否存在且包含 LLM 配置
        try:
            import yaml
            with open("config/settings.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if 'llm' in config and config['llm'].get('provider') == 'azure':
                return True, 'settings.yaml'
        except:
            pass
        return False, 'settings.yaml'

    if provider == 'openai':
        env_var = 'OPENAI_API_KEY'
        return os.getenv(env_var) is not None, env_var

    elif provider == 'ollama':
        # 若设置了 base_url 或使用默认值，则假定 Ollama 可用
        return True, 'OLLAMA_BASE_URL'

    return False, ''


# 测试用例

@pytest.mark.integration
@pytest.mark.parametrize("provider,env_var", [
    ("openai", "OPENAI_API_KEY"),
    ("azure", "AZURE_OPENAI_API_KEY"),
    ("ollama", "OLLAMA_BASE_URL"),
])
def test_multiple_providers_if_available(provider, env_var, sample_noisy_chunk):
    """测试使用多个 LLM provider 进行精炼（若已配置）。

    未配置凭据的 provider 将跳过该测试。
    """
    available, env_name = is_provider_available(provider)

    if not available:
        pytest.skip(f"Skipping {provider} test: {env_name} not set")

    # 为 provider 创建配置
    settings = create_settings_for_provider(provider)

    # 创建 refiner
    refiner = ChunkRefiner(settings)
    trace = TraceContext()

    # 执行精炼
    result = refiner.transform([sample_noisy_chunk], trace=trace)

    # 断言
    assert len(result) == 1
    refined_chunk = result[0]

    # 应标记为由 LLM 精炼
    assert refined_chunk.metadata['refined_by'] == 'llm', \
        f"Expected LLM refinement for {provider}, got {refined_chunk.metadata.get('refined_by')}"

    # 精炼后的文本应更干净（无分隔线、无 HTML）
    assert '────────────' not in refined_chunk.text
    assert '<!-- ' not in refined_chunk.text
    assert '<div' not in refined_chunk.text
    assert 'Footer:' not in refined_chunk.text.lower()

    # 应保留关键内容
    for keyword in EXPECTED_CLEAN_RESULT_KEYWORDS:
        assert keyword in refined_chunk.text, \
            f"Expected keyword '{keyword}' not found in refined text"

    # Trace 应记录 LLM 使用情况
    stage_data = trace.get_stage_data('chunk_refiner')
    assert stage_data is not None
    assert stage_data['data']['llm_enhanced_count'] == 1
    assert stage_data['data']['fallback_count'] == 0

    # 打印以便人工检查
    print(f"\n{'='*60}")
    print(f"Provider: {provider}")
    print(f"{'='*60}")
    print("ORIGINAL TEXT (first 200 chars):")
    print(sample_noisy_chunk.text[:200])
    print(f"\n{'-'*60}")
    print("REFINED TEXT:")
    print(refined_chunk.text)
    print(f"{'='*60}\n")


@pytest.mark.integration
def test_graceful_fallback_with_invalid_model(sample_noisy_chunk):
    """测试 LLM 失败时 refiner 回退到基于规则的实现。"""
    # 创建故意使用无效模型的配置
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {'use_llm': True}
    settings.llm = Mock()
    settings.llm.provider = 'openai'
    settings.llm.model = 'nonexistent-model-xyz'
    settings.llm.api_key = os.getenv('OPENAI_API_KEY', 'fake-key')

    refiner = ChunkRefiner(settings)
    trace = TraceContext()

    # 不应崩溃，应回退
    result = refiner.transform([sample_noisy_chunk], trace=trace)

    assert len(result) == 1
    # 应回退到基于规则的方式
    assert result[0].metadata['refined_by'] == 'rule'
    assert 'refine_fallback_reason' in result[0].metadata

    # 仍应应用基于规则的清洗
    assert '────────────' not in result[0].text
    assert 'Footer:' not in result[0].text


@pytest.mark.integration
@pytest.mark.skipif(not is_provider_available('openai')[0], reason="OPENAI_API_KEY not set")
def test_refinement_quality_comparison(sample_noisy_chunk):
    """对比基于规则与基于 LLM 的精炼质量。

    该测试提供可视化对比，便于人工评估质量。
    """
    # 仅基于规则
    settings_rule = Mock(spec=Settings)
    settings_rule.ingestion = Mock()
    settings_rule.ingestion.chunk_refiner = {'use_llm': False}

    refiner_rule = ChunkRefiner(settings_rule)
    result_rule = refiner_rule.transform([sample_noisy_chunk])

    # LLM 增强
    settings_llm = create_settings_for_provider('openai')
    refiner_llm = ChunkRefiner(settings_llm)
    result_llm = refiner_llm.transform([sample_noisy_chunk])

    # 打印对比
    print(f"\n{'='*60}")
    print("QUALITY COMPARISON")
    print(f"{'='*60}")
    print("\nORIGINAL (first 300 chars):")
    print(sample_noisy_chunk.text[:300])
    print(f"\n{'-'*60}")
    print("RULE-BASED REFINEMENT:")
    print(result_rule[0].text)
    print(f"\n{'-'*60}")
    print("LLM-ENHANCED REFINEMENT:")
    print(result_llm[0].text)
    print(f"{'='*60}\n")
    
    # 基础断言
    assert len(result_rule[0].text) > 0
    assert len(result_llm[0].text) > 0
    assert result_rule[0].metadata['refined_by'] == 'rule'
    assert result_llm[0].metadata['refined_by'] == 'llm'


@pytest.mark.integration
@pytest.mark.skipif(not is_provider_available('openai')[0], reason="OPENAI_API_KEY not set")
def test_batch_refinement_performance(sample_noisy_chunk):
    """测试批量精炼多个 chunk。"""
    settings = create_settings_for_provider('openai')
    refiner = ChunkRefiner(settings)
    trace = TraceContext()

    # 创建 3 个测试 chunk
    chunks = [
        sample_noisy_chunk,
        Chunk(
            id="chunk_002",
            text="Another  chunk   with   noise\n\n\n\nand issues.",
            metadata={"source": "test2.pdf", "source_path": "test2.pdf"}
        ),
        Chunk(
            id="chunk_003",
            text="Clean chunk without much noise.",
            metadata={"source": "test3.pdf", "source_path": "test3.pdf"}
        )
    ]
    
    # 精炼全部
    import time
    start_time = time.time()
    result = refiner.transform(chunks, trace=trace)
    elapsed_time = time.time() - start_time

    # 断言
    assert len(result) == 3
    assert all(r.metadata['refined_by'] == 'llm' for r in result)

    # Trace
    stage_data = trace.get_stage_data('chunk_refiner')
    assert stage_data['data']['llm_enhanced_count'] == 3
    
    print(f"\n{'='*60}")
    print(f"Refined {len(chunks)} chunks in {elapsed_time:.2f} seconds")
    print(f"Average: {elapsed_time/len(chunks):.2f} seconds per chunk")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
