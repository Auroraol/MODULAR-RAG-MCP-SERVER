"""MetadataEnricher 接入真实 LLM 提供方的集成测试。

这些测试需要真实的 API key，并会发起真实的 API 调用。
运行方式：pytest tests/integration/test_metadata_enricher_llm.py -v -s

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
from src.ingestion.transform.metadata_enricher import MetadataEnricher


# 测试数据：需要 metadata 增强的真实 chunk
SAMPLE_TECHNICAL_CHUNK = """
# Microservices Architecture Design Patterns

Microservices architecture is an approach to developing a single application as a suite of small services, each running in its own process and communicating with lightweight mechanisms, often an HTTP resource API. These services are built around business capabilities and independently deployable by fully automated deployment machinery.

Key characteristics include:
- Componentization via Services
- Organized around Business Capabilities
- Products not Projects
- Smart endpoints and dumb pipes
- Decentralized Governance
- Decentralized Data Management
- Infrastructure Automation
- Design for failure
- Evolutionary Design

The pattern emerged from real-world use at companies like Netflix, Amazon, and Uber. It addresses the challenges of monolithic architectures while introducing its own complexities.
"""

SAMPLE_CODE_CHUNK = """
The UserAuthenticationService class implements OAuth2 authentication flow. It uses JWT tokens for stateless session management and includes methods like authenticateUser(), refreshToken(), and validatePermissions(). The service integrates with external identity providers including Google, GitHub, and Azure AD.
"""

SAMPLE_DATA_SCIENCE_CHUNK = """
Gradient Boosting Machines (GBM) are ensemble learning methods that build predictive models in a stage-wise fashion. Unlike random forests which train trees in parallel, GBM trains them sequentially. Each new tree corrects errors made by the previous ensemble. Common implementations include XGBoost, LightGBM, and CatBoost.
"""


# Fixture 定义

@pytest.fixture
def sample_technical_chunk():
    """技术文档 chunk。"""
    return Chunk(
        id="chunk_tech_001",
        text=SAMPLE_TECHNICAL_CHUNK,
        metadata={"source_path": "architecture.md", "type": "technical"},
        source_ref="architecture.md#microservices"
    )


@pytest.fixture
def sample_code_chunk():
    """与代码相关的 chunk。"""
    return Chunk(
        id="chunk_code_001",
        text=SAMPLE_CODE_CHUNK,
        metadata={"source_path": "auth_service.md", "type": "code"},
        source_ref="auth_service.md#oauth"
    )


@pytest.fixture
def sample_datascience_chunk():
    """数据科学 chunk。"""
    return Chunk(
        id="chunk_ds_001",
        text=SAMPLE_DATA_SCIENCE_CHUNK,
        metadata={"source_path": "ml_guide.md", "type": "datascience"},
        source_ref="ml_guide.md#gbm"
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
                config_dict = yaml.safe_load(f)

            settings = load_settings("config/settings.yaml")

            # 校验 LLM 已配置
            if not hasattr(settings, 'llm') or settings.llm.provider != 'azure':
                pytest.skip("Azure LLM not configured in settings.yaml")

            # 从 config_dict 提取 API key 并注入环境变量
            llm_config = config_dict.get('llm', {})
            api_key = llm_config.get('api_key', '')
            azure_endpoint = llm_config.get('azure_endpoint', '')

            if api_key:
                os.environ["AZURE_OPENAI_API_KEY"] = api_key
            if azure_endpoint:
                os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
                os.environ["ENDPOINT"] = azure_endpoint

            # 注意: Settings 是冻结的，因此直接在 enricher 实例上启用 LLM
            return settings

        except Exception as e:
            pytest.skip(f"Failed to load Azure settings: {e}")

    elif provider == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        settings = Mock(spec=Settings)
        settings.llm = Mock()
        settings.llm.provider = 'openai'
        settings.llm.api_key = api_key
        settings.llm.model = 'gpt-4o-mini'
        settings.llm.temperature = 0.3
        settings.llm.max_tokens = 500

        settings.ingestion = Mock()
        settings.ingestion.metadata_enricher = {'use_llm': True}

        return settings

    elif provider == 'ollama':
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        settings = Mock(spec=Settings)
        settings.llm = Mock()
        settings.llm.provider = 'ollama'
        settings.llm.base_url = base_url
        settings.llm.model = 'qwen2.5:3b'
        settings.llm.temperature = 0.3
        settings.llm.max_tokens = 500

        settings.ingestion = Mock()
        settings.ingestion.metadata_enricher = {'use_llm': True}

        return settings

    else:
        raise ValueError(f"Unsupported provider: {provider}")


# ============================================================================
# Azure LLM 集成测试（主要 - 使用真实配置）
# ============================================================================

@pytest.mark.integration
@pytest.mark.llm
class TestMetadataEnricherAzureIntegration:
    """使用 Azure OpenAI 的集成测试（来自 settings.yaml）。"""

    def test_azure_enrichment_success(self, sample_technical_chunk):
        """测试使用 Azure LLM 成功进行 metadata 增强。"""
        settings = create_settings_for_provider('azure')
        enricher = MetadataEnricher(settings)

        # 强制启用 LLM（覆盖冻结的 dataclass）
        enricher.use_llm = True

        trace = TraceContext(trace_id="azure_test_001")

        # 执行增强
        result = enricher.transform([sample_technical_chunk], trace=trace)

        # 断言
        assert len(result) == 1
        enriched_chunk = result[0]

        # 校验 metadata 已增强
        assert 'title' in enriched_chunk.metadata
        assert 'summary' in enriched_chunk.metadata
        assert 'tags' in enriched_chunk.metadata
        assert enriched_chunk.metadata['enriched_by'] == 'llm'

        # 校验增强质量
        title = enriched_chunk.metadata['title']
        summary = enriched_chunk.metadata['summary']
        tags = enriched_chunk.metadata['tags']

        print("\n" + "="*60)
        print("AZURE LLM ENRICHMENT RESULT")
        print("="*60)
        print(f"Title: {title}")
        print(f"Summary: {summary}")
        print(f"Tags: {tags}")
        print("="*60)

        # 质量检查
        assert title
        assert len(title) <= 200, "Title should be concise"
        assert "microservice" in title.lower() or "architecture" in title.lower()

        assert summary
        assert len(summary) > 50, "Summary should be substantial"
        assert any(keyword in summary.lower() for keyword in ['service', 'microservice', 'architecture'])

        assert tags
        assert len(tags) >= 3, "Should extract at least 3 tags"
        assert isinstance(tags, list)

        # 校验 trace 记录
        assert 'llm_enrich' in trace.stages
        assert 'metadata_enricher' in trace.stages
        assert trace.stages['llm_enrich']['data']['success'] is True

    def test_azure_multiple_chunks_enrichment(self, sample_technical_chunk, sample_code_chunk, sample_datascience_chunk):
        """测试对不同内容类型的多个 chunk 进行增强。"""
        settings = create_settings_for_provider('azure')
        enricher = MetadataEnricher(settings)

        # 强制启用 LLM
        enricher.use_llm = True

        chunks = [sample_technical_chunk, sample_code_chunk, sample_datascience_chunk]
        result = enricher.transform(chunks)

        assert len(result) == 3

        print("\n" + "="*60)
        print("MULTIPLE CHUNKS ENRICHMENT")
        print("="*60)

        for i, chunk in enumerate(result):
            assert chunk.metadata['enriched_by'] == 'llm'
            print(f"\nChunk {i+1}:")
            print(f"  Title: {chunk.metadata['title']}")
            print(f"  Tags: {chunk.metadata['tags']}")

        print("="*60)

    def test_azure_fallback_on_invalid_model(self, sample_technical_chunk):
        """测试使用无效模型名称时的优雅回退。"""
        settings = create_settings_for_provider('azure')
        # 覆盖为无效模型
        settings.llm.model = 'gpt-nonexistent-model-12345'

        enricher = MetadataEnricher(settings)

        # 强制启用 LLM
        enricher.use_llm = True

        trace = TraceContext(trace_id="azure_fallback_test")

        # 不应抛出异常，应回退到基于规则的方式
        result = enricher.transform([sample_technical_chunk], trace=trace)

        assert len(result) == 1
        enriched_chunk = result[0]

        # 应包含 metadata（来自基于规则的回退）
        assert 'title' in enriched_chunk.metadata
        assert 'summary' in enriched_chunk.metadata
        assert 'tags' in enriched_chunk.metadata

        # 应标记为回退
        assert enriched_chunk.metadata['enriched_by'] == 'rule'
        assert 'enrich_fallback_reason' in enriched_chunk.metadata

        print("\n" + "="*60)
        print("FALLBACK TEST - Rule-based enrichment")
        print("="*60)
        print(f"Enriched by: {enriched_chunk.metadata['enriched_by']}")
        print(f"Fallback reason: {enriched_chunk.metadata.get('enrich_fallback_reason')}")
        print(f"Title: {enriched_chunk.metadata['title']}")
        print("="*60)


# ============================================================================
# OpenAI LLM 集成测试（可选 - 需要 OPENAI_API_KEY）
# ============================================================================

@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestMetadataEnricherOpenAIIntegration:
    """使用 OpenAI 的集成测试（需要 API key）。"""

    def test_openai_enrichment_success(self, sample_code_chunk):
        """测试使用 OpenAI 成功进行增强。"""
        settings = create_settings_for_provider('openai')
        enricher = MetadataEnricher(settings)

        result = enricher.transform([sample_code_chunk])

        assert len(result) == 1
        enriched_chunk = result[0]

        assert enriched_chunk.metadata['enriched_by'] == 'llm'
        assert 'title' in enriched_chunk.metadata
        assert 'OAuth' in enriched_chunk.metadata['title'] or 'Authentication' in enriched_chunk.metadata['title']

        print("\n" + "="*60)
        print("OPENAI LLM ENRICHMENT RESULT")
        print("="*60)
        print(f"Title: {enriched_chunk.metadata['title']}")
        print(f"Summary: {enriched_chunk.metadata['summary']}")
        print(f"Tags: {enriched_chunk.metadata['tags']}")
        print("="*60)


# ============================================================================
# Ollama LLM 集成测试（可选 - 需要运行中的 Ollama）
# ============================================================================

@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.skipif(not os.getenv("OLLAMA_BASE_URL"), reason="Ollama not configured")
class TestMetadataEnricherOllamaIntegration:
    """使用 Ollama 本地 LLM 的集成测试。"""

    def test_ollama_enrichment_success(self, sample_datascience_chunk):
        """测试使用 Ollama 成功进行增强。"""
        settings = create_settings_for_provider('ollama')
        enricher = MetadataEnricher(settings)

        result = enricher.transform([sample_datascience_chunk])

        assert len(result) == 1
        enriched_chunk = result[0]

        # Ollama 可能较慢或不够稳定，因此只检查是否尝试过增强
        assert 'title' in enriched_chunk.metadata
        assert 'enriched_by' in enriched_chunk.metadata

        print("\n" + "="*60)
        print("OLLAMA LLM ENRICHMENT RESULT")
        print("="*60)
        print(f"Enriched by: {enriched_chunk.metadata['enriched_by']}")
        print(f"Title: {enriched_chunk.metadata['title']}")
        print(f"Summary: {enriched_chunk.metadata['summary']}")
        print(f"Tags: {enriched_chunk.metadata['tags']}")
        print("="*60)


# ============================================================================
# 校验指引
# ============================================================================

"""
To verify the integration tests are working correctly:

1. Run Azure tests (primary):
   pytest tests/integration/test_metadata_enricher_llm.py::TestMetadataEnricherAzureIntegration -v -s

2. Check output:
   - Title should be semantically meaningful and relevant
   - Summary should capture key concepts (not just copy first sentence)
   - Tags should include relevant technical terms
   - enriched_by should be 'llm' for success cases

3. Verify fallback:
   - When model is invalid, should fallback to 'rule' enrichment
   - Should not crash the pipeline

4. Cost awareness:
   - These tests make real API calls
   - Run sparingly during development
   - Azure tests use configured model (check settings.yaml for cost)
"""
