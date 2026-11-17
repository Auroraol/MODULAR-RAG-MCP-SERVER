"""DenseEncoder 接入真实 Azure Embedding 的集成测试。

本测试套件通过真实 API 调用验证 DenseEncoder 与 Azure Embedding provider
的正确集成。这些测试验证:
- 真实 Embedding API 连通性
- 正确的向量维度
- 使用真实 provider 的批处理
- 类生产场景下的错误处理

⚠️ 警告: 这些测试会发起真实 API 调用并产生费用。
仅在验证 Azure Embedding 配置时运行。
"""

import pytest
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.core.types import Chunk
from src.core.settings import load_settings
from src.libs.embedding.embedding_factory import EmbeddingFactory


@pytest.fixture(scope="module")
def settings():
    """从配置文件加载配置。"""
    return load_settings("config/settings.yaml")


@pytest.fixture(scope="module")
def azure_embedding(settings):
    """根据配置创建 Azure Embedding 实例。

    该 fixture 验证:
    1. 配置包含有效的 Azure Embedding 配置
    2. EmbeddingFactory 能从 settings.yaml 创建 Azure provider
    3. 该 provider 已就绪可用

    所有配置（endpoint、api_key、model）均来自 settings.yaml。
    不使用硬编码值或环境变量覆盖。
    """
    # 校验 Azure 配置存在
    assert settings.embedding.provider == "azure", \
        "Integration test requires Azure embedding provider in settings"
    
    assert settings.embedding.azure_endpoint, \
        "Azure endpoint must be configured in settings.yaml"
    
    assert settings.embedding.api_key, \
        "Azure API key must be configured in settings.yaml"
    
    # 通过工厂创建 embedding provider（从配置读取全部配置）
    embedding = EmbeddingFactory.create(settings)
    
    return embedding


@pytest.fixture
def encoder(azure_embedding):
    """创建使用 Azure Embedding provider 的 DenseEncoder。"""
    return DenseEncoder(azure_embedding, batch_size=10)


# ============================================================================
# 真实 API 调用测试
# ============================================================================

def test_encode_single_chunk_with_azure(encoder):
    """测试使用真实 Azure Embedding API 编码单个 chunk。

    该测试验证:
    - API 连通性
    - 正确的响应格式
    - 预期的向量维度（text-embedding-ada-002 为 1536）
    """
    chunks = [
        Chunk(
            id="test_1",
            text="This is a test chunk for Azure Embedding integration.",
            metadata={'source_path': 'integration_test'}
        )
    ]
    
    vectors = encoder.encode(chunks)
    
    # 验证输出结构
    assert len(vectors) == 1, "Should return exactly 1 vector"
    assert len(vectors[0]) == 1536, \
        "text-embedding-ada-002 should return 1536-dimensional vectors"

    # 验证所有值均为 float
    assert all(isinstance(v, float) for v in vectors[0]), \
        "All vector components should be floats"

    # 验证向量非零（合理性检查）
    assert any(v != 0.0 for v in vectors[0]), \
        "Vector should contain non-zero values"


def test_encode_multiple_chunks_with_azure(encoder):
    """测试在单个批次中编码多个 chunk。

    该测试验证:
    - 批处理能与真实 API 配合工作
    - 所有 chunk 均被处理
    - 向量在语义上可区分
    """
    chunks = [
        Chunk(id="1", text="Artificial intelligence is transforming technology.", metadata={'source_path': 'integration_test'}),
        Chunk(id="2", text="Machine learning enables computers to learn from data.", metadata={'source_path': 'integration_test'}),
        Chunk(id="3", text="Python is a popular programming language.", metadata={'source_path': 'integration_test'}),
    ]
    
    vectors = encoder.encode(chunks)
    
    # 验证向量数量正确
    assert len(vectors) == 3, "Should return 3 vectors for 3 chunks"

    # 验证所有向量维度正确
    assert all(len(v) == 1536 for v in vectors), \
        "All vectors should have 1536 dimensions"

    # 验证向量彼此不同（语义差异）
    # Chunk 1 和 2 相关（AI/ML），chunk 3 不同（Python）
    # 这里不校验精确的相似度值，只要求向量不同
    assert vectors[0] != vectors[1] != vectors[2], \
        "Vectors should be distinct for different texts"


def test_encode_with_batching(azure_embedding):
    """测试使用真实 API 进行多批次编码。

    该测试验证:
    - 多批次处理正常工作
    - 批次边界不会引发问题
    - 所有 chunk 跨批次均被处理
    """
    encoder = DenseEncoder(azure_embedding, batch_size=2)
    
    chunks = [
        Chunk(id=f"chunk_{i}", text=f"Test chunk number {i} with unique content.", metadata={'source_path': 'integration_test'})
        for i in range(5)
    ]
    
    vectors = encoder.encode(chunks)
    
    # 应分 3 个批次处理：[0:2]、[2:4]、[4:5]
    assert len(vectors) == 5, "All 5 chunks should be processed"
    assert all(len(v) == 1536 for v in vectors), "All vectors should have correct dimension"

    # 验证批处理未打乱顺序
    # （在不知道向量内容的情况下无法验证精确顺序，
    #  但可以验证数量与维度）
    assert encoder.get_batch_count(5) == 3, "Should calculate 3 batches"


def test_encode_realistic_text_lengths(encoder):
    """测试编码不同真实长度的 chunk。

    该测试验证:
    - 短 chunk（标题/表头）
    - 中等 chunk（段落）
    - 长 chunk（完整章节）
    """
    chunks = [
        Chunk(
            id="short",
            text="Introduction",
            metadata={'source_path': 'integration_test'}
        ),
        Chunk(
            id="medium",
            text=(
                "This is a medium-length chunk representing a typical paragraph. "
                "It contains multiple sentences with varying content and structure. "
                "The embedding should capture the semantic meaning effectively."
            ),
            metadata={'source_path': 'integration_test'}
        ),
        Chunk(
            id="long",
            text=(
                "This is a longer chunk that might represent a full section of a document. "
                "It contains multiple paragraphs with different topics and concepts. "
                "The first paragraph introduces the main theme and sets the context. "
                "The second paragraph provides detailed explanations and examples. "
                "The third paragraph summarizes key points and conclusions. "
                "This tests the embedding model's ability to capture meaning from longer texts."
            ),
            metadata={'source_path': 'integration_test'}
        ),
    ]
    
    vectors = encoder.encode(chunks)
    
    assert len(vectors) == 3, "Should process all 3 chunks"
    assert all(len(v) == 1536 for v in vectors), "All vectors should have correct dimension"


def test_encode_special_characters(encoder):
    """测试编码包含特殊字符和格式的文本。

    该测试验证:
    - Unicode 处理
    - 特殊字符
    - 代码片段
    - 数学符号
    """
    chunks = [
        Chunk(
            id="unicode",
            text="多语言文本支持：中文、日本語、한국어、العربية",
            metadata={'source_path': 'integration_test'}
        ),
        Chunk(
            id="code",
            text='def hello_world():\n    print("Hello, World!")\n    return True',
            metadata={'source_path': 'integration_test'}
        ),
        Chunk(
            id="math",
            text="Mathematical formula: E = mc² and α + β = γ",
            metadata={'source_path': 'integration_test'}
        ),
    ]
    
    vectors = encoder.encode(chunks)
    
    assert len(vectors) == 3, "Should handle special characters correctly"
    assert all(len(v) == 1536 for v in vectors), "Vector dimensions should be consistent"


# ============================================================================
# 配置校验测试
# ============================================================================

def test_azure_configuration_is_valid(settings):
    """校验 settings.yaml 中的 Azure Embedding 配置是否正确。

    该测试校验:
    - provider 设置为 'azure'
    - 指定了模型名称
    - 维度与模型匹配（ada-002 为 1536）
    - 可通过环境变量提供凭据
    """
    assert settings.embedding.provider == "azure", \
        "Provider should be 'azure' for integration tests"
    
    assert settings.embedding.model == "text-embedding-ada-002", \
        "Model should be text-embedding-ada-002 for this test suite"
    
    assert settings.embedding.dimensions == 1536, \
        "Dimensions should be 1536 for text-embedding-ada-002"


def test_factory_creates_azure_embedding(settings):
    """校验 EmbeddingFactory 能正确创建 Azure provider。"""
    embedding = EmbeddingFactory.create(settings)

    # 验证类型正确（AzureEmbedding）
    assert embedding.__class__.__name__ == "AzureEmbedding", \
        "Factory should create AzureEmbedding instance"


# ============================================================================
# 错误处理测试（优雅降级）
# ============================================================================

def test_encode_handles_empty_text_gracefully(encoder):
    """测试空/非法 chunk 在调用 API 前被拦截。

    以避免在非法请求上浪费 API 配额。
    """
    chunks = [
        Chunk(id="empty", text="", metadata={'source_path': 'integration_test'})
    ]
    
    with pytest.raises(ValueError, match="has empty or whitespace-only text"):
        encoder.encode(chunks)


def test_encode_validates_before_api_call(encoder):
    """测试校验在执行昂贵的 API 调用之前进行。"""
    chunks = [
        Chunk(id="valid", text="Valid chunk", metadata={'source_path': 'integration_test'}),
        Chunk(id="invalid", text="   \n  ", metadata={'source_path': 'integration_test'}),  # 仅空白字符
    ]

    # 应在发起任何 API 调用之前校验失败
    with pytest.raises(ValueError, match="whitespace-only"):
        encoder.encode(chunks)


# ============================================================================
# 性能与质量观察
# ============================================================================

def test_encode_performance_observation(encoder):
    """观察真实 API 的编码性能。

    该测试不做性能断言，只记录耗时信息，
    供人工观察与优化参考。
    """
    import time
    
    chunks = [
        Chunk(id=f"perf_{i}", text=f"Performance test chunk {i}", metadata={'source_path': 'integration_test'})
        for i in range(10)
    ]
    
    start_time = time.time()
    vectors = encoder.encode(chunks)
    elapsed = time.time() - start_time
    
    assert len(vectors) == 10
    
    # 记录用于观察（非断言）
    print(f"\n⏱️  Encoded 10 chunks in {elapsed:.2f}s ({elapsed/10:.2f}s per chunk)")


def test_semantic_similarity_observation(encoder):
    """观察相关与不相关 chunk 的语义相似度。

    该测试计算余弦相似度，以校验 Embedding 是否正确捕捉语义关系。
    """
    import math
    
    chunks = [
        Chunk(id="1", text="Dogs are loyal pets and great companions.", metadata={'source_path': 'integration_test'}),
        Chunk(id="2", text="Cats are independent animals that enjoy solitude.", metadata={'source_path': 'integration_test'}),
        Chunk(id="3", text="Canines make excellent guard animals for homes.", metadata={'source_path': 'integration_test'}),
    ]
    
    vectors = encoder.encode(chunks)
    
    # 计算余弦相似度的辅助函数
    def cosine_similarity(v1, v2):
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(b * b for b in v2))
        return dot_product / (magnitude1 * magnitude2)
    
    # 计算相似度
    sim_dogs_cats = cosine_similarity(vectors[0], vectors[1])  # 不同：dogs vs cats
    sim_dogs_canines = cosine_similarity(vectors[0], vectors[2])  # 相似：dogs vs canines

    # 记录观察结果
    print(f"\n📊 Semantic Similarity:")
    print(f"   Dogs vs Cats: {sim_dogs_cats:.4f}")
    print(f"   Dogs vs Canines: {sim_dogs_canines:.4f}")
    
    # 预期：Dogs 与 Canines 的相似度应高于与 Cats 的相似度
    # （这是一个较弱的断言 —— 真实阈值取决于模型）
    assert sim_dogs_canines > sim_dogs_cats, \
        "Semantically related chunks should have higher similarity"
