"""ResponseBuilder 与 CitationGenerator 的单元测试。

本模块测试 MCP 工具用于生成带引用格式化输出的响应构建组件。
"""

import pytest
from typing import Dict, Any, List

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.response.response_builder import ResponseBuilder, MCPToolResponse
from src.core.types import RetrievalResult


# =============================================================================
# 测试夹具
# =============================================================================

@pytest.fixture
def sample_retrieval_results() -> List[RetrievalResult]:
    """创建用于测试的示例检索结果。"""
    return [
        RetrievalResult(
            chunk_id="doc1_chunk_001",
            score=0.95,
            text="Azure OpenAI 是微软提供的人工智能服务，提供 GPT-4 和 DALL-E 等模型的访问能力。它与 Azure 云服务深度集成，支持企业级安全和合规性需求。",
            metadata={
                "source_path": "docs/azure-guide.pdf",
                "page": 5,
                "chunk_index": 0,
                "title": "Azure OpenAI 简介",
                "doc_type": "pdf",
            },
        ),
        RetrievalResult(
            chunk_id="doc1_chunk_002",
            score=0.87,
            text="配置 Azure OpenAI 需要以下步骤：1. 创建 Azure 订阅；2. 在 Azure Portal 创建 OpenAI 资源；3. 获取 API 密钥和端点地址。",
            metadata={
                "source_path": "docs/azure-guide.pdf",
                "page": 10,
                "chunk_index": 5,
                "title": "配置步骤",
            },
        ),
        RetrievalResult(
            chunk_id="doc2_chunk_003",
            score=0.72,
            text="GPT-4 模型支持多轮对话、代码生成、文本分析等多种任务。在 Azure 平台上，您可以通过 REST API 或 SDK 进行调用。",
            metadata={
                "source_path": "docs/gpt4-usage.md",
                "chunk_index": 2,
                "title": "GPT-4 使用指南",
            },
        ),
    ]


@pytest.fixture
def empty_retrieval_results() -> List[RetrievalResult]:
    """空结果列表。"""
    return []


@pytest.fixture
def citation_generator() -> CitationGenerator:
    """创建 CitationGenerator 实例。"""
    return CitationGenerator()


@pytest.fixture
def response_builder() -> ResponseBuilder:
    """创建 ResponseBuilder 实例。"""
    return ResponseBuilder()


# =============================================================================
# CitationGenerator 测试
# =============================================================================

class TestCitationGenerator:
    """CitationGenerator 类的测试。"""

    def test_generate_citations_basic(
        self,
        citation_generator: CitationGenerator,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试基本的引用生成。"""
        citations = citation_generator.generate(sample_retrieval_results)
        
        assert len(citations) == 3
        
        # 检查第一个引用
        assert citations[0].index == 1
        assert citations[0].chunk_id == "doc1_chunk_001"
        assert citations[0].source == "docs/azure-guide.pdf"
        assert citations[0].score == 0.95
        assert citations[0].page == 5
        
        # 检查索引从 1 开始
        assert citations[1].index == 2
        assert citations[2].index == 3
    
    def test_generate_citations_empty_results(
        self,
        citation_generator: CitationGenerator,
        empty_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试结果为空时的引用生成。"""
        citations = citation_generator.generate(empty_retrieval_results)
        assert citations == []
    
    def test_citation_to_dict(
        self,
        citation_generator: CitationGenerator,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试 Citation.to_dict() 序列化。"""
        citations = citation_generator.generate(sample_retrieval_results)
        
        citation_dict = citations[0].to_dict()
        
        assert citation_dict["index"] == 1
        assert citation_dict["chunk_id"] == "doc1_chunk_001"
        assert citation_dict["source"] == "docs/azure-guide.pdf"
        assert citation_dict["score"] == 0.95
        assert citation_dict["page"] == 5
        assert "text_snippet" in citation_dict
    
    def test_text_snippet_truncation(self) -> None:
        """测试片段中的长文本会被截断。"""
        generator = CitationGenerator(snippet_max_length=50)
        
        result = RetrievalResult(
            chunk_id="test_001",
            score=0.9,
            text="这是一段非常长的文本，用于测试文本截断功能是否正常工作。" * 10,
            metadata={"source_path": "test.pdf"},
        )
        
        citations = generator.generate([result])
        
        # 片段应被截断
        assert len(citations[0].text_snippet) <= 60  # 50 + "..."
        assert citations[0].text_snippet.endswith("...")
    
    def test_metadata_extraction(
        self,
        citation_generator: CitationGenerator,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试元数据字段被正确提取。"""
        citations = citation_generator.generate(sample_retrieval_results)
        
        # 第一个引用的元数据中应包含 title
        assert "title" in citations[0].metadata
        assert citations[0].metadata["title"] == "Azure OpenAI 简介"
        
        # 检查包含 chunk_index
        assert "chunk_index" in citations[0].metadata
        assert citations[0].metadata["chunk_index"] == 0
    
    def test_page_number_extraction(self) -> None:
        """测试从不同元数据格式中提取页码。"""
        generator = CitationGenerator()
        
        # 测试使用 'page' 键
        result1 = RetrievalResult(
            chunk_id="test_001",
            score=0.9,
            text="Test content",
            metadata={"source_path": "test.pdf", "page": 5},
        )
        
        # 测试使用 'page_num' 键
        result2 = RetrievalResult(
            chunk_id="test_002",
            score=0.8,
            text="Test content",
            metadata={"source_path": "test.pdf", "page_num": 10},
        )
        
        # 测试不包含页码信息
        result3 = RetrievalResult(
            chunk_id="test_003",
            score=0.7,
            text="Test content",
            metadata={"source_path": "test.md"},
        )
        
        citations = generator.generate([result1, result2, result3])
        
        assert citations[0].page == 5
        assert citations[1].page == 10
        assert citations[2].page is None
    
    def test_format_citation_marker(
        self,
        citation_generator: CitationGenerator,
    ) -> None:
        """测试引用标记的格式化。"""
        assert citation_generator.format_citation_marker(1) == "[1]"
        assert citation_generator.format_citation_marker(10) == "[10]"
        assert citation_generator.format_citation_marker(99) == "[99]"


# =============================================================================
# ResponseBuilder 测试
# =============================================================================

class TestResponseBuilder:
    """ResponseBuilder 类的测试。"""

    def test_build_basic_response(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试基本的响应构建。"""
        response = response_builder.build(
            results=sample_retrieval_results,
            query="Azure OpenAI 配置",
            collection="docs",
        )
        
        assert isinstance(response, MCPToolResponse)
        assert response.is_empty is False
        assert len(response.citations) == 3
        assert response.metadata["query"] == "Azure OpenAI 配置"
        assert response.metadata["collection"] == "docs"
        assert response.metadata["result_count"] == 3
    
    def test_build_empty_response(
        self,
        response_builder: ResponseBuilder,
        empty_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试结果为空时的响应构建。"""
        response = response_builder.build(
            results=empty_retrieval_results,
            query="不存在的查询",
            collection="docs",
        )
        
        assert response.is_empty is True
        assert len(response.citations) == 0
        assert "未找到相关结果" in response.content
        assert "建议" in response.content
    
    def test_content_contains_citation_markers(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试内容中包含引用标记。"""
        response = response_builder.build(
            results=sample_retrieval_results,
            query="Azure 配置",
        )
        
        # 应包含引用标记
        assert "[1]" in response.content
        assert "[2]" in response.content
        assert "[3]" in response.content
    
    def test_content_contains_source_info(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试内容中包含来源信息。"""
        response = response_builder.build(
            results=sample_retrieval_results,
            query="Azure",
        )
        
        # 应包含来源路径
        assert "azure-guide.pdf" in response.content
        assert "gpt4-usage.md" in response.content
    
    def test_content_contains_scores(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试内容中包含相关度分数。"""
        response = response_builder.build(
            results=sample_retrieval_results,
            query="Azure",
        )
        
        # 应包含分数指示符
        assert "相关度" in response.content or "95%" in response.content
    
    def test_response_to_dict(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试 MCPToolResponse.to_dict() 序列化。"""
        response = response_builder.build(
            results=sample_retrieval_results,
            query="Azure",
        )
        
        response_dict = response.to_dict()
        
        assert "content" in response_dict
        assert "structuredContent" in response_dict
        assert "citations" in response_dict["structuredContent"]
        assert len(response_dict["structuredContent"]["citations"]) == 3
    
    def test_response_to_mcp_content(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试面向 MCP 协议的 MCPToolResponse.to_mcp_content()。"""
        from mcp import types
        
        response = response_builder.build(
            results=sample_retrieval_results,
            query="Azure",
        )
        
        content_blocks = response.to_mcp_content()
        
        assert len(content_blocks) >= 1
        # 现在返回的是 TextContent 对象而不是字典
        assert isinstance(content_blocks[0], types.TextContent)
        assert content_blocks[0].type == "text"
        assert "检索结果" in content_blocks[0].text
    
    def test_max_results_in_content(self) -> None:
        """测试 max_results_in_content 参数生效。"""
        builder = ResponseBuilder(max_results_in_content=2)
        
        results = [
            RetrievalResult(
                chunk_id=f"chunk_{i}",
                score=0.9 - i * 0.1,
                text=f"Content {i}",
                metadata={"source_path": f"doc{i}.pdf"},
            )
            for i in range(5)
        ]
        
        response = builder.build(results=results, query="test")
        
        # 内容中应提示还有更多结果
        assert "还有" in response.content or "未显示" in response.content
        
        # 但所有引用仍应被包含
        assert len(response.citations) == 5
    
    def test_collection_in_metadata(
        self,
        response_builder: ResponseBuilder,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试提供 collection 时它会被包含在元数据中。"""
        # 提供 collection
        response1 = response_builder.build(
            results=sample_retrieval_results,
            query="test",
            collection="my_collection",
        )
        assert response1.metadata["collection"] == "my_collection"
        
        # 不提供 collection
        response2 = response_builder.build(
            results=sample_retrieval_results,
            query="test",
        )
        assert "collection" not in response2.metadata
    
    def test_empty_response_with_collection(
        self,
        response_builder: ResponseBuilder,
    ) -> None:
        """测试空响应中会提及 collection 名称。"""
        response = response_builder.build(
            results=[],
            query="test",
            collection="my_docs",
        )
        
        assert "my_docs" in response.content


# =============================================================================
# Citation 数据类测试
# =============================================================================

class TestCitationDataclass:
    """Citation 数据类的测试。"""

    def test_citation_creation(self) -> None:
        """测试 Citation 对象的创建。"""
        citation = Citation(
            index=1,
            chunk_id="test_001",
            source="test.pdf",
            score=0.95,
            text_snippet="Test content...",
            page=5,
            metadata={"title": "Test"},
        )
        
        assert citation.index == 1
        assert citation.chunk_id == "test_001"
        assert citation.source == "test.pdf"
        assert citation.score == 0.95
        assert citation.page == 5
    
    def test_citation_to_dict_without_page(self) -> None:
        """测试 page 为 None 时的 Citation.to_dict()。"""
        citation = Citation(
            index=1,
            chunk_id="test_001",
            source="test.md",
            score=0.8,
            text_snippet="Test",
        )
        
        citation_dict = citation.to_dict()
        
        assert "page" not in citation_dict
    
    def test_citation_score_rounding(self) -> None:
        """测试 to_dict() 中分数会被四舍五入。"""
        citation = Citation(
            index=1,
            chunk_id="test_001",
            source="test.pdf",
            score=0.123456789,
            text_snippet="Test",
        )
        
        citation_dict = citation.to_dict()
        
        # 分数应四舍五入到 4 位小数
        assert citation_dict["score"] == 0.1235


# =============================================================================
# MCPToolResponse 数据类测试
# =============================================================================

class TestMCPToolResponseDataclass:
    """MCPToolResponse 数据类的测试。"""

    def test_response_creation(self) -> None:
        """测试 MCPToolResponse 对象的创建。"""
        response = MCPToolResponse(
            content="Test content",
            citations=[],
            metadata={"query": "test"},
            is_empty=False,
        )
        
        assert response.content == "Test content"
        assert response.citations == []
        assert response.metadata["query"] == "test"
        assert response.is_empty is False
    
    def test_response_default_values(self) -> None:
        """测试 MCPToolResponse 的默认值。"""
        response = MCPToolResponse(content="Test")
        
        assert response.citations == []
        assert response.metadata == {}
        assert response.is_empty is False


# =============================================================================
# 集成测试
# =============================================================================

class TestResponseBuilderIntegration:
    """ResponseBuilder 与 CitationGenerator 的集成测试。"""

    def test_full_response_generation_flow(
        self,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试完整的响应生成流程。"""
        builder = ResponseBuilder()
        
        response = builder.build(
            results=sample_retrieval_results,
            query="如何配置 Azure OpenAI？",
            collection="technical_docs",
        )
        
        # 验证响应结构
        assert isinstance(response, MCPToolResponse)
        assert not response.is_empty
        
        # 验证内容
        assert "检索结果" in response.content
        assert "Azure" in response.content
        
        # 验证引用
        assert len(response.citations) == 3
        assert all(isinstance(c, Citation) for c in response.citations)
        
        # 验证元数据
        assert response.metadata["query"] == "如何配置 Azure OpenAI？"
        assert response.metadata["collection"] == "technical_docs"
        assert response.metadata["result_count"] == 3
        
        # 验证序列化
        response_dict = response.to_dict()
        assert "content" in response_dict
        assert "structuredContent" in response_dict
    
    def test_custom_citation_generator(
        self,
        sample_retrieval_results: List[RetrievalResult],
    ) -> None:
        """测试使用自定义 CitationGenerator 的 ResponseBuilder。"""
        custom_generator = CitationGenerator(
            snippet_max_length=50,
            include_metadata_fields=["title"],
        )
        
        builder = ResponseBuilder(citation_generator=custom_generator)
        response = builder.build(
            results=sample_retrieval_results,
            query="test",
        )
        
        # 片段应更短
        for citation in response.citations:
            assert len(citation.text_snippet) <= 60  # 50 + "..."
        
        # 元数据中只应包含 'title'
        for citation in response.citations:
            if citation.metadata:
                assert "chunk_index" not in citation.metadata
