"""list_collections MCP 工具的单元测试。

本模块测试 ListCollectionsTool 类，该类通过 MCP 协议
提供 collection 列表能力。
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path

from src.mcp_server.tools.list_collections import (
    ListCollectionsTool,
    ListCollectionsConfig,
    CollectionInfo,
    TOOL_NAME,
    TOOL_DESCRIPTION,
    TOOL_INPUT_SCHEMA,
    register_tool,
)


# =============================================================================
# 测试夹具
# =============================================================================

@pytest.fixture
def mock_settings() -> Mock:
    """创建 mock 配置对象。"""
    settings = Mock()
    settings.vector_store = Mock()
    settings.vector_store.persist_directory = "./data/db/chroma"
    return settings


@pytest.fixture
def mock_config() -> ListCollectionsConfig:
    """创建测试配置。"""
    return ListCollectionsConfig(
        persist_directory="./test_data/chroma",
        include_stats_default=True,
    )


@pytest.fixture
def tool_with_mock_settings(mock_settings: Mock) -> ListCollectionsTool:
    """使用 mock 配置创建 ListCollectionsTool。"""
    return ListCollectionsTool(settings=mock_settings)


@pytest.fixture
def tool_with_config(mock_config: ListCollectionsConfig) -> ListCollectionsTool:
    """使用显式配置创建 ListCollectionsTool。"""
    return ListCollectionsTool(config=mock_config)


@pytest.fixture
def sample_collections() -> List[CollectionInfo]:
    """创建示例 collection 信息列表。"""
    return [
        CollectionInfo(
            name="knowledge_hub",
            count=150,
            metadata={"hnsw:space": "cosine"},
        ),
        CollectionInfo(
            name="documents",
            count=75,
            metadata={"description": "Main documents"},
        ),
        CollectionInfo(
            name="test_collection",
            count=0,
            metadata=None,
        ),
    ]


# =============================================================================
# CollectionInfo 测试
# =============================================================================

class TestCollectionInfo:
    """CollectionInfo 数据类的测试。"""

    def test_basic_creation(self) -> None:
        """测试基本的 CollectionInfo 创建。"""
        info = CollectionInfo(name="test_coll")
        
        assert info.name == "test_coll"
        assert info.count is None
        assert info.metadata is None
    
    def test_creation_with_all_fields(self) -> None:
        """测试填充所有字段的 CollectionInfo 创建。"""
        info = CollectionInfo(
            name="documents",
            count=100,
            metadata={"type": "pdf"},
        )
        
        assert info.name == "documents"
        assert info.count == 100
        assert info.metadata == {"type": "pdf"}
    
    def test_to_dict_minimal(self) -> None:
        """测试仅含最少字段的 to_dict。"""
        info = CollectionInfo(name="test")
        result = info.to_dict()
        
        assert result == {"name": "test"}
        assert "count" not in result
        assert "metadata" not in result
    
    def test_to_dict_with_count(self) -> None:
        """测试带 count 的 to_dict。"""
        info = CollectionInfo(name="test", count=50)
        result = info.to_dict()
        
        assert result == {"name": "test", "count": 50}
    
    def test_to_dict_full(self) -> None:
        """测试包含所有字段的 to_dict。"""
        info = CollectionInfo(
            name="docs",
            count=25,
            metadata={"source": "local"},
        )
        result = info.to_dict()
        
        assert result == {
            "name": "docs",
            "count": 25,
            "metadata": {"source": "local"},
        }
    
    def test_to_dict_empty_metadata(self) -> None:
        """测试 metadata 为空字典时的 to_dict。"""
        info = CollectionInfo(name="test", metadata={})
        result = info.to_dict()
        
        # 空 metadata 不应包含在结果中
        assert result == {"name": "test"}


# =============================================================================
# ListCollectionsConfig 测试
# =============================================================================

class TestListCollectionsConfig:
    """ListCollectionsConfig 数据类的测试。"""

    def test_default_values(self) -> None:
        """测试默认配置值。"""
        config = ListCollectionsConfig()
        
        assert config.persist_directory == "./data/db/chroma"
        assert config.include_stats_default is True
    
    def test_custom_values(self) -> None:
        """测试自定义配置值。"""
        config = ListCollectionsConfig(
            persist_directory="/custom/path",
            include_stats_default=False,
        )
        
        assert config.persist_directory == "/custom/path"
        assert config.include_stats_default is False


# =============================================================================
# ListCollectionsTool 初始化测试
# =============================================================================

class TestListCollectionsToolInit:
    """ListCollectionsTool 初始化的测试。"""

    def test_init_with_settings(self, mock_settings: Mock) -> None:
        """测试使用 settings 初始化。"""
        tool = ListCollectionsTool(settings=mock_settings)
        
        assert tool._settings == mock_settings
        assert tool._config is None
    
    def test_init_with_config(self, mock_config: ListCollectionsConfig) -> None:
        """测试使用显式配置初始化。"""
        tool = ListCollectionsTool(config=mock_config)
        
        assert tool._settings is None
        assert tool._config == mock_config
    
    def test_init_no_args(self) -> None:
        """测试无参数初始化。"""
        tool = ListCollectionsTool()
        
        assert tool._settings is None
        assert tool._config is None
    
    def test_settings_lazy_load(self) -> None:
        """测试 settings 惰性加载。"""
        tool = ListCollectionsTool()
        
        with patch('src.core.settings.load_settings') as mock_load:
            mock_settings = Mock()
            mock_load.return_value = mock_settings
            
            # 访问 settings 属性
            result = tool.settings
            
            mock_load.assert_called_once()
            assert result == mock_settings
    
    def test_config_derived_from_settings(self, mock_settings: Mock) -> None:
        """测试从 settings 派生 config。"""
        tool = ListCollectionsTool(settings=mock_settings)
        
        config = tool.config
        
        assert config.persist_directory == "./data/db/chroma"
    
    def test_config_fallback_no_vector_store(self) -> None:
        """测试 vector_store 配置缺失时 config 的回退。"""
        settings = Mock(spec=[])  # 无 vector_store 属性
        tool = ListCollectionsTool(settings=settings)
        
        config = tool.config
        
        assert config.persist_directory == "./data/db/chroma"


# =============================================================================
# ListCollectionsTool ChromaDB 客户端测试
# =============================================================================

class TestListCollectionsToolChromaClient:
    """ChromaDB 客户端管理的测试。"""

    def test_get_chroma_client_chromadb_not_installed(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试未安装 chromadb 时的错误。"""
        with patch.dict('sys.modules', {'chromadb': None}):
            with patch('builtins.__import__', side_effect=ImportError("No chromadb")):
                with pytest.raises(ImportError) as exc_info:
                    tool_with_config._get_chroma_client()
        
                assert "chromadb package is required" in str(exc_info.value)
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    def test_get_chroma_client_success(
        self,
        mock_chroma_settings: Mock,
        mock_client_class: Mock,
        tool_with_config: ListCollectionsTool,
        tmp_path: Path,
    ) -> None:
        """测试成功创建 ChromaDB 客户端。"""
        # 将配置更新为临时路径
        tool_with_config._config = ListCollectionsConfig(
            persist_directory=str(tmp_path / "chroma")
        )
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        result = tool_with_config._get_chroma_client()
        
        assert result == mock_client
        mock_client_class.assert_called_once()
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    def test_get_chroma_client_creates_directory(
        self,
        mock_chroma_settings: Mock,
        mock_client_class: Mock,
        tool_with_config: ListCollectionsTool,
        tmp_path: Path,
    ) -> None:
        """测试缺失的目录被创建。"""
        new_path = tmp_path / "new_chroma_dir"
        tool_with_config._config = ListCollectionsConfig(
            persist_directory=str(new_path)
        )
        
        mock_client_class.return_value = Mock()
        
        tool_with_config._get_chroma_client()
        
        assert new_path.exists()
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.config.Settings')
    def test_get_chroma_client_init_failure(
        self,
        mock_chroma_settings: Mock,
        mock_client_class: Mock,
        tool_with_config: ListCollectionsTool,
        tmp_path: Path,
    ) -> None:
        """测试客户端初始化失败时的错误处理。"""
        tool_with_config._config = ListCollectionsConfig(
            persist_directory=str(tmp_path)
        )
        
        mock_client_class.side_effect = Exception("Connection failed")
        
        with pytest.raises(RuntimeError) as exc_info:
            tool_with_config._get_chroma_client()
        
        assert "Failed to initialize ChromaDB client" in str(exc_info.value)


# =============================================================================
# ListCollectionsTool list_collections 方法测试
# =============================================================================

class TestListCollectionsMethod:
    """list_collections 方法的测试。"""

    def test_list_collections_empty(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试无 collection 时的列表查询。"""
        mock_client = Mock()
        mock_client.list_collections.return_value = []
        
        with patch.object(tool_with_config, '_get_chroma_client', return_value=mock_client):
            result = tool_with_config.list_collections()
        
        assert result == []
    
    def test_list_collections_with_stats(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试带统计信息的 collection 列表。"""
        mock_coll1 = Mock()
        mock_coll1.name = "collection1"
        mock_coll1.metadata = {"type": "docs"}
        mock_coll1.count.return_value = 100
        
        mock_coll2 = Mock()
        mock_coll2.name = "collection2"
        mock_coll2.metadata = {}
        mock_coll2.count.return_value = 50
        
        mock_client = Mock()
        mock_client.list_collections.return_value = [mock_coll1, mock_coll2]
        
        with patch.object(tool_with_config, '_get_chroma_client', return_value=mock_client):
            result = tool_with_config.list_collections(include_stats=True)
        
        assert len(result) == 2
        assert result[0].name == "collection1"
        assert result[0].count == 100
        assert result[0].metadata == {"type": "docs"}
        assert result[1].name == "collection2"
        assert result[1].count == 50
    
    def test_list_collections_without_stats(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试不带统计信息的 collection 列表。"""
        mock_coll = Mock()
        mock_coll.name = "test"
        mock_coll.metadata = None
        
        mock_client = Mock()
        mock_client.list_collections.return_value = [mock_coll]
        
        with patch.object(tool_with_config, '_get_chroma_client', return_value=mock_client):
            result = tool_with_config.list_collections(include_stats=False)
        
        assert len(result) == 1
        assert result[0].name == "test"
        assert result[0].count is None  # 未获取
        mock_coll.count.assert_not_called()
    
    def test_list_collections_count_error_graceful(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试 count() 失败时的优雅处理。"""
        mock_coll = Mock()
        mock_coll.name = "problematic"
        mock_coll.metadata = {}
        mock_coll.count.side_effect = Exception("Count failed")
        
        mock_client = Mock()
        mock_client.list_collections.return_value = [mock_coll]
        
        with patch.object(tool_with_config, '_get_chroma_client', return_value=mock_client):
            result = tool_with_config.list_collections(include_stats=True)
        
        # 仍应返回 collection，但 count 为 None
        assert len(result) == 1
        assert result[0].name == "problematic"
        assert result[0].count is None
    
    def test_list_collections_client_error(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试客户端失败时的错误处理。"""
        with patch.object(
            tool_with_config,
            '_get_chroma_client',
            side_effect=RuntimeError("Client error")
        ):
            result = tool_with_config.list_collections()
        
        assert result == []
    
    def test_list_collections_list_error(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试 list_collections 失败时的错误处理。"""
        mock_client = Mock()
        mock_client.list_collections.side_effect = Exception("List failed")
        
        with patch.object(tool_with_config, '_get_chroma_client', return_value=mock_client):
            result = tool_with_config.list_collections()
        
        assert result == []


# =============================================================================
# ListCollectionsTool format_response 方法测试
# =============================================================================

class TestFormatResponse:
    """format_response 方法的测试。"""

    def test_format_empty_collections(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试格式化空 collection 列表。"""
        result = tool_with_config.format_response([])
        
        assert result == "No collections found in the knowledge base."
    
    def test_format_single_collection(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试格式化单个 collection。"""
        collections = [CollectionInfo(name="docs", count=50)]
        result = tool_with_config.format_response(collections)
        
        assert "## Available Collections (1 total)" in result
        assert "1. **docs** - 50 documents" in result
    
    def test_format_multiple_collections(
        self,
        tool_with_config: ListCollectionsTool,
        sample_collections: List[CollectionInfo]
    ) -> None:
        """测试格式化多个 collection。"""
        result = tool_with_config.format_response(sample_collections)
        
        assert "## Available Collections (3 total)" in result
        assert "1. **knowledge_hub** - 150 documents" in result
        assert "2. **documents** - 75 documents" in result
        assert "3. **test_collection** - 0 documents" in result
    
    def test_format_with_metadata(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试格式化用户 metadata。"""
        collections = [
            CollectionInfo(
                name="research",
                count=30,
                metadata={"category": "papers", "year": 2024},
            )
        ]
        result = tool_with_config.format_response(collections)
        
        assert "**research**" in result
        assert "30 documents" in result
        assert "category=papers" in result
        assert "year=2024" in result
    
    def test_format_filters_internal_metadata(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试内部 metadata 被过滤。"""
        collections = [
            CollectionInfo(
                name="test",
                count=10,
                metadata={
                    "hnsw:space": "cosine",
                    "_internal": "value",
                    "user_field": "visible",
                },
            )
        ]
        result = tool_with_config.format_response(collections)
        
        # 内部 metadata 应被过滤
        assert "hnsw:space" not in result
        assert "_internal" not in result
        # 用户 metadata 应可见
        assert "user_field=visible" in result
    
    def test_format_without_count(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试 count 为 None 时的格式化。"""
        collections = [CollectionInfo(name="no_count")]
        result = tool_with_config.format_response(collections)
        
        assert "1. **no_count**" in result
        assert "documents" not in result  # 不显示 count


# =============================================================================
# ListCollectionsTool execute 方法测试
# =============================================================================

class TestExecuteMethod:
    """execute 异步方法的测试。"""

    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        tool_with_config: ListCollectionsTool,
        sample_collections: List[CollectionInfo]
    ) -> None:
        """测试成功执行。"""
        with patch.object(
            tool_with_config,
            'list_collections',
            return_value=sample_collections
        ):
            result = await tool_with_config.execute(include_stats=True)
        
        assert result.is_error is False
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert "Available Collections (3 total)" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_execute_empty_result(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试无 collection 时执行。"""
        with patch.object(tool_with_config, 'list_collections', return_value=[]):
            result = await tool_with_config.execute()
        
        assert result.is_error is False
        assert "No collections found" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_execute_error(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试执行时的错误处理。"""
        with patch.object(
            tool_with_config,
            'list_collections',
            side_effect=Exception("Unexpected error")
        ):
            result = await tool_with_config.execute()
        
        assert result.is_error is True
        assert "Error listing collections" in result.content[0].text
    
    @pytest.mark.asyncio
    async def test_execute_passes_include_stats(
        self,
        tool_with_config: ListCollectionsTool
    ) -> None:
        """测试 include_stats 被正确传递。"""
        mock_list = Mock(return_value=[])
        
        with patch.object(tool_with_config, 'list_collections', mock_list):
            await tool_with_config.execute(include_stats=False)
        
        mock_list.assert_called_once_with(include_stats=False)


# =============================================================================
# register_tool 函数测试
# =============================================================================

class TestRegisterTool:
    """register_tool 函数的测试。"""

    def test_register_tool_success(self) -> None:
        """测试工具注册成功。"""
        mock_handler = Mock()
        
        register_tool(mock_handler)
        
        mock_handler.register_tool.assert_called_once()
        call_args = mock_handler.register_tool.call_args
        
        assert call_args.kwargs['name'] == TOOL_NAME
        assert call_args.kwargs['description'] == TOOL_DESCRIPTION
        assert call_args.kwargs['input_schema'] == TOOL_INPUT_SCHEMA
        assert callable(call_args.kwargs['handler'])
    
    @pytest.mark.asyncio
    async def test_registered_handler_callable(self) -> None:
        """测试注册的 handler 可调用。"""
        mock_handler = Mock()

        register_tool(mock_handler)

        # 获取注册的 handler
        handler = mock_handler.register_tool.call_args.kwargs['handler']

        # mock 工具的 execute 方法
        with patch.object(
            ListCollectionsTool,
            'execute',
            new_callable=AsyncMock
        ) as mock_execute:
            mock_result = Mock()
            mock_execute.return_value = mock_result
            
            result = await handler(include_stats=True)
            
            # handler 应已被调用
            assert result == mock_result


# =============================================================================
# 工具元数据测试
# =============================================================================

class TestToolMetadata:
    """工具元数据常量的测试。"""

    def test_tool_name(self) -> None:
        """测试工具名称常量。"""
        assert TOOL_NAME == "list_collections"
    
    def test_tool_description_content(self) -> None:
        """测试工具描述包含关键信息。"""
        assert "collection" in TOOL_DESCRIPTION.lower()
        assert "knowledge base" in TOOL_DESCRIPTION.lower()
    
    def test_input_schema_structure(self) -> None:
        """测试输入 schema 结构正确。"""
        assert TOOL_INPUT_SCHEMA["type"] == "object"
        assert "properties" in TOOL_INPUT_SCHEMA
        assert "include_stats" in TOOL_INPUT_SCHEMA["properties"]
    
    def test_input_schema_include_stats(self) -> None:
        """测试 include_stats 属性的 schema。"""
        prop = TOOL_INPUT_SCHEMA["properties"]["include_stats"]
        
        assert prop["type"] == "boolean"
        assert prop["default"] is True
        assert "description" in prop
    
    def test_no_required_params(self) -> None:
        """测试没有必需参数。"""
        assert TOOL_INPUT_SCHEMA["required"] == []
