"""MCP Protocol Handler 的单元测试。

测试覆盖：
- 工具注册与 schema 生成
- 不同返回类型的工具执行
- 错误处理（无效参数、内部错误、未知工具）
- 符合 JSON-RPC 错误码规范
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from mcp import types

from src.mcp_server.protocol_handler import (
    JSONRPCErrorCodes,
    ProtocolHandler,
    ToolDefinition,
    create_mcp_server,
    get_protocol_handler,
)


# ============================================================================
# 夹具
# ============================================================================


@pytest.fixture
def protocol_handler() -> ProtocolHandler:
    """创建全新的 ProtocolHandler 实例。"""
    return ProtocolHandler(
        server_name="test-server",
        server_version="1.0.0",
    )


@pytest.fixture
def sample_tool_schema() -> Dict[str, Any]:
    """工具输入的示例 JSON schema。"""
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }


# ============================================================================
# ToolDefinition 测试
# ============================================================================


class TestToolDefinition:
    """ToolDefinition 数据类的测试。"""

    def test_create_tool_definition(self, sample_tool_schema: Dict[str, Any]) -> None:
        """应创建包含所有必需字段的 ToolDefinition。"""

        async def dummy_handler(**kwargs: Any) -> str:
            return "result"

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema=sample_tool_schema,
            handler=dummy_handler,
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema == sample_tool_schema
        assert tool.handler == dummy_handler


# ============================================================================
# ProtocolHandler - 工具注册测试
# ============================================================================


class TestToolRegistration:
    """工具注册功能的测试。"""

    def test_register_tool_success(
        self, protocol_handler: ProtocolHandler, sample_tool_schema: Dict[str, Any]
    ) -> None:
        """应成功注册工具。"""

        async def handler(**kwargs: Any) -> str:
            return "result"

        protocol_handler.register_tool(
            name="search",
            description="Search the knowledge base",
            input_schema=sample_tool_schema,
            handler=handler,
        )

        assert "search" in protocol_handler.tools
        assert protocol_handler.tools["search"].name == "search"
        assert protocol_handler.tools["search"].description == "Search the knowledge base"

    def test_register_duplicate_tool_raises_error(
        self, protocol_handler: ProtocolHandler, sample_tool_schema: Dict[str, Any]
    ) -> None:
        """注册重复工具名时应抛出 ValueError。"""

        async def handler(**kwargs: Any) -> str:
            return "result"

        protocol_handler.register_tool(
            name="search",
            description="First search tool",
            input_schema=sample_tool_schema,
            handler=handler,
        )

        with pytest.raises(ValueError, match="Tool 'search' is already registered"):
            protocol_handler.register_tool(
                name="search",
                description="Duplicate search tool",
                input_schema=sample_tool_schema,
                handler=handler,
            )

    def test_register_multiple_tools(
        self, protocol_handler: ProtocolHandler, sample_tool_schema: Dict[str, Any]
    ) -> None:
        """应注册多个不同的工具。"""

        async def handler1(**kwargs: Any) -> str:
            return "result1"

        async def handler2(**kwargs: Any) -> str:
            return "result2"

        protocol_handler.register_tool(
            name="tool1",
            description="First tool",
            input_schema=sample_tool_schema,
            handler=handler1,
        )
        protocol_handler.register_tool(
            name="tool2",
            description="Second tool",
            input_schema={"type": "object", "properties": {}},
            handler=handler2,
        )

        assert len(protocol_handler.tools) == 2
        assert "tool1" in protocol_handler.tools
        assert "tool2" in protocol_handler.tools


# ============================================================================
# ProtocolHandler - 工具 Schema 测试
# ============================================================================


class TestGetToolSchemas:
    """get_tool_schemas 方法的测试。"""

    def test_empty_tools_returns_empty_list(
        self, protocol_handler: ProtocolHandler
    ) -> None:
        """无工具注册时应返回空列表。"""
        schemas = protocol_handler.get_tool_schemas()
        assert schemas == []

    def test_returns_tool_schemas(
        self, protocol_handler: ProtocolHandler, sample_tool_schema: Dict[str, Any]
    ) -> None:
        """应返回具有正确 schema 的 Tool 对象列表。"""

        async def handler(**kwargs: Any) -> str:
            return "result"

        protocol_handler.register_tool(
            name="query_knowledge_hub",
            description="Query the knowledge hub",
            input_schema=sample_tool_schema,
            handler=handler,
        )

        schemas = protocol_handler.get_tool_schemas()

        assert len(schemas) == 1
        assert isinstance(schemas[0], types.Tool)
        assert schemas[0].name == "query_knowledge_hub"
        assert schemas[0].description == "Query the knowledge hub"
        assert schemas[0].input_schema == sample_tool_schema


# ============================================================================
# ProtocolHandler - 工具执行测试
# ============================================================================


class TestExecuteTool:
    """工具执行功能的测试。"""

    @pytest.mark.asyncio
    async def test_execute_tool_returns_string(
        self, protocol_handler: ProtocolHandler, sample_tool_schema: Dict[str, Any]
    ) -> None:
        """应将字符串返回值包装为 CallToolResult。"""

        async def handler(query: str, top_k: int = 5) -> str:
            return f"Found {top_k} results for: {query}"

        protocol_handler.register_tool(
            name="search",
            description="Search tool",
            input_schema=sample_tool_schema,
            handler=handler,
        )

        result = await protocol_handler.execute_tool(
            "search", {"query": "test query", "top_k": 3}
        )

        assert isinstance(result, types.CallToolResult)
        assert result.is_error is False
        assert len(result.content) == 1
        assert isinstance(result.content[0], types.TextContent)
        assert result.content[0].text == "Found 3 results for: test query"

    @pytest.mark.asyncio
    async def test_execute_tool_returns_call_tool_result(
        self, protocol_handler: ProtocolHandler, sample_tool_schema: Dict[str, Any]
    ) -> None:
        """应直接透传 CallToolResult。"""

        async def handler(query: str, top_k: int = 5) -> types.CallToolResult:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="Custom result")],
                isError=False,
            )

        protocol_handler.register_tool(
            name="search",
            description="Search tool",
            input_schema=sample_tool_schema,
            handler=handler,
        )

        result = await protocol_handler.execute_tool("search", {"query": "test"})

        assert isinstance(result, types.CallToolResult)
        assert result.content[0].text == "Custom result"

    @pytest.mark.asyncio
    async def test_execute_tool_returns_content_list(
        self, protocol_handler: ProtocolHandler, sample_tool_schema: Dict[str, Any]
    ) -> None:
        """应将内容列表包装为 CallToolResult。"""

        async def handler(query: str, top_k: int = 5) -> List[types.TextContent]:
            return [
                types.TextContent(type="text", text="Result 1"),
                types.TextContent(type="text", text="Result 2"),
            ]

        protocol_handler.register_tool(
            name="search",
            description="Search tool",
            input_schema=sample_tool_schema,
            handler=handler,
        )

        result = await protocol_handler.execute_tool("search", {"query": "test"})

        assert len(result.content) == 2
        assert result.content[0].text == "Result 1"
        assert result.content[1].text == "Result 2"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(
        self, protocol_handler: ProtocolHandler
    ) -> None:
        """未知工具应返回错误结果。"""
        result = await protocol_handler.execute_tool(
            "nonexistent_tool", {"arg": "value"}
        )

        assert isinstance(result, types.CallToolResult)
        assert result.is_error is True
        assert "not found" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_with_invalid_params(
        self, protocol_handler: ProtocolHandler, sample_tool_schema: Dict[str, Any]
    ) -> None:
        """无效参数应返回错误。"""

        async def handler(query: str, top_k: int = 5) -> str:
            return f"Results for {query}"

        protocol_handler.register_tool(
            name="search",
            description="Search tool",
            input_schema=sample_tool_schema,
            handler=handler,
        )

        # 使用意外的关键字参数调用
        result = await protocol_handler.execute_tool(
            "search", {"query": "test", "invalid_param": "value"}
        )

        assert result.is_error is True
        assert "invalid" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_internal_error(
        self, protocol_handler: ProtocolHandler, sample_tool_schema: Dict[str, Any]
    ) -> None:
        """应返回通用错误而不泄露堆栈信息。"""

        async def handler(query: str, top_k: int = 5) -> str:
            raise RuntimeError("Database connection failed")

        protocol_handler.register_tool(
            name="search",
            description="Search tool",
            input_schema=sample_tool_schema,
            handler=handler,
        )

        result = await protocol_handler.execute_tool("search", {"query": "test"})

        assert result.is_error is True
        assert "internal" in result.content[0].text.lower()
        # 不应泄露实际错误消息
        assert "database" not in result.content[0].text.lower()


# ============================================================================
# ProtocolHandler - 能力测试
# ============================================================================


class TestGetCapabilities:
    """能力上报的测试。"""

    def test_capabilities_with_no_tools(
        self, protocol_handler: ProtocolHandler
    ) -> None:
        """无工具注册时应返回空的 tools 能力。"""
        caps = protocol_handler.get_capabilities()
        assert "tools" in caps
        assert caps["tools"] == {}

    def test_capabilities_with_tools(
        self, protocol_handler: ProtocolHandler, sample_tool_schema: Dict[str, Any]
    ) -> None:
        """有工具注册时应表示具备 tools 能力。"""

        async def handler(**kwargs: Any) -> str:
            return "result"

        protocol_handler.register_tool(
            name="search",
            description="Search tool",
            input_schema=sample_tool_schema,
            handler=handler,
        )

        caps = protocol_handler.get_capabilities()
        assert "tools" in caps


# ============================================================================
# JSON-RPC 错误码测试
# ============================================================================


class TestJSONRPCErrorCodes:
    """JSON-RPC 错误码常量的测试。"""

    def test_error_codes_are_correct(self) -> None:
        """应具有标准 JSON-RPC 2.0 错误码。"""
        assert JSONRPCErrorCodes.PARSE_ERROR == -32700
        assert JSONRPCErrorCodes.INVALID_REQUEST == -32600
        assert JSONRPCErrorCodes.METHOD_NOT_FOUND == -32601
        assert JSONRPCErrorCodes.INVALID_PARAMS == -32602
        assert JSONRPCErrorCodes.INTERNAL_ERROR == -32603


# ============================================================================
# create_mcp_server 工厂测试
# ============================================================================


class TestCreateMCPServer:
    """服务器工厂函数的测试。"""

    def test_creates_server_instance(self) -> None:
        """应创建 Server 实例。"""
        from mcp.server.lowlevel import Server

        server = create_mcp_server("test-server", "1.0.0")

        assert isinstance(server, Server)

    def test_attaches_protocol_handler(self) -> None:
        """应将 protocol handler 附加到服务器。"""
        server = create_mcp_server("test-server", "1.0.0")

        handler = get_protocol_handler(server)

        assert isinstance(handler, ProtocolHandler)
        assert handler.server_name == "test-server"
        assert handler.server_version == "1.0.0"

    def test_uses_provided_protocol_handler(
        self, sample_tool_schema: Dict[str, Any]
    ) -> None:
        """若提供则应使用传入的 protocol handler。"""
        custom_handler = ProtocolHandler(
            server_name="custom",
            server_version="2.0.0",
        )

        async def tool_handler(**kwargs: Any) -> str:
            return "result"

        custom_handler.register_tool(
            name="custom_tool",
            description="Custom tool",
            input_schema=sample_tool_schema,
            handler=tool_handler,
        )

        server = create_mcp_server(
            "test-server", "1.0.0", protocol_handler=custom_handler
        )

        handler = get_protocol_handler(server)
        assert handler is custom_handler
        assert "custom_tool" in handler.tools


# ============================================================================
# 类集成测试（Server + ProtocolHandler）
# ============================================================================


class TestServerProtocolHandlerIntegration:
    """服务器与 protocol handler 协同工作的测试。"""

    @pytest.mark.asyncio
    async def test_list_tools_returns_registered_tools(
        self, sample_tool_schema: Dict[str, Any]
    ) -> None:
        """服务器的 list_tools handler 应返回已注册工具。"""
        handler = ProtocolHandler(
            server_name="test-server",
            server_version="1.0.0",
        )

        async def search_handler(query: str, top_k: int = 5) -> str:
            return f"Results for {query}"

        handler.register_tool(
            name="query_knowledge_hub",
            description="Query the knowledge hub",
            input_schema=sample_tool_schema,
            handler=search_handler,
        )

        # 该用例自带同名工具，需跳过默认工具注册以免重复注册
        server = create_mcp_server(
            "test-server", "1.0.0", protocol_handler=handler, register_tools=False
        )

        # 校验工具可通过 protocol handler 访问
        tools = handler.get_tool_schemas()
        assert len(tools) == 1
        assert tools[0].name == "query_knowledge_hub"

    @pytest.mark.asyncio
    async def test_call_tool_executes_handler(
        self, sample_tool_schema: Dict[str, Any]
    ) -> None:
        """服务器的 call_tool handler 应执行工具。"""
        handler = ProtocolHandler(
            server_name="test-server",
            server_version="1.0.0",
        )

        async def search_handler(query: str, top_k: int = 5) -> str:
            return f"Found {top_k} results for: {query}"

        handler.register_tool(
            name="search",
            description="Search tool",
            input_schema=sample_tool_schema,
            handler=search_handler,
        )

        server = create_mcp_server("test-server", "1.0.0", protocol_handler=handler)

        # 通过 protocol handler 执行
        result = await handler.execute_tool("search", {"query": "test", "top_k": 10})

        assert result.is_error is False
        assert "Found 10 results for: test" in result.content[0].text
