"""通过 stdio 对真实服务器模拟 MCP 客户端的 E2E 测试。

以子进程方式启动 ``src.mcp_server.server``，并驱动完整的
JSON-RPC / MCP 生命周期：

    initialize → notifications/initialized → tools/list → tools/call

每个测试都在**协议传输层**验证 JSON-RPC 契约，因此服务器、协议处理器
或已注册工具中的任何破坏性变更都能被 CI 捕获。

用法::

    pytest tests/e2e/test_mcp_client.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ── 辅助函数 ─────────────────────────────────────────────────────────


def _start_server() -> subprocess.Popen:
    """以 stdio 传输方式启动 MCP 服务器子进程。

    返回:
        带有 stdin/stdout/stderr 管道的运行中子进程。
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def _send_jsonrpc(
    proc: subprocess.Popen,
    messages: List[Dict[str, Any]],
    expected_responses: int,
    timeout: float = 15.0,
) -> List[Dict[str, Any]]:
    """发送 JSON-RPC 消息并收集预期数量的响应。

    使用后台线程读取 stdout，从而在 ``readline()`` 阻塞时
    仍能遵守 ``timeout``。

    参数:
        proc: 带有 stdin/stdout 管道的子进程。
        messages: JSON-RPC 请求 / 通知列表。
        expected_responses: 需要等待的 JSON-RPC *响应*（带 ``id``）数量。
        timeout: 最长等待秒数。

    返回:
        解析后的 JSON-RPC 响应字典（仅包含带 ``id`` 的条目）。
    """
    assert proc.stdin is not None
    assert proc.stdout is not None

    for msg in messages:
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    responses: List[Dict[str, Any]] = []
    stop_event = threading.Event()

    def _reader() -> None:
        """在守护线程中读取 stdout 行，以便超时能够中断。"""
        while not stop_event.is_set():
            line = proc.stdout.readline()  # type: ignore[union-attr]
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if "id" in data and ("result" in data or "error" in data):
                responses.append(data)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    # 等待直到收集到足够的响应 *或* 超时
    deadline = time.time() + timeout
    while len(responses) < expected_responses and time.time() < deadline:
        time.sleep(0.1)

    stop_event.set()
    return responses


def _find(responses: List[Dict[str, Any]], req_id: int) -> Optional[Dict[str, Any]]:
    """按请求 id 查找响应。"""
    for r in responses:
        if r.get("id") == req_id:
            return r
    return None


def _terminate(proc: subprocess.Popen) -> None:
    """优雅地终止服务器子进程。"""
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ── 夹具 ─────────────────────────────────────────────────────────────

INIT_REQUEST: Dict[str, Any] = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "clientInfo": {"name": "e2e-pytest-client", "version": "1.0.0"},
        "capabilities": {},
    },
}

INITIALIZED_NOTIFICATION: Dict[str, Any] = {
    "jsonrpc": "2.0",
    "method": "notifications/initialized",
}


@pytest.fixture()
def mcp_server():
    """提供一个运行中的 MCP 服务器子进程；测试结束后拆除。"""
    proc = _start_server()
    yield proc
    _terminate(proc)


# ── 测试 ─────────────────────────────────────────────────────────────


class TestMCPClientE2E:
    """模拟完整 MCP 客户端会话的 E2E 测试套件。"""

    # ------------------------------------------------------------------
    # 1. 生命周期：initialize → tools/list
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_initialize_and_tools_list(self, mcp_server: subprocess.Popen) -> None:
        """服务器响应 initialize 和 tools/list，并返回全部 3 个已注册工具。"""
        messages = [
            INIT_REQUEST,
            INITIALIZED_NOTIFICATION,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            },
        ]

        responses = _send_jsonrpc(mcp_server, messages, expected_responses=2)

        # -- initialize ------------------------------------------------
        init_resp = _find(responses, 1)
        assert init_resp is not None, f"Missing initialize response. Got: {responses}"
        assert "result" in init_resp
        assert "serverInfo" in init_resp["result"]
        assert "capabilities" in init_resp["result"]
        assert init_resp["result"]["capabilities"].get("tools") is not None

        # -- tools/list ------------------------------------------------
        tools_resp = _find(responses, 2)
        assert tools_resp is not None, f"Missing tools/list response. Got: {responses}"
        assert "result" in tools_resp
        tools = tools_resp["result"]["tools"]
        assert isinstance(tools, list)

        tool_names = {t["name"] for t in tools}
        assert "query_knowledge_hub" in tool_names
        assert "list_collections" in tool_names
        assert "get_document_summary" in tool_names

        # 每个工具都必须声明合法的 inputSchema
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            schema = tool["inputSchema"]
            assert schema.get("type") == "object"
            assert "properties" in schema

    # ------------------------------------------------------------------
    # 2. tools/call – query_knowledge_hub（协议往返）
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_tools_call_query_knowledge_hub(
        self, mcp_server: subprocess.Popen
    ) -> None:
        """query_knowledge_hub 的 tools/call 返回格式正确的 CallToolResult。

        即使没有已索引的数据，该工具也应返回合法的（可能为空的）结果
        或优雅的错误——绝不能出现协议级失败。
        """
        messages = [
            INIT_REQUEST,
            INITIALIZED_NOTIFICATION,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "query_knowledge_hub",
                    "arguments": {
                        "query": "What is Azure OpenAI?",
                        "top_k": 3,
                    },
                },
            },
        ]

        responses = _send_jsonrpc(mcp_server, messages, expected_responses=2, timeout=60.0)

        init_resp = _find(responses, 1)
        assert init_resp is not None, "Missing initialize response"

        call_resp = _find(responses, 2)
        assert call_resp is not None, f"Missing tools/call response. Got: {responses}"
        assert "result" in call_resp, f"Expected result in response: {call_resp}"

        result = call_resp["result"]
        # MCP CallToolResult 必须包含 ``content`` 列表
        assert "content" in result
        assert isinstance(result["content"], list)
        assert len(result["content"]) >= 1

        # 每个内容块都必须包含 ``type`` 和 ``text``（图片则为 ``data``）
        for block in result["content"]:
            assert "type" in block
            assert block["type"] in ("text", "image", "resource")

        # 若存在 ``isError``，它必须是布尔值
        if "isError" in result:
            assert isinstance(result["isError"], bool)

    # ------------------------------------------------------------------
    # 3. tools/call – list_collections
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_tools_call_list_collections(
        self, mcp_server: subprocess.Popen
    ) -> None:
        """list_collections 的 tools/call 返回合法的集合列表。"""
        messages = [
            INIT_REQUEST,
            INITIALIZED_NOTIFICATION,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "list_collections",
                    "arguments": {"include_stats": True},
                },
            },
        ]

        responses = _send_jsonrpc(mcp_server, messages, expected_responses=2, timeout=15.0)

        call_resp = _find(responses, 2)
        assert call_resp is not None, f"Missing tools/call response. Got: {responses}"
        assert "result" in call_resp

        result = call_resp["result"]
        assert "content" in result
        assert isinstance(result["content"], list)
        assert len(result["content"]) >= 1

        # 第一个内容块应为带集合信息的文本（或空提示信息）
        first = result["content"][0]
        assert first["type"] == "text"
        assert isinstance(first.get("text", ""), str)

    # ------------------------------------------------------------------
    # 4. tools/call – get_document_summary（不存在的文档 → 优雅处理）
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_tools_call_get_document_summary_missing(
        self, mcp_server: subprocess.Popen
    ) -> None:
        """使用非法 doc_id 调用 get_document_summary 时返回格式正确的错误。"""
        messages = [
            INIT_REQUEST,
            INITIALIZED_NOTIFICATION,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "get_document_summary",
                    "arguments": {
                        "doc_id": "nonexistent_doc_id_12345",
                    },
                },
            },
        ]

        responses = _send_jsonrpc(mcp_server, messages, expected_responses=2, timeout=15.0)

        call_resp = _find(responses, 2)
        assert call_resp is not None, f"Missing tools/call response. Got: {responses}"
        assert "result" in call_resp

        result = call_resp["result"]
        assert "content" in result
        assert isinstance(result["content"], list)
        assert len(result["content"]) >= 1

        # 工具应通过 isError 或错误文本表示出错
        first_text = result["content"][0].get("text", "")
        is_error = result.get("isError", False)
        # isError 标志被置位，或文本中包含错误消息
        assert is_error or "not found" in first_text.lower() or "error" in first_text.lower(), (
            f"Expected an error indication for missing doc. Got isError={is_error}, text={first_text!r}"
        )

    # ------------------------------------------------------------------
    # 5. tools/call – 未知工具返回错误
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_tools_call_unknown_tool(
        self, mcp_server: subprocess.Popen
    ) -> None:
        """调用不存在的工具时返回错误响应，而不是崩溃。"""
        messages = [
            INIT_REQUEST,
            INITIALIZED_NOTIFICATION,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "nonexistent_tool",
                    "arguments": {},
                },
            },
        ]

        responses = _send_jsonrpc(mcp_server, messages, expected_responses=2, timeout=15.0)

        call_resp = _find(responses, 2)
        assert call_resp is not None, f"Missing tools/call response. Got: {responses}"
        assert "result" in call_resp

        result = call_resp["result"]
        assert "content" in result
        assert result.get("isError") is True or "error" in result["content"][0].get("text", "").lower()

    # ------------------------------------------------------------------
    # 6. 完整会话：列出工具 → 调用查询 → 校验引用格式
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_full_session_query_with_citations_format(
        self, mcp_server: subprocess.Popen
    ) -> None:
        """完整客户端会话：初始化 → 列表 → 查询，并校验引用结构。

        query_knowledge_hub 返回的文本应当满足以下之一：
        - 存在结果时包含引用标记（例如 [1]、[2]），或
        - 未索引任何数据时返回“无结果”消息。
        无论哪种情况，协议契约都必须得到满足。
        """
        messages = [
            INIT_REQUEST,
            INITIALIZED_NOTIFICATION,
            # 步骤 1：发现工具
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            },
            # 步骤 2：调用 query_knowledge_hub
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "query_knowledge_hub",
                    "arguments": {
                        "query": "RAG pipeline architecture",
                        "top_k": 5,
                    },
                },
            },
        ]

        responses = _send_jsonrpc(mcp_server, messages, expected_responses=3, timeout=60.0)

        # 校验 tools/list
        tools_resp = _find(responses, 2)
        assert tools_resp is not None, "Missing tools/list response"
        tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
        assert "query_knowledge_hub" in tool_names

        # 校验查询响应
        query_resp = _find(responses, 3)
        assert query_resp is not None, f"Missing query response. Got: {responses}"
        assert "result" in query_resp

        result = query_resp["result"]
        assert "content" in result
        assert isinstance(result["content"], list)
        assert len(result["content"]) >= 1

        # 收集所有文本内容
        all_text = " ".join(
            block.get("text", "")
            for block in result["content"]
            if block.get("type") == "text"
        )
        assert len(all_text) > 0, "Response should contain non-empty text content"

        # 响应应包含引用标记，或表明没有结果
        has_citations = "[1]" in all_text or "**Sources**" in all_text or "citation" in all_text.lower()
        has_no_results = (
            "no results" in all_text.lower()
            or "no relevant" in all_text.lower()
            or "no documents" in all_text.lower()
            or "not found" in all_text.lower()
            or "0 result" in all_text.lower()
            or result.get("isError") is True
        )
        assert has_citations or has_no_results, (
            f"Expected citations or 'no results' indication in response text: {all_text[:300]}"
        )

    # ------------------------------------------------------------------
    # 7. 同一会话内的多次顺序调用
    # ------------------------------------------------------------------

    @pytest.mark.e2e
    def test_multiple_tool_calls_same_session(
        self, mcp_server: subprocess.Popen
    ) -> None:
        """服务器能够在一个会话中处理多次 tools/call 调用。"""
        messages = [
            INIT_REQUEST,
            INITIALIZED_NOTIFICATION,
            # 调用 1：list_collections
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "list_collections",
                    "arguments": {"include_stats": False},
                },
            },
            # 调用 2：query_knowledge_hub
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "query_knowledge_hub",
                    "arguments": {"query": "test query", "top_k": 2},
                },
            },
            # 调用 3：get_document_summary（预期返回优雅的错误）
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "get_document_summary",
                    "arguments": {"doc_id": "does_not_exist"},
                },
            },
        ]

        responses = _send_jsonrpc(mcp_server, messages, expected_responses=4, timeout=60.0)

        # 四个响应（init + 3 次工具调用）都应到达
        for req_id in (1, 2, 3, 4):
            resp = _find(responses, req_id)
            assert resp is not None, f"Missing response for id={req_id}. Got: {responses}"
            assert "result" in resp, f"Response id={req_id} missing 'result': {resp}"

        # 每次工具调用的结果都应包含合法内容
        for req_id in (2, 3, 4):
            resp = _find(responses, req_id)
            assert resp is not None
            content = resp["result"]["content"]
            assert isinstance(content, list)
            assert len(content) >= 1
            assert all("type" in block for block in content)
