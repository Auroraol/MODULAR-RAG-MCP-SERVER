"""MCP server stdio 入口点的集成测试。"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import pytest


def send_and_receive(
    proc: subprocess.Popen,
    requests: List[Dict[str, Any]],
    timeout: float = 5.0,
    expected_responses: int = 0,
) -> List[str]:
    """向 proc 的 stdin 发送请求并收集 stdout 行。

    参数:
        proc: 带 stdin/stdout 管道的子进程。
        requests: 要发送的 JSON-RPC 请求/通知列表。
        timeout: 等待响应的最长时间。
        expected_responses: 等待的响应数量（0 = 等待到超时/EOF）。

    返回:
        从 stdout 读取的行列表。
    """
    assert proc.stdin is not None
    assert proc.stdout is not None

    # 发送所有请求
    for req in requests:
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()

    # 带超时地读取 stdout
    lines = []
    start = time.time()
    response_count = 0

    # 统计预期响应数（含 'id' 字段的请求，不含通知）
    if expected_responses == 0:
        expected_responses = sum(1 for req in requests if 'id' in req)

    while time.time() - start < timeout:
        # 检查是否已收到足够响应
        if expected_responses > 0 and response_count >= expected_responses:
            break

        line = proc.stdout.readline()
        if not line:
            # 为较慢的响应多留一些时间
            time.sleep(0.1)
            continue
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
            # 统计 JSON-RPC 响应（含 'id' 以及 'result' 或 'error'）
            try:
                data = json.loads(stripped)
                if 'id' in data and ('result' in data or 'error' in data):
                    response_count += 1
            except json.JSONDecodeError:
                pass

    return lines


def find_response(lines: List[str], request_id: int) -> Optional[Dict[str, Any]]:
    """在 lines 中查找指定 id 的 JSON-RPC 响应。"""
    for line in lines:
        if not line.startswith('{"jsonrpc"'):
            continue
        try:
            data = json.loads(line)
            if data.get("id") == request_id:
                return data
        except json.JSONDecodeError:
            continue
    return None


@pytest.mark.integration
def test_mcp_server_initialize_stdio() -> None:
    """确保 initialize 正常工作，且 stdout 为干净的 JSON-RPC 输出。"""

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "clientInfo": {"name": "pytest", "version": "0.0.0"},
            "capabilities": {},
        },
    }

    try:
        lines = send_and_receive(proc, [request], timeout=5.0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    assert len(lines) > 0, "No stdout lines received."

    response = find_response(lines, 1)
    assert response is not None, f"No initialize response found in: {lines}"

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    assert "serverInfo" in response["result"]
    assert "capabilities" in response["result"]


@pytest.mark.integration
def test_mcp_server_tools_list_stdio() -> None:
    """确保 tools/list 正常工作，并返回空的工具数组。"""

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    requests = [
        # 初始化请求
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "clientInfo": {"name": "pytest", "version": "0.0.0"},
                "capabilities": {},
            },
        },
        # 初始化完成通知（MCP 协议要求）
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        },
        # 工具列表请求
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        },
    ]

    try:
        lines = send_and_receive(proc, requests, timeout=10.0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    assert len(lines) > 0, "No stdout lines received."

    # 校验 initialize 响应
    init_response = find_response(lines, 1)
    assert init_response is not None, f"No initialize response found in: {lines}"
    assert "result" in init_response

    # 校验 tools/list 响应
    tools_response = find_response(lines, 2)
    assert tools_response is not None, f"No tools/list response found in: {lines}"

    assert tools_response["jsonrpc"] == "2.0"
    assert tools_response["id"] == 2
    assert "result" in tools_response
    assert "tools" in tools_response["result"]
    # 应至少注册了 query_knowledge_hub 和 list_collections 两个工具
    assert isinstance(tools_response["result"]["tools"], list)
    assert len(tools_response["result"]["tools"]) >= 2

    # 校验已注册的工具存在
    tool_names = [t["name"] for t in tools_response["result"]["tools"]]
    assert "query_knowledge_hub" in tool_names
    assert "list_collections" in tool_names


# =============================================================================
# 多模态响应测试（E6）
# =============================================================================


@pytest.mark.integration
@pytest.mark.image
def test_multimodal_assembler_image_content_structure() -> None:
    """测试 MultimodalAssembler 生成正确的 MCP ImageContent 结构。

    校验:
    - ImageContent 块的 type="image"
    - mimeType 设置正确（例如 "image/png"）
    - data 字段包含有效的 base64 字符串
    """
    import base64
    import tempfile
    from pathlib import Path
    
    from mcp import types
    
    from src.core.response.multimodal_assembler import MultimodalAssembler
    from src.core.types import RetrievalResult
    
    # 创建测试图像（最小有效 PNG）
    png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test_img.png"
        img_path.write_bytes(png_data)
        
        assembler = MultimodalAssembler()
        
        result = RetrievalResult(
            chunk_id="test_chunk",
            score=0.95,
            text="Content with [IMAGE: test_img]",
            metadata={
                "source_path": "test.pdf",
                "images": [{"id": "test_img", "path": str(img_path)}],
            },
        )
        
        blocks = assembler.assemble_for_result(result)
        
        # 查找 ImageContent 块
        image_blocks = [b for b in blocks if isinstance(b, types.ImageContent)]
        assert len(image_blocks) >= 1, "Should produce at least one ImageContent block"
        
        img_block = image_blocks[0]
        assert img_block.type == "image", "ImageContent type should be 'image'"
        assert img_block.mime_type == "image/png", "MIME type should be 'image/png'"
        assert img_block.data, "data should not be empty"
        
        # 校验 base64 有效
        decoded = base64.b64decode(img_block.data)
        assert decoded.startswith(b"\x89PNG"), "Decoded data should be valid PNG"


@pytest.mark.integration
@pytest.mark.image
def test_response_builder_multimodal_integration() -> None:
    """测试 ResponseBuilder 正确集成多模态内容。

    校验:
    - ResponseBuilder 生成带 image_contents 的 MCPToolResponse
    - 存在图像时 to_mcp_content() 返回 ImageContent 块
    - metadata 包含 has_images 与 image_count
    """
    import tempfile
    from pathlib import Path
    
    from mcp import types
    
    from src.core.response.multimodal_assembler import MultimodalAssembler
    from src.core.response.response_builder import ResponseBuilder
    from src.core.types import RetrievalResult
    
    # 创建测试图像
    png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "builder_test_img.png"
        img_path.write_bytes(png_data)
        
        assembler = MultimodalAssembler()
        builder = ResponseBuilder(
            multimodal_assembler=assembler,
            enable_multimodal=True,
        )
        
        results = [
            RetrievalResult(
                chunk_id="chunk_001",
                score=0.92,
                text="Result with image content",
                metadata={
                    "source_path": "docs/guide.pdf",
                    "images": [{"id": "builder_test_img", "path": str(img_path)}],
                },
            ),
        ]
        
        response = builder.build(results, query="test query")
        
        # 检查 MCPToolResponse 是否包含图像
        assert response.has_images, "Response should have images"
        assert len(response.image_contents) >= 1, "Should have at least one ImageContent"

        # 检查 metadata
        assert response.metadata.get("has_images") is True
        assert response.metadata.get("image_count", 0) >= 1

        # 检查 to_mcp_content() 输出
        mcp_blocks = response.to_mcp_content()
        image_blocks = [b for b in mcp_blocks if isinstance(b, types.ImageContent)]
        assert len(image_blocks) >= 1, "MCP content should include ImageContent blocks"


@pytest.mark.integration
@pytest.mark.image
def test_mcp_tool_response_image_content_format() -> None:
    """测试 MCPToolResponse.to_mcp_content() 返回正确的图像格式。

    校验 MCP 协议期望的精确结构:
    - ImageContent 块的 type、mimeType、data 字段正确
    - 多种内容类型（文本 + 图像）可以共存
    """
    from mcp import types
    
    from src.core.response.citation_generator import Citation
    from src.core.response.response_builder import MCPToolResponse
    
    # 创建带 mock 图像内容的响应
    mock_image = types.ImageContent(
        type="image",
        data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        mimeType="image/png",
    )
    
    response = MCPToolResponse(
        content="# Test Result\n\nContent with image reference.",
        citations=[
            Citation(
                index=1,
                chunk_id="chunk_001",
                source="test.pdf",
                page=1,
                score=0.95,
                text_snippet="Sample text snippet from the document...",
            )
        ],
        metadata={"query": "test", "result_count": 1},
        image_contents=[mock_image],
    )
    
    # 获取 MCP 内容块
    blocks = response.to_mcp_content()

    # 应同时包含文本块和图像块
    text_blocks = [b for b in blocks if isinstance(b, types.TextContent)]
    image_blocks = [b for b in blocks if isinstance(b, types.ImageContent)]

    assert len(text_blocks) >= 1, "Should have at least one TextContent"
    assert len(image_blocks) == 1, "Should have exactly one ImageContent"

    # 校验图像块结构
    img = image_blocks[0]
    assert img.type == "image"
    assert img.mime_type == "image/png"
    assert img.data.startswith("iVBORw0KGgo")  # Base64 PNG 文件头


@pytest.mark.integration
@pytest.mark.image
def test_multimodal_mime_type_detection() -> None:
    """测试不同图像格式的 MIME 类型检测。

    校验:
    - PNG 文件返回 image/png
    - JPEG 文件返回 image/jpeg
    - 扩展名与魔数字节均可作为检测依据
    """
    import tempfile
    from pathlib import Path
    
    from src.core.response.multimodal_assembler import MultimodalAssembler
    
    assembler = MultimodalAssembler()
    
    test_cases = [
        # (文件名, 数据字节, 预期 MIME)
        ("test.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "image/png"),
        ("test.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 50, "image/jpeg"),
        ("test.gif", b"GIF89a" + b"\x00" * 50, "image/gif"),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, data, expected_mime in test_cases:
            img_path = Path(tmpdir) / filename
            img_path.write_bytes(data)
            
            content = assembler.load_image(str(img_path))
            
            assert content is not None, f"Failed to load {filename}"
            assert content.mime_type == expected_mime, (
                f"MIME type mismatch for {filename}: "
                f"expected {expected_mime}, got {content.mime_type}"
            )