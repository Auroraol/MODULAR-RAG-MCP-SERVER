"""使用官方 MCP SDK 的 MCP 服务器入口。

该模块使用官方 Python MCP SDK 实现 MCP 服务器，支持两种传输方式：

- ``stdio``（默认）：供 Claude Code / Cursor / Copilot 等 MCP 客户端
  以子进程方式拉起。stdout 只包含协议消息，所有日志输出到 stderr。
- ``http``：Streamable HTTP 传输，常驻监听端口，适合远程或容器化部署。

用法::

    python -m src.mcp_server.server                     # stdio（默认）
    python -m src.mcp_server.server --transport http    # http://127.0.0.1:8000/mcp
    python -m src.mcp_server.server --transport http --port 9000
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from src.mcp_server.protocol_handler import create_mcp_server
from src.observability.logger import get_logger

SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"

DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000
DEFAULT_HTTP_PATH = "/mcp"


def _redirect_all_loggers_to_stderr() -> None:
    """将所有根 logger 的 handler 重定向到 stderr。

    MCP stdio 传输将 stdout 保留给 JSON-RPC 消息。
    任何输出到 stdout 的日志都会破坏协议流。
    """
    import logging as _logging

    root = _logging.getLogger()
    stderr_handler = _logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        _logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    # 替换任何可能指向 stdout 的现有 stream handler
    for handler in root.handlers[:]:
        if isinstance(handler, _logging.StreamHandler) and not isinstance(
            handler, _logging.FileHandler
        ):
            root.removeHandler(handler)
    root.addHandler(stderr_handler)


def _preload_heavy_imports() -> None:
    """在 **主线程** 中提前导入重量级第三方模块。

    MCP SDK 使用 anyio + 后台线程进行 stdin/stdout I/O。
    当工具处理函数执行 ``asyncio.to_thread(fn)`` 时，*fn* 会在
    新的工作线程中运行。如果它尝试 ``import chromadb``（会间接
    引入 onnxruntime、numpy、sqlite3 C 扩展……），该导入可能与
    stdin 读取线程发生死锁，因为二者都会争抢 Python 的全局
    *import lock*。

    在此处预先导入（早于 anyio 启动其 I/O 线程）即可完全避免
    死锁：后续工作线程中的 ``import`` 语句只需命中 ``sys.modules``
    并立即返回。
    """
    # chromadb 是最重的来源（onnxruntime、numpy……）
    try:
        import chromadb  # noqa: F401
        import chromadb.config  # noqa: F401
    except ImportError:
        pass  # 安装时可选项

    # 工具在 asyncio.to_thread 中延迟导入的内部模块
    try:
        import src.core.query_engine.retriever.dense_retriever  # noqa: F401
        import src.core.query_engine.hybrid_search  # noqa: F401
        import src.core.query_engine.query_processor  # noqa: F401
        import src.core.query_engine.reranker  # noqa: F401
        import src.core.query_engine.retriever.sparse_retriever  # noqa: F401
        import src.ingestion.storage.bm25_indexer  # noqa: F401
        import src.libs.embedding.embedding_factory  # noqa: F401
        import src.libs.vector_store.vector_store_factory  # noqa: F401
    except ImportError:
        pass


async def run_stdio_server_async() -> int:
    """以异步方式通过 stdio 运行 MCP 服务器。

    返回:
        退出码。
    """
    # 在此导入，避免 mcp 未安装时的导入错误
    import mcp.server.stdio

    # 确保所有日志都输出到 stderr（stdout 保留给 JSON-RPC）
    _redirect_all_loggers_to_stderr()

    # 在主线程中预加载重量级依赖，防止之后工具处理函数调用
    # asyncio.to_thread() 时发生 import 锁死锁。
    _preload_heavy_imports()

    logger = get_logger(log_level="INFO")
    logger.info("Starting MCP server (stdio transport) with official SDK.")

    # 创建带协议处理器的服务器
    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)

    # 以 stdio 传输运行
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    logger.info("MCP server shutting down.")
    return 0


def run_stdio_server() -> int:
    """通过 stdio 运行 MCP 服务器（同步包装）。

    返回:
        退出码。
    """
    return asyncio.run(run_stdio_server_async())


def run_http_server(host: str = DEFAULT_HTTP_HOST, port: int = DEFAULT_HTTP_PORT) -> int:
    """通过 Streamable HTTP 传输运行 MCP 服务器（阻塞）。

    参数:
        host: 监听地址。
        port: 监听端口。

    返回:
        退出码。
    """
    import uvicorn

    _redirect_all_loggers_to_stderr()
    _preload_heavy_imports()

    logger = get_logger(log_level="INFO")
    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)
    app = server.streamable_http_app(
        streamable_http_path=DEFAULT_HTTP_PATH,
        host=host,
    )

    logger.info(
        "Starting MCP server (streamable http) at http://%s:%s%s",
        host,
        port,
        DEFAULT_HTTP_PATH,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    参数:
        argv: 参数列表（默认取 ``sys.argv[1:]``）。

    返回:
        解析后的参数命名空间。
    """
    parser = argparse.ArgumentParser(
        description="Modular RAG MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--transport",
        "-t",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport to use (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HTTP_HOST,
        help=f"HTTP bind address, http transport only (default: {DEFAULT_HTTP_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_HTTP_PORT,
        help=f"HTTP port, http transport only (default: {DEFAULT_HTTP_PORT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """MCP 服务器入口点。

    参数:
        argv: 可选参数列表（默认取 ``sys.argv[1:]``）。

    返回:
        退出码。
    """
    args = parse_args(argv)
    if args.transport == "http":
        return run_http_server(host=args.host, port=args.port)
    return run_stdio_server()


if __name__ == "__main__":
    sys.exit(main())
