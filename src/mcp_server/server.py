"""使用官方 MCP SDK 的 MCP 服务器入口。

该模块使用官方 Python MCP SDK 实现 MCP 服务器，
采用 stdio 传输。它确保 stdout 只包含协议消息，
而所有日志都输出到 stderr。
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from src.mcp_server.protocol_handler import create_mcp_server
from src.observability.logger import get_logger

if TYPE_CHECKING:
    pass


SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"


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
        import src.core.query_engine.query_processor  # noqa: F401
        import src.core.query_engine.hybrid_search  # noqa: F401
        import src.core.query_engine.dense_retriever  # noqa: F401
        import src.core.query_engine.sparse_retriever  # noqa: F401
        import src.core.query_engine.reranker  # noqa: F401
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


def main() -> int:
    """stdio MCP 服务器的入口点。"""
    return run_stdio_server()


if __name__ == "__main__":
    sys.exit(main())