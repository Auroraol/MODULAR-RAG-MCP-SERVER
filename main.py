"""
Modular RAG MCP Server - 主入口

MCP Server 的入口点。负责初始化配置、设置日志并启动服务器。

Transport 由命令行参数决定（默认 stdio）::

    python main.py                          # stdio（供 MCP 客户端拉起）
    python main.py --transport http         # http://127.0.0.1:8000/mcp
    python main.py --transport http --port 9000

注意: 启动前请确保已填写 ``config/settings.yaml`` 中的 API Key
（或设置 ``DEEPSEEK_API_KEY`` / ``OPENAI_API_KEY`` 环境变量），
否则摄取与查询会返回 401。
"""

import sys
from pathlib import Path

from src.core.settings import SettingsError, load_settings
from src.observability.logger import get_logger


def main() -> int:
    """
    MCP Server 的主入口点。

    返回:
        int: 退出码（成功为 0，失败为非零）
    """
    settings_path = Path(__file__).resolve().parent / "config" / "settings.yaml"
    try:
        settings = load_settings(settings_path)
    except SettingsError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    logger = get_logger(log_level=settings.observability.log_level)
    logger.info("Settings loaded: llm=%s embedding=%s", settings.llm.provider, settings.embedding.provider)

    # 委托给真正的 MCP 服务器实现（stdio 或 http）
    from src.mcp_server.server import main as server_main

    return server_main()


if __name__ == "__main__":
    sys.exit(main())
