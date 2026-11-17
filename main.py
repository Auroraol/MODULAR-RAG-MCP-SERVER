"""
Modular RAG MCP Server - 主入口

MCP Server 的入口点。负责初始化配置、设置日志并启动服务器。
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
    print("Modular RAG MCP Server - Starting...")

    settings_path = Path("config/settings.yaml")
    try:
        settings = load_settings(settings_path)
    except SettingsError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    logger = get_logger(log_level=settings.observability.log_level)
    logger.info("Settings loaded successfully.")
    logger.info("MCP Server will be implemented in Phase E.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
