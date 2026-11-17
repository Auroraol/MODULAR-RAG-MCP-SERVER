"""可观测性日志工具。

提供：
- ``get_logger``：标准的人类可读 logger（与 C 阶段一致）。
- ``JSONFormatter``：输出 JSON 的自定义 :class:`logging.Formatter`。
- ``get_trace_logger``：返回由 JSON Lines 文件处理器支撑的 logger。
- ``write_trace``：将 trace 字典追加到 ``logs/traces.jsonl`` 的便捷函数。
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.settings import resolve_path

# traces 文件的默认路径（绝对路径，与 CWD 无关）
_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


# ── 人类可读 logger（现有实现） ────────────────────────────────────


def get_logger(name: str = "modular-rag", log_level: Optional[str] = None) -> logging.Logger:
    """获取配置好的 logger。

    参数:
        name: Logger 名称。
        log_level: 可选的日志级别字符串（例如 "INFO"）。

    返回:
        配置好的 logger 实例。
    """

    if log_level:
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    # 抑制 httpx 日志（其中包含敏感的 endpoint URL）
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logging.getLogger(name)


# ── JSON Lines 格式化器 ────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """每行输出一个 JSON 对象的日志格式化器。

    每条日志记录被序列化为一个 dict，至少包含：
    ``timestamp``、``level``、``logger``、``message``。如果记录带有
    ``exc_info`` 元组，则 traceback 会作为 ``exception`` 字段包含在内。

    通过 logger 调用的 *extra=* 附加的额外属性会合并到顶层 dict 中
    （Python 内部字段除外）。
    """

    _INTERNAL_ATTRS = frozenset({
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module",
        "msecs", "message", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        """将日志记录返回为单行 JSON 字符串。"""
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 合并调用方附加的额外字段
        for key, val in record.__dict__.items():
            if key not in self._INTERNAL_ATTRS and key not in payload:
                try:
                    json.dumps(val)  # 低成本的可序列化性测试
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = str(val)

        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# ── Trace 日志记录器 ────────────────────────────────────────


def get_trace_logger(
    traces_path: str | Path = _DEFAULT_TRACES_PATH,
    *,
    name: str = "modular-rag.trace",
) -> logging.Logger:
    """返回一个将 JSON Lines 写入 *traces_path* 的 logger。

    该 logger 使用 :class:`JSONFormatter` 和配置为追加模式的
    :class:`FileHandler`。使用相同 *name* 的重复调用会返回同一个
    logger（标准 :mod:`logging` 语义）。

    参数:
        traces_path: JSONL 输出的文件路径。父目录会自动创建。
        name: Logger 名称。

    返回:
        一个可用于输出 JSON Lines 的 :class:`logging.Logger`。
    """
    path = Path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复调用时添加重复的 handler
    if not logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False  # 不输出到控制台

    return logger


# ── trace 字典的便捷写入器 ──────────────────────────────────────────


def write_trace(
    trace_dict: Dict[str, Any],
    traces_path: str | Path = _DEFAULT_TRACES_PATH,
) -> None:
    """将单个 trace 字典追加为一行 JSON。

    这是一个直接写入的薄封装——不涉及日志框架——因此输出与
    :class:`~src.core.trace.trace_collector.TraceCollector` 产生的
    内容完全一致。

    参数:
        trace_dict: 可 JSON 序列化的字典（通常来自
            ``TraceContext.to_dict()``）。
        traces_path: 输出文件路径；父目录会自动创建。
    """
    path = Path(traces_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(trace_dict, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")