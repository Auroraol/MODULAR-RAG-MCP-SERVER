"""Trace 收集器 – 接收已完成的 TraceContext 并持久化。

收集器是内存中 TraceContext 对象与 Dashboard 使用的磁盘
JSON Lines 日志之间的桥梁。它有意与 logging 模块解耦，
以保证 Trace 持久化的可预期性与可测试性。
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.core.settings import resolve_path
from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# Trace 文件的默认绝对路径（与 CWD 无关）
_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceCollector:
    """收集已完成的 Trace 并将其追加到 JSON Lines 文件。

    参数:
        traces_path: ``traces.jsonl`` 输出的文件路径。
            父目录会自动创建。
    """

    def __init__(self, traces_path: str | Path = _DEFAULT_TRACES_PATH) -> None:
        self._path = Path(traces_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def collect(self, trace: TraceContext) -> None:
        """将单条 Trace 持久化为一行 JSON。

        若 Trace 尚未结束，会自动调用 ``finish()``，
        以保证输出始终包含耗时数据。

        参数:
            trace: 已填充的 :class:`TraceContext`。
        """
        if trace.finished_at is None:
            trace.finish()

        line = json.dumps(trace.to_dict(), ensure_ascii=False)
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            logger.exception("Failed to write trace %s", trace.trace_id)

    @property
    def path(self) -> Path:
        """返回 Trace 文件解析后的路径。"""
        return self._path
