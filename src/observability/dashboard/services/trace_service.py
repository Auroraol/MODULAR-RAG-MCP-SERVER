"""TraceService – 从 logs/traces.jsonl 读取并解析 trace。

在原始 JSONL trace 日志之上提供带类型、可过滤的接口。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path

logger = logging.getLogger(__name__)

# traces 文件的默认路径（绝对路径，与 CWD 无关）
DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceService:
    """用于查询已记录 trace 的只读服务。

    参数:
        traces_path: JSONL 文件路径。默认为
            ``logs/traces.jsonl``。
    """

    def __init__(self, traces_path: Optional[str | Path] = None) -> None:
        self.traces_path = Path(traces_path) if traces_path else DEFAULT_TRACES_PATH

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def list_traces(
        self,
        trace_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """按倒序时间返回 trace。

        参数:
            trace_type: 按 ``trace_type`` 字段过滤（例如
                ``"ingestion"`` 或 ``"query"``）。``None`` 表示全部。
            limit: 返回的 trace 最大数量。

        返回:
            trace dict 列表（最新的在前）。
        """
        traces = self._load_all()

        if trace_type:
            traces = [t for t in traces if t.get("trace_type") == trace_type]

        # 最新的在前
        traces.sort(key=lambda t: t.get("started_at", ""), reverse=True)

        return traces[:limit]

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """按 ``trace_id`` 获取单条 trace。

        返回:
            trace dict，未找到则返回 ``None``。
        """
        for t in self._load_all():
            if t.get("trace_id") == trace_id:
                return t
        return None

    def get_stage_timings(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从 trace 中提取各阶段耗时。

        返回:
            包含 stage_name、elapsed_ms、data 键的 dict 列表。
            按出现顺序排列。
        """
        stages = trace.get("stages", [])
        timings: List[Dict[str, Any]] = []
        for s in stages:
            # 原始阶段 dict 包含：stage、timestamp、data（dict）、elapsed_ms
            # 直接提取内层的 'data' dict，而不是将其展平
            stage_data = s.get("data", {})
            if not isinstance(stage_data, dict):
                stage_data = {}
            timings.append(
                {
                    "stage_name": s.get("stage"),
                    "elapsed_ms": s.get("elapsed_ms", 0),
                    "data": stage_data,
                }
            )
        return timings

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _load_all(self) -> List[Dict[str, Any]]:
        """解析 JSONL 文件中的每一行。

        静默跳过格式错误的行。
        """
        if not self.traces_path.exists():
            return []

        traces: List[Dict[str, Any]] = []
        with self.traces_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed trace line: %s", line[:80])
        return traces
