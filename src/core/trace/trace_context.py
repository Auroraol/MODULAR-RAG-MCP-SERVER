"""用于跨 Pipeline 阶段可观测性的 Trace 上下文。

提供 trace_id、trace_type（query/ingestion）、分阶段耗时、
finish() 生命周期，以及用于 JSON Lines 输出的 to_dict() 序列化。
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional


@dataclass
class TraceContext:
    """请求级 Trace 上下文，记录 Pipeline 阶段与耗时。

    属性:
        trace_id: 该 Trace 的唯一标识。
        trace_type: 取值为 ``"query"`` 或 ``"ingestion"``。
        started_at: Trace 创建时的 ISO-8601 时间戳。
        finished_at: 调用 ``finish()`` 时的 ISO-8601 时间戳，或 None。
        stages: 按顺序记录的各阶段字典列表。
        metadata: 附加到 Trace 上的任意键值对。
    """

    trace_type: Literal["query", "ingestion"] = "query"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = field(default=None)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 用于精确计算耗时的内部单调时钟
    # 注意：Windows 上 time.monotonic() 的分辨率约 15.6ms，
    # 短耗时会量化为 0；perf_counter() 分辨率为 100ns，故此处使用它。
    _start_mono: float = field(default_factory=time.perf_counter, repr=False)
    _finish_mono: Optional[float] = field(default=None, repr=False)
    _stage_timings: Dict[str, float] = field(default_factory=dict, repr=False)

    # ---- 记录 ---------------------------------------------------

    def record_stage(
        self,
        stage_name: str,
        data: Dict[str, Any],
        elapsed_ms: Optional[float] = None,
    ) -> None:
        """记录某个 Pipeline 阶段的数据。

        参数:
            stage_name: 阶段名称（例如 ``"dense_retrieval"``）。
            data: 阶段相关的数据负载（method、provider、details 等）。
            elapsed_ms: 预先计算的毫秒耗时。若为 *None*，
                调用方应在外部测量，或交由
                ``stage_timer`` 上下文管理器处理。
        """
        entry: Dict[str, Any] = {
            "stage": stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(elapsed_ms, 2)
            self._stage_timings[stage_name] = elapsed_ms
        self.stages.append(entry)

    # ---- 生命周期 ----------------------------------------------------

    def finish(self) -> None:
        """将 Trace 标记为已结束并记录墙钟结束时间。"""
        self._finish_mono = time.perf_counter()
        self.finished_at = datetime.now(timezone.utc).isoformat()

    # ---- 耗时辅助函数 -----------------------------------------------

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """返回以毫秒计的耗时。

        参数:
            stage_name: 若给定，则返回该阶段记录的耗时。
                若为 *None*，返回 Trace 的总耗时
                （开始 → 结束，若尚未结束则为开始 → 当前）。

        返回:
            毫秒耗时。

        异常:
            KeyError: 提供了 *stage_name* 但未找到时。
        """
        if stage_name is not None:
            if stage_name not in self._stage_timings:
                raise KeyError(f"Stage '{stage_name}' has no recorded timing")
            return self._stage_timings[stage_name]

        end = self._finish_mono if self._finish_mono is not None else time.perf_counter()
        return (end - self._start_mono) * 1000.0

    # ---- 序列化 ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """将 Trace 序列化为适合 ``json.dumps`` 的普通字典。

        返回:
            包含全部 Trace 数据的字典。
        """
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": round(self.elapsed_ms(), 2),
            "stages": list(self.stages),
            "metadata": dict(self.metadata),
        }

    # ---- C5 / C6 使用的向后兼容辅助函数 -----------------------

    def get_stage_data(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """获取指定阶段记录的数据。

        搜索 stages 列表（同名阶段以最后一次写入为准）。

        参数:
            stage_name: 要获取的阶段名称。

        返回:
            匹配阶段的 ``data`` 字典，未找到时返回 *None*。
        """
        for entry in reversed(self.stages):
            if entry.get("stage") == stage_name:
                return entry.get("data")
        return None
