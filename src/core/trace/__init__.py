"""
Trace 模块。

本包包含 Trace 相关组件：
- Trace 上下文
- Trace 收集器
"""

from src.core.trace.trace_context import TraceContext
from src.core.trace.trace_collector import TraceCollector

__all__ = ['TraceContext', 'TraceCollector']
