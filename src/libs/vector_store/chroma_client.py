"""ChromaDB PersistentClient 的线程安全创建工具。

背景（chromadb 1.x）：
    ``chromadb.PersistentClient(path=...)`` 内部要走 Rust bindings 的
    初始化流程，**同一个进程内并发创建多个指向同一路径的客户端**
    会触发底层竞态，表现为以下两种随机错误之一：

    - ``ValueError: Could not connect to tenant default_tenant.``
    - ``AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'``

    在 MCP Server 场景下很容易触发：同一会话里并发调用
    ``list_collections`` / ``query_knowledge_hub`` / ``get_document_summary``，
    每个工具都会在 ``asyncio.to_thread`` 里各自创建一个客户端。

解决：
    用进程级互斥锁把「创建客户端」这一步串行化。创建完成后客户端本身
    可以并发使用（chromadb 的读写接口是线程安全的），因此只需锁住
    构造过程，不会成为性能瓶颈。
"""

from __future__ import annotations

import threading
from typing import Any

# 进程级创建锁：串行化 PersistentClient 的构造过程
_CLIENT_CREATION_LOCK = threading.Lock()


def create_persistent_client(path: str, **kwargs: Any) -> Any:
    """线程安全地创建 ChromaDB PersistentClient。

    参数:
        path: 持久化目录。
        **kwargs: 透传给 ``chromadb.PersistentClient``（例如 ``settings``）。

    返回:
        ChromaDB PersistentClient 实例。

    异常:
        ImportError: chromadb 未安装。
    """
    # 在函数内导入，便于测试通过 sys.modules / patch 替换 chromadb
    import chromadb

    with _CLIENT_CREATION_LOCK:
        return chromadb.PersistentClient(path=path, **kwargs)
