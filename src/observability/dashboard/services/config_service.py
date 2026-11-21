"""Dashboard 配置读取服务。

封装 :class:`Settings`，为 Overview 页面提供格式化后的
组件信息。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import Settings, load_settings


@dataclass
class ComponentInfo:
    """单个已配置组件的摘要。"""

    name: str
    provider: str
    model: str
    extra: Dict[str, Any]


class ConfigService:
    """暴露应用配置的只读服务。

    参数:
        settings_path: ``settings.yaml`` 的路径。
    """

    def __init__(self, settings_path: Optional[str] = None) -> None:
        self._settings_path = settings_path
        self._settings: Optional[Settings] = None

    # ── 懒加载 ──────────────────────────────────────────────────────

    def _load(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings(self._settings_path)
        return self._settings

    def reload(self) -> None:
        """强制从磁盘重新加载配置。"""
        self._settings = None

    @property
    def settings(self) -> Settings:
        return self._load()

    # ── 组件卡片 ────────────────────────────────────────────────────

    def get_component_cards(self) -> List[ComponentInfo]:
        """返回供 Overview 页面使用的组件摘要列表。"""
        s = self._load()
        cards: List[ComponentInfo] = []

        # LLM
        cards.append(ComponentInfo(
            name="LLM(大模型)",
            provider=s.llm.provider,
            model=s.llm.model,
            extra={"temperature": s.llm.temperature, "max_tokens": s.llm.max_tokens},
        ))

        # Embedding
        cards.append(ComponentInfo(
            name="Embedding(向量化)",
            provider=s.embedding.provider,
            model=s.embedding.model,
            extra={"dimensions": s.embedding.dimensions},
        ))

        # VectorStore
        cards.append(ComponentInfo(
            name="Vector Store(向量数据库)",
            provider=s.vector_store.provider,
            model=s.vector_store.collection_name,
            extra={"persist_directory": s.vector_store.persist_directory},
        ))

        # 检索
        cards.append(ComponentInfo(
            name="Retrieval(检索)",
            provider="hybrid",
            model="dense(稠密检索——语义匹配) + sparse(稀疏检索——关键词匹配) + RRF(Reciprocal Rank Fusion，倒数排名融合——合并两路结果)",
            extra={
                "dense_top_k": s.retrieval.dense_top_k,
                "sparse_top_k": s.retrieval.sparse_top_k,
                "fusion_top_k": s.retrieval.fusion_top_k,
            },
        ))

        # Rerank
        cards.append(ComponentInfo(
            name="Reranker(重排序)",
            provider=s.rerank.provider if s.rerank.enabled else "disabled",
            model=s.rerank.model if s.rerank.enabled else "-",
            extra={"enabled": s.rerank.enabled, "top_k": s.rerank.top_k},
        ))

        # 视觉 LLM
        if s.vision_llm and s.vision_llm.enabled:
            cards.append(ComponentInfo(
                name="Vision LLM(视觉 LLM)",
                provider=s.vision_llm.provider,
                model=s.vision_llm.model,
                extra={"max_image_size": s.vision_llm.max_image_size},
            ))


        # 摄入
        if s.ingestion:
            cards.append(ComponentInfo(
                name="Ingestion(数据摄取+切块)",
                provider=s.ingestion.splitter,
                model="-",
                extra={
                    "chunk_size": s.ingestion.chunk_size,
                    "chunk_overlap": s.ingestion.chunk_overlap,
                    "batch_size": s.ingestion.batch_size,
                },
            ))

        return cards
