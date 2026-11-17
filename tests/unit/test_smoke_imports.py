"""包导入的冒烟测试。

本模块验证所有关键包都能成功导入。
它作为项目结构的基本健康检查。
"""

import pytest


@pytest.mark.unit
class TestSmokeImports:
    """冒烟测试，验证所有关键包均可导入。"""

    def test_import_src_package(self) -> None:
        """测试能够导入 src 包。"""
        import src
        assert src is not None

    def test_import_mcp_server(self) -> None:
        """测试能够导入 mcp_server 包。"""
        from src import mcp_server
        assert mcp_server is not None

    def test_import_mcp_server_tools(self) -> None:
        """测试能够导入 mcp_server.tools 子包。"""
        from src.mcp_server import tools
        assert tools is not None

    def test_import_core(self) -> None:
        """测试能够导入 core 包。"""
        from src import core
        assert core is not None

    def test_import_core_query_engine(self) -> None:
        """测试能够导入 core.query_engine 子包。"""
        from src.core import query_engine
        assert query_engine is not None

    def test_import_core_response(self) -> None:
        """测试能够导入 core.response 子包。"""
        from src.core import response
        assert response is not None

    def test_import_core_trace(self) -> None:
        """测试能够导入 core.trace 子包。"""
        from src.core import trace
        assert trace is not None

    def test_import_ingestion(self) -> None:
        """测试能够导入 ingestion 包。"""
        from src import ingestion
        assert ingestion is not None

    def test_import_ingestion_embedding(self) -> None:
        """测试能够导入 ingestion.embedding 子包。"""
        from src.ingestion import embedding
        assert embedding is not None

    def test_import_ingestion_storage(self) -> None:
        """测试能够导入 ingestion.storage 子包。"""
        from src.ingestion import storage
        assert storage is not None

    def test_import_ingestion_transform(self) -> None:
        """测试能够导入 ingestion.transform 子包。"""
        from src.ingestion import transform
        assert transform is not None

    def test_import_libs(self) -> None:
        """测试能够导入 libs 包。"""
        from src import libs
        assert libs is not None

    def test_import_libs_embedding(self) -> None:
        """测试能够导入 libs.embedding 子包。"""
        from src.libs import embedding
        assert embedding is not None

    def test_import_libs_evaluator(self) -> None:
        """测试能够导入 libs.evaluator 子包。"""
        from src.libs import evaluator
        assert evaluator is not None

    def test_import_libs_llm(self) -> None:
        """测试能够导入 libs.llm 子包。"""
        from src.libs import llm
        assert llm is not None

    def test_import_libs_loader(self) -> None:
        """测试能够导入 libs.loader 子包。"""
        from src.libs import loader
        assert loader is not None

    def test_import_libs_reranker(self) -> None:
        """测试能够导入 libs.reranker 子包。"""
        from src.libs import reranker
        assert reranker is not None

    def test_import_libs_splitter(self) -> None:
        """测试能够导入 libs.splitter 子包。"""
        from src.libs import splitter
        assert splitter is not None

    def test_import_libs_vector_store(self) -> None:
        """测试能够导入 libs.vector_store 子包。"""
        from src.libs import vector_store
        assert vector_store is not None

    def test_import_observability(self) -> None:
        """测试能够导入 observability 包。"""
        from src import observability
        assert observability is not None

    def test_import_observability_dashboard(self) -> None:
        """测试能够导入 observability.dashboard 子包。"""
        from src.observability import dashboard
        assert dashboard is not None

    def test_import_observability_evaluation(self) -> None:
        """测试能够导入 observability.evaluation 子包。"""
        from src.observability import evaluation
        assert evaluation is not None
