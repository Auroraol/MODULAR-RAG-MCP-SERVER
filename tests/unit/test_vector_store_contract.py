"""VectorStore 工厂与 BaseVectorStore 的单元测试。

测试覆盖：
- 工厂模式：provider 注册、创建与路由
- 配置驱动的实例化
- 未知/缺失 provider 的错误处理
- BaseVectorStore 中的校验逻辑
- 接口一致性的契约测试
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeVectorStore(BaseVectorStore):
    """用于测试的假 Vector Store provider。

    维护一个内存字典来模拟 upsert/query 操作。
    """

    def __init__(self, settings: Any = None, **kwargs: Any):
        """初始化假 Vector Store。

        参数:
            settings: 可选的配置（假实现中未使用）。
            **kwargs: 额外参数（未使用）。
        """
        self.settings = settings
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.upsert_count = 0
        self.query_count = 0
    
    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """将记录存入内存。"""
        self.validate_records(records)
        self.upsert_count += 1
        
        for record in records:
            self.storage[record['id']] = record
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """返回已存储的记录，并附带假相似度分数。"""
        self.validate_query_vector(vector, top_k)
        self.query_count += 1
        
        # 简单假实现：返回所有记录，分数递减
        results = []
        for i, (record_id, record) in enumerate(self.storage.items()):
            if i >= top_k:
                break
            
            # 如果提供了元数据过滤条件则应用
            if filters:
                metadata = record.get('metadata', {})
                if not all(metadata.get(k) == v for k, v in filters.items()):
                    continue
            
            results.append({
                'id': record_id,
                'score': 1.0 - (i * 0.1),  # 假分数递减
                'metadata': record.get('metadata', {}),
                'vector': record.get('vector'),
            })
        
        return results


class TestBaseVectorStore:
    """BaseVectorStore 抽象类的测试。"""

    def test_validate_records_success(self):
        """合法记录应通过校验。"""
        store = FakeVectorStore()
        records = [
            {'id': 'doc1', 'vector': [0.1, 0.2, 0.3]},
            {'id': 'doc2', 'vector': [0.4, 0.5, 0.6], 'metadata': {'source': 'test.pdf'}},
        ]
        # 不应抛出异常
        store.validate_records(records)
    
    def test_validate_records_empty_list(self):
        """空的记录列表应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="cannot be empty"):
            store.validate_records([])
    
    def test_validate_records_non_dict(self):
        """非字典类型的记录应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="not a dict"):
            store.validate_records([{'id': 'doc1', 'vector': [0.1]}, "invalid"])  # type: ignore
    
    def test_validate_records_missing_id(self):
        """缺少 'id' 字段的记录应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="missing required field: 'id'"):
            store.validate_records([{'vector': [0.1, 0.2]}])
    
    def test_validate_records_missing_vector(self):
        """缺少 'vector' 字段的记录应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="missing required field: 'vector'"):
            store.validate_records([{'id': 'doc1'}])
    
    def test_validate_records_invalid_vector_type(self):
        """类型错误的向量应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="invalid vector type"):
            store.validate_records([{'id': 'doc1', 'vector': "not a list"}])  # type: ignore
    
    def test_validate_records_empty_vector(self):
        """空向量应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="empty vector"):
            store.validate_records([{'id': 'doc1', 'vector': []}])
    
    def test_validate_query_vector_success(self):
        """合法的查询向量应通过校验。"""
        store = FakeVectorStore()
        # 不应抛出异常
        store.validate_query_vector([0.1, 0.2, 0.3], top_k=10)
    
    def test_validate_query_vector_invalid_type(self):
        """类型错误的查询向量应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="must be a list or tuple"):
            store.validate_query_vector("not a list", top_k=10)  # type: ignore
    
    def test_validate_query_vector_empty(self):
        """空的查询向量应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="cannot be empty"):
            store.validate_query_vector([], top_k=10)
    
    def test_validate_query_vector_invalid_top_k(self):
        """非法的 top_k 应抛出 ValueError。"""
        store = FakeVectorStore()
        with pytest.raises(ValueError, match="must be a positive integer"):
            store.validate_query_vector([0.1, 0.2], top_k=0)
        
        with pytest.raises(ValueError, match="must be a positive integer"):
            store.validate_query_vector([0.1, 0.2], top_k=-5)
    
    def test_delete_not_implemented(self):
        """默认情况下 delete() 应抛出 NotImplementedError。"""
        store = FakeVectorStore()
        with pytest.raises(NotImplementedError, match="does not implement delete"):
            store.delete(['doc1'])
    
    def test_clear_not_implemented(self):
        """默认情况下 clear() 应抛出 NotImplementedError。"""
        store = FakeVectorStore()
        with pytest.raises(NotImplementedError, match="does not implement clear"):
            store.clear()


class TestFakeVectorStore:
    """FakeVectorStore 实现的测试。"""

    def test_upsert_single_record(self):
        """Upsert 单条记录应将其存储。"""
        store = FakeVectorStore()
        records = [{'id': 'doc1', 'vector': [0.1, 0.2, 0.3]}]
        store.upsert(records)
        
        assert store.upsert_count == 1
        assert 'doc1' in store.storage
        assert store.storage['doc1']['vector'] == [0.1, 0.2, 0.3]
    
    def test_upsert_multiple_records(self):
        """Upsert 多条记录应全部存储。"""
        store = FakeVectorStore()
        records = [
            {'id': 'doc1', 'vector': [0.1, 0.2]},
            {'id': 'doc2', 'vector': [0.3, 0.4], 'metadata': {'source': 'test.pdf'}},
        ]
        store.upsert(records)
        
        assert len(store.storage) == 2
        assert 'doc1' in store.storage
        assert 'doc2' in store.storage
        assert store.storage['doc2']['metadata']['source'] == 'test.pdf'
    
    def test_upsert_idempotent(self):
        """多次 upsert 同一条记录应保持幂等。"""
        store = FakeVectorStore()
        records = [{'id': 'doc1', 'vector': [0.1, 0.2]}]
        
        store.upsert(records)
        store.upsert(records)
        
        assert store.upsert_count == 2
        assert len(store.storage) == 1  # 仍然只有一条记录
    
    def test_upsert_validates_input(self):
        """upsert() 应校验记录，并在输入非法时抛出异常。"""
        store = FakeVectorStore()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            store.upsert([])
        
        with pytest.raises(ValueError, match="missing required field: 'id'"):
            store.upsert([{'vector': [0.1]}])
    
    def test_query_returns_results(self):
        """query() 应返回带分数的已存储记录。"""
        store = FakeVectorStore()
        store.upsert([
            {'id': 'doc1', 'vector': [0.1, 0.2]},
            {'id': 'doc2', 'vector': [0.3, 0.4]},
        ])
        
        results = store.query(vector=[0.5, 0.6], top_k=10)
        
        assert len(results) == 2
        assert results[0]['id'] == 'doc1'
        assert results[0]['score'] == 1.0
        assert results[1]['id'] == 'doc2'
        assert results[1]['score'] == 0.9
    
    def test_query_respects_top_k(self):
        """query() 应将结果数量限制为 top_k。"""
        store = FakeVectorStore()
        store.upsert([
            {'id': f'doc{i}', 'vector': [float(i), float(i)]}
            for i in range(10)
        ])
        
        results = store.query(vector=[0.0, 0.0], top_k=3)
        
        assert len(results) == 3
    
    def test_query_with_filters(self):
        """query() 应应用元数据过滤条件。"""
        store = FakeVectorStore()
        store.upsert([
            {'id': 'doc1', 'vector': [0.1], 'metadata': {'source': 'a.pdf'}},
            {'id': 'doc2', 'vector': [0.2], 'metadata': {'source': 'b.pdf'}},
            {'id': 'doc3', 'vector': [0.3], 'metadata': {'source': 'a.pdf'}},
        ])
        
        results = store.query(vector=[0.0], top_k=10, filters={'source': 'a.pdf'})
        
        # 应只返回 doc1 和 doc3
        result_ids = [r['id'] for r in results]
        assert 'doc1' in result_ids
        assert 'doc3' in result_ids
        assert 'doc2' not in result_ids
    
    def test_query_increments_count(self):
        """每次查询都应递增计数器。"""
        store = FakeVectorStore()
        assert store.query_count == 0
        
        store.query(vector=[0.1], top_k=5)
        assert store.query_count == 1
        
        store.query(vector=[0.2], top_k=5)
        assert store.query_count == 2
    
    def test_query_validates_input(self):
        """query() 应校验输入，并在参数非法时抛出异常。"""
        store = FakeVectorStore()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            store.query(vector=[], top_k=5)
        
        with pytest.raises(ValueError, match="must be a positive integer"):
            store.query(vector=[0.1], top_k=0)


class TestVectorStoreFactory:
    """VectorStoreFactory 的测试。"""

    def setup_method(self):
        """每个测试前重置工厂注册表。"""
        VectorStoreFactory._PROVIDERS.clear()
    
    def test_register_provider_success(self):
        """注册合法的 provider 应将其加入注册表。"""
        VectorStoreFactory.register_provider('fake', FakeVectorStore)
        
        assert 'fake' in VectorStoreFactory._PROVIDERS
        assert VectorStoreFactory._PROVIDERS['fake'] is FakeVectorStore
    
    def test_register_provider_case_insensitive(self):
        """provider 名称应大小写不敏感。"""
        VectorStoreFactory.register_provider('FakeStore', FakeVectorStore)
        
        assert 'fakestore' in VectorStoreFactory._PROVIDERS
    
    def test_register_provider_invalid_class(self):
        """注册非 BaseVectorStore 子类应抛出 ValueError。"""
        
        class NotAVectorStore:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseVectorStore"):
            VectorStoreFactory.register_provider('invalid', NotAVectorStore)  # type: ignore
    
    def test_create_with_registered_provider(self):
        """使用已注册的 provider 创建实例应正常工作。"""
        VectorStoreFactory.register_provider('fake', FakeVectorStore)

        # Mock 配置
        settings = MagicMock()
        settings.vector_store.provider = 'fake'
        
        store = VectorStoreFactory.create(settings)
        
        assert isinstance(store, FakeVectorStore)
        assert store.settings is settings
    
    def test_create_case_insensitive_provider(self):
        """provider 查找应大小写不敏感。"""
        VectorStoreFactory.register_provider('fake', FakeVectorStore)
        
        settings = MagicMock()
        settings.vector_store.provider = 'FAKE'
        
        store = VectorStoreFactory.create(settings)
        
        assert isinstance(store, FakeVectorStore)
    
    def test_create_unknown_provider(self):
        """使用未知 provider 创建实例应抛出 ValueError。"""
        settings = MagicMock()
        settings.vector_store.provider = 'nonexistent'
        
        with pytest.raises(ValueError, match="Unsupported VectorStore provider: 'nonexistent'"):
            VectorStoreFactory.create(settings)
    
    def test_create_unknown_provider_shows_available(self):
        """错误信息应列出可用的 provider。"""
        VectorStoreFactory.register_provider('chroma', FakeVectorStore)
        VectorStoreFactory.register_provider('qdrant', FakeVectorStore)
        
        settings = MagicMock()
        settings.vector_store.provider = 'unknown'
        
        with pytest.raises(ValueError, match="Available providers: chroma, qdrant"):
            VectorStoreFactory.create(settings)
    
    def test_create_missing_provider_field(self):
        """缺少 provider 字段应抛出带有明确提示的 ValueError。"""
        settings = MagicMock()
        del settings.vector_store  # 模拟配置段缺失
        
        with pytest.raises(ValueError) as exc_info:
            VectorStoreFactory.create(settings)
        
        error_message = str(exc_info.value)
        assert "Missing required configuration" in error_message
        assert "settings.vector_store.provider" in error_message
        assert "settings.yaml" in error_message
    
    def test_create_with_override_kwargs(self):
        """工厂应将覆盖参数传递给 provider 构造函数。"""
        
        class ConfigurableVectorStore(BaseVectorStore):
            def __init__(self, settings: Any, custom_param: str = "default", **kwargs: Any):
                self.settings = settings
                self.custom_param = custom_param
            
            def upsert(self, records: List[Dict[str, Any]], trace: Optional[Any] = None, **kwargs: Any) -> None:
                pass
            
            def query(self, vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None, trace: Optional[Any] = None, **kwargs: Any) -> List[Dict[str, Any]]:
                return []
        
        VectorStoreFactory.register_provider('configurable', ConfigurableVectorStore)
        
        settings = MagicMock()
        settings.vector_store.provider = 'configurable'
        
        store = VectorStoreFactory.create(settings, custom_param="overridden")
        
        assert isinstance(store, ConfigurableVectorStore)
        assert store.custom_param == "overridden"
    
    def test_create_provider_instantiation_error(self):
        """provider 实例化过程中的错误应被包装并附带上下文。"""
        
        class FailingVectorStore(BaseVectorStore):
            def __init__(self, settings: Any, **kwargs: Any):
                raise RuntimeError("Simulated initialization failure")
            
            def upsert(self, records: List[Dict[str, Any]], trace: Optional[Any] = None, **kwargs: Any) -> None:
                pass
            
            def query(self, vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None, trace: Optional[Any] = None, **kwargs: Any) -> List[Dict[str, Any]]:
                return []
        
        VectorStoreFactory.register_provider('failing', FailingVectorStore)
        
        settings = MagicMock()
        settings.vector_store.provider = 'failing'
        
        with pytest.raises(RuntimeError, match="Failed to instantiate VectorStore provider 'failing'"):
            VectorStoreFactory.create(settings)
    
    def test_list_providers_empty(self):
        """未注册任何 provider 时 list_providers() 应返回空列表。"""
        assert VectorStoreFactory.list_providers() == []
    
    def test_list_providers_with_providers(self):
        """list_providers() 应返回 provider 名称的排序列表。"""
        VectorStoreFactory.register_provider('chroma', FakeVectorStore)
        VectorStoreFactory.register_provider('qdrant', FakeVectorStore)
        VectorStoreFactory.register_provider('milvus', FakeVectorStore)
        
        providers = VectorStoreFactory.list_providers()
        
        assert providers == ['chroma', 'milvus', 'qdrant']  # 按字母顺序排序


# ── 边界 / 契约测试（I4） ──────────────────────────────────────────

class TestBaseVectorStoreContractBoundary:
    """BaseVectorStore 契约的边界测试。"""

    def test_get_by_ids_not_implemented(self):
        """默认情况下 get_by_ids() 应抛出 NotImplementedError。"""
        store = FakeVectorStore()
        with pytest.raises(NotImplementedError, match="does not implement get_by_ids"):
            store.get_by_ids(['doc1'])

    def test_validate_records_vector_non_numeric_accepted(self):
        """含非数值元素的向量在基类层面是被接受的。

        注意：基类只检查类型（list/tuple）与非空。
        对单个元素的类型约束由各 provider 自行实现。
        """
        store = FakeVectorStore()
        # 基类校验不会拒绝非数值元素
        store.validate_records([{'id': 'doc1', 'vector': ['a', 'b']}])

    def test_validate_records_single_element_vector(self):
        """单元素向量应合法。"""
        store = FakeVectorStore()
        store.validate_records([{'id': 'doc1', 'vector': [0.5]}])

    def test_query_top_k_one(self):
        """top_k=1 时最多返回 1 条结果。"""
        store = FakeVectorStore()
        store.upsert([
            {'id': 'a', 'vector': [0.1]},
            {'id': 'b', 'vector': [0.2]},
        ])
        results = store.query([0.1], top_k=1)
        assert len(results) <= 1

    def test_query_empty_store(self):
        """查询空存储应返回空列表。"""
        store = FakeVectorStore()
        results = store.query([0.1, 0.2], top_k=5)
        assert results == []

    def test_upsert_preserves_metadata(self):
        """元数据应在 upsert → query 的完整流程中被保留。"""
        store = FakeVectorStore()
        store.upsert([{
            'id': 'doc1',
            'vector': [0.1],
            'metadata': {'source': 'test.pdf', 'page': 3},
        }])
        results = store.query([0.1], top_k=1)
        assert results[0]['metadata']['source'] == 'test.pdf'
        assert results[0]['metadata']['page'] == 3

    def test_query_filters_empty_dict(self):
        """空的过滤字典应匹配所有记录。"""
        store = FakeVectorStore()
        store.upsert([{'id': 'a', 'vector': [0.1], 'metadata': {'k': 'v'}}])
        results = store.query([0.1], top_k=10, filters={})
        assert len(results) == 1

    def test_delete_requires_list(self):
        """delete() 接收 ID 列表（契约形状检查）。"""
        store = FakeVectorStore()
        with pytest.raises(NotImplementedError):
            store.delete(['id1', 'id2'])
