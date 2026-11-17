"""pytest 配置与共享 fixture。

本模块包含所有测试模块共用的 pytest 配置和 fixture。
"""

import sys
from pathlib import Path

import pytest

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    """返回项目根目录路径。

    返回:
        项目根目录的路径。
    """
    return PROJECT_ROOT


@pytest.fixture
def sample_documents_dir(project_root: Path) -> Path:
    """返回示例文档目录路径。

    参数:
        project_root: 项目根目录路径。

    返回:
        示例文档目录的路径。
    """
    return project_root / "tests" / "fixtures" / "sample_documents"


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """返回配置目录路径。

    参数:
        project_root: 项目根目录路径。

    返回:
        配置目录的路径。
    """
    return project_root / "config"
