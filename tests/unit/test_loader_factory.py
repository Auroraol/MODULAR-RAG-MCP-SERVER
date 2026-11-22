"""LoaderFactory 扩展名分发的单元测试。"""

import pytest

from src.libs.loader.loader_factory import LoaderFactory


@pytest.fixture()
def factory(tmp_path):
    return LoaderFactory(image_storage_dir=tmp_path / "images")


def test_text_formats(factory, tmp_path):
    md = tmp_path / "a.md"
    md.write_text("# 标题\n\n正文", encoding="utf-8")
    doc = factory.load(md)
    assert doc.metadata["doc_type"] == "md"
    assert "正文" in doc.text

    csv = tmp_path / "b.csv"
    csv.write_text("a,b\n1,2", encoding="utf-8")
    assert factory.load(csv).metadata["doc_type"] == "csv"


def test_gbk_fallback(factory, tmp_path):
    txt = tmp_path / "gbk.txt"
    txt.write_bytes("中文测试".encode("gb18030"))
    assert factory.load(txt).text == "中文测试"


def test_unsupported_extension(factory, tmp_path):
    f = tmp_path / "x.xyz"
    f.write_text("x")
    with pytest.raises(ValueError, match="Unsupported file type"):
        factory.load(f)


def test_empty_text_rejected(factory, tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n")
    with pytest.raises(ValueError, match="No text"):
        factory.load(f)