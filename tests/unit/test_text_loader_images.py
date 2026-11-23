"""TextLoader 本地图片提取测试（Markdown）。"""

import base64
from pathlib import Path

from src.libs.loader.text_loader import TextLoader

PNG = b"\x89PNG fake bytes for tests"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_markdown_local_images_become_placeholders(tmp_path):
    img = tmp_path / "pics" / "a.png"
    img.parent.mkdir()
    img.write_bytes(PNG)
    data_uri = "data:image/png;base64," + base64.b64encode(PNG).decode()

    md = _write(
        tmp_path / "doc.md",
        "# Title\n\n"
        "![local](pics/a.png)\n"
        "![missing](pics/nope.png)\n"
        "![remote](https://example.com/x.png)\n"
        f"![inline]({data_uri})\n"
        '<img src="pics/a.png" alt="html">\n',
    )

    doc = TextLoader(image_storage_dir=tmp_path / "images").load(md)

    images = doc.metadata["images"]
    assert len(images) == 3  # local + data uri + html local
    for image in images:
        assert f"[IMAGE: {image['id']}]" in doc.text
        assert Path(image["path"]).read_bytes() == PNG

    # 远程与缺失文件保持原样
    assert "![remote](https://example.com/x.png)" in doc.text
    assert "![missing](pics/nope.png)" in doc.text
    # 本地文件的 id 按出现顺序编号
    assert [i["id"].rsplit("_", 1)[1] for i in images] == ["1", "2", "3"]


def test_markdown_images_disabled_keeps_text(tmp_path):
    md = _write(tmp_path / "doc.md", "# Title\n\n![local](pics/a.png)\n")

    doc = TextLoader(extract_images=False, image_storage_dir=tmp_path / "images").load(md)

    assert "images" not in doc.metadata
    assert "![local](pics/a.png)" in doc.text


def test_non_markdown_keeps_image_syntax(tmp_path):
    txt = _write(tmp_path / "doc.txt", "![local](pics/a.png)\n")

    doc = TextLoader(image_storage_dir=tmp_path / "images").load(txt)

    assert "images" not in doc.metadata
    assert "![local](pics/a.png)" in doc.text