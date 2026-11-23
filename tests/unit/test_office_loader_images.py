"""OfficeLoader 内嵌图片提取测试（docx / pptx）。"""

import zipfile
from pathlib import Path

import pytest

pytest.importorskip("markitdown")
pytest.importorskip("mammoth")
PIL_Image = pytest.importorskip("PIL.Image")

from src.libs.loader.office_loader import OfficeLoader


@pytest.fixture
def png_bytes(tmp_path: Path) -> bytes:
    png = tmp_path / "pic.png"
    PIL_Image.new("RGB", (32, 16), "red").save(png)
    return png.read_bytes()


def _make_docx(path: Path, png: bytes) -> None:
    """手工构造带一张图片的最小 docx（避免依赖 python-docx）。"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    document = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
<w:body>
<w:p><w:r><w:t>Before image.</w:t></w:r></w:p>
<w:p><w:r><w:drawing><wp:inline><wp:extent cx="320000" cy="160000"/>
<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic><pic:nvPicPr><pic:cNvPr id="1" name="pic.png"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="rId2"/></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="320000" cy="160000"/></a:xfrm></pic:spPr></pic:pic>
</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>
<w:p><w:r><w:t>After image.</w:t></w:r></w:p>
</w:body></w:document>"""
    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
</Relationships>"""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", document_rels)
        z.writestr("word/media/image1.png", png)


def test_docx_embedded_image_becomes_placeholder(tmp_path, png_bytes):
    docx_path = tmp_path / "t.docx"
    _make_docx(docx_path, png_bytes)

    loader = OfficeLoader(image_storage_dir=tmp_path / "images")
    doc = loader.load(docx_path)

    images = doc.metadata["images"]
    assert len(images) == 1
    assert f"[IMAGE: {images[0]['id']}]" in doc.text
    assert "data:image" not in doc.text
    assert Path(images[0]["path"]).is_file()
    assert Path(images[0]["path"]).read_bytes() == png_bytes


def test_docx_images_disabled_strips_markdown(tmp_path, png_bytes):
    docx_path = tmp_path / "t.docx"
    _make_docx(docx_path, png_bytes)

    loader = OfficeLoader(extract_images=False, image_storage_dir=tmp_path / "images")
    doc = loader.load(docx_path)

    assert "images" not in doc.metadata
    assert "data:image" not in doc.text
    assert "![" not in doc.text
    assert "Before image." in doc.text


def test_pptx_image_gets_slide_number(tmp_path, png_bytes):
    pptx = pytest.importorskip("pptx")
    from pptx.util import Inches

    png_path = tmp_path / "pic.png"
    png_path.write_bytes(png_bytes)

    prs = pptx.Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[5])
    s1.shapes.title.text = "Slide One"
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    s2.shapes.title.text = "Slide Two"
    s2.shapes.add_picture(str(png_path), Inches(1), Inches(2))
    pptx_path = tmp_path / "t.pptx"
    prs.save(pptx_path)

    loader = OfficeLoader(image_storage_dir=tmp_path / "images")
    doc = loader.load(pptx_path)

    images = doc.metadata["images"]
    assert len(images) == 1
    assert images[0]["page"] == 2
    assert f"[IMAGE: {images[0]['id']}]" in doc.text