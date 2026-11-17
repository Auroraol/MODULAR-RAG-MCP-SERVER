"""生成用于测试的示例 PDF 文件。

该脚本会创建：
1. simple.pdf - 带标题和段落的纯文本 PDF
2. with_images.pdf - 包含文本和图片的 PDF（目前为占位实现）
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.colors import blue, black
from reportlab.platypus import Image as RLImage
from PIL import Image
import io


def create_simple_pdf():
    """创建一个仅含文本的简单 PDF。"""
    filepath = Path(__file__).parent / "simple.pdf"
    
    c = canvas.Canvas(str(filepath), pagesize=letter)
    width, height = letter
    
    # 标题
    c.setFont("Helvetica-Bold", 24)
    c.drawString(1*inch, height - 1*inch, "Sample Document")
    
    # 副标题
    c.setFont("Helvetica", 14)
    c.drawString(1*inch, height - 1.5*inch, "A Simple Test PDF")
    
    # 第 1 段
    c.setFont("Helvetica", 11)
    y_position = height - 2.5*inch
    
    text_lines = [
        "This is a sample PDF document for testing the PDF loader.",
        "It contains multiple paragraphs of text to verify that",
        "the MarkItDown conversion works correctly.",
        "",
        "This document should be parsed into Markdown format,",
        "with the title extracted and metadata populated.",
    ]
    
    for line in text_lines:
        c.drawString(1*inch, y_position, line)
        y_position -= 0.25*inch
    
    # 添加章节标题
    y_position -= 0.5*inch
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1*inch, y_position, "Section 1: Introduction")
    
    y_position -= 0.4*inch
    c.setFont("Helvetica", 11)
    c.drawString(1*inch, y_position, "This section contains introductory text.")
    y_position -= 0.25*inch
    c.drawString(1*inch, y_position, "The loader should handle this correctly.")
    
    c.save()
    print(f"✅ Created: {filepath}")


def create_pdf_with_images():
    """创建一个包含文本和简单图片的 PDF。"""
    filepath = Path(__file__).parent / "with_images.pdf"

    # 创建一个简单的测试图片
    img = Image.new('RGB', (200, 100), color='lightblue')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)

    # 保存临时图片
    temp_img = Path(__file__).parent / "temp_test_image.png"
    with open(temp_img, 'wb') as f:
        f.write(img_buffer.getvalue())

    c = canvas.Canvas(str(filepath), pagesize=letter)
    width, height = letter

    # 标题
    c.setFont("Helvetica-Bold", 24)
    c.drawString(1*inch, height - 1*inch, "Document with Images")

    # 图片前的文本
    c.setFont("Helvetica", 11)
    y_position = height - 1.8*inch
    c.drawString(1*inch, y_position, "This document contains an embedded image below:")

    # 添加图片
    y_position -= 1.5*inch
    c.drawImage(str(temp_img), 1*inch, y_position, width=2*inch, height=1*inch)

    # 图片后的文本
    y_position -= 0.5*inch
    c.drawString(1*inch, y_position, "Text continues after the image.")
    y_position -= 0.25*inch
    c.drawString(1*inch, y_position, "The loader should detect and extract this image.")
    
    c.save()
    
    # 清理临时图片
    temp_img.unlink()
    
    print(f"✅ Created: {filepath}")


if __name__ == "__main__":
    create_simple_pdf()
    create_pdf_with_images()
    print("\n✅ All sample PDF files generated successfully!")
