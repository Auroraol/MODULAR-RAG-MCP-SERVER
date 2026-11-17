import pytest
from pathlib import Path
from src.core.settings import load_settings
from src.core.types import Chunk
from src.ingestion.transform.image_captioner import ImageCaptioner

@pytest.mark.integration
def test_image_captioner_azure_integration():
    """ImageCaptioner 接入真实 Azure OpenAI Vision LLM 的集成测试。

    需要在 config/settings.yaml 中配置有效凭据（由用户配置）。
    """
    # 1. 加载配置
    settings = load_settings("config/settings.yaml")

    # 若 vision 未启用或 provider 未配置则跳过
    if not settings.vision_llm or not settings.vision_llm.enabled:
        pytest.skip("Vision LLM not enabled in settings")

    if settings.vision_llm.provider != "azure":
        pytest.skip("Test specific for Azure provider (as requested)")

    # 2. 检查测试图像
    image_path = Path("tests/fixtures/sample_documents/test_vision_llm.jpg")
    if not image_path.exists():
        pytest.fail(f"Test image not found at {image_path}")

    # 3. 创建样本 chunk
    # 模拟来自 loader 的 chunk
    chunk = Chunk(
        id="chunk_test_001",
        text="Here is an image: [IMAGE: img_001]",
        metadata={
            "source_path": str(image_path),
            "images": [
                {
                    "id": "img_001",
                    "path": str(image_path),
                    "page": 1
                }
            ]
        }
    )
    
    # 4. 初始化 ImageCaptioner
    captioner = ImageCaptioner(settings=settings)

    # 5. 执行转换
    # 此处会调用真实 API
    processed_chunks = captioner.transform([chunk])

    # 6. 校验结果
    assert len(processed_chunks) == 1
    processed_chunk = processed_chunks[0]

    # 检查文本修改
    print(f"\nOriginal Text: 'Here is an image: [IMAGE: img_001]'")
    print(f"New Text: '{processed_chunk.text}'")
    
    assert "[IMAGE: img_001]" in processed_chunk.text
    assert "(Description:" in processed_chunk.text
    assert "image_captions" in processed_chunk.metadata
    assert len(processed_chunk.metadata["image_captions"]) == 1
    
    caption = processed_chunk.metadata["image_captions"][0]["caption"]
    print(f"Generated Caption: {caption}")
    assert len(caption) > 10
