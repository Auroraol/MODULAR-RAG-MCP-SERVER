"""Vision LLM 工厂与基础接口的单元测试。

本模块使用假实现测试 Vision LLM 工厂模式、provider 注册，
以及 BaseVisionLLM 抽象接口。
"""

import pytest
from pathlib import Path
from typing import Any, Optional

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory


# ================================
# 用于测试的假实现
# ================================

class FakeVisionLLM(BaseVisionLLM):
    """用于测试的假 Vision LLM 实现。

    该实现返回确定性的响应，便于测试，
    并记录调用次数以供校验。
    """

    def __init__(
        self,
        settings: Any = None,
        response_template: str = "I see: {text} | Image: {image_type}",
        **kwargs: Any
    ):
        """初始化假 Vision LLM。

        参数:
            settings: 可选的配置对象（假实现中未使用）。
            response_template: 用于生成响应的模板。
            **kwargs: 额外参数（未使用）。
        """
        self.settings = settings
        self.response_template = response_template
        self.call_count = 0
        self.last_text = None
        self.last_image = None
        self.last_messages = None
    
    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: Optional[list[Message]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """根据输入生成假响应。"""
        # 使用基类方法校验输入
        self.validate_text(text)
        self.validate_image(image)

        # 记录调用
        self.call_count += 1
        self.last_text = text
        self.last_image = image
        self.last_messages = messages

        # 确定用于响应的图片类型
        if image.path:
            image_type = f"path({image.path})"
        elif image.data:
            image_type = f"bytes({len(image.data)} bytes)"
        elif image.base64:
            image_type = f"base64({len(image.base64)} chars)"
        else:
            image_type = "unknown"
        
        # 生成确定性响应
        content = self.response_template.format(
            text=text,
            image_type=image_type
        )
        
        return ChatResponse(
            content=content,
            model="fake-vision-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )


# =========================
# 测试 BaseVisionLLM
# =========================

class TestBaseVisionLLM:
    """测试 BaseVisionLLM 抽象接口。"""

    def test_abstract_cannot_instantiate(self):
        """BaseVisionLLM 不能被直接实例化。"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseVisionLLM()
    
    def test_validate_text_success(self):
        """validate_text 接受合法文本。"""
        fake = FakeVisionLLM()
        fake.validate_text("valid text")  # 不应抛出异常

    def test_validate_text_empty(self):
        """validate_text 拒绝空文本。"""
        fake = FakeVisionLLM()
        with pytest.raises(ValueError, match="Text prompt cannot be empty"):
            fake.validate_text("")
    
    def test_validate_text_whitespace_only(self):
        """validate_text 拒绝仅含空白的文本。"""
        fake = FakeVisionLLM()
        with pytest.raises(ValueError, match="Text prompt cannot be empty"):
            fake.validate_text("   \n\t  ")
    
    def test_validate_text_non_string(self):
        """validate_text 拒绝非字符串输入。"""
        fake = FakeVisionLLM()
        with pytest.raises(ValueError, match="Text must be a string"):
            fake.validate_text(123)  # type: ignore
    
    def test_validate_image_success(self):
        """validate_image 接受合法的 ImageInput。"""
        fake = FakeVisionLLM()
        image = ImageInput(path="test.png")
        fake.validate_image(image)  # 不应抛出异常

    def test_validate_image_invalid_type(self):
        """validate_image 拒绝非 ImageInput 输入。"""
        fake = FakeVisionLLM()
        with pytest.raises(ValueError, match="Image must be an ImageInput instance"):
            fake.validate_image("not_an_image_input")  # type: ignore
    
    def test_preprocess_image_default(self):
        """默认情况下 preprocess_image 原样返回图片。"""
        fake = FakeVisionLLM()
        image = ImageInput(path="test.png")
        result = fake.preprocess_image(image)
        assert result is image  # 同一实例


# =========================
# 测试 ImageInput
# =========================

class TestImageInput:
    """测试 ImageInput 数据类。"""

    def test_image_input_path(self):
        """ImageInput 可以使用 path 创建。"""
        image = ImageInput(path="test.png")
        assert image.path == "test.png"
        assert image.data is None
        assert image.base64 is None
        assert image.mime_type == "image/png"
    
    def test_image_input_path_as_pathlib(self):
        """ImageInput 接受 pathlib.Path。"""
        path = Path("test.png")
        image = ImageInput(path=path)
        assert image.path == path
    
    def test_image_input_data(self):
        """ImageInput 可以使用 bytes 数据创建。"""
        data = b"fake_image_bytes"
        image = ImageInput(data=data, mime_type="image/jpeg")
        assert image.data == data
        assert image.path is None
        assert image.base64 is None
        assert image.mime_type == "image/jpeg"
    
    def test_image_input_base64(self):
        """ImageInput 可以使用 base64 字符串创建。"""
        base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAUA"
        image = ImageInput(base64=base64_str)
        assert image.base64 == base64_str
        assert image.path is None
        assert image.data is None
    
    def test_image_input_no_input(self):
        """未提供任何输入时 ImageInput 会抛出错误。"""
        with pytest.raises(ValueError, match="Must provide one of: path, data, or base64"):
            ImageInput()
    
    def test_image_input_multiple_inputs(self):
        """提供多个输入时 ImageInput 会抛出错误。"""
        with pytest.raises(ValueError, match="Must provide exactly one of"):
            ImageInput(path="test.png", data=b"bytes")


# =========================
# 测试 FakeVisionLLM
# =========================

class TestFakeVisionLLM:
    """测试 FakeVisionLLM 实现。"""

    def test_chat_with_image_path(self):
        """FakeVisionLLM 为基于 path 的图片生成响应。"""
        fake = FakeVisionLLM()
        image = ImageInput(path="diagram.png")

        response = fake.chat_with_image(
            text="Describe this diagram",
            image=image
        )

        assert response.content == "I see: Describe this diagram | Image: path(diagram.png)"
        assert response.model == "fake-vision-model"
        assert fake.call_count == 1
        assert fake.last_text == "Describe this diagram"
        assert fake.last_image == image

    def test_chat_with_image_bytes(self):
        """FakeVisionLLM 为基于 bytes 的图片生成响应。"""
        fake = FakeVisionLLM()
        data = b"fake_image_data_12345"
        image = ImageInput(data=data)

        response = fake.chat_with_image(
            text="What is this?",
            image=image
        )

        assert "bytes(21 bytes)" in response.content
        assert fake.call_count == 1

    def test_chat_with_image_base64(self):
        """FakeVisionLLM 为基于 base64 的图片生成响应。"""
        fake = FakeVisionLLM()
        base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAUA"
        image = ImageInput(base64=base64_str)

        response = fake.chat_with_image(
            text="Analyze this",
            image=image
        )

        assert f"base64({len(base64_str)} chars)" in response.content
        assert fake.call_count == 1

    def test_chat_with_image_with_messages(self):
        """FakeVisionLLM 接受对话历史。"""
        fake = FakeVisionLLM()
        messages = [
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="Previous question")
        ]
        image = ImageInput(path="test.png")

        response = fake.chat_with_image(
            text="New question",
            image=image,
            messages=messages
        )

        assert response.content is not None
        assert fake.last_messages == messages

    def test_chat_with_image_validates_text(self):
        """FakeVisionLLM 会校验文本输入。"""
        fake = FakeVisionLLM()
        image = ImageInput(path="test.png")

        with pytest.raises(ValueError, match="Text prompt cannot be empty"):
            fake.chat_with_image(text="", image=image)

    def test_chat_with_image_validates_image(self):
        """FakeVisionLLM 会校验图片输入。"""
        fake = FakeVisionLLM()

        with pytest.raises(ValueError, match="Image must be an ImageInput instance"):
            fake.chat_with_image(text="test", image="not_an_image")  # type: ignore

    def test_custom_response_template(self):
        """FakeVisionLLM 接受自定义响应模板。"""
        fake = FakeVisionLLM(response_template="Custom: {text}")
        image = ImageInput(path="test.png")

        response = fake.chat_with_image(text="Hello", image=image)

        assert response.content == "Custom: Hello"


# ================================
# 测试 Vision LLM 工厂
# ================================

class TestVisionLLMFactory:
    """测试 Vision LLM 工厂模式。"""

    def setup_method(self):
        """每个测试前清理注册表。"""
        LLMFactory._VISION_PROVIDERS.clear()

    def test_register_vision_provider_success(self):
        """register_vision_provider 注册合法的 provider。"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)
        
        assert "fake" in LLMFactory._VISION_PROVIDERS
        assert LLMFactory._VISION_PROVIDERS["fake"] == FakeVisionLLM
    
    def test_register_vision_provider_case_insensitive(self):
        """register_vision_provider 会将 provider 名称规范化为小写。"""
        LLMFactory.register_vision_provider("FakeVision", FakeVisionLLM)
        
        assert "fakevision" in LLMFactory._VISION_PROVIDERS
        assert "FakeVision" not in LLMFactory._VISION_PROVIDERS
    
    def test_register_vision_provider_invalid_class(self):
        """register_vision_provider 拒绝非 BaseVisionLLM 子类。"""
        class NotAVisionLLM:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseVisionLLM"):
            LLMFactory.register_vision_provider("invalid", NotAVisionLLM)  # type: ignore
    
    def test_list_vision_providers_empty(self):
        """未注册任何 provider 时 list_vision_providers 返回空列表。"""
        assert LLMFactory.list_vision_providers() == []
    
    def test_list_vision_providers_sorted(self):
        """list_vision_providers 返回排序后的 provider 名称。"""
        LLMFactory.register_vision_provider("zebra", FakeVisionLLM)
        LLMFactory.register_vision_provider("alpha", FakeVisionLLM)
        LLMFactory.register_vision_provider("beta", FakeVisionLLM)
        
        providers = LLMFactory.list_vision_providers()
        assert providers == ["alpha", "beta", "zebra"]
    
    def test_create_vision_llm_success(self):
        """create_vision_llm 根据 vision_llm 配置创建实例。"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)

        # 带 vision_llm 配置段的 Mock 配置
        class FakeSettings:
            class VisionLLM:
                provider = "fake"
            vision_llm = VisionLLM()
        
        settings = FakeSettings()
        vision_llm = LLMFactory.create_vision_llm(settings)
        
        assert isinstance(vision_llm, FakeVisionLLM)
        assert vision_llm.settings == settings
    
    def test_create_vision_llm_fallback_to_llm_config(self):
        """vision_llm 配置不存在时，create_vision_llm 回退到 llm.provider。"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)

        # 只带 llm 配置段（无 vision_llm）的 Mock 配置
        class FakeSettings:
            class LLM:
                provider = "fake"
            llm = LLM()
        
        settings = FakeSettings()
        vision_llm = LLMFactory.create_vision_llm(settings)
        
        assert isinstance(vision_llm, FakeVisionLLM)
    
    def test_create_vision_llm_case_insensitive(self):
        """create_vision_llm 能处理 provider 名称大小写不敏感的情况。"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)

        class FakeSettings:
            class VisionLLM:
                provider = "FAKE"  # 大写
            vision_llm = VisionLLM()
        
        settings = FakeSettings()
        vision_llm = LLMFactory.create_vision_llm(settings)
        
        assert isinstance(vision_llm, FakeVisionLLM)
    
    def test_create_vision_llm_unknown_provider(self):
        """create_vision_llm 对未知 provider 抛出错误。"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)

        class FakeSettings:
            class VisionLLM:
                provider = "unknown"
            vision_llm = VisionLLM()

        settings = FakeSettings()

        with pytest.raises(ValueError, match="Unsupported Vision LLM provider: 'unknown'"):
            LLMFactory.create_vision_llm(settings)

    def test_create_vision_llm_missing_config(self):
        """配置缺失时 create_vision_llm 抛出错误。"""
        class FakeSettings:
            pass  # 没有 vision_llm 或 llm 配置段

        settings = FakeSettings()

        with pytest.raises(ValueError, match="Missing required configuration"):
            LLMFactory.create_vision_llm(settings)

    def test_create_vision_llm_no_providers_registered(self):
        """注册表为空时 create_vision_llm 显示 available: none。"""
        class FakeSettings:
            class VisionLLM:
                provider = "fake"
            vision_llm = VisionLLM()

        settings = FakeSettings()

        with pytest.raises(ValueError, match="Available Vision LLM providers: none"):
            LLMFactory.create_vision_llm(settings)

    def test_create_vision_llm_with_overrides(self):
        """create_vision_llm 将覆盖参数传递给 provider。"""
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)

        class FakeSettings:
            class VisionLLM:
                provider = "fake"
            vision_llm = VisionLLM()

        settings = FakeSettings()
        vision_llm = LLMFactory.create_vision_llm(
            settings,
            response_template="Override: {text}"
        )

        assert isinstance(vision_llm, FakeVisionLLM)
        assert vision_llm.response_template == "Override: {text}"

    def test_create_vision_llm_provider_instantiation_failure(self):
        """create_vision_llm 能处理 provider 实例化错误。"""
        class BrokenVisionLLM(BaseVisionLLM):
            def __init__(self, settings, **kwargs):
                raise RuntimeError("Intentional failure")

            def chat_with_image(self, text, image, messages=None, trace=None, **kwargs):
                pass

        LLMFactory.register_vision_provider("broken", BrokenVisionLLM)

        class FakeSettings:
            class VisionLLM:
                provider = "broken"
            vision_llm = VisionLLM()

        settings = FakeSettings()

        with pytest.raises(RuntimeError, match="Failed to instantiate Vision LLM provider 'broken'"):
            LLMFactory.create_vision_llm(settings)


# ================================
# 集成测试
# ================================

class TestVisionLLMIntegration:
    """结合工厂与实现的集成测试。"""

    def setup_method(self):
        """清理注册表并注册假 provider。"""
        LLMFactory._VISION_PROVIDERS.clear()
        LLMFactory.register_vision_provider("fake", FakeVisionLLM)

    def test_end_to_end_vision_workflow(self):
        """完整流程：从工厂创建 -> 调用 vision 方法。"""
        class FakeSettings:
            class VisionLLM:
                provider = "fake"
            vision_llm = VisionLLM()

        settings = FakeSettings()

        # 从工厂创建 Vision LLM
        vision_llm = LLMFactory.create_vision_llm(settings)

        # 用它进行图片描述
        image = ImageInput(path="document.pdf.page1.png")
        response = vision_llm.chat_with_image(
            text="Describe the main content of this page",
            image=image
        )
        
        assert "Describe the main content" in response.content
        assert "document.pdf.page1.png" in response.content
        assert response.model == "fake-vision-model"
