"""Dashboard i18n 模块的单元测试。

校验：

1. 中英文文案键集合完全一致（避免漏译）。
2. 所有占位符在两个语言版本中匹配。
3. ``tr()`` 的默认回退行为。
4. 页面源码中不再残留硬编码的英文 UI 文案。
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import pytest

from src.observability.dashboard import i18n

PAGES_DIR = Path(i18n.__file__).parent / "pages"

_PLACEHOLDER = re.compile(r"\{(\w+)")


def _fmt_fields(template: str) -> set[str]:
    """提取模板中的 ``str.format`` 字段名。"""
    return set(_PLACEHOLDER.findall(template))


def _dummy_value(field_name: str, spec: str) -> object:
    """按格式说明推断占位符应传入的值类型。

    例如 ``{count:,}`` 需要 int，``{elapsed:.0f}`` 需要 float，
    其余按字符串处理。这样测试才能真正验证模板可格式化。
    """
    if spec:
        # 去掉字段名与对齐/填充，只看类型字符
        if spec.endswith("f"):
            return 1.0
        if spec.rstrip("d").count(",") or spec.endswith("d") or "," in spec:
            return 1
    return "x"


def _sample_kwargs(template: str) -> dict[str, object]:
    """为模板中的每个字段生成类型合适的示例值。"""
    kwargs: dict[str, object] = {}
    for _, field_name, spec, _ in string.Formatter().parse(template):
        if field_name:
            kwargs[field_name] = _dummy_value(field_name, spec or "")
    return kwargs


class TestTranslationTables:
    """文案表的一致性校验。"""

    def test_languages_have_identical_keys(self) -> None:
        """中英文键集合必须完全一致。"""
        en_keys = set(i18n._EN)
        zh_keys = set(i18n._ZH)
        assert not en_keys - zh_keys, f"中文缺少键: {sorted(en_keys - zh_keys)}"
        assert not zh_keys - en_keys, f"英文缺少键: {sorted(zh_keys - en_keys)}"

    def test_placeholders_match_across_languages(self) -> None:
        """每条文案的占位符在中英文版本中必须相同。"""
        mismatched = {
            key: (_fmt_fields(i18n._EN[key]), _fmt_fields(i18n._ZH[key]))
            for key in i18n._EN
            if _fmt_fields(i18n._EN[key]) != _fmt_fields(i18n._ZH[key])
        }
        assert not mismatched, f"占位符不一致: {mismatched}"

    def test_no_empty_translations(self) -> None:
        """不允许空文案。"""
        empty = [k for k, v in {**i18n._EN, **i18n._ZH}.items() if not v.strip()]
        assert not empty, f"空文案: {empty}"


class TestTranslateFunction:
    """``tr()`` 行为校验。"""

    def test_default_language_is_chinese(self) -> None:
        """默认语言为中文。"""
        assert i18n.DEFAULT_LANGUAGE == "zh"

    def test_unknown_key_falls_back_to_key(self) -> None:
        """未知键原样返回，不抛异常。"""
        assert i18n.tr("no.such.key") == "no.such.key"

    def test_placeholder_substitution(self) -> None:
        """占位符按关键字参数替换。"""
        assert i18n.tr("overview.config_failed", exc="boom").endswith("boom")

    def test_template_without_kwargs_is_not_formatted(self) -> None:
        """不传参数时原样返回，避免正文花括号被误解析。"""
        text = i18n.tr("overview.no_collections")
        assert "{exc}" not in text

    @pytest.mark.parametrize("key", sorted(i18n._EN))
    def test_every_key_is_formattable(self, key: str) -> None:
        """每个键在传入合适类型的占位符值时都能安全格式化。"""
        for language in ("en", "zh"):
            template = i18n._TRANSLATIONS[language][key]
            kwargs = _sample_kwargs(template)
            rendered = template.format(**kwargs) if kwargs else template
            assert isinstance(rendered, str)


class TestNoHardcodedUiStrings:
    """页面源码中不应再残留硬编码的 UI 文案。"""

    #: 允许保留的非文案字符串（HTML 类名、CSS、占位标记等）
    ALLOWED = {
        "raw_text",
        "split_{i}",
        "before_{i}",
        "after_{i}",
        "Content",
        "---",
    }

    _CALL = re.compile(
        r"st\.(?:header|subheader|title|markdown|info|warning|error|success|caption)"
        r"\(\s*(?:[a-z_]+=\s*)?[\"']([^\"']{3,})"
    )

    def test_pages_have_no_literal_ui_text(self) -> None:
        """页面渲染调用不应直接传英文字面量。"""
        offenders: list[tuple[str, str]] = []
        for path in sorted(PAGES_DIR.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            for literal in self._CALL.findall(source):
                if literal in self.ALLOWED:
                    continue
                # 纯符号/数字/emoji 不计
                if not any(ch.isalpha() for ch in literal):
                    continue
                offenders.append((path.name, literal))
        assert not offenders, f"仍有硬编码文案: {offenders}"
