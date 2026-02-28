"""Tests for internationalization system."""
from unittest.mock import patch

import pytest
from src.i18n import detect_system_language, get_translator, SUPPORTED_LANGUAGES


class TestGetTranslator:
    """Tests for get_translator function."""

    def test_english_translator(self):
        t = get_translator("en")
        assert t("messages.complete") == "Transcription complete"

    def test_portuguese_translator(self):
        t = get_translator("pt")
        assert t("messages.complete") == "Transcrição completa"

    def test_nested_keys(self):
        t = get_translator("en")
        assert t("stages.loading") == "Loading model"

    def test_invalid_language_falls_back_to_english(self):
        t = get_translator("xx")
        assert t("messages.complete") == "Transcription complete"

    def test_missing_key_returns_key(self):
        t = get_translator("en")
        assert t("nonexistent.key") == "nonexistent.key"

    def test_translator_lang_attribute_english(self):
        t = get_translator("en")
        assert t.lang == "en"

    def test_translator_lang_attribute_portuguese(self):
        t = get_translator("pt")
        assert t.lang == "pt"

    def test_translator_lang_attribute_fallback(self):
        t = get_translator("xx")
        assert t.lang == "en"

    def test_translator_lang_attribute_auto_detect(self):
        with patch("src.i18n.detect_system_language", return_value="pt"):
            t = get_translator()
            assert t.lang == "pt"


class TestDetectSystemLanguage:
    """Tests for detect_system_language function."""

    def test_portuguese_locale(self):
        with patch("src.i18n.locale.getlocale", return_value=("pt_BR", "UTF-8")):
            assert detect_system_language() == "pt"

    def test_english_locale(self):
        with patch("src.i18n.locale.getlocale", return_value=("en_US", "UTF-8")):
            assert detect_system_language() == "en"

    def test_none_locale_falls_back_to_english(self):
        with patch("src.i18n.locale.getlocale", return_value=(None, None)):
            assert detect_system_language() == "en"

    def test_exception_falls_back_to_english(self):
        with patch(
            "src.i18n.locale.getlocale", side_effect=ValueError("bad locale")
        ):
            assert detect_system_language() == "en"

    def test_other_locale_falls_back_to_english(self):
        with patch("src.i18n.locale.getlocale", return_value=("es_ES", "UTF-8")):
            assert detect_system_language() == "en"


class TestSupportedLanguages:
    """Tests for supported languages."""

    def test_english_supported(self):
        assert "en" in SUPPORTED_LANGUAGES

    def test_portuguese_supported(self):
        assert "pt" in SUPPORTED_LANGUAGES
