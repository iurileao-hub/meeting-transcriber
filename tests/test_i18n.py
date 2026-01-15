"""Tests for internationalization system."""
import pytest
from src.i18n import get_translator, SUPPORTED_LANGUAGES


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


class TestSupportedLanguages:
    """Tests for supported languages."""

    def test_english_supported(self):
        assert "en" in SUPPORTED_LANGUAGES

    def test_portuguese_supported(self):
        assert "pt" in SUPPORTED_LANGUAGES
