"""Internationalization (i18n) support for Meeting Transcriber."""
import json
import locale
from pathlib import Path
from typing import Callable

SUPPORTED_LANGUAGES = {"en", "pt"}
_DEFAULT_LANGUAGE = "en"


def _load_strings(lang: str) -> dict:
    """Load translation strings for a language."""
    i18n_dir = Path(__file__).parent
    lang_file = i18n_dir / f"{lang}.json"

    if not lang_file.exists():
        lang_file = i18n_dir / f"{_DEFAULT_LANGUAGE}.json"

    return json.loads(lang_file.read_text(encoding="utf-8"))


def _get_nested(data: dict, key: str) -> str:
    """Get nested value from dict using dot notation."""
    keys = key.split(".")
    value = data
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return key  # Return key itself if not found


def get_translator(lang: str | None = None) -> Callable[[str], str]:
    """Get a translator function for the specified language.

    Args:
        lang: Language code ('en', 'pt') or None for auto-detect.

    Returns:
        Translator function t(key) -> translated string.
    """
    if lang is None:
        # Auto-detect from system locale
        system_locale = locale.getdefaultlocale()[0] or ""
        lang = "pt" if system_locale.startswith("pt") else "en"

    if lang not in SUPPORTED_LANGUAGES:
        lang = _DEFAULT_LANGUAGE

    strings = _load_strings(lang)

    def translate(key: str) -> str:
        return _get_nested(strings, key)

    return translate


def detect_system_language() -> str:
    """Detect language from system locale.

    Returns:
        Language code ('en' or 'pt').
    """
    system_locale = locale.getdefaultlocale()[0] or ""
    return "pt" if system_locale.startswith("pt") else "en"
