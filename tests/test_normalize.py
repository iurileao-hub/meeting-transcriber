"""Tests for text normalization."""
import pytest
from src.normalize import normalize_text


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_capitalizes_first_character(self):
        assert normalize_text("hello world") == "Hello world"

    def test_capitalizes_after_period(self):
        assert normalize_text("hello. world") == "Hello. World"

    def test_capitalizes_after_exclamation(self):
        assert normalize_text("hello! world") == "Hello! World"

    def test_capitalizes_after_question(self):
        assert normalize_text("hello? world") == "Hello? World"

    def test_handles_multiple_sentences(self):
        text = "first sentence. second sentence! third sentence?"
        expected = "First sentence. Second sentence! Third sentence?"
        assert normalize_text(text) == expected

    def test_preserves_existing_caps(self):
        assert normalize_text("Hello. World") == "Hello. World"

    def test_handles_empty_string(self):
        assert normalize_text("") == ""

    def test_handles_single_word(self):
        assert normalize_text("hello") == "Hello"

    def test_preserves_acronyms(self):
        assert normalize_text("the NASA team. they went to space") == "The NASA team. They went to space"
