"""Tests for vocabulary system."""
import pytest
from pathlib import Path
from src.vocabulary import (
    load_vocabulary,
    parse_vocab_file,
    _is_safe_vocab_path,
    FORBIDDEN_VOCAB_PREFIXES,
)


class TestParseVocabFile:
    """Tests for parse_vocab_file function."""

    def test_parse_simple_file(self, tmp_path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("word1\nword2\nword3")
        words = parse_vocab_file(vocab_file)
        assert words == ["word1", "word2", "word3"]

    def test_ignores_comments(self, tmp_path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("# comment\nword1\n# another comment\nword2")
        words = parse_vocab_file(vocab_file)
        assert words == ["word1", "word2"]

    def test_ignores_empty_lines(self, tmp_path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("word1\n\n\nword2\n")
        words = parse_vocab_file(vocab_file)
        assert words == ["word1", "word2"]

    def test_strips_whitespace(self, tmp_path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("  word1  \n\tword2\t")
        words = parse_vocab_file(vocab_file)
        assert words == ["word1", "word2"]


class TestLoadVocabulary:
    """Tests for load_vocabulary function."""

    def test_returns_empty_if_no_files(self, tmp_path):
        words = load_vocabulary(vocab_dir=tmp_path)
        assert words == []

    def test_loads_default_if_exists(self, tmp_path):
        default_file = tmp_path / "default.txt"
        default_file.write_text("word1\nword2")
        words = load_vocabulary(vocab_dir=tmp_path)
        assert "word1" in words
        assert "word2" in words

    def test_loads_extra_files(self, tmp_path):
        extra_file = tmp_path / "extra.txt"
        extra_file.write_text("extra1\nextra2")
        words = load_vocabulary(vocab_dir=tmp_path, extra_files=[str(extra_file)])
        assert "extra1" in words
        assert "extra2" in words

    def test_deduplicates_words(self, tmp_path):
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("word1\nword2")
        file2.write_text("word2\nword3")
        words = load_vocabulary(vocab_dir=tmp_path, extra_files=[str(file1), str(file2)])
        assert words.count("word2") == 1


class TestVocabPathSecurity:
    """Security tests for vocabulary path validation."""

    def test_safe_path_is_allowed(self, tmp_path):
        safe_path = tmp_path / "vocab.txt"
        assert _is_safe_vocab_path(safe_path) is True

    def test_root_is_blocked(self):
        forbidden_path = Path("/root")
        assert _is_safe_vocab_path(forbidden_path) is False

    def test_var_log_is_blocked(self):
        forbidden_path = Path("/var/log")
        assert _is_safe_vocab_path(forbidden_path) is False

    def test_subdir_of_root_is_blocked(self):
        """Subdirectory of /root should be blocked."""
        forbidden_path = Path("/root/.ssh/authorized_keys")
        assert _is_safe_vocab_path(forbidden_path) is False

    def test_var_log_subdir_is_blocked(self):
        """Subdirectory of /var/log should be blocked."""
        forbidden_path = Path("/var/log/auth.log")
        assert _is_safe_vocab_path(forbidden_path) is False

    def test_forbidden_prefixes_constant_populated(self):
        """FORBIDDEN_VOCAB_PREFIXES should contain sensitive paths."""
        assert "/root" in FORBIDDEN_VOCAB_PREFIXES
        assert "/var/log" in FORBIDDEN_VOCAB_PREFIXES

    def test_forbidden_paths_skipped_in_load(self, tmp_path):
        """Forbidden paths should be silently skipped during loading."""
        # Create a valid vocab file
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("valid_word")

        # Try to load with a forbidden path (should be skipped)
        words = load_vocabulary(
            vocab_dir=tmp_path,
            extra_files=[str(valid_file), "/root/forbidden.txt"]
        )

        # Valid word should be loaded, forbidden path skipped
        assert "valid_word" in words
