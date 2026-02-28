"""Tests for vocabulary system."""
import logging
import pytest
from pathlib import Path
from unittest.mock import patch
from src.vocabulary import (
    load_vocabulary,
    parse_vocab_file,
    _is_safe_vocab_path,
    MAX_VOCAB_FILE_SIZE,
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

    def test_rejects_file_exceeding_size_limit(self, tmp_path):
        """Files larger than MAX_VOCAB_FILE_SIZE should raise ValueError."""
        big_file = tmp_path / "big.txt"
        # Write slightly more than 1 MB
        big_file.write_text("x" * (MAX_VOCAB_FILE_SIZE + 1))
        with pytest.raises(ValueError, match="too large"):
            parse_vocab_file(big_file)

    def test_accepts_file_at_size_limit(self, tmp_path):
        """Files exactly at the limit should be accepted."""
        ok_file = tmp_path / "ok.txt"
        ok_file.write_text("x" * MAX_VOCAB_FILE_SIZE)
        # Should not raise
        words = parse_vocab_file(ok_file)
        assert len(words) == 1


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

    @patch("src.vocabulary._is_safe_vocab_path", return_value=True)
    def test_loads_extra_files(self, mock_safe, tmp_path):
        extra_file = tmp_path / "extra.txt"
        extra_file.write_text("extra1\nextra2")
        words = load_vocabulary(vocab_dir=tmp_path, extra_files=[str(extra_file)])
        assert "extra1" in words
        assert "extra2" in words

    @patch("src.vocabulary._is_safe_vocab_path", return_value=True)
    def test_deduplicates_words(self, mock_safe, tmp_path):
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("word1\nword2")
        file2.write_text("word2\nword3")
        words = load_vocabulary(vocab_dir=tmp_path, extra_files=[str(file1), str(file2)])
        assert words.count("word2") == 1


class TestVocabPathSecurity:
    """Security tests for vocabulary path validation."""

    def test_path_inside_project_root_is_allowed(self, tmp_path):
        """A path within the allowed base directory is permitted."""
        safe_path = tmp_path / "vocab.txt"
        assert _is_safe_vocab_path(safe_path, allowed_base=tmp_path) is True

    def test_path_outside_project_root_is_blocked(self, tmp_path):
        """A path outside the allowed base directory is blocked."""
        outside_path = Path("/etc/passwd")
        assert _is_safe_vocab_path(outside_path, allowed_base=tmp_path) is False

    def test_root_is_blocked(self, tmp_path):
        """The /root directory should be blocked."""
        forbidden_path = Path("/root")
        assert _is_safe_vocab_path(forbidden_path, allowed_base=tmp_path) is False

    def test_var_log_is_blocked(self, tmp_path):
        """The /var/log directory should be blocked."""
        forbidden_path = Path("/var/log")
        assert _is_safe_vocab_path(forbidden_path, allowed_base=tmp_path) is False

    def test_subdir_of_allowed_base_is_allowed(self, tmp_path):
        """Subdirectories within the allowed base are permitted."""
        subdir = tmp_path / "sub" / "deep" / "file.txt"
        assert _is_safe_vocab_path(subdir, allowed_base=tmp_path) is True

    def test_traversal_attack_is_blocked(self, tmp_path):
        """Path traversal (../) beyond the allowed base should be blocked."""
        traversal_path = tmp_path / ".." / ".." / "etc" / "passwd"
        assert _is_safe_vocab_path(traversal_path, allowed_base=tmp_path) is False

    def test_default_allowed_base_uses_project_root(self):
        """When no allowed_base is given, project root is used."""
        # A path inside the project root should be allowed
        project_root = Path(__file__).parent.parent
        safe_path = project_root / "vocab" / "test.txt"
        assert _is_safe_vocab_path(safe_path) is True

        # A path outside the project root should be blocked
        outside_path = Path("/tmp/outside.txt")
        assert _is_safe_vocab_path(outside_path) is False

    def test_forbidden_paths_skipped_in_load_with_warning(self, tmp_path, caplog):
        """Forbidden paths should be skipped with a warning log."""
        # Create a valid vocab file inside tmp_path
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("valid_word")

        # Use an outside path that will be blocked
        outside_path = "/root/forbidden.txt"

        # Mock _is_safe_vocab_path to allow valid_file but block outside_path
        original_is_safe = _is_safe_vocab_path

        def selective_safe(path, allowed_base=None):
            if str(path) == str(valid_file):
                return True
            return False

        with patch("src.vocabulary._is_safe_vocab_path", side_effect=selective_safe):
            with caplog.at_level(logging.WARNING):
                words = load_vocabulary(
                    vocab_dir=tmp_path,
                    extra_files=[str(valid_file), outside_path],
                )

        # Valid word should be loaded, forbidden path skipped
        assert "valid_word" in words
        # Warning should have been logged
        assert any(
            "Skipped vocabulary path outside project directory" in r.message
            for r in caplog.records
        )

    def test_warning_logged_for_blocked_path(self, tmp_path, caplog):
        """A warning should be logged when a path is blocked."""
        with caplog.at_level(logging.WARNING):
            load_vocabulary(
                vocab_dir=tmp_path,
                extra_files=["/etc/shadow"],
            )

        assert (
            len(
                [
                    r
                    for r in caplog.records
                    if "Skipped vocabulary path" in r.message
                ]
            )
            == 1
        )

    def test_safety_check_before_existence_check(self, tmp_path):
        """Safety check should run before checking if the file exists.

        This prevents timing oracles that could reveal file existence.
        """
        # Even if a file exists outside allowed_base, it should be blocked
        # (and never read). We test by providing a path that exists but is
        # outside the allowed base. The function should skip it without
        # attempting to parse it.
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("secret_data")
            outside_file = f.name

        try:
            words = load_vocabulary(
                vocab_dir=tmp_path,
                extra_files=[outside_file],
            )
            # The outside file should not be loaded
            assert "secret_data" not in words
        finally:
            Path(outside_file).unlink()
