"""Tests for notification system."""
import pytest
from unittest.mock import patch, MagicMock
from src.notify import notify, is_macos, _escape_applescript_string


class TestIsMacos:
    """Tests for macOS detection."""

    @patch("platform.system", return_value="Darwin")
    def test_darwin_is_macos(self, mock_system):
        assert is_macos() is True

    @patch("platform.system", return_value="Linux")
    def test_linux_is_not_macos(self, mock_system):
        assert is_macos() is False

    @patch("platform.system", return_value="Windows")
    def test_windows_is_not_macos(self, mock_system):
        assert is_macos() is False


class TestNotify:
    """Tests for notify function."""

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    def test_notify_calls_osascript_on_macos(self, mock_run, mock_system):
        notify("Title", "Message")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "osascript"

    @patch("platform.system", return_value="Linux")
    @patch("subprocess.run")
    def test_notify_does_nothing_on_linux(self, mock_run, mock_system):
        notify("Title", "Message")
        mock_run.assert_not_called()

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    def test_notify_includes_title_and_message(self, mock_run, mock_system):
        notify("Test Title", "Test Message")
        args = mock_run.call_args[0][0]
        script = args[2]  # osascript -e "script"
        assert "Test Title" in script
        assert "Test Message" in script


class TestAppleScriptEscaping:
    """Security tests for AppleScript string escaping."""

    def test_escapes_double_quotes(self):
        result = _escape_applescript_string('Hello "World"')
        assert result == 'Hello \\"World\\"'

    def test_escapes_backslashes(self):
        result = _escape_applescript_string("path\\to\\file")
        assert result == "path\\\\to\\\\file"

    def test_escapes_newlines(self):
        result = _escape_applescript_string("line1\nline2")
        assert result == "line1\\nline2"

    def test_escapes_carriage_returns(self):
        result = _escape_applescript_string("line1\rline2")
        assert result == "line1\\rline2"

    def test_escapes_tabs(self):
        result = _escape_applescript_string("col1\tcol2")
        assert result == "col1\\tcol2"

    def test_truncates_long_strings(self):
        long_string = "a" * 600
        result = _escape_applescript_string(long_string)
        assert len(result) == 500
        assert result.endswith("...")

    def test_handles_injection_attempt(self):
        # Attempt to inject AppleScript command
        malicious = 'test" with title "hacked" --'
        result = _escape_applescript_string(malicious)
        # Double quotes should be escaped
        assert '\\"' in result
        # Should not contain unescaped quotes
        assert '" with title "' not in result

    def test_backslash_before_quote_handled(self):
        # This is a tricky case: \" in input should become \\"
        tricky = 'test\\"end'
        result = _escape_applescript_string(tricky)
        # Backslash escaped, then quote escaped
        assert '\\\\\\"' in result


class TestNotifySecurity:
    """Security tests for notify function."""

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    def test_quotes_are_escaped(self, mock_run, mock_system):
        notify('Title with "quotes"', 'Message with "quotes"')
        args = mock_run.call_args[0][0]
        script = args[2]
        # Quotes should be escaped
        assert '\\"' in script

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    def test_long_title_truncated(self, mock_run, mock_system):
        long_title = "a" * 200
        notify(long_title, "message")
        args = mock_run.call_args[0][0]
        script = args[2]
        # Title should be truncated to 100 chars
        assert "a" * 101 not in script

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    def test_timeout_is_set(self, mock_run, mock_system):
        notify("Title", "Message")
        kwargs = mock_run.call_args[1]
        assert kwargs.get("timeout") == 5
