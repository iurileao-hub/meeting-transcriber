"""Tests for notification system."""
import pytest
from unittest.mock import patch, MagicMock
from src.notify import notify, is_macos


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
