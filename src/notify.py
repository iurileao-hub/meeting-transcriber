"""System notifications for Meeting Transcriber."""
import platform
import subprocess


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system() == "Darwin"


def _escape_applescript_string(s: str) -> str:
    """Escape a string for safe use in AppleScript.

    AppleScript strings use double quotes and backslash escaping.
    This function handles all necessary escape sequences.

    Args:
        s: String to escape.

    Returns:
        Escaped string safe for AppleScript.
    """
    # Order matters: escape backslashes first, then other characters
    s = s.replace("\\", "\\\\")  # Backslashes
    s = s.replace('"', '\\"')    # Double quotes
    s = s.replace("\n", "\\n")   # Newlines
    s = s.replace("\r", "\\r")   # Carriage returns
    s = s.replace("\t", "\\t")   # Tabs

    # Limit length to prevent excessively long notifications
    max_len = 500
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."

    return s


def notify(title: str, message: str) -> None:
    """Send system notification.

    On macOS, uses osascript to display native notification.
    On other systems, does nothing (silent no-op).

    Args:
        title: Notification title (max 100 chars).
        message: Notification body text (max 500 chars).
    """
    if not is_macos():
        return

    # Escape for AppleScript string safety
    safe_title = _escape_applescript_string(title[:100])
    safe_message = _escape_applescript_string(message)

    script = f'display notification "{safe_message}" with title "{safe_title}"'

    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            timeout=5,  # Prevent hanging
        )
    except Exception:
        # Silent failure - notifications are non-critical
        pass
