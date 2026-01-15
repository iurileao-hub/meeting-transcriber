"""System notifications for Meeting Transcriber."""
import platform
import subprocess


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system() == "Darwin"


def notify(title: str, message: str) -> None:
    """Send system notification.

    On macOS, uses osascript to display native notification.
    On other systems, does nothing (silent no-op).

    Args:
        title: Notification title.
        message: Notification body text.
    """
    if not is_macos():
        return

    # Escape quotes in title and message
    title = title.replace('"', '\\"')
    message = message.replace('"', '\\"')

    script = f'display notification "{message}" with title "{title}"'

    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
        )
    except Exception:
        # Silent failure - notifications are non-critical
        pass
