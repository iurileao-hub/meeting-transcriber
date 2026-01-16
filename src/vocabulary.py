"""Custom vocabulary support for transcription."""
from pathlib import Path


# Sensitive path prefixes that should not be read as vocabulary files (security)
# These are prefixes, not exact paths
# Note: macOS resolves /var to /private/var, so we include both variants
FORBIDDEN_VOCAB_PREFIXES = [
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/etc/ssh",
    "/root",
    "/var/log",
    "/private/var/log",  # macOS resolved path
    "/Library/Keychains",  # macOS
]


def _is_safe_vocab_path(path: Path) -> bool:
    """Check if vocabulary path is safe to read.

    Args:
        path: Path to check.

    Returns:
        True if path is safe, False otherwise.
    """
    resolved = path.resolve()
    path_str = str(resolved)

    for forbidden in FORBIDDEN_VOCAB_PREFIXES:
        if path_str == forbidden or path_str.startswith(forbidden + "/"):
            return False

    return True


def parse_vocab_file(path: Path) -> list[str]:
    """Parse vocabulary file.

    File format:
    - One word/term per line
    - Lines starting with # are comments
    - Empty lines are ignored
    - Whitespace is stripped

    Args:
        path: Path to vocabulary file.

    Returns:
        List of vocabulary words.
    """
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line)
    return words


def load_vocabulary(
    vocab_dir: Path | str | None = None,
    extra_files: list[str] | None = None,
) -> list[str]:
    """Load vocabulary from default file and extra files.

    Args:
        vocab_dir: Directory containing default.txt. Defaults to project's vocab/.
        extra_files: List of additional vocabulary file paths.

    Returns:
        Deduplicated list of vocabulary words.
    """
    words = []

    # Determine vocab directory
    if vocab_dir is None:
        vocab_dir = Path(__file__).parent.parent / "vocab"
    else:
        vocab_dir = Path(vocab_dir)

    # Load default vocabulary if exists
    default_file = vocab_dir / "default.txt"
    if default_file.exists():
        words.extend(parse_vocab_file(default_file))

    # Load extra files (with security validation)
    for filepath in extra_files or []:
        path = Path(filepath)
        if path.exists():
            if not _is_safe_vocab_path(path):
                # Silently skip forbidden paths to avoid information disclosure
                continue
            words.extend(parse_vocab_file(path))

    # Deduplicate while preserving order
    seen = set()
    unique_words = []
    for word in words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)

    return unique_words
