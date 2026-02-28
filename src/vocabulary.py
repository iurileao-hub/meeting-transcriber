"""Custom vocabulary support for transcription."""
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

MAX_VOCAB_FILE_SIZE = 1 * 1024 * 1024  # 1 MB


def _is_safe_vocab_path(path: Path, allowed_base: Path | None = None) -> bool:
    """Check if vocabulary path is within the allowed directory.

    Uses an allowlist approach: only paths within the project directory
    (or an explicitly provided base directory) are permitted.

    Args:
        path: Path to check.
        allowed_base: Base directory to allow. Defaults to project root.

    Returns:
        True if path is within the allowed directory, False otherwise.
    """
    resolved = path.resolve()
    if allowed_base is None:
        allowed_base = Path(__file__).parent.parent  # project root
    try:
        resolved.relative_to(allowed_base.resolve())
        return True
    except ValueError:
        return False


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

    Raises:
        ValueError: If the file exceeds MAX_VOCAB_FILE_SIZE.
    """
    if path.stat().st_size > MAX_VOCAB_FILE_SIZE:
        raise ValueError(
            f"Vocabulary file too large (>{MAX_VOCAB_FILE_SIZE // 1024}KB): {path}"
        )

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
        if not _is_safe_vocab_path(path):
            logger.warning(
                "Skipped vocabulary path outside project directory: %s", filepath
            )
            continue
        if path.exists():
            words.extend(parse_vocab_file(path))

    # Deduplicate while preserving order
    seen = set()
    unique_words = []
    for word in words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)

    return unique_words
