"""Text normalization for transcriptions."""
import re


def normalize_text(text: str) -> str:
    """Apply basic normalization to transcribed text.

    Normalizations applied:
    - Capitalize first character
    - Capitalize after sentence endings (. ! ?)

    Does NOT:
    - Modify words (no spell correction)
    - Change punctuation
    - Alter spacing

    Args:
        text: Raw transcribed text.

    Returns:
        Normalized text.
    """
    if not text:
        return text

    # Capitalize after sentence endings
    text = re.sub(
        r"([.!?])\s+(\w)",
        lambda m: m.group(1) + " " + m.group(2).upper(),
        text,
    )

    # Capitalize first character
    text = text[0].upper() + text[1:]

    return text
