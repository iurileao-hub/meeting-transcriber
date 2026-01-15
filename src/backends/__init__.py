"""Transcription backends for Meeting Transcriber."""
from .base import TranscriptionBackend, TranscriptionResult
from .whisperx_backend import WhisperXBackend
from .mlx_backend import MLXBackend
from .granite_backend import GraniteBackend

__all__ = [
    "TranscriptionBackend",
    "TranscriptionResult",
    "WhisperXBackend",
    "MLXBackend",
    "GraniteBackend",
    "get_backend",
]

# Mode to backend mapping
_BACKENDS = {
    "fast": MLXBackend,
    "meeting": WhisperXBackend,
    "precise": GraniteBackend,
}


def get_backend(mode: str) -> TranscriptionBackend:
    """Get transcription backend for the specified mode.

    Args:
        mode: Transcription mode ('fast', 'meeting', 'precise').

    Returns:
        Configured TranscriptionBackend instance.

    Raises:
        ValueError: If mode is not recognized.
    """
    if mode not in _BACKENDS:
        valid_modes = ", ".join(_BACKENDS.keys())
        raise ValueError(f"Invalid mode: '{mode}'. Choose from: {valid_modes}")

    backend_class = _BACKENDS[mode]
    return backend_class()
