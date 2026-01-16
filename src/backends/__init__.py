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


def get_backend(mode: str, **kwargs) -> TranscriptionBackend:
    """Get transcription backend for the specified mode.

    Args:
        mode: Transcription mode ('fast', 'meeting', 'precise').
        **kwargs: Backend-specific configuration. Common options:
            - device: Processing device ('cpu', 'cuda', 'mps')
            - model_size: Whisper model size (for WhisperX/MLX)
            - hf_token: HuggingFace token (for pyannote diarization)

    Returns:
        Configured TranscriptionBackend instance.

    Raises:
        ValueError: If mode is not recognized.

    Example:
        >>> backend = get_backend("meeting", device="mps", model_size="large-v3")
        >>> backend = get_backend("precise", device="cpu", hf_token="hf_xxx")
    """
    if mode not in _BACKENDS:
        valid_modes = ", ".join(_BACKENDS.keys())
        raise ValueError(f"Invalid mode: '{mode}'. Choose from: {valid_modes}")

    backend_class = _BACKENDS[mode]

    # Filter kwargs to only include parameters accepted by this backend's __init__
    # This prevents TypeError from unexpected keyword arguments
    import inspect
    sig = inspect.signature(backend_class.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return backend_class(**filtered_kwargs)
