"""Transcription backends for Meeting Transcriber."""
from .base import TranscriptionBackend, TranscriptionResult

__all__ = [
    "TranscriptionBackend",
    "TranscriptionResult",
    "get_backend",
]


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
        ValueError: If mode is not recognized or backend is unavailable.

    Example:
        >>> backend = get_backend("meeting", device="mps", model_size="large-v3")
        >>> backend = get_backend("precise", device="cpu", hf_token="hf_xxx")
    """
    _BACKEND_PARAMS = {
        "fast": {"model_size", "enable_diarization", "device", "hf_token"},
        "meeting": {"model_size", "device", "compute_type", "batch_size", "hf_token"},
        "precise": {"model_name", "device", "hf_token"},
    }

    if mode not in _BACKEND_PARAMS:
        valid_modes = ", ".join(_BACKEND_PARAMS.keys())
        raise ValueError(f"Invalid mode: '{mode}'. Choose from: {valid_modes}")

    # Lazy imports — only load the requested backend
    if mode == "fast":
        from .mlx_backend import MLXBackend

        backend_class = MLXBackend
    elif mode == "meeting":
        from .whisperx_backend import WhisperXBackend

        backend_class = WhisperXBackend
    elif mode == "precise":
        from .granite_backend import GraniteBackend

        backend_class = GraniteBackend

    valid_params = _BACKEND_PARAMS[mode]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    backend = backend_class(**filtered_kwargs)
    if not backend.is_available():
        raise ValueError(
            f"Backend '{mode}' is not available. "
            f"Install missing dependencies. See README for details."
        )
    return backend
