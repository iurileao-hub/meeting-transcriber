"""MLX-Whisper backend for fast mode (Apple Silicon optimized)."""
from .base import TranscriptionBackend, TranscriptionResult


class MLXBackend(TranscriptionBackend):
    """MLX-Whisper backend optimized for Apple Silicon.

    Used for: --mode fast
    Features: Fast transcription, no diarization
    """

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        num_speakers: int | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio without diarization."""
        # TODO: Implement in Task 9
        raise NotImplementedError("MLX backend not yet implemented")

    @property
    def supports_diarization(self) -> bool:
        return False
