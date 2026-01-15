"""Granite backend for precise mode (high accuracy)."""
from .base import TranscriptionBackend, TranscriptionResult


class GraniteBackend(TranscriptionBackend):
    """IBM Granite Speech backend for high-accuracy transcription.

    Used for: --mode precise
    Features: High accuracy + separate pyannote diarization
    """

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        num_speakers: int | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio with high accuracy."""
        # TODO: Implement in Task 10
        raise NotImplementedError("Granite backend not yet implemented")

    @property
    def supports_diarization(self) -> bool:
        return True  # Via separate pyannote pipeline
