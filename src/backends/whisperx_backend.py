"""WhisperX backend for meeting mode (with diarization)."""
from .base import TranscriptionBackend, TranscriptionResult


class WhisperXBackend(TranscriptionBackend):
    """WhisperX backend with integrated diarization.

    Used for: --mode meeting (default)
    Features: Transcription + word alignment + speaker diarization
    """

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        num_speakers: int | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio with speaker diarization."""
        # TODO: Implement in Task 3
        raise NotImplementedError("WhisperX backend not yet implemented")

    @property
    def supports_diarization(self) -> bool:
        return True
