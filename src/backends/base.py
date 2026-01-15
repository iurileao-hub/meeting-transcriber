"""Base class for transcription backends."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptionResult:
    """Result from a transcription backend.

    Attributes:
        segments: List of transcribed segments with text, timestamps, and optional speaker.
        language: Detected or specified language code.
        metadata: Optional metadata about the transcription.
    """

    segments: list[dict[str, Any]]
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends.

    All backends must implement:
    - transcribe(): Process audio and return TranscriptionResult
    - supports_diarization: Property indicating if backend can identify speakers
    """

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        num_speakers: int | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file.
            language: Language code (e.g., 'pt', 'en') or None for auto-detect.
            num_speakers: Expected number of speakers (for diarization).
            **kwargs: Backend-specific options.

        Returns:
            TranscriptionResult with segments and metadata.
        """
        pass

    @property
    @abstractmethod
    def supports_diarization(self) -> bool:
        """Whether this backend supports speaker diarization."""
        pass

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        return self.__class__.__name__.replace("Backend", "")
