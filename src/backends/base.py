"""Base class for transcription backends."""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable


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
    @abstractmethod
    def total_stages(self) -> int:
        """Total number of progress stages for this backend."""
        pass

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        return self.__class__.__name__.replace("Backend", "")


@contextmanager
def pyannote_progress_hook(
    progress_callback: Callable[[str, float], None] | None,
    stage_name: str,
):
    """Patch Inference.slide to inject progress hook for pyannote operations.

    Monkey-patches pyannote's Inference.slide method to intercept the
    hook(completed, total) calls and forward them to progress_callback.

    Args:
        progress_callback: Callback(stage_name, percent) or None.
        stage_name: Stage name to report (e.g., 'vad', 'diarizing').
    """
    if not progress_callback:
        yield
        return

    from pyannote.audio.core.inference import Inference

    original_slide = Inference.slide

    def slide_with_hook(self, waveform, sample_rate, hook=None):
        def our_hook(completed, total):
            if total > 0:
                pct = (completed / total) * 99
                progress_callback(stage_name, min(pct, 99))
            if hook:
                hook(completed, total)

        return original_slide(self, waveform, sample_rate, hook=our_hook)

    Inference.slide = slide_with_hook
    try:
        yield
    finally:
        Inference.slide = original_slide
