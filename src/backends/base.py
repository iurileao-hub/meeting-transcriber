"""Base class for transcription backends."""
import functools
import inspect
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


_hf_compat_applied = False


def patch_hf_hub_compat() -> None:
    """Patch hf_hub_download to accept deprecated use_auth_token parameter.

    Newer versions of huggingface_hub removed the use_auth_token parameter
    in favor of 'token'. Since pyannote.audio 3.4.0 still uses the old
    parameter throughout its codebase, this patch translates it transparently.

    Safe to call multiple times — only applies once.
    """
    global _hf_compat_applied
    if _hf_compat_applied:
        return

    import huggingface_hub

    original = huggingface_hub.hf_hub_download

    # Check if patch is needed (use_auth_token still accepted)
    sig = inspect.signature(original)
    if "use_auth_token" in sig.parameters:
        _hf_compat_applied = True
        return

    @functools.wraps(original)
    def _patched_hf_hub_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return original(*args, **kwargs)

    # Patch the source module
    huggingface_hub.hf_hub_download = _patched_hf_hub_download

    # Patch already-imported references in pyannote modules
    for mod_name in (
        "pyannote.audio.core.pipeline",
        "pyannote.audio.core.model",
    ):
        try:
            import sys

            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, "hf_hub_download"):
                mod.hf_hub_download = _patched_hf_hub_download
        except Exception:
            pass

    _hf_compat_applied = True


@contextmanager
def diarization_progress_hook(
    progress_callback: Callable[[str, float], None] | None,
    stage_name: str = "diarizing",
):
    """Inject native pyannote hook into SpeakerDiarization.apply for granular progress.

    Patches SpeakerDiarization.apply to pass a hook that maps pyannote's internal
    sub-stages (segmentation, embeddings, discrete_diarization) into a single
    0-99% progress range with weighted distribution.

    Args:
        progress_callback: Callback(stage_name, percent) or None.
        stage_name: Stage name to report (e.g., 'diarizing').
    """
    if not progress_callback:
        yield
        return

    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

    original_apply = SpeakerDiarization.apply

    # Sub-stage weight distribution (approximate time split)
    STAGE_RANGES = {
        "segmentation": (0, 45),
        "embeddings": (45, 85),
        "discrete_diarization": (85, 99),
    }

    def apply_with_hook(self, file, **kwargs):
        def our_hook(step_name, step_artifact, file=None, completed=0, total=0):
            rng = STAGE_RANGES.get(step_name)
            if rng and total > 0:
                start_pct, end_pct = rng
                pct = start_pct + (completed / total) * (end_pct - start_pct)
                progress_callback(stage_name, min(pct, 99))

        kwargs["hook"] = our_hook
        return original_apply(self, file, **kwargs)

    SpeakerDiarization.apply = apply_with_hook
    try:
        yield
    finally:
        SpeakerDiarization.apply = original_apply


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
