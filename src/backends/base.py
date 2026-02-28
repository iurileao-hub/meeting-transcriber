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


def get_default_device() -> str:
    """Auto-detect the best available processing device.

    Returns 'mps' on Apple Silicon, 'cuda' with NVIDIA GPUs,
    or 'cpu' as fallback.

    Returns:
        Device string: 'mps', 'cuda', or 'cpu'.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, AttributeError):
        pass
    return "cpu"


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


@contextmanager
def tqdm_progress_hook(
    progress_callback: Callable[[str, float], None] | None,
    stage_name: str,
):
    """Intercept tqdm progress bars to forward updates to progress_callback.

    Temporarily replaces tqdm.tqdm with a subclass that intercepts update()
    calls, converting frame-level progress into percentage-based callbacks.
    Used by MLX backend to capture mlx_whisper's internal progress bar.

    Args:
        progress_callback: Callback(stage_name, percent) or None.
        stage_name: Stage name to report (e.g., 'transcribing').
    """
    if not progress_callback:
        yield
        return

    import tqdm as tqdm_module

    OriginalTqdm = tqdm_module.tqdm

    class _ProgressTqdm(OriginalTqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._hook_n = 0

        def update(self, n=1):
            super().update(n)
            self._hook_n += n
            if self.total and self.total > 0:
                pct = (self._hook_n / self.total) * 99
                progress_callback(stage_name, min(pct, 99))

    tqdm_module.tqdm = _ProgressTqdm
    try:
        yield
    finally:
        tqdm_module.tqdm = OriginalTqdm


class ProgressStreamer:
    """Streamer for transformers model.generate() that reports progress.

    Implements the BaseStreamer interface (put/end) expected by
    transformers' generate() method. Counts generated tokens and
    reports progress as a percentage of max_tokens.

    Used by Granite backend for per-token transcription progress.

    Note:
        Standalone implementation compatible with transformers BaseStreamer.
        Does not inherit from BaseStreamer to avoid import dependency.
    """

    def __init__(
        self,
        max_tokens: int,
        progress_callback: Callable[[str, float], None],
        stage_name: str = "transcribing",
    ):
        """Initialize progress streamer.

        Args:
            max_tokens: Maximum tokens to generate (for percentage calculation).
            progress_callback: Callback(stage_name, percent).
            stage_name: Stage name to report.
        """
        self.max_tokens = max_tokens
        self.tokens_generated = 0
        self.progress_callback = progress_callback
        self.stage_name = stage_name

    def put(self, value) -> None:
        """Called by generate() for each new token batch.

        Args:
            value: Tensor of newly generated token IDs.
        """
        if hasattr(value, "shape") and len(value.shape) > 0:
            self.tokens_generated += value.shape[0]
        else:
            self.tokens_generated += 1
        pct = (self.tokens_generated / self.max_tokens) * 99
        self.progress_callback(self.stage_name, min(pct, 99))

    def end(self) -> None:
        """Called by generate() when generation is complete."""
        pass
