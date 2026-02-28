# src/backends/mlx_backend.py
"""MLX-Whisper backend for fast mode (Apple Silicon optimized)."""
import gc
import os
from pathlib import Path

from dotenv import load_dotenv

from .base import (
    TranscriptionBackend,
    TranscriptionResult,
    diarization_progress_hook,
    patch_hf_hub_compat,
    tqdm_progress_hook,
)

# Model name → HuggingFace repo mapping
MLX_MODEL_REPOS = {
    # Standard
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    # Extended
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
    "large-v3-8bit": "mlx-community/whisper-large-v3-mlx-8bit",
}


class MLXBackend(TranscriptionBackend):
    """MLX-Whisper backend optimized for Apple Silicon.

    Used for: --mode fast
    Features: Fast transcription (10-15x realtime), optional diarization
    Requirements: Apple Silicon Mac, mlx-whisper package
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        enable_diarization: bool = False,
        device: str = "cpu",
        hf_token: str | None = None,
    ):
        """Initialize MLX backend.

        Args:
            model_size: Whisper model size or MLX model name.
            enable_diarization: Enable speaker diarization via pyannote.
            device: Processing device for diarization (cpu, cuda, mps).
            hf_token: HuggingFace token for pyannote. Loaded from env if None.
        """
        self.model_size = model_size
        self._enable_diarization = enable_diarization
        self.device = device
        self._hf_token = hf_token

    def is_available(self) -> bool:
        """Check if MLX-Whisper is available.

        Returns:
            True if mlx_whisper can be imported.
        """
        try:
            import mlx_whisper
            return True
        except ImportError:
            return False

    def _load_hf_token(self) -> str:
        """Load HuggingFace token from environment or .env file."""
        if self._hf_token:
            return self._hf_token

        env_file = Path(__file__).parent.parent.parent / ".env"
        load_dotenv(env_file)

        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "HuggingFace token not found.\n"
                "Set HF_TOKEN in .env or pass hf_token parameter.\n"
                "Get your token at: https://huggingface.co/settings/tokens"
            )
        return token

    def _get_model_repo(self) -> str:
        """Get HuggingFace repo name for the model size.

        Returns:
            HuggingFace repo path.

        Raises:
            ValueError: If model size is not recognized.
        """
        repo = MLX_MODEL_REPOS.get(self.model_size)
        if not repo:
            valid = ", ".join(MLX_MODEL_REPOS.keys())
            raise ValueError(
                f"Unknown MLX model: '{self.model_size}'. Valid: {valid}"
            )
        return repo

    @property
    def supports_diarization(self) -> bool:
        return self._enable_diarization

    @property
    def total_stages(self) -> int:
        # With diarization: loading → transcribing → diarizing → saving
        # Without: loading → transcribing → saving
        return 4 if self._enable_diarization else 3

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        progress_callback=None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio with optional diarization.

        Args:
            audio_path: Path to audio file.
            language: Language code or None for auto-detect.
            num_speakers: Exact number of speakers (for diarization).
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.
            progress_callback: Optional callback(stage, percent) for progress.
            **kwargs: Additional options.

        Returns:
            TranscriptionResult with segments and optional speaker labels.

        Raises:
            ImportError: If mlx-whisper is not installed.
        """
        if not self.is_available():
            raise ImportError(
                "mlx-whisper not installed.\n"
                "Install with: pip install mlx-whisper\n"
                "Note: Requires Apple Silicon Mac."
            )

        import mlx_whisper

        if progress_callback:
            progress_callback("loading", 0)

        model_name = self._get_model_repo()

        if progress_callback:
            progress_callback("loading", 50)

        if progress_callback:
            progress_callback("transcribing", 0)

        with tqdm_progress_hook(progress_callback, "transcribing"):
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=model_name,
                language=language,
                word_timestamps=self._enable_diarization,
            )

        if progress_callback:
            progress_callback("transcribing", 100)

        if self._enable_diarization:
            result = self._run_diarization(
                audio_path, result, num_speakers, min_speakers,
                max_speakers, progress_callback,
            )

        # Convert to standard format
        segments = []
        for seg in result.get("segments", []):
            entry = {
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
            }
            if "speaker" in seg:
                entry["speaker"] = seg["speaker"]
            segments.append(entry)

        return TranscriptionResult(
            segments=segments,
            language=result.get("language", language or "unknown"),
            metadata={
                "model": self.model_size,
                "backend": "mlx-whisper",
                "diarization": self._enable_diarization,
            },
        )

    def _run_diarization(
        self,
        audio_path: str,
        result: dict,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
        progress_callback,
    ) -> dict:
        """Run pyannote diarization and assign speakers to segments.

        Args:
            audio_path: Path to audio file.
            result: MLX transcription result dict.
            num_speakers: Exact number of speakers.
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.
            progress_callback: Progress callback.

        Returns:
            Result dict with speaker labels assigned to segments.
        """
        if progress_callback:
            progress_callback("diarizing", 0)

        hf_token = self._load_hf_token()
        patch_hf_hub_compat()

        import whisperx
        from whisperx.diarize import DiarizationPipeline

        audio = whisperx.load_audio(audio_path)
        diarize_model = DiarizationPipeline(
            use_auth_token=hf_token,
            device=self.device,
        )

        diarize_kwargs = {}
        if num_speakers:
            diarize_kwargs["num_speakers"] = num_speakers
        if min_speakers:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers:
            diarize_kwargs["max_speakers"] = max_speakers

        with diarization_progress_hook(progress_callback, "diarizing"):
            diarize_segments = diarize_model(audio, **diarize_kwargs)

        # Combine transcription with diarization
        whisperx_result = {"segments": result.get("segments", [])}
        whisperx_result = whisperx.assign_word_speakers(
            diarize_segments, whisperx_result
        )
        result["segments"] = whisperx_result["segments"]

        del diarize_model
        gc.collect()

        if progress_callback:
            progress_callback("diarizing", 100)

        return result
