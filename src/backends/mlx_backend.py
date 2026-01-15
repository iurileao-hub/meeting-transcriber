# src/backends/mlx_backend.py
"""MLX-Whisper backend for fast mode (Apple Silicon optimized)."""
from .base import TranscriptionBackend, TranscriptionResult


class MLXBackend(TranscriptionBackend):
    """MLX-Whisper backend optimized for Apple Silicon.

    Used for: --mode fast
    Features: Fast transcription (10-15x realtime), no diarization
    Requirements: Apple Silicon Mac, mlx-whisper package
    """

    def __init__(self, model_size: str = "large-v3"):
        """Initialize MLX backend.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3).
        """
        self.model_size = model_size

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

    @property
    def supports_diarization(self) -> bool:
        return False

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        num_speakers: int | None = None,
        progress_callback=None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio without diarization.

        Args:
            audio_path: Path to audio file.
            language: Language code or None for auto-detect.
            num_speakers: Ignored (no diarization support).
            progress_callback: Optional callback(stage, percent) for progress.
            **kwargs: Additional options.

        Returns:
            TranscriptionResult with segments (no speaker labels).

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

        # MLX model names use different format
        model_name = f"mlx-community/whisper-{self.model_size}-mlx"

        if progress_callback:
            progress_callback("loading", 50)

        if progress_callback:
            progress_callback("transcribing", 0)

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_name,
            language=language,
        )

        if progress_callback:
            progress_callback("transcribing", 100)

        # Convert to standard format
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
                # No speaker - MLX doesn't support diarization
            })

        return TranscriptionResult(
            segments=segments,
            language=result.get("language", language or "unknown"),
            metadata={
                "model": self.model_size,
                "backend": "mlx-whisper",
            },
        )
