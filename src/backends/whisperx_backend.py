"""WhisperX backend for meeting mode (with diarization)."""
import gc
import os
from pathlib import Path

from dotenv import load_dotenv

from .base import TranscriptionBackend, TranscriptionResult


class WhisperXBackend(TranscriptionBackend):
    """WhisperX backend with integrated diarization.

    Used for: --mode meeting (default)
    Features: Transcription + word alignment + speaker diarization
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str | None = None,
        batch_size: int | None = None,
        hf_token: str | None = None,
    ):
        """Initialize WhisperX backend.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3).
            device: Processing device (cpu, cuda, mps).
            compute_type: Compute type (int8, float16). Auto-detected if None.
            batch_size: Batch size for processing. Auto-detected if None.
            hf_token: HuggingFace token for pyannote. Loaded from env if None.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type or self._get_compute_type()
        self.batch_size = batch_size or self._get_batch_size()
        self._hf_token = hf_token

    def _get_compute_type(self) -> str:
        """Get optimal compute type for device."""
        return "int8" if self.device == "cpu" else "float16"

    def _get_batch_size(self) -> int:
        """Get optimal batch size for device and model."""
        if self.device == "cpu":
            return 8
        if self.model_size in ("tiny", "base", "small"):
            return 32
        return 16

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

    @property
    def supports_diarization(self) -> bool:
        return True

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
        """Transcribe audio with speaker diarization.

        Args:
            audio_path: Path to audio file.
            language: Language code or None for auto-detect.
            num_speakers: Exact number of speakers (if known).
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.
            progress_callback: Optional callback(stage, percent) for progress.
            **kwargs: Additional options.

        Returns:
            TranscriptionResult with segments and speaker labels.
        """
        import whisperx
        from whisperx.diarize import DiarizationPipeline

        # Stage 1: Load model
        if progress_callback:
            progress_callback("loading", 0)

        model = whisperx.load_model(
            self.model_size,
            self.device,
            compute_type=self.compute_type,
            language=language,
        )

        if progress_callback:
            progress_callback("loading", 100)

        # Stage 2: Transcribe
        if progress_callback:
            progress_callback("transcribing", 0)

        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=self.batch_size)
        detected_language = result.get("language", language or "unknown")

        if progress_callback:
            progress_callback("transcribing", 100)

        # Free memory
        del model
        gc.collect()

        # Stage 3: Align
        if progress_callback:
            progress_callback("aligning", 0)

        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=self.device,
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        if progress_callback:
            progress_callback("aligning", 100)

        # Free memory
        del model_a
        gc.collect()

        # Stage 4: Diarize
        if progress_callback:
            progress_callback("diarizing", 0)

        hf_token = self._load_hf_token()

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

        diarize_segments = diarize_model(audio, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        if progress_callback:
            progress_callback("diarizing", 100)

        # Free memory
        del diarize_model
        gc.collect()

        # Count speakers
        speakers = {
            seg.get("speaker")
            for seg in result.get("segments", [])
            if "speaker" in seg
        }

        return TranscriptionResult(
            segments=result.get("segments", []),
            language=detected_language,
            metadata={
                "model": self.model_size,
                "device": self.device,
                "compute_type": self.compute_type,
                "num_speakers": len(speakers),
                "backend": "whisperx",
            },
        )
