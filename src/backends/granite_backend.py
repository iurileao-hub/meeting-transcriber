# src/backends/granite_backend.py
"""Granite backend for precise mode (high accuracy)."""
import gc
import os
from pathlib import Path

from dotenv import load_dotenv

from .base import TranscriptionBackend, TranscriptionResult


class GraniteBackend(TranscriptionBackend):
    """IBM Granite Speech backend for high-accuracy transcription.

    Used for: --mode precise
    Features: High accuracy transcription + separate pyannote diarization
    Requirements: transformers, accelerate, pyannote.audio
    """

    # Default Granite Speech model (requires ~16GB RAM)
    DEFAULT_MODEL = "ibm-granite/granite-speech-3.3-8b"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        hf_token: str | None = None,
    ):
        """Initialize Granite backend.

        Args:
            model_name: HuggingFace model name.
            device: Processing device (cpu, cuda, mps).
            hf_token: HuggingFace token for pyannote.

        Note:
            Granite 8B requires ~16GB RAM. For machines with less memory,
            use --mode meeting (WhisperX) instead.
        """
        self.model_name = model_name
        self.device = device
        self._hf_token = hf_token

    def is_available(self) -> bool:
        """Check if required packages are available.

        Returns:
            True if transformers can be imported.
        """
        try:
            import transformers
            return True
        except ImportError:
            return False

    def _load_hf_token(self) -> str:
        """Load HuggingFace token from environment."""
        if self._hf_token:
            return self._hf_token

        env_file = Path(__file__).parent.parent.parent / ".env"
        load_dotenv(env_file)

        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "HuggingFace token not found.\n"
                "Set HF_TOKEN in .env or pass hf_token parameter."
            )
        return token

    @property
    def supports_diarization(self) -> bool:
        return True  # Via separate pyannote pipeline

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
        """Transcribe audio with high accuracy.

        Args:
            audio_path: Path to audio file.
            language: Language code or None for auto-detect.
            num_speakers: Exact number of speakers.
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.
            progress_callback: Optional callback(stage, percent).
            **kwargs: Additional options.

        Returns:
            TranscriptionResult with segments and speaker labels.

        Raises:
            ImportError: If required packages not installed.
        """
        if not self.is_available():
            raise ImportError(
                "transformers not installed.\n"
                "Install with: pip install transformers accelerate torchaudio"
            )

        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        import torch
        import torchaudio

        # Stage 1: Load Granite model
        if progress_callback:
            progress_callback("loading", 0)

        processor = AutoProcessor.from_pretrained(self.model_name)
        tokenizer = processor.tokenizer

        # Use bfloat16 for better compatibility, float32 for CPU
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(self.device)

        if progress_callback:
            progress_callback("loading", 100)

        # Stage 2: Transcribe with Granite
        if progress_callback:
            progress_callback("transcribing", 0)

        # Load audio with torchaudio (Granite requires mono 16kHz)
        wav, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
            sr = 16000

        # Normalize audio
        wav = wav / wav.abs().max()

        audio_duration = wav.shape[1] / sr

        # Granite uses a chat-based interface with audio
        system_prompt = "You are a helpful AI assistant for transcription."
        user_prompt = "<|audio|>Transcribe the speech into written text."

        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # Process audio + text together
        model_inputs = processor(
            prompt, wav, device=self.device, return_tensors="pt"
        ).to(self.device)

        # Generate transcription
        with torch.no_grad():
            model_outputs = model.generate(
                **model_inputs,
                max_new_tokens=448,
                do_sample=False,
                num_beams=1,
            )

        # Extract only the new tokens (skip input tokens)
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:]

        transcription = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        if progress_callback:
            progress_callback("transcribing", 100)

        # Free Granite model memory
        del model, processor, tokenizer
        gc.collect()

        # Stage 3: Diarize with pyannote
        if progress_callback:
            progress_callback("diarizing", 0)

        hf_token = self._load_hf_token()
        from whisperx.diarize import DiarizationPipeline
        import whisperx

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

        diarize_segments = diarize_model(audio, **diarize_kwargs)

        if progress_callback:
            progress_callback("diarizing", 100)

        # Free diarization model
        del diarize_model
        gc.collect()

        # Combine transcription with diarization
        # For now, create a single segment (Granite doesn't provide timestamps)
        segments = [{
            "start": 0,
            "end": audio_duration,
            "text": transcription.strip(),
            "speaker": "SPEAKER_00",  # Basic assignment
        }]

        # Count speakers from diarization
        speakers = set()
        for seg in diarize_segments.itertracks(yield_label=True):
            speakers.add(seg[2])

        return TranscriptionResult(
            segments=segments,
            language=language or "en",  # Granite focuses on English
            metadata={
                "model": self.model_name,
                "device": self.device,
                "num_speakers": len(speakers),
                "backend": "granite",
            },
        )
