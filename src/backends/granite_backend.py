# src/backends/granite_backend.py
"""Granite backend for precise mode (high accuracy)."""
import gc
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from .base import TranscriptionBackend, TranscriptionResult


class GraniteBackend(TranscriptionBackend):
    """IBM Granite Speech backend for high-accuracy transcription.

    Used for: --mode precise
    Features: High accuracy transcription + separate pyannote diarization
    Requirements: transformers, accelerate, pyannote.audio

    Note:
        Diarization in precise mode uses sentence-level speaker assignment
        based on pyannote segments. This is an approximation since Granite
        doesn't provide word-level timestamps for precise speaker alignment.
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

        # Use device_map for efficient memory loading (avoids loading to CPU first)
        # This is the recommended approach per IBM Granite documentation
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=self.device,
        )

        if progress_callback:
            progress_callback("loading", 100)

        # Stage 2: Transcribe with Granite
        if progress_callback:
            progress_callback("transcribing", 0)

        # Load audio with torchaudio (Granite requires mono 16kHz)
        # normalize=True ensures audio values are in [-1, 1] range
        wav, sr = torchaudio.load(audio_path, normalize=True)

        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
            sr = 16000

        # Normalize audio to peak amplitude (avoid division by zero for silent audio)
        max_val = wav.abs().max()
        if max_val > 0:
            wav = wav / max_val
        # If max_val is 0, audio is silent - keep as-is (will produce empty transcription)

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
        # Since Granite doesn't provide word-level timestamps, we use
        # sentence-level approximation based on diarization segments
        segments = self._align_transcription_with_diarization(
            transcription=transcription.strip(),
            diarize_segments=diarize_segments,
            audio_duration=audio_duration,
        )

        # Count unique speakers
        speakers = {seg["speaker"] for seg in segments}

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

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using basic punctuation rules.

        Args:
            text: Full transcription text.

        Returns:
            List of sentences.
        """
        if not text:
            return []

        # Split on sentence-ending punctuation followed by space or end
        # Handles: . ! ? and common abbreviations
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter empty strings and strip whitespace
        return [s.strip() for s in sentences if s.strip()]

    def _align_transcription_with_diarization(
        self,
        transcription: str,
        diarize_segments,
        audio_duration: float,
    ) -> list[dict]:
        """Align transcription text with diarization segments.

        Since Granite doesn't provide word-level timestamps, we use a
        sentence-level approximation: split text into sentences, distribute
        them across the audio duration, and assign speakers based on
        which diarization segment has the most overlap.

        Args:
            transcription: Full transcription text.
            diarize_segments: Pyannote diarization result.
            audio_duration: Total audio duration in seconds.

        Returns:
            List of segment dicts with start, end, text, and speaker.
        """
        # Handle empty transcription
        if not transcription:
            return [{
                "start": 0.0,
                "end": audio_duration,
                "text": "",
                "speaker": "SPEAKER_00",
            }]

        # Split transcription into sentences
        sentences = self._split_into_sentences(transcription)
        if not sentences:
            return [{
                "start": 0.0,
                "end": audio_duration,
                "text": transcription,
                "speaker": "SPEAKER_00",
            }]

        # Build diarization timeline: [(start, end, speaker), ...]
        diarize_timeline = []
        for turn, _, speaker in diarize_segments.itertracks(yield_label=True):
            diarize_timeline.append((turn.start, turn.end, speaker))

        # If no diarization segments, assign all to SPEAKER_00
        if not diarize_timeline:
            return [{
                "start": 0.0,
                "end": audio_duration,
                "text": transcription,
                "speaker": "SPEAKER_00",
            }]

        # Sort by start time
        diarize_timeline.sort(key=lambda x: x[0])

        # Distribute sentences proportionally across audio duration
        # Estimate: each sentence takes proportional time based on character count
        total_chars = sum(len(s) for s in sentences)
        if total_chars == 0:
            total_chars = 1  # Avoid division by zero

        segments = []
        current_time = 0.0

        for sentence in sentences:
            # Estimate duration based on character proportion
            char_proportion = len(sentence) / total_chars
            sentence_duration = audio_duration * char_proportion

            start_time = current_time
            end_time = min(current_time + sentence_duration, audio_duration)

            # Find speaker with most overlap in this time range
            speaker = self._find_dominant_speaker(
                start_time, end_time, diarize_timeline
            )

            segments.append({
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "text": sentence,
                "speaker": speaker,
            })

            current_time = end_time

        return segments

    def _find_dominant_speaker(
        self,
        start: float,
        end: float,
        diarize_timeline: list[tuple[float, float, str]],
    ) -> str:
        """Find the speaker with most overlap in a time range.

        Args:
            start: Segment start time.
            end: Segment end time.
            diarize_timeline: List of (start, end, speaker) tuples.

        Returns:
            Speaker label with most overlap, or 'SPEAKER_00' if none.
        """
        speaker_overlap = {}

        for d_start, d_end, speaker in diarize_timeline:
            # Calculate overlap
            overlap_start = max(start, d_start)
            overlap_end = min(end, d_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > 0:
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0) + overlap

        if not speaker_overlap:
            return "SPEAKER_00"

        # Return speaker with maximum overlap
        return max(speaker_overlap, key=speaker_overlap.get)
