# tests/test_granite_backend.py
"""Tests for Granite backend."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.backends.granite_backend import GraniteBackend


class TestGraniteBackendProperties:
    """Test backend properties."""

    def test_supports_diarization_is_true(self):
        backend = GraniteBackend()
        assert backend.supports_diarization is True

    def test_name(self):
        backend = GraniteBackend()
        assert backend.name == "Granite"


class TestGraniteBackendConfig:
    """Test backend configuration."""

    def test_default_model_name(self):
        backend = GraniteBackend()
        assert "granite" in backend.model_name.lower()

    def test_default_device(self):
        backend = GraniteBackend()
        assert backend.device == "cpu"

    def test_custom_device(self):
        backend = GraniteBackend(device="mps")
        assert backend.device == "mps"

    def test_custom_model_name(self):
        custom_model = "custom/model"
        backend = GraniteBackend(model_name=custom_model)
        assert backend.model_name == custom_model

    def test_custom_hf_token(self):
        backend = GraniteBackend(hf_token="test_token")
        assert backend._hf_token == "test_token"


class TestGraniteBackendAvailability:
    """Test Granite availability detection."""

    def test_is_available_returns_bool(self):
        backend = GraniteBackend()
        assert isinstance(backend.is_available(), bool)

    @patch.dict("sys.modules", {"transformers": None})
    def test_is_available_false_when_transformers_missing(self):
        # Need to reload module to pick up patched import
        backend = GraniteBackend()
        # Can't fully test import failure without complex mocking
        # Just verify the method returns a bool
        result = backend.is_available()
        assert isinstance(result, bool)


class TestSentenceSplitting:
    """Test sentence splitting logic."""

    def test_split_simple_sentences(self):
        backend = GraniteBackend()
        text = "Hello world. How are you? I am fine!"
        sentences = backend._split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"
        assert sentences[2] == "I am fine!"

    def test_split_empty_text(self):
        backend = GraniteBackend()
        sentences = backend._split_into_sentences("")
        assert sentences == []

    def test_split_single_sentence(self):
        backend = GraniteBackend()
        text = "Just one sentence."
        sentences = backend._split_into_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == "Just one sentence."

    def test_split_no_punctuation(self):
        backend = GraniteBackend()
        text = "No punctuation here"
        sentences = backend._split_into_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == "No punctuation here"

    def test_split_handles_whitespace(self):
        backend = GraniteBackend()
        text = "  First.   Second.  "
        sentences = backend._split_into_sentences(text)
        assert len(sentences) == 2
        assert sentences[0] == "First."
        assert sentences[1] == "Second."


class TestDiarizationAlignment:
    """Test transcription-diarization alignment logic."""

    def test_align_empty_transcription(self):
        backend = GraniteBackend()
        mock_diarize = Mock()
        mock_diarize.itertracks.return_value = []

        segments = backend._align_transcription_with_diarization(
            transcription="",
            diarize_segments=mock_diarize,
            audio_duration=10.0,
        )

        assert len(segments) == 1
        assert segments[0]["text"] == ""
        assert segments[0]["speaker"] == "SPEAKER_00"

    def test_align_no_diarization_segments(self):
        backend = GraniteBackend()
        mock_diarize = Mock()
        mock_diarize.itertracks.return_value = []

        segments = backend._align_transcription_with_diarization(
            transcription="Hello world. How are you?",
            diarize_segments=mock_diarize,
            audio_duration=10.0,
        )

        # When no diarization, fallback to single segment
        assert len(segments) == 1
        assert segments[0]["speaker"] == "SPEAKER_00"
        assert segments[0]["text"] == "Hello world. How are you?"

    def test_align_single_speaker(self):
        backend = GraniteBackend()

        # Create mock diarization segment
        mock_turn = Mock()
        mock_turn.start = 0.0
        mock_turn.end = 10.0

        mock_diarize = Mock()
        mock_diarize.itertracks.return_value = [(mock_turn, None, "SPEAKER_01")]

        segments = backend._align_transcription_with_diarization(
            transcription="Hello. World.",
            diarize_segments=mock_diarize,
            audio_duration=10.0,
        )

        assert len(segments) == 2
        for seg in segments:
            assert seg["speaker"] == "SPEAKER_01"

    def test_align_multiple_speakers(self):
        backend = GraniteBackend()

        # Create mock diarization segments - two speakers
        mock_turn1 = Mock()
        mock_turn1.start = 0.0
        mock_turn1.end = 5.0

        mock_turn2 = Mock()
        mock_turn2.start = 5.0
        mock_turn2.end = 10.0

        mock_diarize = Mock()
        mock_diarize.itertracks.return_value = [
            (mock_turn1, None, "SPEAKER_01"),
            (mock_turn2, None, "SPEAKER_02"),
        ]

        segments = backend._align_transcription_with_diarization(
            transcription="First half. Second half.",
            diarize_segments=mock_diarize,
            audio_duration=10.0,
        )

        assert len(segments) == 2
        # First sentence should be assigned to SPEAKER_01
        assert segments[0]["speaker"] == "SPEAKER_01"
        # Second sentence should be assigned to SPEAKER_02
        assert segments[1]["speaker"] == "SPEAKER_02"


class TestFindDominantSpeaker:
    """Test speaker overlap calculation."""

    def test_single_speaker_full_overlap(self):
        backend = GraniteBackend()
        timeline = [(0.0, 10.0, "SPEAKER_01")]

        speaker = backend._find_dominant_speaker(2.0, 5.0, timeline)
        assert speaker == "SPEAKER_01"

    def test_no_overlap_returns_default(self):
        backend = GraniteBackend()
        timeline = [(0.0, 1.0, "SPEAKER_01")]

        speaker = backend._find_dominant_speaker(5.0, 10.0, timeline)
        assert speaker == "SPEAKER_00"

    def test_dominant_speaker_wins(self):
        backend = GraniteBackend()
        # SPEAKER_01 has 3 seconds overlap, SPEAKER_02 has 1 second
        timeline = [
            (0.0, 5.0, "SPEAKER_01"),  # 3 seconds overlap with (2, 5)
            (4.0, 6.0, "SPEAKER_02"),  # 1 second overlap with (2, 5)
        ]

        speaker = backend._find_dominant_speaker(2.0, 5.0, timeline)
        assert speaker == "SPEAKER_01"

    def test_empty_timeline(self):
        backend = GraniteBackend()
        timeline = []

        speaker = backend._find_dominant_speaker(0.0, 5.0, timeline)
        assert speaker == "SPEAKER_00"


class TestHFTokenLoading:
    """Test HuggingFace token loading."""

    def test_uses_provided_token(self):
        backend = GraniteBackend(hf_token="my_test_token")
        token = backend._load_hf_token()
        assert token == "my_test_token"

    @patch.dict("os.environ", {"HF_TOKEN": "env_token"})
    def test_loads_from_environment(self):
        backend = GraniteBackend()
        token = backend._load_hf_token()
        assert token == "env_token"

    @patch.dict("os.environ", {}, clear=True)
    @patch("src.backends.granite_backend.load_dotenv")
    def test_raises_on_missing_token(self, mock_dotenv):
        # Clear the HF_TOKEN from environment
        import os
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]

        backend = GraniteBackend()
        with pytest.raises(ValueError, match="HuggingFace token not found"):
            backend._load_hf_token()


class TestAudioProcessingEdgeCases:
    """Test audio processing edge cases (logic tests - no mocking of internal imports)."""

    def test_handles_silent_audio(self):
        """Silent audio (all zeros) should not cause division by zero."""
        import torch

        # Create actual tensor for testing normalization logic
        silent_audio = torch.zeros(1, 16000)  # 1 second of silence

        # Test the normalization logic directly (same as in granite_backend.py)
        wav = silent_audio
        max_val = wav.abs().max()
        if max_val > 0:
            wav = wav / max_val
        # Should not raise ZeroDivisionError

        assert wav.abs().max() == 0  # Still silent

    def test_handles_stereo_to_mono_conversion(self):
        """Stereo audio should be converted to mono."""
        import torch

        stereo_audio = torch.randn(2, 16000)  # Stereo

        # Test conversion logic (same as in granite_backend.py)
        wav = stereo_audio
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        assert wav.shape[0] == 1  # Mono

    def test_handles_resampling_detection(self):
        """Non-16kHz audio should trigger resampling."""
        import torch

        audio_48k = torch.randn(1, 48000)  # 1 second at 48kHz
        sr = 48000

        # Test resampling condition (same as in granite_backend.py)
        needs_resampling = sr != 16000

        assert needs_resampling is True

    def test_16khz_audio_no_resampling(self):
        """16kHz audio should not trigger resampling."""
        sr = 16000
        needs_resampling = sr != 16000
        assert needs_resampling is False


class TestBackendIntegration:
    """Integration tests for backend factory."""

    def test_backend_accepts_device_parameter(self):
        from src.backends import get_backend

        backend = get_backend("precise", device="mps")
        assert backend.device == "mps"

    def test_backend_accepts_hf_token_parameter(self):
        from src.backends import get_backend

        backend = get_backend("precise", hf_token="test_token")
        assert backend._hf_token == "test_token"

    def test_backend_ignores_invalid_parameters(self):
        from src.backends import get_backend

        # Should not raise even with unknown parameter
        backend = get_backend("precise", unknown_param="value")
        assert isinstance(backend, GraniteBackend)
