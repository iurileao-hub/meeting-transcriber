"""Tests for transcription backends."""
import pytest
from src.backends.base import TranscriptionBackend, TranscriptionResult


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_result_has_segments(self):
        result = TranscriptionResult(segments=[], language="en")
        assert result.segments == []

    def test_result_has_language(self):
        result = TranscriptionResult(segments=[], language="pt")
        assert result.language == "pt"

    def test_result_has_optional_metadata(self):
        result = TranscriptionResult(segments=[], language="en", metadata={"model": "test"})
        assert result.metadata["model"] == "test"


class TestTranscriptionBackend:
    """Tests for TranscriptionBackend ABC."""

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            TranscriptionBackend()

    def test_subclass_must_implement_transcribe(self):
        class IncompleteBackend(TranscriptionBackend):
            @property
            def supports_diarization(self) -> bool:
                return False

        with pytest.raises(TypeError):
            IncompleteBackend()

    def test_subclass_must_implement_supports_diarization(self):
        class IncompleteBackend(TranscriptionBackend):
            def transcribe(self, audio_path, **kwargs):
                pass

        with pytest.raises(TypeError):
            IncompleteBackend()
