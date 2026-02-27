"""Tests for transcription backends."""
import pytest
from unittest.mock import patch, MagicMock
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

            @property
            def total_stages(self) -> int:
                return 3

        with pytest.raises(TypeError):
            IncompleteBackend()

    def test_subclass_must_implement_supports_diarization(self):
        class IncompleteBackend(TranscriptionBackend):
            def transcribe(self, audio_path, **kwargs):
                pass

            @property
            def total_stages(self) -> int:
                return 3

        with pytest.raises(TypeError):
            IncompleteBackend()

    def test_subclass_must_implement_total_stages(self):
        class IncompleteBackend(TranscriptionBackend):
            def transcribe(self, audio_path, **kwargs):
                pass

            @property
            def supports_diarization(self) -> bool:
                return False

        with pytest.raises(TypeError):
            IncompleteBackend()


class TestBackendFactory:
    """Tests for get_backend factory function."""

    def test_get_backend_meeting_mode(self):
        from src.backends import get_backend

        backend = get_backend("meeting")
        assert backend.supports_diarization is True

    def test_get_backend_fast_mode(self):
        from src.backends import get_backend

        backend = get_backend("fast")
        assert backend.supports_diarization is False

    def test_get_backend_precise_mode(self):
        from src.backends import get_backend

        backend = get_backend("precise")
        assert backend.supports_diarization is True

    def test_get_backend_invalid_mode_raises(self):
        from src.backends import get_backend

        with pytest.raises(ValueError, match="Invalid mode"):
            get_backend("invalid")

    def test_get_backend_returns_backend_instance(self):
        from src.backends import TranscriptionBackend, get_backend

        backend = get_backend("meeting")
        assert isinstance(backend, TranscriptionBackend)


class TestBackendTotalStages:
    """Tests for total_stages property on each backend."""

    def test_whisperx_total_stages(self):
        from src.backends import get_backend

        backend = get_backend("meeting")
        assert backend.total_stages == 6

    def test_mlx_total_stages(self):
        from src.backends import get_backend

        backend = get_backend("fast")
        assert backend.total_stages == 3

    def test_granite_total_stages(self):
        from src.backends import get_backend

        backend = get_backend("precise")
        assert backend.total_stages == 4


class TestHfHubCompatPatch:
    """Tests for huggingface_hub compatibility patch."""

    def setup_method(self):
        """Reset the patch flag before each test."""
        import src.backends.base as base_mod

        base_mod._hf_compat_applied = False

    def test_patch_translates_use_auth_token_to_token(self):
        """Patch should translate use_auth_token kwarg to token."""
        import src.backends.base as base_mod

        # Create a mock hf_hub_download without use_auth_token param
        mock_original = MagicMock()
        mock_original.__signature__ = MagicMock()

        mock_module = MagicMock()
        mock_module.hf_hub_download = mock_original

        with patch.dict("sys.modules", {"huggingface_hub": mock_module}):
            with patch("inspect.signature") as mock_sig:
                # Simulate: use_auth_token NOT in signature (newer version)
                mock_params = MagicMock()
                mock_params.parameters = {}
                mock_sig.return_value = mock_params

                base_mod.patch_hf_hub_compat()

            # Now call the patched function with use_auth_token
            patched_fn = mock_module.hf_hub_download
            patched_fn("model_id", "filename", use_auth_token="my_token")

            # The original should have received 'token' instead
            mock_original.assert_called_once_with(
                "model_id", "filename", token="my_token"
            )

    def test_patch_skips_when_use_auth_token_supported(self):
        """Patch should not apply if use_auth_token is still supported."""
        import src.backends.base as base_mod

        mock_original = MagicMock()
        mock_module = MagicMock()
        mock_module.hf_hub_download = mock_original

        with patch.dict("sys.modules", {"huggingface_hub": mock_module}):
            with patch("inspect.signature") as mock_sig:
                # Simulate: use_auth_token IN signature (older version)
                mock_params = MagicMock()
                mock_params.parameters = {"use_auth_token": MagicMock()}
                mock_sig.return_value = mock_params

                base_mod.patch_hf_hub_compat()

            # Function should not have been replaced
            assert mock_module.hf_hub_download is mock_original

    def test_patch_only_applies_once(self):
        """Calling patch_hf_hub_compat multiple times should only patch once."""
        import src.backends.base as base_mod

        call_count = 0

        def counting_patch():
            nonlocal call_count
            call_count += 1

        with patch("inspect.signature") as mock_sig:
            mock_params = MagicMock()
            mock_params.parameters = {"use_auth_token": MagicMock()}
            mock_sig.return_value = mock_params

            with patch.dict("sys.modules", {"huggingface_hub": MagicMock()}):
                base_mod.patch_hf_hub_compat()
                base_mod.patch_hf_hub_compat()
                base_mod.patch_hf_hub_compat()

        # inspect.signature should only be called once (guard prevents re-entry)
        assert mock_sig.call_count == 1

    def test_patch_passes_through_other_kwargs(self):
        """Patch should pass through all other kwargs unchanged."""
        import src.backends.base as base_mod

        mock_original = MagicMock()
        mock_module = MagicMock()
        mock_module.hf_hub_download = mock_original

        with patch.dict("sys.modules", {"huggingface_hub": mock_module}):
            with patch("inspect.signature") as mock_sig:
                mock_params = MagicMock()
                mock_params.parameters = {}
                mock_sig.return_value = mock_params

                base_mod.patch_hf_hub_compat()

            patched_fn = mock_module.hf_hub_download
            patched_fn(
                "model_id",
                "filename",
                use_auth_token="my_token",
                revision="main",
                cache_dir="/tmp",
            )

            mock_original.assert_called_once_with(
                "model_id",
                "filename",
                token="my_token",
                revision="main",
                cache_dir="/tmp",
            )
