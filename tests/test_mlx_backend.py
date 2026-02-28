# tests/test_mlx_backend.py
"""Tests for MLX backend."""
import pytest
from src.backends.mlx_backend import MLXBackend, MLX_MODEL_REPOS


class TestMLXBackendProperties:
    """Test backend properties."""

    def test_supports_diarization_is_false_by_default(self):
        backend = MLXBackend()
        assert backend.supports_diarization is False

    def test_supports_diarization_when_enabled(self):
        backend = MLXBackend(enable_diarization=True)
        assert backend.supports_diarization is True

    def test_name(self):
        backend = MLXBackend()
        assert backend.name == "MLX"

    def test_total_stages_without_diarization(self):
        backend = MLXBackend()
        assert backend.total_stages == 3

    def test_total_stages_with_diarization(self):
        backend = MLXBackend(enable_diarization=True)
        assert backend.total_stages == 4


class TestMLXBackendConfig:
    """Test backend configuration."""

    def test_default_model_size(self):
        backend = MLXBackend()
        assert backend.model_size == "large-v3"

    def test_custom_model_size(self):
        backend = MLXBackend(model_size="small")
        assert backend.model_size == "small"

    def test_default_device(self):
        backend = MLXBackend()
        assert backend.device == "cpu"

    def test_custom_device(self):
        backend = MLXBackend(device="mps")
        assert backend.device == "mps"

    def test_diarization_disabled_by_default(self):
        backend = MLXBackend()
        assert backend._enable_diarization is False

    def test_hf_token_stored(self):
        backend = MLXBackend(hf_token="test_token")
        assert backend._hf_token == "test_token"


class TestMLXBackendAvailability:
    """Test MLX availability detection."""

    def test_is_available_returns_bool(self):
        backend = MLXBackend()
        assert isinstance(backend.is_available(), bool)


class TestMLXModelRepos:
    """Test MLX model repository mapping."""

    def test_standard_models_exist(self):
        for name in ["tiny", "base", "small", "medium", "large-v3"]:
            assert name in MLX_MODEL_REPOS

    def test_extended_models_exist(self):
        for name in ["large-v3-turbo", "distil-large-v3", "large-v3-8bit"]:
            assert name in MLX_MODEL_REPOS

    def test_all_repos_are_mlx_community(self):
        for name, repo in MLX_MODEL_REPOS.items():
            assert repo.startswith("mlx-community/"), f"{name} repo doesn't start with mlx-community/"

    def test_get_model_repo_standard(self):
        backend = MLXBackend(model_size="small")
        assert backend._get_model_repo() == "mlx-community/whisper-small-mlx"

    def test_get_model_repo_extended(self):
        backend = MLXBackend(model_size="large-v3-turbo")
        assert backend._get_model_repo() == "mlx-community/whisper-large-v3-turbo"

    def test_get_model_repo_invalid_raises(self):
        backend = MLXBackend(model_size="nonexistent")
        with pytest.raises(ValueError, match="Unknown MLX model"):
            backend._get_model_repo()

    def test_get_model_repo_error_lists_valid_models(self):
        backend = MLXBackend(model_size="bad")
        with pytest.raises(ValueError, match="tiny"):
            backend._get_model_repo()


class TestMLXBackendHfToken:
    """Test HuggingFace token loading (via base class _load_hf_token)."""

    def test_load_hf_token_returns_stored_token(self):
        backend = MLXBackend(hf_token="hf_test123")
        assert backend._load_hf_token() == "hf_test123"

    def test_load_hf_token_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        from unittest.mock import patch
        backend = MLXBackend()
        # Patch load_dotenv on the base module (where _load_hf_token lives now)
        with patch("src.backends.base.load_dotenv"):
            with pytest.raises(ValueError, match="HuggingFace token not found"):
                backend._load_hf_token()


import sys
from unittest.mock import patch, MagicMock


class TestMLXGranularProgress:
    """Test granular progress in MLX transcription."""

    def test_transcribe_uses_tqdm_hook(self):
        """Transcription should use tqdm_progress_hook for granular progress."""
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {
            "segments": [{"start": 0, "end": 1, "text": "test"}],
            "language": "en",
        }

        reported_stages = []

        def callback(stage, pct):
            reported_stages.append(stage)

        backend = MLXBackend(model_size="base")

        with patch.dict(sys.modules, {"mlx_whisper": mock_mlx}), \
             patch.object(backend, "is_available", return_value=True):
            backend.transcribe("test.wav", progress_callback=callback)

        # Should have reported "transcribing" stage (not just 0 and 100)
        assert "transcribing" in reported_stages

    def test_transcribe_works_without_callback(self):
        """Transcription should work fine without progress callback."""
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {
            "segments": [{"start": 0, "end": 1, "text": "test"}],
            "language": "en",
        }

        backend = MLXBackend(model_size="base")

        with patch.dict(sys.modules, {"mlx_whisper": mock_mlx}), \
             patch.object(backend, "is_available", return_value=True):
            result = backend.transcribe("test.wav", progress_callback=None)

        assert len(result.segments) == 1
