# tests/test_mlx_backend.py
"""Tests for MLX backend."""
import pytest
from src.backends.mlx_backend import MLXBackend


class TestMLXBackendProperties:
    """Test backend properties."""

    def test_supports_diarization_is_false(self):
        backend = MLXBackend()
        assert backend.supports_diarization is False

    def test_name(self):
        backend = MLXBackend()
        assert backend.name == "MLX"


class TestMLXBackendConfig:
    """Test backend configuration."""

    def test_default_model_size(self):
        backend = MLXBackend()
        assert backend.model_size == "large-v3"

    def test_custom_model_size(self):
        backend = MLXBackend(model_size="small")
        assert backend.model_size == "small"


class TestMLXBackendAvailability:
    """Test MLX availability detection."""

    def test_is_available_returns_bool(self):
        backend = MLXBackend()
        assert isinstance(backend.is_available(), bool)
