# tests/test_granite_backend.py
"""Tests for Granite backend."""
import pytest
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


class TestGraniteBackendAvailability:
    """Test Granite availability detection."""

    def test_is_available_returns_bool(self):
        backend = GraniteBackend()
        assert isinstance(backend.is_available(), bool)
