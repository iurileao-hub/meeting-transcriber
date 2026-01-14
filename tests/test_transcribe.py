"""
Testes unitários para o módulo de transcrição.

Uso:
    pytest tests/ -v
    pytest tests/ -v --tb=short  # Traceback curto
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Adiciona src ao path para importar o módulo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transcribe import (
    SUPPORTED_AUDIO_FORMATS,
    TranscriptionError,
    get_batch_size,
    get_compute_type,
    load_hf_token,
    validate_audio_file,
)


class TestLoadHfToken:
    """Testes para a função load_hf_token()."""

    def test_load_token_from_env(self, monkeypatch):
        """Deve carregar token da variável de ambiente."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_token_123")
        assert load_hf_token() == "hf_test_token_123"

    def test_load_token_missing_raises_error(self, monkeypatch):
        """Deve levantar TranscriptionError se token não encontrado."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        # Mock load_dotenv para não carregar .env real
        with patch("transcribe.load_dotenv"):
            with pytest.raises(TranscriptionError) as exc_info:
                load_hf_token()
            assert "Token HuggingFace não encontrado" in str(exc_info.value)

    def test_error_message_contains_help_url(self, monkeypatch):
        """Mensagem de erro deve conter URL de ajuda."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with patch("transcribe.load_dotenv"):
            with pytest.raises(TranscriptionError) as exc_info:
                load_hf_token()
            assert "huggingface.co/settings/tokens" in str(exc_info.value)


class TestGetComputeType:
    """Testes para a função get_compute_type()."""

    def test_cpu_returns_int8(self):
        """CPU deve usar int8 para melhor performance."""
        assert get_compute_type("cpu") == "int8"

    def test_cuda_returns_float16(self):
        """CUDA deve usar float16."""
        assert get_compute_type("cuda") == "float16"

    def test_mps_returns_float16(self):
        """MPS (Apple Silicon) deve usar float16."""
        assert get_compute_type("mps") == "float16"


class TestGetBatchSize:
    """Testes para a função get_batch_size()."""

    def test_cpu_returns_conservative_size(self):
        """CPU deve usar batch size conservador."""
        assert get_batch_size("cpu", "large-v3") == 8
        assert get_batch_size("cpu", "small") == 8

    def test_gpu_small_models_use_larger_batch(self):
        """Modelos pequenos em GPU podem usar batches maiores."""
        assert get_batch_size("cuda", "tiny") == 32
        assert get_batch_size("cuda", "base") == 32
        assert get_batch_size("cuda", "small") == 32

    def test_gpu_large_models_use_moderate_batch(self):
        """Modelos grandes em GPU usam batch moderado."""
        assert get_batch_size("cuda", "medium") == 16
        assert get_batch_size("cuda", "large-v3") == 16


class TestValidateAudioFile:
    """Testes para a função validate_audio_file()."""

    def test_nonexistent_file_raises_error(self, tmp_path):
        """Deve levantar erro para arquivo inexistente."""
        fake_path = tmp_path / "nao_existe.wav"
        with pytest.raises(TranscriptionError) as exc_info:
            validate_audio_file(fake_path)
        assert "Arquivo não encontrado" in str(exc_info.value)

    def test_unsupported_format_raises_error(self, tmp_path):
        """Deve levantar erro para formato não suportado."""
        invalid_file = tmp_path / "teste.txt"
        invalid_file.touch()
        with pytest.raises(TranscriptionError) as exc_info:
            validate_audio_file(invalid_file)
        assert "Formato não suportado" in str(exc_info.value)
        assert ".txt" in str(exc_info.value)

    def test_valid_formats_pass(self, tmp_path):
        """Deve aceitar todos os formatos suportados."""
        for ext in SUPPORTED_AUDIO_FORMATS:
            valid_file = tmp_path / f"audio{ext}"
            valid_file.touch()
            # Não deve levantar exceção
            validate_audio_file(valid_file)

    def test_case_insensitive_format(self, tmp_path):
        """Deve aceitar formatos em maiúsculas."""
        upper_file = tmp_path / "audio.WAV"
        upper_file.touch()
        # Não deve levantar exceção
        validate_audio_file(upper_file)


class TestSupportedFormats:
    """Testes para os formatos suportados."""

    def test_common_formats_included(self):
        """Formatos comuns devem estar incluídos."""
        assert ".wav" in SUPPORTED_AUDIO_FORMATS
        assert ".mp3" in SUPPORTED_AUDIO_FORMATS
        assert ".m4a" in SUPPORTED_AUDIO_FORMATS
        assert ".flac" in SUPPORTED_AUDIO_FORMATS

    def test_no_video_formats(self):
        """Formatos de vídeo não devem estar incluídos."""
        assert ".mp4" not in SUPPORTED_AUDIO_FORMATS
        assert ".avi" not in SUPPORTED_AUDIO_FORMATS
        assert ".mkv" not in SUPPORTED_AUDIO_FORMATS


class TestTranscriptionError:
    """Testes para a classe TranscriptionError."""

    def test_is_exception(self):
        """Deve ser uma exceção."""
        assert issubclass(TranscriptionError, Exception)

    def test_can_be_raised_with_message(self):
        """Deve suportar mensagem customizada."""
        with pytest.raises(TranscriptionError) as exc_info:
            raise TranscriptionError("Mensagem de teste")
        assert "Mensagem de teste" in str(exc_info.value)
