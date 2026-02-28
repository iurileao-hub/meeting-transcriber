"""
Testes unitários para o módulo de transcrição.

Uso:
    pytest tests/ -v
    pytest tests/ -v --tb=short  # Traceback curto
"""

import json
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Adiciona src ao path para importar o módulo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transcribe import (
    SUPPORTED_AUDIO_FORMATS,
    TranscriptionError,
    SuppressOutput,
    _allow_legacy_torch_load,
    _save_results,
    validate_audio_file,
    validate_output_path,
    format_timestamp,
    save_as_text,
    save_as_markdown,
    FORBIDDEN_OUTPUT_PATHS,
)


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
        assert ".opus" in SUPPORTED_AUDIO_FORMATS  # WhatsApp voice messages

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


class TestPathTraversalSecurity:
    """Security tests for path traversal prevention."""

    def test_valid_relative_path_allowed(self, tmp_path):
        """Valid relative paths should be allowed."""
        output_dir = tmp_path / "output"
        path = validate_output_path(str(output_dir))
        assert path.is_absolute()

    def test_valid_absolute_path_allowed(self, tmp_path):
        """Valid absolute paths should be allowed."""
        path = validate_output_path(str(tmp_path))
        assert path == tmp_path.resolve()

    def test_sys_blocked(self):
        """/sys directory should be blocked."""
        with pytest.raises(TranscriptionError) as exc_info:
            validate_output_path("/sys")
        assert "system directory" in str(exc_info.value).lower()

    def test_proc_blocked(self):
        """/proc directory should be blocked."""
        with pytest.raises(TranscriptionError) as exc_info:
            validate_output_path("/proc")
        assert "system directory" in str(exc_info.value).lower()

    def test_root_blocked(self):
        """/root directory should be blocked."""
        with pytest.raises(TranscriptionError) as exc_info:
            validate_output_path("/root")
        assert "system directory" in str(exc_info.value).lower()

    def test_boot_blocked(self):
        """/boot directory should be blocked."""
        with pytest.raises(TranscriptionError) as exc_info:
            validate_output_path("/boot")
        assert "system directory" in str(exc_info.value).lower()

    def test_root_subdir_blocked(self):
        """Subdirectories of /root should be blocked."""
        with pytest.raises(TranscriptionError) as exc_info:
            validate_output_path("/root/.ssh")
        assert "system directory" in str(exc_info.value).lower()

    def test_forbidden_paths_constant_populated(self):
        """FORBIDDEN_OUTPUT_PATHS should contain common system paths."""
        assert "/sys" in FORBIDDEN_OUTPUT_PATHS
        assert "/proc" in FORBIDDEN_OUTPUT_PATHS
        assert "/root" in FORBIDDEN_OUTPUT_PATHS
        assert "/boot" in FORBIDDEN_OUTPUT_PATHS

    def test_ssh_dir_blocked(self):
        """~/.ssh directory should be blocked."""
        ssh_path = str(Path.home() / ".ssh")
        assert ssh_path in FORBIDDEN_OUTPUT_PATHS
        with pytest.raises(TranscriptionError):
            validate_output_path(ssh_path)

    def test_aws_dir_blocked(self):
        """~/.aws directory should be blocked."""
        aws_path = str(Path.home() / ".aws")
        assert aws_path in FORBIDDEN_OUTPUT_PATHS
        with pytest.raises(TranscriptionError):
            validate_output_path(aws_path)

    def test_kube_dir_blocked(self):
        """~/.kube directory should be blocked."""
        kube_path = str(Path.home() / ".kube")
        assert kube_path in FORBIDDEN_OUTPUT_PATHS
        with pytest.raises(TranscriptionError):
            validate_output_path(kube_path)


class TestFormatTimestamp:
    """Tests for format_timestamp()."""

    def test_zero_seconds(self):
        """Zero seconds should return [00:00]."""
        assert format_timestamp(0) == "[00:00]"

    def test_minutes_and_seconds(self):
        """Should format minutes and seconds."""
        assert format_timestamp(125) == "[02:05]"

    def test_exactly_one_hour(self):
        """One hour should use HH:MM:SS format."""
        assert format_timestamp(3600) == "[01:00:00]"

    def test_hours_greater_than_zero(self):
        """Hours > 0 should use HH:MM:SS format."""
        assert format_timestamp(3661) == "[01:01:01]"

    def test_large_hours(self):
        """Large number of hours should be handled."""
        assert format_timestamp(36000) == "[10:00:00]"

    def test_just_under_one_hour(self):
        """59:59 should use MM:SS format."""
        assert format_timestamp(3599) == "[59:59]"


class TestSaveAsText:
    """Tests for save_as_text()."""

    def test_saves_segments_as_text(self, tmp_path):
        """Should save segments in text format."""
        result = {
            "segments": [
                {"start": 0, "speaker": "SPEAKER_00", "text": "Hello world"},
                {"start": 65, "speaker": "SPEAKER_01", "text": "Hi there"},
            ]
        }
        output_file = tmp_path / "test.txt"
        save_as_text(result, output_file)
        content = output_file.read_text()
        assert "[00:00] SPEAKER_00: Hello world" in content
        assert "[01:05] SPEAKER_01: Hi there" in content

    def test_empty_segments(self, tmp_path):
        """Should handle empty segments list."""
        result = {"segments": []}
        output_file = tmp_path / "test.txt"
        save_as_text(result, output_file)
        content = output_file.read_text()
        assert content == ""

    def test_missing_speaker_defaults_to_unknown(self, tmp_path):
        """Missing speaker should default to UNKNOWN."""
        result = {"segments": [{"start": 0, "text": "Hello"}]}
        output_file = tmp_path / "test.txt"
        save_as_text(result, output_file)
        content = output_file.read_text()
        assert "UNKNOWN" in content

    def test_segments_separated_by_double_newline(self, tmp_path):
        """Segments should be separated by double newlines."""
        result = {
            "segments": [
                {"start": 0, "speaker": "A", "text": "First"},
                {"start": 10, "speaker": "B", "text": "Second"},
            ]
        }
        output_file = tmp_path / "test.txt"
        save_as_text(result, output_file)
        content = output_file.read_text()
        assert "\n\n" in content


class TestSaveAsMarkdown:
    """Tests for save_as_markdown()."""

    def test_saves_header_and_segments(self, tmp_path):
        """Should save markdown with header and segments."""
        result = {
            "metadata": {
                "source_file": "test.wav",
                "transcribed_at": "2026-01-15T10:00:00",
                "language": "en",
                "model": "large-v3",
                "num_speakers": 2,
            },
            "segments": [
                {
                    "start": 0, "end": 5,
                    "speaker": "SPEAKER_00", "text": "Hello"
                },
            ],
        }
        output_file = tmp_path / "test.md"
        save_as_markdown(result, output_file)
        content = output_file.read_text()
        assert "# Transcrição: test.wav" in content
        assert "**SPEAKER_00**" in content
        assert "> Hello" in content

    def test_empty_segments(self, tmp_path):
        """Should handle empty segments list."""
        result = {"metadata": {}, "segments": []}
        output_file = tmp_path / "test.md"
        save_as_markdown(result, output_file)
        content = output_file.read_text()
        assert "---" in content


class TestSuppressOutput:
    """Tests for SuppressOutput context manager."""

    def test_restores_stdout_after_suppression(self):
        """Should restore sys.stdout after exiting context."""
        original_stdout = sys.stdout
        with SuppressOutput(suppress_stdout=True, suppress_stderr=False):
            assert sys.stdout is not original_stdout
        assert sys.stdout is original_stdout

    def test_restores_stderr_after_suppression(self):
        """Should restore sys.stderr after exiting context."""
        original_stderr = sys.stderr
        with SuppressOutput(suppress_stdout=False, suppress_stderr=True):
            assert sys.stderr is not original_stderr
        assert sys.stderr is original_stderr

    def test_restores_both_on_exception(self):
        """Should restore stdout and stderr even if exception occurs."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with pytest.raises(RuntimeError):
            with SuppressOutput(suppress_stdout=True, suppress_stderr=True):
                raise RuntimeError("test")
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr

    def test_no_suppression_leaves_streams_unchanged(self):
        """With both suppress flags False, streams should be unchanged."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with SuppressOutput(suppress_stdout=False, suppress_stderr=False):
            assert sys.stdout is original_stdout
            assert sys.stderr is original_stderr
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr

    def test_does_not_suppress_exceptions(self):
        """SuppressOutput should not suppress exceptions."""
        with pytest.raises(ValueError):
            with SuppressOutput(suppress_stdout=True, suppress_stderr=True):
                raise ValueError("should propagate")

    def test_uses_is_not_none_for_restore(self):
        """Should use 'is not None' check to handle falsy-but-valid streams."""
        # This tests the fix: using 'is not None' instead of truthy check
        original_stdout = sys.stdout
        with SuppressOutput(suppress_stdout=True, suppress_stderr=False):
            pass
        # If the code used truthy check and stdout was somehow falsy,
        # it would fail to restore. With 'is not None', it always restores.
        assert sys.stdout is original_stdout


class TestAllowLegacyTorchLoad:
    """Tests for _allow_legacy_torch_load context manager."""

    def test_patches_torch_load_inside_context(self):
        """torch.load should be patched inside the context."""
        import torch

        original = torch.load
        with _allow_legacy_torch_load():
            assert torch.load is not original
        assert torch.load is original

    def test_patches_serialization_load_inside_context(self):
        """torch.serialization.load should be patched inside context."""
        import torch.serialization as torch_ser

        original = torch_ser.load
        with _allow_legacy_torch_load():
            assert torch_ser.load is not original
        assert torch_ser.load is original

    def test_restores_on_exception(self):
        """Should restore original functions even if exception occurs."""
        import torch
        import torch.serialization as torch_ser

        original_load = torch.load
        original_ser = torch_ser.load
        with pytest.raises(RuntimeError):
            with _allow_legacy_torch_load():
                raise RuntimeError("test error")
        assert torch.load is original_load
        assert torch_ser.load is original_ser

    def test_patched_load_sets_weights_only_false(self):
        """Patched torch.load should force weights_only=False."""
        import torch

        calls = []

        def mock_load(*args, **kwargs):
            calls.append(kwargs)

        # Temporarily replace torch.load
        real_load = torch.load
        torch.load = mock_load
        try:
            with _allow_legacy_torch_load():
                torch.load("dummy_path")
        finally:
            torch.load = real_load

        assert len(calls) == 1
        assert calls[0]["weights_only"] is False


class TestSaveResults:
    """Tests for _save_results()."""

    def _make_result(self):
        """Create a sample result dict for testing."""
        return {
            "segments": [
                {
                    "start": 0, "end": 5,
                    "speaker": "SPEAKER_00",
                    "text": "Hello world",
                },
            ],
            "metadata": {
                "source_file": "test.wav",
                "transcribed_at": "2026-01-15T10:00:00",
                "language": "en",
                "model": "large-v3",
                "num_speakers": 1,
            },
        }

    def test_save_json_format(self, tmp_path):
        """Should save JSON file."""
        result = self._make_result()
        files = _save_results(result, str(tmp_path), "test", "json")
        assert len(files) == 1
        assert files[0].suffix == ".json"
        assert files[0].exists()
        data = json.loads(files[0].read_text())
        assert "segments" in data

    def test_save_txt_format(self, tmp_path):
        """Should save TXT file."""
        result = self._make_result()
        files = _save_results(result, str(tmp_path), "test", "txt")
        assert len(files) == 1
        assert files[0].suffix == ".txt"
        assert files[0].exists()
        content = files[0].read_text()
        assert "SPEAKER_00" in content

    def test_save_md_format(self, tmp_path):
        """Should save Markdown file."""
        result = self._make_result()
        files = _save_results(result, str(tmp_path), "test", "md")
        assert len(files) == 1
        assert files[0].suffix == ".md"
        assert files[0].exists()
        content = files[0].read_text()
        assert "**SPEAKER_00**" in content

    def test_save_all_formats(self, tmp_path):
        """Should save all three formats."""
        result = self._make_result()
        files = _save_results(result, str(tmp_path), "test", "all")
        assert len(files) == 3
        suffixes = {f.suffix for f in files}
        assert suffixes == {".json", ".txt", ".md"}

    def test_creates_output_directory(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        output_dir = tmp_path / "new" / "nested" / "dir"
        result = self._make_result()
        files = _save_results(result, str(output_dir), "test", "json")
        assert len(files) == 1
        assert output_dir.exists()

    def test_forbidden_path_raises_error(self):
        """Should raise TranscriptionError for forbidden paths."""
        result = self._make_result()
        with pytest.raises(TranscriptionError):
            _save_results(result, "/sys", "test", "json")
