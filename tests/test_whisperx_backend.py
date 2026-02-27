"""Tests for WhisperX backend."""
import builtins
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.backends.whisperx_backend import WhisperXBackend


class TestWhisperXBackendProperties:
    """Test backend properties."""

    def test_supports_diarization(self):
        backend = WhisperXBackend()
        assert backend.supports_diarization is True

    def test_name(self):
        backend = WhisperXBackend()
        assert backend.name == "WhisperX"


class TestWhisperXBackendConfig:
    """Test backend configuration."""

    def test_default_model_size(self):
        backend = WhisperXBackend()
        assert backend.model_size == "large-v3"

    def test_custom_model_size(self):
        backend = WhisperXBackend(model_size="small")
        assert backend.model_size == "small"

    def test_default_device(self):
        backend = WhisperXBackend()
        assert backend.device == "cpu"

    def test_custom_device(self):
        backend = WhisperXBackend(device="cuda")
        assert backend.device == "cuda"

    def test_default_compute_type_cpu(self):
        backend = WhisperXBackend(device="cpu")
        assert backend.compute_type == "int8"

    def test_default_compute_type_cuda(self):
        backend = WhisperXBackend(device="cuda")
        assert backend.compute_type == "float16"

    def test_custom_compute_type(self):
        backend = WhisperXBackend(compute_type="float32")
        assert backend.compute_type == "float32"

    def test_default_batch_size_cpu(self):
        backend = WhisperXBackend(device="cpu")
        assert backend.batch_size == 8

    def test_default_batch_size_gpu_small_model(self):
        backend = WhisperXBackend(device="cuda", model_size="small")
        assert backend.batch_size == 32

    def test_default_batch_size_gpu_large_model(self):
        backend = WhisperXBackend(device="cuda", model_size="large-v3")
        assert backend.batch_size == 16

    def test_custom_batch_size(self):
        backend = WhisperXBackend(batch_size=4)
        assert backend.batch_size == 4


class TestWhisperXBackendHFToken:
    """Test HuggingFace token handling."""

    def test_custom_hf_token(self):
        backend = WhisperXBackend(hf_token="test_token")
        assert backend._load_hf_token() == "test_token"

    @patch.dict("os.environ", {"HF_TOKEN": "env_token"}, clear=True)
    def test_hf_token_from_env(self):
        backend = WhisperXBackend()
        assert backend._load_hf_token() == "env_token"

    @patch.dict("os.environ", {}, clear=True)
    @patch("src.backends.whisperx_backend.load_dotenv")
    def test_hf_token_missing_raises(self, mock_dotenv):
        backend = WhisperXBackend()
        with pytest.raises(ValueError, match="HuggingFace token not found"):
            backend._load_hf_token()


class TestWhisperXBackendTranscribe:
    """Test transcription functionality with mocked whisperx."""

    def test_transcribe_returns_result(self):
        """Test that transcribe returns TranscriptionResult."""
        # Create a mock whisperx module
        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"text": "Hello", "start": 0, "end": 1}],
            "language": "en",
        }
        mock_whisperx.load_model.return_value = mock_model
        mock_whisperx.load_audio.return_value = "audio_data"
        mock_whisperx.load_align_model.return_value = (MagicMock(), {})
        mock_whisperx.align.return_value = {
            "segments": [{"text": "Hello", "start": 0, "end": 1}]
        }
        mock_whisperx.assign_word_speakers.return_value = {
            "segments": [
                {"text": "Hello", "start": 0, "end": 1, "speaker": "SPEAKER_00"}
            ]
        }

        # Mock diarization
        mock_diarize_module = MagicMock()
        mock_diarize = MagicMock()
        mock_diarize.return_value = {"segments": []}
        mock_diarize_module.DiarizationPipeline.return_value = mock_diarize

        with patch.dict(
            sys.modules,
            {
                "whisperx": mock_whisperx,
                "whisperx.diarize": mock_diarize_module,
            },
        ):
            backend = WhisperXBackend(hf_token="test_token")
            result = backend.transcribe("test.wav")

            assert result.language == "en"
            assert len(result.segments) == 1
            assert result.segments[0]["speaker"] == "SPEAKER_00"
            assert result.metadata["backend"] == "whisperx"

    def test_transcribe_with_progress_callback(self):
        """Test that progress callback is called at each stage."""
        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [],
            "language": "en",
        }
        mock_whisperx.load_model.return_value = mock_model
        mock_whisperx.load_audio.return_value = "audio_data"
        mock_whisperx.load_align_model.return_value = (MagicMock(), {})
        mock_whisperx.align.return_value = {"segments": []}
        mock_whisperx.assign_word_speakers.return_value = {"segments": []}

        mock_diarize_module = MagicMock()
        mock_diarize = MagicMock()
        mock_diarize.return_value = {"segments": []}
        mock_diarize_module.DiarizationPipeline.return_value = mock_diarize

        with patch.dict(
            sys.modules,
            {
                "whisperx": mock_whisperx,
                "whisperx.diarize": mock_diarize_module,
            },
        ):
            callback = MagicMock()
            backend = WhisperXBackend(hf_token="test_token")
            backend.transcribe("test.wav", progress_callback=callback)

            # Verify callback was called for each stage
            stages = [call[0][0] for call in callback.call_args_list]
            assert "loading" in stages
            assert "transcribing" in stages
            assert "aligning" in stages
            assert "diarizing" in stages

    def test_transcribe_with_num_speakers(self):
        """Test that num_speakers is passed to diarization."""
        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": [], "language": "en"}
        mock_whisperx.load_model.return_value = mock_model
        mock_whisperx.load_audio.return_value = "audio_data"
        mock_whisperx.load_align_model.return_value = (MagicMock(), {})
        mock_whisperx.align.return_value = {"segments": []}
        mock_whisperx.assign_word_speakers.return_value = {"segments": []}

        mock_diarize_module = MagicMock()
        mock_diarize = MagicMock()
        mock_diarize.return_value = {"segments": []}
        mock_diarize_module.DiarizationPipeline.return_value = mock_diarize

        with patch.dict(
            sys.modules,
            {
                "whisperx": mock_whisperx,
                "whisperx.diarize": mock_diarize_module,
            },
        ):
            backend = WhisperXBackend(hf_token="test_token")
            backend.transcribe("test.wav", num_speakers=3)

            # Verify num_speakers was passed
            mock_diarize.assert_called_once()
            call_kwargs = mock_diarize.call_args[1]
            assert call_kwargs.get("num_speakers") == 3

    def test_transcribe_with_language(self):
        """Test that language is passed to model."""
        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": [], "language": "pt"}
        mock_whisperx.load_model.return_value = mock_model
        mock_whisperx.load_audio.return_value = "audio_data"
        mock_whisperx.load_align_model.return_value = (MagicMock(), {})
        mock_whisperx.align.return_value = {"segments": []}
        mock_whisperx.assign_word_speakers.return_value = {"segments": []}

        mock_diarize_module = MagicMock()
        mock_diarize = MagicMock()
        mock_diarize.return_value = {"segments": []}
        mock_diarize_module.DiarizationPipeline.return_value = mock_diarize

        with patch.dict(
            sys.modules,
            {
                "whisperx": mock_whisperx,
                "whisperx.diarize": mock_diarize_module,
            },
        ):
            backend = WhisperXBackend(hf_token="test_token")
            result = backend.transcribe("test.wav", language="pt")

            # Verify language was passed to load_model
            mock_whisperx.load_model.assert_called_once()
            call_kwargs = mock_whisperx.load_model.call_args[1]
            assert call_kwargs.get("language") == "pt"
            assert result.language == "pt"


class TestWhisperXProgressInterception:
    """Test print_progress interception for granular transcription progress."""

    def _make_mocks(self, print_side_effect=None):
        """Create standard whisperx mocks. Optionally inject side_effect on transcribe."""
        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"text": "Hello", "start": 0, "end": 1}],
            "language": "en",
        }
        if print_side_effect:
            mock_model.transcribe.side_effect = print_side_effect
        mock_whisperx.load_model.return_value = mock_model
        mock_whisperx.load_audio.return_value = "audio_data"
        mock_whisperx.load_align_model.return_value = (MagicMock(), {})
        mock_whisperx.align.return_value = {
            "segments": [{"text": "Hello", "start": 0, "end": 1}]
        }
        mock_whisperx.assign_word_speakers.return_value = {
            "segments": [
                {"text": "Hello", "start": 0, "end": 1, "speaker": "SPEAKER_00"}
            ]
        }
        mock_diarize_module = MagicMock()
        mock_diarize = MagicMock()
        mock_diarize.return_value = {"segments": []}
        mock_diarize_module.DiarizationPipeline.return_value = mock_diarize
        return mock_whisperx, mock_diarize_module

    def test_progress_callback_receives_intermediate_values(self):
        """Test that whisperX progress prints are forwarded to callback."""

        def simulate_progress(*args, **kwargs):
            # Simulate what whisperX does internally with print_progress=True
            builtins.print("Progress: 25.00%...")
            builtins.print("Progress: 50.00%...")
            builtins.print("Progress: 75.00%...")
            return {
                "segments": [{"text": "Hello", "start": 0, "end": 1}],
                "language": "en",
            }

        mock_whisperx, mock_diarize_module = self._make_mocks(simulate_progress)

        with patch.dict(
            sys.modules,
            {"whisperx": mock_whisperx, "whisperx.diarize": mock_diarize_module},
        ):
            callback = MagicMock()
            backend = WhisperXBackend(hf_token="test_token")
            backend.transcribe("test.wav", progress_callback=callback)

            # Extract transcribing calls with intermediate percentages
            transcribing_calls = [
                call[0][1]
                for call in callback.call_args_list
                if call[0][0] == "transcribing"
            ]
            # Should have: 0, 25, 50, 75, 100
            assert 0 in transcribing_calls
            assert 25.0 in transcribing_calls
            assert 50.0 in transcribing_calls
            assert 75.0 in transcribing_calls
            assert 100 in transcribing_calls

    def test_print_progress_flag_passed_to_transcribe(self):
        """Test that print_progress=True is passed to model.transcribe."""
        mock_whisperx, mock_diarize_module = self._make_mocks()

        with patch.dict(
            sys.modules,
            {"whisperx": mock_whisperx, "whisperx.diarize": mock_diarize_module},
        ):
            callback = MagicMock()
            backend = WhisperXBackend(hf_token="test_token")
            backend.transcribe("test.wav", progress_callback=callback)

            # Verify print_progress=True was passed
            mock_model = mock_whisperx.load_model.return_value
            call_kwargs = mock_model.transcribe.call_args[1]
            assert call_kwargs.get("print_progress") is True

    def test_builtins_print_restored_after_exception(self):
        """Test that builtins.print is restored even if transcribe raises."""
        original_print = builtins.print

        def raise_error(*args, **kwargs):
            raise RuntimeError("Transcription failed")

        mock_whisperx, mock_diarize_module = self._make_mocks(raise_error)

        with patch.dict(
            sys.modules,
            {"whisperx": mock_whisperx, "whisperx.diarize": mock_diarize_module},
        ):
            callback = MagicMock()
            backend = WhisperXBackend(hf_token="test_token")

            with pytest.raises(RuntimeError, match="Transcription failed"):
                backend.transcribe("test.wav", progress_callback=callback)

        # builtins.print must be restored
        assert builtins.print is original_print

    def test_non_progress_prints_pass_through(self):
        """Test that non-progress print calls are forwarded to original print."""
        captured = []

        def capture_print(*args, **kwargs):
            captured.append(args)

        def simulate_mixed_output(*args, **kwargs):
            builtins.print("Loading VAD model...")
            builtins.print("Progress: 50.00%...")
            builtins.print("Processing complete")
            return {
                "segments": [{"text": "Hello", "start": 0, "end": 1}],
                "language": "en",
            }

        mock_whisperx, mock_diarize_module = self._make_mocks(simulate_mixed_output)

        with patch.dict(
            sys.modules,
            {"whisperx": mock_whisperx, "whisperx.diarize": mock_diarize_module},
        ):
            with patch("builtins.print", capture_print):
                callback = MagicMock()
                backend = WhisperXBackend(hf_token="test_token")
                backend.transcribe("test.wav", progress_callback=callback)

            # Non-progress messages should have passed through
            flat = [str(a) for args in captured for a in args]
            assert any("Loading VAD model" in s for s in flat)
            assert any("Processing complete" in s for s in flat)
            # Progress messages should NOT pass through
            assert not any("Progress: 50" in s for s in flat)

    def test_progress_capped_at_99(self):
        """Test that intercepted progress never exceeds 99 before completion."""

        def simulate_100_progress(*args, **kwargs):
            builtins.print("Progress: 100.00%...")
            return {
                "segments": [{"text": "Hello", "start": 0, "end": 1}],
                "language": "en",
            }

        mock_whisperx, mock_diarize_module = self._make_mocks(simulate_100_progress)

        with patch.dict(
            sys.modules,
            {"whisperx": mock_whisperx, "whisperx.diarize": mock_diarize_module},
        ):
            callback = MagicMock()
            backend = WhisperXBackend(hf_token="test_token")
            backend.transcribe("test.wav", progress_callback=callback)

            # The intercepted 100% should be capped at 99
            transcribing_calls = [
                call[0]
                for call in callback.call_args_list
                if call[0][0] == "transcribing"
            ]
            # The intercepted one should be ("transcribing", 99)
            # The final explicit one should be ("transcribing", 100)
            intercepted = [c for c in transcribing_calls if c[1] == 99]
            final = [c for c in transcribing_calls if c[1] == 100]
            assert len(intercepted) == 1
            assert len(final) == 1
