# Phase 3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multiple transcription backends, progress reporting, notifications, vocabulary support, and i18n.

**Architecture:** Backend abstraction with factory pattern. Each mode (fast/meeting/precise) maps to a backend class. Shared utilities for progress, notifications, vocabulary, and i18n.

**Tech Stack:** Python 3.12, whisperX, mlx-whisper, transformers (Granite), pyannote.audio, pytest

---

## Task 1: Create Backend Base Class

**Files:**
- Create: `src/backends/__init__.py`
- Create: `src/backends/base.py`
- Create: `tests/test_backends.py`

**Step 1: Create backends directory**

```bash
mkdir -p src/backends
```

**Step 2: Write the failing test**

```python
# tests/test_backends.py
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
```

**Step 3: Run test to verify it fails**

```bash
cd /Users/iurileao/Documents/Projects/meeting-transcriber/.worktrees/phase3
source venv/bin/activate
pytest tests/test_backends.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'src.backends'"

**Step 4: Write minimal implementation**

```python
# src/backends/__init__.py
"""Transcription backends for Meeting Transcriber."""
from .base import TranscriptionBackend, TranscriptionResult

__all__ = ["TranscriptionBackend", "TranscriptionResult"]
```

```python
# src/backends/base.py
"""Base class for transcription backends."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptionResult:
    """Result from a transcription backend.

    Attributes:
        segments: List of transcribed segments with text, timestamps, and optional speaker.
        language: Detected or specified language code.
        metadata: Optional metadata about the transcription.
    """

    segments: list[dict[str, Any]]
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends.

    All backends must implement:
    - transcribe(): Process audio and return TranscriptionResult
    - supports_diarization: Property indicating if backend can identify speakers
    """

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        num_speakers: int | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file.
            language: Language code (e.g., 'pt', 'en') or None for auto-detect.
            num_speakers: Expected number of speakers (for diarization).
            **kwargs: Backend-specific options.

        Returns:
            TranscriptionResult with segments and metadata.
        """
        pass

    @property
    @abstractmethod
    def supports_diarization(self) -> bool:
        """Whether this backend supports speaker diarization."""
        pass

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        return self.__class__.__name__.replace("Backend", "")
```

**Step 5: Run test to verify it passes**

```bash
pytest tests/test_backends.py -v
```

Expected: PASS (6 tests)

**Step 6: Commit**

```bash
git add src/backends/ tests/test_backends.py
git commit -m "feat(backends): add TranscriptionBackend base class and TranscriptionResult

- Add abstract base class for all transcription backends
- Add TranscriptionResult dataclass for standardized output
- Add tests for base class behavior

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create Backend Factory

**Files:**
- Modify: `src/backends/__init__.py`
- Modify: `tests/test_backends.py`

**Step 1: Write the failing test**

Add to `tests/test_backends.py`:

```python
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
        from src.backends import get_backend, TranscriptionBackend
        backend = get_backend("meeting")
        assert isinstance(backend, TranscriptionBackend)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_backends.py::TestBackendFactory -v
```

Expected: FAIL with "cannot import name 'get_backend'"

**Step 3: Write minimal implementation**

First, create stub backends (we'll implement them fully in later tasks):

```python
# src/backends/whisperx_backend.py
"""WhisperX backend for meeting mode (with diarization)."""
from .base import TranscriptionBackend, TranscriptionResult


class WhisperXBackend(TranscriptionBackend):
    """WhisperX backend with integrated diarization.

    Used for: --mode meeting (default)
    Features: Transcription + word alignment + speaker diarization
    """

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        num_speakers: int | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio with speaker diarization."""
        # TODO: Implement in Task 3
        raise NotImplementedError("WhisperX backend not yet implemented")

    @property
    def supports_diarization(self) -> bool:
        return True
```

```python
# src/backends/mlx_backend.py
"""MLX-Whisper backend for fast mode (Apple Silicon optimized)."""
from .base import TranscriptionBackend, TranscriptionResult


class MLXBackend(TranscriptionBackend):
    """MLX-Whisper backend optimized for Apple Silicon.

    Used for: --mode fast
    Features: Fast transcription, no diarization
    """

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        num_speakers: int | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio without diarization."""
        # TODO: Implement in Task 4
        raise NotImplementedError("MLX backend not yet implemented")

    @property
    def supports_diarization(self) -> bool:
        return False
```

```python
# src/backends/granite_backend.py
"""Granite backend for precise mode (high accuracy)."""
from .base import TranscriptionBackend, TranscriptionResult


class GraniteBackend(TranscriptionBackend):
    """IBM Granite Speech backend for high-accuracy transcription.

    Used for: --mode precise
    Features: High accuracy + separate pyannote diarization
    """

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        num_speakers: int | None = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio with high accuracy."""
        # TODO: Implement in Task 5
        raise NotImplementedError("Granite backend not yet implemented")

    @property
    def supports_diarization(self) -> bool:
        return True  # Via separate pyannote pipeline
```

Update factory in `__init__.py`:

```python
# src/backends/__init__.py
"""Transcription backends for Meeting Transcriber."""
from .base import TranscriptionBackend, TranscriptionResult
from .whisperx_backend import WhisperXBackend
from .mlx_backend import MLXBackend
from .granite_backend import GraniteBackend

__all__ = [
    "TranscriptionBackend",
    "TranscriptionResult",
    "WhisperXBackend",
    "MLXBackend",
    "GraniteBackend",
    "get_backend",
]

# Mode to backend mapping
_BACKENDS = {
    "fast": MLXBackend,
    "meeting": WhisperXBackend,
    "precise": GraniteBackend,
}


def get_backend(mode: str) -> TranscriptionBackend:
    """Get transcription backend for the specified mode.

    Args:
        mode: Transcription mode ('fast', 'meeting', 'precise').

    Returns:
        Configured TranscriptionBackend instance.

    Raises:
        ValueError: If mode is not recognized.
    """
    if mode not in _BACKENDS:
        valid_modes = ", ".join(_BACKENDS.keys())
        raise ValueError(f"Invalid mode: '{mode}'. Choose from: {valid_modes}")

    backend_class = _BACKENDS[mode]
    return backend_class()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_backends.py -v
```

Expected: PASS (11 tests)

**Step 5: Commit**

```bash
git add src/backends/
git commit -m "feat(backends): add backend factory and stub implementations

- Add get_backend() factory function
- Add WhisperXBackend stub (meeting mode)
- Add MLXBackend stub (fast mode)
- Add GraniteBackend stub (precise mode)
- Add factory tests

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Implement WhisperX Backend

**Files:**
- Modify: `src/backends/whisperx_backend.py`
- Create: `tests/test_whisperx_backend.py`

**Step 1: Write the failing test**

```python
# tests/test_whisperx_backend.py
"""Tests for WhisperX backend."""
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_whisperx_backend.py -v
```

Expected: FAIL (missing attributes)

**Step 3: Write implementation**

```python
# src/backends/whisperx_backend.py
"""WhisperX backend for meeting mode (with diarization)."""
import gc
import os
from pathlib import Path

from dotenv import load_dotenv

from .base import TranscriptionBackend, TranscriptionResult


class WhisperXBackend(TranscriptionBackend):
    """WhisperX backend with integrated diarization.

    Used for: --mode meeting (default)
    Features: Transcription + word alignment + speaker diarization
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str | None = None,
        batch_size: int | None = None,
        hf_token: str | None = None,
    ):
        """Initialize WhisperX backend.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3).
            device: Processing device (cpu, cuda, mps).
            compute_type: Compute type (int8, float16). Auto-detected if None.
            batch_size: Batch size for processing. Auto-detected if None.
            hf_token: HuggingFace token for pyannote. Loaded from env if None.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type or self._get_compute_type()
        self.batch_size = batch_size or self._get_batch_size()
        self._hf_token = hf_token

    def _get_compute_type(self) -> str:
        """Get optimal compute type for device."""
        return "int8" if self.device == "cpu" else "float16"

    def _get_batch_size(self) -> int:
        """Get optimal batch size for device and model."""
        if self.device == "cpu":
            return 8
        if self.model_size in ("tiny", "base", "small"):
            return 32
        return 16

    def _load_hf_token(self) -> str:
        """Load HuggingFace token from environment or .env file."""
        if self._hf_token:
            return self._hf_token

        env_file = Path(__file__).parent.parent.parent / ".env"
        load_dotenv(env_file)

        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "HuggingFace token not found.\n"
                "Set HF_TOKEN in .env or pass hf_token parameter.\n"
                "Get your token at: https://huggingface.co/settings/tokens"
            )
        return token

    @property
    def supports_diarization(self) -> bool:
        return True

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
        """Transcribe audio with speaker diarization.

        Args:
            audio_path: Path to audio file.
            language: Language code or None for auto-detect.
            num_speakers: Exact number of speakers (if known).
            min_speakers: Minimum expected speakers.
            max_speakers: Maximum expected speakers.
            progress_callback: Optional callback(stage, percent) for progress.
            **kwargs: Additional options.

        Returns:
            TranscriptionResult with segments and speaker labels.
        """
        import whisperx

        # Stage 1: Load model
        if progress_callback:
            progress_callback("loading", 0)

        model = whisperx.load_model(
            self.model_size,
            self.device,
            compute_type=self.compute_type,
            language=language,
        )

        if progress_callback:
            progress_callback("loading", 100)

        # Stage 2: Transcribe
        if progress_callback:
            progress_callback("transcribing", 0)

        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=self.batch_size)
        detected_language = result.get("language", language or "unknown")

        if progress_callback:
            progress_callback("transcribing", 100)

        # Free memory
        del model
        gc.collect()

        # Stage 3: Align
        if progress_callback:
            progress_callback("aligning", 0)

        model_a, metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=self.device,
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        if progress_callback:
            progress_callback("aligning", 100)

        # Free memory
        del model_a
        gc.collect()

        # Stage 4: Diarize
        if progress_callback:
            progress_callback("diarizing", 0)

        hf_token = self._load_hf_token()
        from whisperx.diarize import DiarizationPipeline

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
        result = whisperx.assign_word_speakers(diarize_segments, result)

        if progress_callback:
            progress_callback("diarizing", 100)

        # Free memory
        del diarize_model
        gc.collect()

        # Count speakers
        speakers = {seg.get("speaker") for seg in result.get("segments", []) if "speaker" in seg}

        return TranscriptionResult(
            segments=result.get("segments", []),
            language=detected_language,
            metadata={
                "model": self.model_size,
                "device": self.device,
                "compute_type": self.compute_type,
                "num_speakers": len(speakers),
                "backend": "whisperx",
            },
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_whisperx_backend.py -v
```

Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/backends/whisperx_backend.py tests/test_whisperx_backend.py
git commit -m "feat(backends): implement WhisperX backend

- Full transcription with word alignment
- Speaker diarization via pyannote
- Progress callback support
- Memory management with gc.collect()
- Configurable model, device, compute type

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement i18n System

**Files:**
- Create: `src/i18n/__init__.py`
- Create: `src/i18n/en.json`
- Create: `src/i18n/pt.json`
- Create: `tests/test_i18n.py`

**Step 1: Create i18n directory**

```bash
mkdir -p src/i18n
```

**Step 2: Write the failing test**

```python
# tests/test_i18n.py
"""Tests for internationalization system."""
import pytest
from src.i18n import get_translator, SUPPORTED_LANGUAGES


class TestGetTranslator:
    """Tests for get_translator function."""

    def test_english_translator(self):
        t = get_translator("en")
        assert t("messages.complete") == "Transcription complete"

    def test_portuguese_translator(self):
        t = get_translator("pt")
        assert t("messages.complete") == "Transcrição completa"

    def test_nested_keys(self):
        t = get_translator("en")
        assert t("stages.loading") == "Loading model"

    def test_invalid_language_falls_back_to_english(self):
        t = get_translator("xx")
        assert t("messages.complete") == "Transcription complete"

    def test_missing_key_returns_key(self):
        t = get_translator("en")
        assert t("nonexistent.key") == "nonexistent.key"


class TestSupportedLanguages:
    """Tests for supported languages."""

    def test_english_supported(self):
        assert "en" in SUPPORTED_LANGUAGES

    def test_portuguese_supported(self):
        assert "pt" in SUPPORTED_LANGUAGES
```

**Step 3: Run test to verify it fails**

```bash
pytest tests/test_i18n.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 4: Write implementation**

```json
// src/i18n/en.json
{
  "stages": {
    "loading": "Loading model",
    "transcribing": "Transcribing",
    "aligning": "Aligning",
    "diarizing": "Diarizing",
    "saving": "Saving"
  },
  "messages": {
    "complete": "Transcription complete",
    "error": "Error",
    "file_not_found": "File not found",
    "invalid_mode": "Invalid mode. Choose: fast, meeting, precise",
    "cancelled": "Transcription cancelled by user"
  },
  "cli": {
    "description": "Transcribe meeting audio with speaker identification",
    "help_audio": "Path to audio file",
    "help_mode": "Transcription mode (fast/meeting/precise)",
    "help_model": "Whisper model size",
    "help_language": "Audio language (auto-detect if not specified)",
    "help_num_speakers": "Exact number of speakers (if known)",
    "help_output": "Output directory",
    "help_format": "Output format (json/txt/md/all)",
    "help_ui_lang": "Interface language (en/pt)",
    "help_notify": "Send system notification when done",
    "help_vocab": "Additional vocabulary file",
    "help_verbose": "Show detailed logs"
  }
}
```

```json
// src/i18n/pt.json
{
  "stages": {
    "loading": "Carregando modelo",
    "transcribing": "Transcrevendo",
    "aligning": "Alinhando",
    "diarizing": "Diarizando",
    "saving": "Salvando"
  },
  "messages": {
    "complete": "Transcrição completa",
    "error": "Erro",
    "file_not_found": "Arquivo não encontrado",
    "invalid_mode": "Modo inválido. Escolha: fast, meeting, precise",
    "cancelled": "Transcrição cancelada pelo usuário"
  },
  "cli": {
    "description": "Transcreve áudio de reuniões com identificação de speakers",
    "help_audio": "Caminho para o arquivo de áudio",
    "help_mode": "Modo de transcrição (fast/meeting/precise)",
    "help_model": "Tamanho do modelo Whisper",
    "help_language": "Idioma do áudio (detecta automaticamente se não especificado)",
    "help_num_speakers": "Número exato de speakers (se conhecido)",
    "help_output": "Diretório de saída",
    "help_format": "Formato de saída (json/txt/md/all)",
    "help_ui_lang": "Idioma da interface (en/pt)",
    "help_notify": "Enviar notificação do sistema ao terminar",
    "help_vocab": "Arquivo de vocabulário adicional",
    "help_verbose": "Mostrar logs detalhados"
  }
}
```

```python
# src/i18n/__init__.py
"""Internationalization (i18n) support for Meeting Transcriber."""
import json
import locale
from pathlib import Path
from typing import Callable

SUPPORTED_LANGUAGES = {"en", "pt"}
_DEFAULT_LANGUAGE = "en"


def _load_strings(lang: str) -> dict:
    """Load translation strings for a language."""
    i18n_dir = Path(__file__).parent
    lang_file = i18n_dir / f"{lang}.json"

    if not lang_file.exists():
        lang_file = i18n_dir / f"{_DEFAULT_LANGUAGE}.json"

    return json.loads(lang_file.read_text(encoding="utf-8"))


def _get_nested(data: dict, key: str) -> str:
    """Get nested value from dict using dot notation."""
    keys = key.split(".")
    value = data
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return key  # Return key itself if not found


def get_translator(lang: str | None = None) -> Callable[[str], str]:
    """Get a translator function for the specified language.

    Args:
        lang: Language code ('en', 'pt') or None for auto-detect.

    Returns:
        Translator function t(key) -> translated string.
    """
    if lang is None:
        # Auto-detect from system locale
        system_locale = locale.getdefaultlocale()[0] or ""
        lang = "pt" if system_locale.startswith("pt") else "en"

    if lang not in SUPPORTED_LANGUAGES:
        lang = _DEFAULT_LANGUAGE

    strings = _load_strings(lang)

    def translate(key: str) -> str:
        return _get_nested(strings, key)

    return translate


def detect_system_language() -> str:
    """Detect language from system locale.

    Returns:
        Language code ('en' or 'pt').
    """
    system_locale = locale.getdefaultlocale()[0] or ""
    return "pt" if system_locale.startswith("pt") else "en"
```

**Step 5: Run test to verify it passes**

```bash
pytest tests/test_i18n.py -v
```

Expected: PASS (7 tests)

**Step 6: Commit**

```bash
git add src/i18n/ tests/test_i18n.py
git commit -m "feat(i18n): add internationalization support (en/pt)

- Add English and Portuguese translations
- Auto-detect system language
- Fallback to English for unknown languages
- Support nested keys with dot notation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Implement Progress Reporter

**Files:**
- Create: `src/progress.py`
- Create: `tests/test_progress.py`

**Step 1: Write the failing test**

```python
# tests/test_progress.py
"""Tests for progress reporting."""
import pytest
from io import StringIO
from src.progress import ProgressReporter, Stage


class TestStage:
    """Tests for Stage enum."""

    def test_stage_has_english_label(self):
        assert Stage.LOADING.label("en") == "Loading model"

    def test_stage_has_portuguese_label(self):
        assert Stage.LOADING.label("pt") == "Carregando modelo"

    def test_all_stages_have_labels(self):
        for stage in Stage:
            assert stage.label("en") is not None
            assert stage.label("pt") is not None


class TestProgressReporter:
    """Tests for ProgressReporter."""

    def test_update_renders_progress_bar(self, capsys):
        reporter = ProgressReporter(total_stages=4, lang="en")
        reporter.update(Stage.TRANSCRIBING, 50)
        captured = capsys.readouterr()
        assert "Transcribing" in captured.out
        assert "50%" in captured.out

    def test_stage_number_increments(self, capsys):
        reporter = ProgressReporter(total_stages=4, lang="en")
        reporter.update(Stage.LOADING, 100)
        reporter.advance()
        reporter.update(Stage.TRANSCRIBING, 0)
        captured = capsys.readouterr()
        assert "[2/4]" in captured.out

    def test_complete_shows_checkmark(self, capsys):
        reporter = ProgressReporter(total_stages=4, lang="en")
        reporter.complete("output.json", 120.5)
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "output.json" in captured.out

    def test_portuguese_labels(self, capsys):
        reporter = ProgressReporter(total_stages=4, lang="pt")
        reporter.update(Stage.TRANSCRIBING, 50)
        captured = capsys.readouterr()
        assert "Transcrevendo" in captured.out
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_progress.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

```python
# src/progress.py
"""Progress reporting for transcription pipeline."""
import sys
from enum import Enum


class Stage(Enum):
    """Transcription pipeline stages."""

    LOADING = ("loading", "Loading model", "Carregando modelo")
    TRANSCRIBING = ("transcribing", "Transcribing", "Transcrevendo")
    ALIGNING = ("aligning", "Aligning", "Alinhando")
    DIARIZING = ("diarizing", "Diarizing", "Diarizando")
    SAVING = ("saving", "Saving", "Salvando")

    def __init__(self, key: str, en: str, pt: str):
        self.key = key
        self._labels = {"en": en, "pt": pt}

    def label(self, lang: str = "en") -> str:
        """Get localized label for this stage."""
        return self._labels.get(lang, self._labels["en"])


class ProgressReporter:
    """Reports progress during transcription.

    Displays progress like:
        [2/4] Transcribing... [████████░░░░] 65%
    """

    def __init__(self, total_stages: int, lang: str = "en", width: int = 20):
        """Initialize progress reporter.

        Args:
            total_stages: Total number of stages in pipeline.
            lang: Language for labels ('en' or 'pt').
            width: Width of progress bar in characters.
        """
        self.total_stages = total_stages
        self.lang = lang
        self.width = width
        self.current_stage = 1

    def _render_bar(self, percent: float) -> str:
        """Render progress bar."""
        filled = int(self.width * percent / 100)
        empty = self.width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    def update(self, stage: Stage, percent: float) -> None:
        """Update progress display.

        Args:
            stage: Current pipeline stage.
            percent: Progress percentage (0-100).
        """
        label = stage.label(self.lang)
        bar = self._render_bar(percent)
        line = f"\r[{self.current_stage}/{self.total_stages}] {label}... {bar} {percent:.0f}%"
        sys.stdout.write(line)
        sys.stdout.flush()

    def advance(self) -> None:
        """Move to next stage."""
        self.current_stage += 1
        # New line after completing a stage
        print()

    def complete(self, output_path: str, duration_seconds: float) -> None:
        """Show completion message.

        Args:
            output_path: Path to output file.
            duration_seconds: Total duration in seconds.
        """
        print()  # New line after progress bar

        # Format duration
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        if minutes > 0:
            duration_str = f"{minutes}m{seconds:02d}s"
        else:
            duration_str = f"{seconds}s"

        if self.lang == "pt":
            print(f"✓ Transcrição completa: {output_path} ({duration_str})")
        else:
            print(f"✓ Transcription complete: {output_path} ({duration_str})")

    def error(self, message: str) -> None:
        """Show error message.

        Args:
            message: Error message to display.
        """
        print()  # New line after progress bar
        if self.lang == "pt":
            print(f"✗ Erro: {message}", file=sys.stderr)
        else:
            print(f"✗ Error: {message}", file=sys.stderr)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_progress.py -v
```

Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/progress.py tests/test_progress.py
git commit -m "feat(progress): add progress reporter with stages

- Stage-based progress bar: [2/4] Transcribing... [████████░░░░] 65%
- Bilingual support (en/pt)
- Completion message with duration
- Error message formatting

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Implement Notification System

**Files:**
- Create: `src/notify.py`
- Create: `tests/test_notify.py`

**Step 1: Write the failing test**

```python
# tests/test_notify.py
"""Tests for notification system."""
import pytest
from unittest.mock import patch, MagicMock
from src.notify import notify, is_macos


class TestIsMacos:
    """Tests for macOS detection."""

    @patch("platform.system", return_value="Darwin")
    def test_darwin_is_macos(self, mock_system):
        assert is_macos() is True

    @patch("platform.system", return_value="Linux")
    def test_linux_is_not_macos(self, mock_system):
        assert is_macos() is False

    @patch("platform.system", return_value="Windows")
    def test_windows_is_not_macos(self, mock_system):
        assert is_macos() is False


class TestNotify:
    """Tests for notify function."""

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    def test_notify_calls_osascript_on_macos(self, mock_run, mock_system):
        notify("Title", "Message")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "osascript"

    @patch("platform.system", return_value="Linux")
    @patch("subprocess.run")
    def test_notify_does_nothing_on_linux(self, mock_run, mock_system):
        notify("Title", "Message")
        mock_run.assert_not_called()

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    def test_notify_includes_title_and_message(self, mock_run, mock_system):
        notify("Test Title", "Test Message")
        args = mock_run.call_args[0][0]
        script = args[2]  # osascript -e "script"
        assert "Test Title" in script
        assert "Test Message" in script
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_notify.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

```python
# src/notify.py
"""System notifications for Meeting Transcriber."""
import platform
import subprocess


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system() == "Darwin"


def notify(title: str, message: str) -> None:
    """Send system notification.

    On macOS, uses osascript to display native notification.
    On other systems, does nothing (silent no-op).

    Args:
        title: Notification title.
        message: Notification body text.
    """
    if not is_macos():
        return

    # Escape quotes in title and message
    title = title.replace('"', '\\"')
    message = message.replace('"', '\\"')

    script = f'display notification "{message}" with title "{title}"'

    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
        )
    except Exception:
        # Silent failure - notifications are non-critical
        pass
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_notify.py -v
```

Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/notify.py tests/test_notify.py
git commit -m "feat(notify): add macOS notification support

- Native macOS notifications via osascript
- Silent no-op on non-macOS systems
- Proper escaping of title and message

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Implement Vocabulary System

**Files:**
- Create: `src/vocabulary.py`
- Create: `tests/test_vocabulary.py`

**Step 1: Write the failing test**

```python
# tests/test_vocabulary.py
"""Tests for vocabulary system."""
import pytest
from pathlib import Path
from src.vocabulary import load_vocabulary, parse_vocab_file


class TestParseVocabFile:
    """Tests for parse_vocab_file function."""

    def test_parse_simple_file(self, tmp_path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("word1\nword2\nword3")
        words = parse_vocab_file(vocab_file)
        assert words == ["word1", "word2", "word3"]

    def test_ignores_comments(self, tmp_path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("# comment\nword1\n# another comment\nword2")
        words = parse_vocab_file(vocab_file)
        assert words == ["word1", "word2"]

    def test_ignores_empty_lines(self, tmp_path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("word1\n\n\nword2\n")
        words = parse_vocab_file(vocab_file)
        assert words == ["word1", "word2"]

    def test_strips_whitespace(self, tmp_path):
        vocab_file = tmp_path / "vocab.txt"
        vocab_file.write_text("  word1  \n\tword2\t")
        words = parse_vocab_file(vocab_file)
        assert words == ["word1", "word2"]


class TestLoadVocabulary:
    """Tests for load_vocabulary function."""

    def test_returns_empty_if_no_files(self, tmp_path):
        words = load_vocabulary(vocab_dir=tmp_path)
        assert words == []

    def test_loads_default_if_exists(self, tmp_path):
        default_file = tmp_path / "default.txt"
        default_file.write_text("word1\nword2")
        words = load_vocabulary(vocab_dir=tmp_path)
        assert "word1" in words
        assert "word2" in words

    def test_loads_extra_files(self, tmp_path):
        extra_file = tmp_path / "extra.txt"
        extra_file.write_text("extra1\nextra2")
        words = load_vocabulary(vocab_dir=tmp_path, extra_files=[str(extra_file)])
        assert "extra1" in words
        assert "extra2" in words

    def test_deduplicates_words(self, tmp_path):
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("word1\nword2")
        file2.write_text("word2\nword3")
        words = load_vocabulary(vocab_dir=tmp_path, extra_files=[str(file1), str(file2)])
        assert words.count("word2") == 1
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_vocabulary.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

```python
# src/vocabulary.py
"""Custom vocabulary support for transcription."""
from pathlib import Path


def parse_vocab_file(path: Path) -> list[str]:
    """Parse vocabulary file.

    File format:
    - One word/term per line
    - Lines starting with # are comments
    - Empty lines are ignored
    - Whitespace is stripped

    Args:
        path: Path to vocabulary file.

    Returns:
        List of vocabulary words.
    """
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line)
    return words


def load_vocabulary(
    vocab_dir: Path | str | None = None,
    extra_files: list[str] | None = None,
) -> list[str]:
    """Load vocabulary from default file and extra files.

    Args:
        vocab_dir: Directory containing default.txt. Defaults to project's vocab/.
        extra_files: List of additional vocabulary file paths.

    Returns:
        Deduplicated list of vocabulary words.
    """
    words = []

    # Determine vocab directory
    if vocab_dir is None:
        vocab_dir = Path(__file__).parent.parent / "vocab"
    else:
        vocab_dir = Path(vocab_dir)

    # Load default vocabulary if exists
    default_file = vocab_dir / "default.txt"
    if default_file.exists():
        words.extend(parse_vocab_file(default_file))

    # Load extra files
    for filepath in extra_files or []:
        path = Path(filepath)
        if path.exists():
            words.extend(parse_vocab_file(path))

    # Deduplicate while preserving order
    seen = set()
    unique_words = []
    for word in words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)

    return unique_words
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_vocabulary.py -v
```

Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add src/vocabulary.py tests/test_vocabulary.py
git commit -m "feat(vocabulary): add custom vocabulary support

- Load default.txt automatically if present
- Support extra vocabulary files via --vocab
- Ignore comments and empty lines
- Deduplicate words across files

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Implement Text Normalization

**Files:**
- Create: `src/normalize.py`
- Create: `tests/test_normalize.py`

**Step 1: Write the failing test**

```python
# tests/test_normalize.py
"""Tests for text normalization."""
import pytest
from src.normalize import normalize_text


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_capitalizes_first_character(self):
        assert normalize_text("hello world") == "Hello world"

    def test_capitalizes_after_period(self):
        assert normalize_text("hello. world") == "Hello. World"

    def test_capitalizes_after_exclamation(self):
        assert normalize_text("hello! world") == "Hello! World"

    def test_capitalizes_after_question(self):
        assert normalize_text("hello? world") == "Hello? World"

    def test_handles_multiple_sentences(self):
        text = "first sentence. second sentence! third sentence?"
        expected = "First sentence. Second sentence! Third sentence?"
        assert normalize_text(text) == expected

    def test_preserves_existing_caps(self):
        assert normalize_text("Hello. World") == "Hello. World"

    def test_handles_empty_string(self):
        assert normalize_text("") == ""

    def test_handles_single_word(self):
        assert normalize_text("hello") == "Hello"

    def test_preserves_acronyms(self):
        assert normalize_text("the NASA team. they went to space") == "The NASA team. They went to space"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_normalize.py -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

```python
# src/normalize.py
"""Text normalization for transcriptions."""
import re


def normalize_text(text: str) -> str:
    """Apply basic normalization to transcribed text.

    Normalizations applied:
    - Capitalize first character
    - Capitalize after sentence endings (. ! ?)

    Does NOT:
    - Modify words (no spell correction)
    - Change punctuation
    - Alter spacing

    Args:
        text: Raw transcribed text.

    Returns:
        Normalized text.
    """
    if not text:
        return text

    # Capitalize after sentence endings
    text = re.sub(
        r"([.!?])\s+(\w)",
        lambda m: m.group(1) + " " + m.group(2).upper(),
        text,
    )

    # Capitalize first character
    text = text[0].upper() + text[1:]

    return text
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_normalize.py -v
```

Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add src/normalize.py tests/test_normalize.py
git commit -m "feat(normalize): add basic text normalization

- Capitalize first character
- Capitalize after sentence endings
- Preserve original words (no substitutions)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Implement MLX Backend (Stub with Import Check)

**Files:**
- Modify: `src/backends/mlx_backend.py`
- Create: `tests/test_mlx_backend.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_mlx_backend.py -v
```

Expected: FAIL (missing attributes)

**Step 3: Write implementation**

```python
# src/backends/mlx_backend.py
"""MLX-Whisper backend for fast mode (Apple Silicon optimized)."""
from .base import TranscriptionBackend, TranscriptionResult


class MLXBackend(TranscriptionBackend):
    """MLX-Whisper backend optimized for Apple Silicon.

    Used for: --mode fast
    Features: Fast transcription (10-15x realtime), no diarization
    Requirements: Apple Silicon Mac, mlx-whisper package
    """

    def __init__(self, model_size: str = "large-v3"):
        """Initialize MLX backend.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3).
        """
        self.model_size = model_size

    def is_available(self) -> bool:
        """Check if MLX-Whisper is available.

        Returns:
            True if mlx_whisper can be imported.
        """
        try:
            import mlx_whisper
            return True
        except ImportError:
            return False

    @property
    def supports_diarization(self) -> bool:
        return False

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        num_speakers: int | None = None,
        progress_callback=None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio without diarization.

        Args:
            audio_path: Path to audio file.
            language: Language code or None for auto-detect.
            num_speakers: Ignored (no diarization support).
            progress_callback: Optional callback(stage, percent) for progress.
            **kwargs: Additional options.

        Returns:
            TranscriptionResult with segments (no speaker labels).

        Raises:
            ImportError: If mlx-whisper is not installed.
        """
        if not self.is_available():
            raise ImportError(
                "mlx-whisper not installed.\n"
                "Install with: pip install mlx-whisper\n"
                "Note: Requires Apple Silicon Mac."
            )

        import mlx_whisper

        if progress_callback:
            progress_callback("loading", 0)

        # MLX model names use different format
        model_name = f"mlx-community/whisper-{self.model_size}-mlx"

        if progress_callback:
            progress_callback("loading", 50)

        if progress_callback:
            progress_callback("transcribing", 0)

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_name,
            language=language,
        )

        if progress_callback:
            progress_callback("transcribing", 100)

        # Convert to standard format
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
                # No speaker - MLX doesn't support diarization
            })

        return TranscriptionResult(
            segments=segments,
            language=result.get("language", language or "unknown"),
            metadata={
                "model": self.model_size,
                "backend": "mlx-whisper",
            },
        )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_mlx_backend.py -v
```

Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/backends/mlx_backend.py tests/test_mlx_backend.py
git commit -m "feat(backends): implement MLX backend for fast mode

- Apple Silicon optimized (10-15x realtime)
- No diarization (single speaker use case)
- Availability check for graceful fallback
- Progress callback support

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Implement Granite Backend (Stub with Import Check)

**Files:**
- Modify: `src/backends/granite_backend.py`
- Create: `tests/test_granite_backend.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_granite_backend.py -v
```

Expected: FAIL (missing attributes)

**Step 3: Write implementation**

```python
# src/backends/granite_backend.py
"""Granite backend for precise mode (high accuracy)."""
import gc
import os
from pathlib import Path

from dotenv import load_dotenv

from .base import TranscriptionBackend, TranscriptionResult


class GraniteBackend(TranscriptionBackend):
    """IBM Granite Speech backend for high-accuracy transcription.

    Used for: --mode precise
    Features: High accuracy transcription + separate pyannote diarization
    Requirements: transformers, accelerate, pyannote.audio
    """

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-speech-3.3-8b",
        device: str = "cpu",
        hf_token: str | None = None,
    ):
        """Initialize Granite backend.

        Args:
            model_name: HuggingFace model name.
            device: Processing device (cpu, cuda, mps).
            hf_token: HuggingFace token for pyannote.
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
                "Install with: pip install transformers accelerate"
            )

        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        import torch

        # Stage 1: Load Granite model
        if progress_callback:
            progress_callback("loading", 0)

        processor = AutoProcessor.from_pretrained(self.model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
        )
        model.to(self.device)

        if progress_callback:
            progress_callback("loading", 100)

        # Stage 2: Transcribe with Granite
        if progress_callback:
            progress_callback("transcribing", 0)

        import librosa
        audio_array, sr = librosa.load(audio_path, sr=16000)

        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=448)

        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        if progress_callback:
            progress_callback("transcribing", 100)

        # Free Granite model memory
        del model, processor
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
        # For now, create a single segment (Granite doesn't provide timestamps)
        segments = [{
            "start": 0,
            "end": len(audio_array) / sr,
            "text": transcription.strip(),
            "speaker": "SPEAKER_00",  # Basic assignment
        }]

        # Count speakers from diarization
        speakers = set()
        for seg in diarize_segments.itertracks(yield_label=True):
            speakers.add(seg[2])

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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_granite_backend.py -v
```

Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/backends/granite_backend.py tests/test_granite_backend.py
git commit -m "feat(backends): implement Granite backend for precise mode

- IBM Granite Speech 3.3 8B for high accuracy
- Separate pyannote diarization pipeline
- Memory management between stages
- Progress callback support

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Update Main CLI

**Files:**
- Modify: `src/transcribe.py`

**Step 1: Update CLI with new flags**

This is a larger refactoring task. The main changes:

1. Add `--mode` flag (fast/meeting/precise)
2. Add `--ui-lang` flag (en/pt)
3. Add `--notify` flag
4. Add `--vocab` flag
5. Integrate progress reporter
6. Use backend factory

**Step 2: Test manually**

```bash
python src/transcribe.py --help
```

Should show new flags.

**Step 3: Commit**

```bash
git add src/transcribe.py
git commit -m "feat(cli): add mode, ui-lang, notify, vocab flags

- --mode fast/meeting/precise for backend selection
- --ui-lang en/pt for interface language
- --notify for macOS notifications
- --vocab for custom vocabulary files
- Integrate progress reporter
- Use backend factory pattern

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 12: Update Documentation

**Files:**
- Modify: `README.md` (English)
- Create: `README.pt.md` (Portuguese)
- Modify: `CLAUDE.md`

**Step 1: Update README.md with new features**

**Step 2: Create README.pt.md**

**Step 3: Update CLAUDE.md with new commands**

**Step 4: Commit**

```bash
git add README.md README.pt.md CLAUDE.md
git commit -m "docs: update documentation for Phase 3 features

- Document new --mode flag and backends
- Document --notify, --vocab, --ui-lang flags
- Add Portuguese README
- Update CLAUDE.md with new commands

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 13: Run Full Test Suite

**Step 1: Run all tests**

```bash
pytest tests/ -v
```

**Step 2: Fix any failures**

**Step 3: Final commit**

```bash
git add .
git commit -m "test: ensure all tests pass for Phase 3

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Backend base class | 6 |
| 2 | Backend factory | 5 |
| 3 | WhisperX backend | 6 |
| 4 | i18n system | 7 |
| 5 | Progress reporter | 7 |
| 6 | Notification system | 6 |
| 7 | Vocabulary system | 9 |
| 8 | Text normalization | 9 |
| 9 | MLX backend | 5 |
| 10 | Granite backend | 5 |
| 11 | CLI update | manual |
| 12 | Documentation | - |
| 13 | Full test suite | all |

**Total new tests:** ~65
**Estimated commits:** 13

---

*Plan generated with superpowers:writing-plans skill*
