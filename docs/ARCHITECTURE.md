# Architecture

Technical reference for developers working on Meeting Transcriber.

---

## Overview

Meeting Transcriber is a local-only audio transcription system with speaker identification. The architecture is organized around four subsystems:

- **Backends** -- Pluggable transcription engines (WhisperX, MLX-Whisper, Granite) behind an ABC interface with a factory function.
- **CLI** -- `src/transcribe.py` orchestrates the pipeline: argument parsing, backend selection, progress reporting, output formatting.
- **Progress** -- Real-time progress display with spinner, ETA, and per-backend stage tracking via monkey-patching and streaming hooks.
- **Supporting modules** -- i18n (`src/i18n/`), notifications (`src/notify.py`), vocabulary (`src/vocabulary.py`), text normalization (`src/normalize.py`).

```
CLI (transcribe.py)
  |
  +-- get_backend(mode) --> WhisperXBackend | MLXBackend | GraniteBackend
  |
  +-- ProgressReporter  --> renders [stage/total] bar + spinner + ETA
  |
  +-- _save_results()   --> JSON / TXT / MD output
```

---

## Backend Architecture

### ABC and Interface

All backends extend `TranscriptionBackend` (`src/backends/base.py`):

```python
class TranscriptionBackend(ABC):
    def transcribe(self, audio_path, language, num_speakers,
                   min_speakers, max_speakers, progress_callback, **kwargs)
    @property supports_diarization -> bool
    @property total_stages -> int
    def is_available() -> bool
```

Shared helpers live in the base class:
- `_load_hf_token()` -- loads HuggingFace token from cache, env, or `.env` file. Centralized here to avoid duplication across backends.
- `_build_diarize_kwargs()` -- builds the `num_speakers / min_speakers / max_speakers` dict for pyannote. Static method shared by all diarizing backends.

### Factory Pattern

`get_backend(mode, **kwargs)` in `src/backends/__init__.py`:

1. Validates `mode` against a static `_BACKEND_PARAMS` dict (avoids `inspect.signature()` overhead).
2. **Lazy-imports** the requested backend module only when needed -- this shaves ~3s off startup by not loading torch/transformers/whisperx upfront.
3. Filters `kwargs` to only the parameters that backend accepts.
4. Instantiates the backend and calls `is_available()` to verify dependencies are installed.

### Backend Modes

| Mode | Class | Backend | Diarization | `total_stages` |
|------|-------|---------|-------------|----------------|
| `fast` | `MLXBackend` | mlx-whisper | Optional (`--diarize`) | 3 |
| `meeting` | `WhisperXBackend` | faster-whisper via whisperX | Built-in | 6 |
| `precise` | `GraniteBackend` | IBM Granite Speech (transformers) | pyannote | 4 |

### Dependency Guards

Each backend uses try/except imports at module level. If a dependency is missing, the import assigns `None` and `is_available()` returns `False`. The factory raises `ValueError` with an installation hint.

---

## Progress System

The progress system (`src/progress.py` and hooks in `src/backends/base.py`) provides real-time feedback across all backends.

### ProgressReporter

Renders a live progress line to stdout:

```
[2/5] Transcribing... [========----] 65%  1m23s | ~1m remaining
```

Key implementation details:
- **Background ticker thread** refreshes the display every 250ms to keep the spinner and elapsed timer moving even when no progress updates arrive.
- **100ms render throttle** prevents excessive redraws (previously up to 448 renders per Granite token generation).
- **Stagnation detection** at >=95% switches the display to "finalizing..." after 10 seconds without progress change.
- **`file` parameter** captures `sys.stdout` before `SuppressOutput` can replace it, ensuring the progress bar remains visible while library noise is suppressed.
- **`advance()`** forces a 100% render before moving to the next stage, preventing the bar from appearing stuck at 99%.

### Per-Backend Progress Hooks

Each backend has a different mechanism for reporting granular progress:

**WhisperX (meeting mode):**
- Transcription: intercepts whisperX's `print_progress` callback.
- VAD: `pyannote_progress_hook` monkey-patches `Inference.slide` to intercept `hook(completed, total)` calls.
- Diarization: `diarization_progress_hook` patches `SpeakerDiarization.apply` to inject a hook that maps pyannote's sub-stages (segmentation, embeddings, discrete_diarization) into a weighted 0-99% range.

**MLX (fast mode):**
- `tqdm_progress_hook` replaces `tqdm.tqdm` with a subclass that intercepts `update()` calls and converts frame counts to percentages.

**Granite (precise mode):**
- `ProgressStreamer` implements the transformers `BaseStreamer` interface (`put`/`end`). Counts generated tokens as a fraction of `max_tokens` for per-token progress.

All hooks are context managers that restore original functions in their `finally` block.

---

## Security Decisions

### Path Validation

**Output directory:** `FORBIDDEN_OUTPUT_PATHS` in `transcribe.py` blocks writing to system directories (`/etc`, `/usr/bin`, `/System`, `/Library`) and sensitive user directories (`~/.ssh`, `~/.aws`, `~/.kube`). Resolved paths are checked against this set.

**Vocabulary files:** `_is_safe_vocab_path()` uses an allowlist approach -- only paths within the project root directory are permitted. Files exceeding `MAX_VOCAB_FILE_SIZE` (1 MB) are rejected to prevent DoS.

### torch.load Context Manager

PyTorch 2.6+ defaults to `weights_only=True`, but pyannote models require legacy loading behavior. The `_allow_legacy_torch_load()` context manager temporarily patches `torch.load` and `torch.serialization.load` to set `weights_only=False`, scoped to only the model-loading code path. This replaces an earlier global patch that was applied at import time.

### AppleScript Sanitization

`_escape_applescript_string()` in `src/notify.py`:
1. Strips null bytes and non-printable control characters (keeping `\t`, `\n`, `\r`).
2. Escapes backslashes, double quotes, newlines, carriage returns, and tabs.
3. Truncates to 500 characters.
4. Notifications use `subprocess.run` with `timeout=5` and `capture_output=True`.

### HuggingFace Hub Compatibility

`patch_hf_hub_compat()` in `src/backends/base.py` translates the deprecated `use_auth_token` parameter to `token` for `huggingface_hub` 1.x. The patch is thread-safe (guarded by `threading.Lock`), idempotent, and also patches already-imported references in pyannote modules via `sys.modules`.

---

## Performance Optimizations

- **Lazy imports:** Backend modules are imported only when the corresponding mode is requested. This avoids loading torch, transformers, or whisperX at startup, saving ~3 seconds.
- **Static parameter dict:** The factory uses a hardcoded `_BACKEND_PARAMS` dictionary instead of calling `inspect.signature()` on each backend constructor.
- **Single audio load (Granite):** Audio is loaded from disk once and reused for both transcription and diarization.
- **Binary search speaker matching (Granite):** Speaker labels from pyannote diarization are matched to transcribed segments using `bisect` for O(log N) lookups instead of O(N x M) linear scans.
- **Progress render throttle:** Renders are rate-limited to one per 100ms, preventing hundreds of unnecessary redraws during fast token generation.
- **Memory management:** Each pipeline stage calls `gc.collect()` after model use and deletes large objects (models, audio tensors) explicitly.

---

## Testing Patterns

### Running Tests

```bash
pytest tests/ -v          # all 289 tests
pytest tests/test_backends.py -v   # specific module
```

### How Backends Are Mocked

Tests never load real ML models. Instead, they mock at the library boundary:

```python
# Mocking the factory to return a fake backend
@patch("src.backends.get_backend")
def test_transcription_flow(mock_get_backend):
    mock_backend = MagicMock()
    mock_backend.transcribe.return_value = TranscriptionResult(
        segments=[{"text": "hello", "start": 0, "end": 1}],
        language="en"
    )
    mock_get_backend.return_value = mock_backend
    ...
```

Backend-specific tests (e.g., `test_granite_backend.py`, `test_whisperx_backend.py`) mock the underlying libraries (`transformers`, `whisperx`, `torch`) to test backend logic without GPU or model downloads.

### Common Patterns

- **ABC compliance tests:** Verify that incomplete subclasses raise `TypeError` (enforces the contract).
- **`TranscriptionResult` tests:** Validate the dataclass fields and defaults.
- **Progress tests:** Use a `StringIO` file parameter to capture rendered output without terminal side effects.
- **Security tests:** Verify path validation rejects traversal attempts, vocabulary allowlist blocks external paths, and AppleScript escaping strips control characters.
- **i18n tests:** Check both `translator.lang` property and `detect_system_language()` behavior.

---

## Key Design Decisions

### Why WhisperX for Meeting Mode

WhisperX provides an integrated pipeline: transcription (faster-whisper) + forced phonetic alignment + speaker diarization (pyannote) in a single API. This avoids manually stitching together separate tools. The trade-off is speed (2-3x real-time vs. 10-15x for MLX), which is acceptable for the meeting use case where accuracy matters more.

### Why MLX-Whisper for Fast Mode

MLX-Whisper is optimized for Apple Silicon and achieves 10-15x real-time transcription. It does not include diarization natively, so the `--diarize` flag optionally adds pyannote as a separate step. This mode is ideal for quick transcriptions where speaker identification is not needed.

### Why Granite for Precise Mode

IBM Granite Speech is a transformer-based model accessed via HuggingFace's `transformers` library. It provides higher accuracy at the cost of speed. Diarization is handled separately via pyannote, with segment-to-speaker matching done via binary search on diarization timestamps.

### HuggingFace Hub Compatibility Patches

pyannote.audio 3.4.0 uses the deprecated `use_auth_token` parameter throughout its codebase. When `huggingface_hub` 1.x removed this parameter, diarization broke silently. The `patch_hf_hub_compat()` function transparently translates the old parameter to `token`, patching both the source module and already-imported references in pyannote's modules. The patch auto-detects whether it is needed, making it safe across hub versions.

### Progress via Monkey-Patching

Each backend library reports progress differently (or not at all). Rather than forking libraries, the system uses targeted monkey-patches scoped as context managers:
- `pyannote_progress_hook` patches `Inference.slide`
- `diarization_progress_hook` patches `SpeakerDiarization.apply`
- `tqdm_progress_hook` replaces `tqdm.tqdm`
- `ProgressStreamer` uses transformers' native `BaseStreamer` interface

All patches restore originals in `finally` blocks, keeping the approach safe and reversible.
