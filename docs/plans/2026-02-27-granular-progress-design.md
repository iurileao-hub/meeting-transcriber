# Granular Progress for MLX and Granite Backends — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add granular transcription progress to MLX (via tqdm interception) and Granite (via BaseStreamer) backends.

**Architecture:** Two new utilities in `base.py` following existing patterns (`diarization_progress_hook`, `pyannote_progress_hook`). Each backend wraps its blocking transcription call with the appropriate hook. TDD with tests first.

**Tech Stack:** Python 3.12, tqdm, transformers BaseStreamer, pytest, unittest.mock

---

### Task 1: Add `tqdm_progress_hook` to base.py (tests)

**Files:**
- Test: `tests/test_backends.py` (append new test class)

**Step 1: Write the failing tests**

Append to `tests/test_backends.py`:

```python
class TestTqdmProgressHook:
    """Tests for tqdm_progress_hook context manager."""

    def test_noop_when_no_callback(self):
        """Should yield without error when callback is None."""
        from src.backends.base import tqdm_progress_hook

        with tqdm_progress_hook(None, "transcribing"):
            pass  # No error

    def test_restores_original_tqdm_after_context(self):
        """Should restore tqdm.tqdm after context exits."""
        from src.backends.base import tqdm_progress_hook
        import tqdm as tqdm_module

        original = tqdm_module.tqdm
        callback = MagicMock()

        with tqdm_progress_hook(callback, "transcribing"):
            assert tqdm_module.tqdm is not original

        assert tqdm_module.tqdm is original

    def test_intercepts_tqdm_update_calls(self):
        """Should forward tqdm update() calls to progress_callback."""
        from src.backends.base import tqdm_progress_hook
        import tqdm as tqdm_module

        reported = []

        def callback(stage, pct):
            reported.append((stage, round(pct, 1)))

        with tqdm_progress_hook(callback, "transcribing"):
            bar = tqdm_module.tqdm(total=100, disable=True)
            bar.update(50)  # 50%
            bar.update(30)  # 80%
            bar.close()

        assert len(reported) == 2
        assert reported[0] == ("transcribing", 49.5)
        assert reported[1] == ("transcribing", 79.2)

    def test_caps_at_99_percent(self):
        """Should never report more than 99%."""
        from src.backends.base import tqdm_progress_hook
        import tqdm as tqdm_module

        reported = []

        def callback(stage, pct):
            reported.append(pct)

        with tqdm_progress_hook(callback, "transcribing"):
            bar = tqdm_module.tqdm(total=100, disable=True)
            bar.update(100)  # 100% -> capped at 99
            bar.close()

        assert all(p <= 99 for p in reported)

    def test_handles_zero_total(self):
        """Should not crash when tqdm total is 0 or None."""
        from src.backends.base import tqdm_progress_hook
        import tqdm as tqdm_module

        callback = MagicMock()

        with tqdm_progress_hook(callback, "transcribing"):
            bar = tqdm_module.tqdm(total=0, disable=True)
            bar.update(1)
            bar.close()

        # Should not have called callback (no valid percentage)
        callback.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backends.py::TestTqdmProgressHook -v`
Expected: FAIL with `ImportError: cannot import name 'tqdm_progress_hook'`

---

### Task 2: Implement `tqdm_progress_hook` in base.py

**Files:**
- Modify: `src/backends/base.py` (add after `pyannote_progress_hook`, around line 209)

**Step 3: Write the implementation**

Add to `src/backends/base.py`:

```python
@contextmanager
def tqdm_progress_hook(
    progress_callback: Callable[[str, float], None] | None,
    stage_name: str,
):
    """Intercept tqdm progress bars to forward updates to progress_callback.

    Temporarily replaces tqdm.tqdm with a subclass that intercepts update()
    calls, converting frame-level progress into percentage-based callbacks.
    Used by MLX backend to capture mlx_whisper's internal progress bar.

    Args:
        progress_callback: Callback(stage_name, percent) or None.
        stage_name: Stage name to report (e.g., 'transcribing').
    """
    if not progress_callback:
        yield
        return

    import tqdm as tqdm_module

    OriginalTqdm = tqdm_module.tqdm

    class _ProgressTqdm(OriginalTqdm):
        def update(self, n=1):
            super().update(n)
            if self.total and self.total > 0:
                pct = (self.n / self.total) * 99
                progress_callback(stage_name, min(pct, 99))

    tqdm_module.tqdm = _ProgressTqdm
    try:
        yield
    finally:
        tqdm_module.tqdm = OriginalTqdm
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backends.py::TestTqdmProgressHook -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/backends/base.py tests/test_backends.py
git commit -m "feat(progress): add tqdm_progress_hook for tqdm interception"
```

---

### Task 3: Integrate tqdm hook into MLX backend (tests)

**Files:**
- Test: `tests/test_mlx_backend.py` (append new test class)

**Step 6: Write the failing tests**

Append to `tests/test_mlx_backend.py`:

```python
from unittest.mock import patch, MagicMock


class TestMLXGranularProgress:
    """Test granular progress in MLX transcription."""

    @patch("src.backends.mlx_backend.mlx_whisper", create=True)
    def test_transcribe_uses_tqdm_hook(self, mock_mlx):
        """Transcription should use tqdm_progress_hook for granular progress."""
        mock_mlx.transcribe.return_value = {
            "segments": [{"start": 0, "end": 1, "text": "test"}],
            "language": "en",
        }

        reported_stages = []

        def callback(stage, pct):
            reported_stages.append(stage)

        backend = MLXBackend(model_size="base")

        with patch.object(backend, "is_available", return_value=True):
            backend.transcribe("test.wav", progress_callback=callback)

        # Should have reported "transcribing" stage (not just 0 and 100)
        assert "transcribing" in reported_stages

    @patch("src.backends.mlx_backend.mlx_whisper", create=True)
    def test_transcribe_works_without_callback(self, mock_mlx):
        """Transcription should work fine without progress callback."""
        mock_mlx.transcribe.return_value = {
            "segments": [{"start": 0, "end": 1, "text": "test"}],
            "language": "en",
        }

        backend = MLXBackend(model_size="base")

        with patch.object(backend, "is_available", return_value=True):
            result = backend.transcribe("test.wav", progress_callback=None)

        assert len(result.segments) == 1
```

**Step 7: Run tests to verify they fail**

Run: `pytest tests/test_mlx_backend.py::TestMLXGranularProgress -v`
Expected: FAIL (import error or no tqdm_progress_hook usage)

---

### Task 4: Integrate tqdm hook into MLX backend (implementation)

**Files:**
- Modify: `src/backends/mlx_backend.py:10-14` (add import)
- Modify: `src/backends/mlx_backend.py:149-170` (wrap transcribe call)

**Step 8: Add import**

In `src/backends/mlx_backend.py`, add `tqdm_progress_hook` to the imports from `.base`:

```python
from .base import (
    TranscriptionBackend,
    TranscriptionResult,
    diarization_progress_hook,
    patch_hf_hub_compat,
    tqdm_progress_hook,
)
```

**Step 9: Wrap transcription with tqdm hook**

Replace lines 159-167 in `mlx_backend.py` (the transcribing block):

```python
        if progress_callback:
            progress_callback("transcribing", 0)

        with tqdm_progress_hook(progress_callback, "transcribing"):
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=model_name,
                language=language,
                word_timestamps=self._enable_diarization,
            )

        if progress_callback:
            progress_callback("transcribing", 100)
```

**Step 10: Run tests to verify they pass**

Run: `pytest tests/test_mlx_backend.py -v`
Expected: All tests PASS

**Step 11: Commit**

```bash
git add src/backends/mlx_backend.py tests/test_mlx_backend.py
git commit -m "feat(progress): integrate tqdm hook into MLX backend for granular transcription progress"
```

---

### Task 5: Add `ProgressStreamer` to base.py (tests)

**Files:**
- Test: `tests/test_backends.py` (append new test class)

**Step 12: Write the failing tests**

Append to `tests/test_backends.py`:

```python
class TestProgressStreamer:
    """Tests for ProgressStreamer (transformers BaseStreamer)."""

    def test_put_calls_callback_with_progress(self):
        """Each put() call should report progress percentage."""
        from src.backends.base import ProgressStreamer
        import torch

        reported = []

        def callback(stage, pct):
            reported.append((stage, round(pct, 1)))

        streamer = ProgressStreamer(
            max_tokens=100,
            progress_callback=callback,
            stage_name="transcribing",
        )

        # Simulate 10 tokens generated
        streamer.put(torch.tensor([42]))
        assert len(reported) == 1
        assert reported[0] == ("transcribing", 1.0)

        # Simulate another 10
        for _ in range(9):
            streamer.put(torch.tensor([42]))
        assert len(reported) == 10
        assert reported[-1] == ("transcribing", 9.9)

    def test_caps_at_99(self):
        """Should never report more than 99%."""
        from src.backends.base import ProgressStreamer
        import torch

        reported = []

        def callback(stage, pct):
            reported.append(pct)

        streamer = ProgressStreamer(
            max_tokens=5,
            progress_callback=callback,
            stage_name="transcribing",
        )

        for _ in range(10):  # more than max_tokens
            streamer.put(torch.tensor([42]))

        assert all(p <= 99 for p in reported)

    def test_end_is_noop(self):
        """end() should not crash or call callback."""
        from src.backends.base import ProgressStreamer

        callback = MagicMock()
        streamer = ProgressStreamer(
            max_tokens=100,
            progress_callback=callback,
            stage_name="transcribing",
        )

        streamer.end()
        callback.assert_not_called()

    def test_handles_batch_tokens(self):
        """Should handle batched token tensors."""
        from src.backends.base import ProgressStreamer
        import torch

        reported = []

        def callback(stage, pct):
            reported.append(pct)

        streamer = ProgressStreamer(
            max_tokens=100,
            progress_callback=callback,
            stage_name="transcribing",
        )

        # Batch of 5 tokens
        streamer.put(torch.tensor([1, 2, 3, 4, 5]))
        assert len(reported) == 1
        # 5/100 * 99 = 4.95
        assert round(reported[0], 1) == 5.0
```

**Step 13: Run tests to verify they fail**

Run: `pytest tests/test_backends.py::TestProgressStreamer -v`
Expected: FAIL with `ImportError: cannot import name 'ProgressStreamer'`

---

### Task 6: Implement `ProgressStreamer` in base.py

**Files:**
- Modify: `src/backends/base.py` (add after `tqdm_progress_hook`)

**Step 14: Write the implementation**

Add to `src/backends/base.py`:

```python
class ProgressStreamer:
    """Streamer for transformers model.generate() that reports progress.

    Implements the BaseStreamer interface (put/end) expected by
    transformers' generate() method. Counts generated tokens and
    reports progress as a percentage of max_tokens.

    Used by Granite backend for per-token transcription progress.

    Note:
        Inherits from transformers.BaseStreamer when available,
        falls back to standalone implementation for testing.
    """

    def __init__(
        self,
        max_tokens: int,
        progress_callback: Callable[[str, float], None],
        stage_name: str = "transcribing",
    ):
        """Initialize progress streamer.

        Args:
            max_tokens: Maximum tokens to generate (for percentage calculation).
            progress_callback: Callback(stage_name, percent).
            stage_name: Stage name to report.
        """
        self.max_tokens = max_tokens
        self.tokens_generated = 0
        self.progress_callback = progress_callback
        self.stage_name = stage_name

    def put(self, value) -> None:
        """Called by generate() for each new token batch.

        Args:
            value: Tensor of newly generated token IDs.
        """
        if hasattr(value, "shape") and len(value.shape) > 0:
            self.tokens_generated += value.shape[0]
        else:
            self.tokens_generated += 1
        pct = (self.tokens_generated / self.max_tokens) * 99
        self.progress_callback(self.stage_name, min(pct, 99))

    def end(self) -> None:
        """Called by generate() when generation is complete."""
        pass
```

**Step 15: Run tests to verify they pass**

Run: `pytest tests/test_backends.py::TestProgressStreamer -v`
Expected: All 4 tests PASS

**Step 16: Commit**

```bash
git add src/backends/base.py tests/test_backends.py
git commit -m "feat(progress): add ProgressStreamer for transformers generate() progress"
```

---

### Task 7: Integrate ProgressStreamer into Granite backend (tests)

**Files:**
- Test: `tests/test_granite_backend.py` (append new test class)

**Step 17: Write the failing tests**

Append to `tests/test_granite_backend.py`:

```python
class TestGraniteGranularProgress:
    """Test granular progress in Granite transcription."""

    def test_transcribe_passes_streamer_to_generate(self):
        """model.generate() should receive a ProgressStreamer."""
        from src.backends.base import ProgressStreamer

        # Mock all heavy dependencies
        mock_processor = MagicMock()
        mock_tokenizer = MagicMock()
        mock_processor.tokenizer = mock_tokenizer
        mock_tokenizer.apply_chat_template.return_value = "prompt"

        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.__getitem__ = MagicMock(return_value=MagicMock())
        mock_model.generate.return_value = mock_outputs

        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=(1, 10)))
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_diarize = MagicMock()
        mock_audio = MagicMock()

        callback = MagicMock()

        with patch("src.backends.granite_backend.AutoProcessor") as mock_ap, \
             patch("src.backends.granite_backend.AutoModelForSpeechSeq2Seq") as mock_am, \
             patch("src.backends.granite_backend.torch") as mock_torch, \
             patch("src.backends.granite_backend.torchaudio") as mock_ta, \
             patch("src.backends.granite_backend.whisperx", create=True) as mock_wx, \
             patch("src.backends.granite_backend.DiarizationPipeline", create=True) as mock_dp, \
             patch("src.backends.granite_backend.patch_hf_hub_compat"), \
             patch("src.backends.granite_backend.diarization_progress_hook"):

            mock_ap.from_pretrained.return_value = mock_processor
            mock_am.from_pretrained.return_value = mock_model
            mock_ta.load.return_value = (MagicMock(), 16000)
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()

            backend = GraniteBackend(hf_token="test")

            with patch.object(backend, "is_available", return_value=True), \
                 patch.object(backend, "_align_transcription_with_diarization",
                              return_value=[{"start": 0, "end": 1, "text": "t", "speaker": "S0"}]):
                backend.transcribe("test.wav", progress_callback=callback)

            # Check that generate was called with a streamer
            generate_kwargs = mock_model.generate.call_args
            assert "streamer" in generate_kwargs.kwargs or \
                   any(isinstance(a, ProgressStreamer) for a in generate_kwargs.args)
```

**Step 18: Run tests to verify they fail**

Run: `pytest tests/test_granite_backend.py::TestGraniteGranularProgress -v`
Expected: FAIL (no streamer passed to generate)

---

### Task 8: Integrate ProgressStreamer into Granite backend (implementation)

**Files:**
- Modify: `src/backends/granite_backend.py:10-15` (add import)
- Modify: `src/backends/granite_backend.py:193-200` (add streamer to generate call)

**Step 19: Add import**

In `src/backends/granite_backend.py`, add `ProgressStreamer` to imports:

```python
from .base import (
    TranscriptionBackend,
    TranscriptionResult,
    diarization_progress_hook,
    patch_hf_hub_compat,
    ProgressStreamer,
)
```

**Step 20: Add streamer to model.generate()**

Replace the `model.generate()` block (lines 193-200) in `granite_backend.py`:

```python
        # Generate transcription
        generate_kwargs = dict(
            **model_inputs,
            max_new_tokens=448,
            do_sample=False,
            num_beams=1,
        )
        if progress_callback:
            generate_kwargs["streamer"] = ProgressStreamer(
                max_tokens=448,
                progress_callback=progress_callback,
                stage_name="transcribing",
            )

        with torch.no_grad():
            model_outputs = model.generate(**generate_kwargs)
```

**Step 21: Run tests to verify they pass**

Run: `pytest tests/test_granite_backend.py -v`
Expected: All tests PASS

**Step 22: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (no regressions)

**Step 23: Commit**

```bash
git add src/backends/granite_backend.py tests/test_granite_backend.py
git commit -m "feat(progress): integrate ProgressStreamer into Granite backend for per-token progress"
```

---

### Task 9: Final verification and docs update

**Files:**
- Modify: `CLAUDE.md` (update features list)

**Step 24: Run full test suite one final time**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 25: Update CLAUDE.md**

Add to the "Progresso em Tempo Real" section in CLAUDE.md:

```
- ✅ Progresso granular na transcrição MLX via interceptação do tqdm
- ✅ Progresso granular na transcrição Granite via BaseStreamer (per-token)
```

**Step 26: Final commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with granular progress for MLX and Granite"
```
