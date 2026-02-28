"""Progress reporting for transcription pipeline."""
import sys
import threading
import time
from enum import Enum


class Stage(Enum):
    """Transcription pipeline stages."""

    LOADING = ("loading", "Loading model", "Carregando modelo")
    VAD = ("vad", "Detecting speech", "Detectando fala")
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


SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like '1m23s', '45s', or '2h05m'.
    """
    if seconds < 0:
        return "---"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class ProgressReporter:
    """Reports progress during transcription.

    Displays progress with a live spinner and elapsed timer:
        [2/5] Transcribing... [████████░░░░] 65%  1m23s | ~1m remaining
    When no percentage updates arrive, the spinner and timer keep moving
    to show the process is still alive.
    """

    def __init__(
        self, total_stages: int, lang: str = "en", width: int = 20, file=None
    ):
        """Initialize progress reporter.

        Args:
            total_stages: Total number of stages in pipeline.
            lang: Language for labels ('en' or 'pt').
            width: Width of progress bar in characters.
            file: Explicit output file object. Defaults to sys.stdout.
        """
        self.total_stages = total_stages
        self.lang = lang
        self.width = width
        self.current_stage = 1
        # Accept explicit output file or capture real stdout before
        # SuppressOutput can replace it
        self._stdout = file if file is not None else sys.stdout
        # Timing
        self._start_time = time.monotonic()
        self._stage_start_time = self._start_time
        # Stagnation detection
        self._last_pct_change_time = self._start_time
        self._last_pct_value: float = 0
        # Render throttle
        self._last_render_time: float = 0
        # Spinner state
        self._spinner_idx = 0
        self._current_stage: Stage | None = None
        self._current_percent: float = 0
        self._lock = threading.Lock()
        self._ticker_stop = threading.Event()
        self._ticker_thread: threading.Thread | None = None

    def _start_ticker(self) -> None:
        """Start background thread that refreshes display every 500ms."""
        if self._ticker_thread is not None and self._ticker_thread.is_alive():
            return
        self._ticker_stop.clear()
        self._ticker_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._ticker_thread.start()

    def _stop_ticker(self) -> None:
        """Stop the background ticker thread."""
        self._ticker_stop.set()
        if self._ticker_thread is not None:
            self._ticker_thread.join(timeout=1)
            self._ticker_thread = None

    def _tick_loop(self) -> None:
        """Background loop: refresh display to update spinner and timer."""
        while not self._ticker_stop.is_set():
            try:
                with self._lock:
                    if self._current_stage is not None:
                        self._render()
            except (ValueError, OSError):
                # stdout was closed (e.g., in tests), stop silently
                break
            self._ticker_stop.wait(0.25)

    def _render_bar(self, percent: float) -> str:
        """Render progress bar."""
        filled = int(self.width * percent / 100)
        empty = self.width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    def _estimate_remaining(self, percent: float) -> float | None:
        """Estimate remaining seconds for current stage.

        Args:
            percent: Current progress percentage (0-100).

        Returns:
            Estimated remaining seconds, or None if not enough data.
        """
        if percent <= 0:
            return None
        elapsed = time.monotonic() - self._stage_start_time
        if elapsed < 2:
            return None
        total_estimated = elapsed / (percent / 100)
        return total_estimated - elapsed

    def _is_stagnant(self) -> bool:
        """Detect stagnant progress (no change for >10s at >=95%)."""
        if self._current_percent < 95:
            return False
        return (time.monotonic() - self._last_pct_change_time) > 10

    def _format_time_info(self, percent: float) -> str:
        """Format elapsed time and ETA string.

        Args:
            percent: Current progress percentage (0-100).

        Returns:
            Formatted time info string.
        """
        elapsed = time.monotonic() - self._start_time
        elapsed_str = format_duration(elapsed)

        if self._is_stagnant():
            label = "finalizando..." if self.lang == "pt" else "finalizing..."
            return f"{elapsed_str} | {label}"

        remaining = self._estimate_remaining(percent)
        if remaining is not None and percent < 100:
            remaining_str = format_duration(remaining)
            if self.lang == "pt":
                return f"{elapsed_str} | ~{remaining_str} restante"
            return f"{elapsed_str} | ~{remaining_str} remaining"

        return elapsed_str

    def _render(self) -> None:
        """Render the current progress line to stdout. Must be called with lock held."""
        stage = self._current_stage
        percent = self._current_percent
        spinner = SPINNER_CHARS[self._spinner_idx % len(SPINNER_CHARS)]
        self._spinner_idx += 1

        label = stage.label(self.lang)
        bar = self._render_bar(percent)
        time_info = self._format_time_info(percent)
        # \033[K clears from cursor to end of line, preventing ghost text
        # from longer previous renders
        line = (
            f"\r{spinner} [{self.current_stage}/{self.total_stages}] "
            f"{label}... {bar} {percent:.0f}%  {time_info}\033[K"
        )
        self._stdout.write(line)
        self._stdout.flush()

    def update(self, stage: Stage, percent: float) -> None:
        """Update progress display.

        Args:
            stage: Current pipeline stage.
            percent: Progress percentage (0-100).
        """
        with self._lock:
            if abs(percent - self._last_pct_value) > 0.5:
                self._last_pct_change_time = time.monotonic()
                self._last_pct_value = percent
            self._current_stage = stage
            self._current_percent = percent
            now = time.monotonic()
            if now - self._last_render_time >= 0.1:  # 100ms throttle
                self._render()
                self._last_render_time = now
        self._start_ticker()

    def advance(self) -> None:
        """Move to next stage.

        Forces a final render at 100% before advancing, ensuring the
        completed stage always displays 100% regardless of throttle state.
        """
        self._stop_ticker()
        with self._lock:
            if self._current_stage is not None:
                self._current_percent = 100
                self._last_pct_change_time = time.monotonic()
                self._render()
        self.current_stage += 1
        self._stage_start_time = time.monotonic()
        self._last_pct_change_time = time.monotonic()
        self._last_pct_value = 0
        print(file=self._stdout)

    def complete(self, output_path: str, duration_seconds: float) -> None:
        """Show completion message.

        Args:
            output_path: Path to output file.
            duration_seconds: Total duration in seconds.
        """
        self._stop_ticker()
        print(file=self._stdout)

        duration_str = format_duration(duration_seconds)

        if self.lang == "pt":
            print(f"✓ Transcrição completa: {output_path} ({duration_str})", file=self._stdout)
        else:
            print(f"✓ Transcription complete: {output_path} ({duration_str})", file=self._stdout)

    def error(self, message: str) -> None:
        """Show error message.

        Args:
            message: Error message to display.
        """
        self._stop_ticker()
        print(file=self._stdout)
        if self.lang == "pt":
            print(f"✗ Erro: {message}", file=sys.stderr)
        else:
            print(f"✗ Error: {message}", file=sys.stderr)
