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
