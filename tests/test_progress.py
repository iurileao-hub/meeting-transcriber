"""Tests for progress reporting."""
import time
from io import StringIO
from unittest.mock import patch

import pytest
from src.progress import ProgressReporter, Stage, format_duration


class TestStage:
    """Tests for Stage enum."""

    def test_stage_has_english_label(self):
        assert Stage.LOADING.label("en") == "Loading model"

    def test_stage_has_portuguese_label(self):
        assert Stage.LOADING.label("pt") == "Carregando modelo"

    def test_vad_stage_english(self):
        assert Stage.VAD.label("en") == "Detecting speech"

    def test_vad_stage_portuguese(self):
        assert Stage.VAD.label("pt") == "Detectando fala"

    def test_all_stages_have_labels(self):
        for stage in Stage:
            assert stage.label("en") is not None
            assert stage.label("pt") is not None


class TestFormatDuration:
    """Tests for format_duration()."""

    def test_negative_returns_dashes(self):
        assert format_duration(-1) == "---"

    def test_zero_seconds(self):
        assert format_duration(0) == "0s"

    def test_seconds_only(self):
        assert format_duration(45) == "45s"

    def test_minutes_and_seconds(self):
        assert format_duration(83) == "1m23s"

    def test_hours_and_minutes(self):
        assert format_duration(7500) == "2h05m"


class TestProgressReporter:
    """Tests for ProgressReporter."""

    def test_update_renders_progress_bar(self):
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter.update(Stage.TRANSCRIBING, 50)
        output = buf.getvalue()
        assert "Transcribing" in output
        assert "50%" in output

    def test_stage_number_increments(self):
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter.update(Stage.LOADING, 100)
        reporter.advance()
        reporter.update(Stage.TRANSCRIBING, 0)
        output = buf.getvalue()
        assert "[2/4]" in output

    def test_complete_shows_checkmark(self):
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter.complete("output.json", 120.5)
        output = buf.getvalue()
        assert "output.json" in output

    def test_portuguese_labels(self):
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="pt", file=buf)
        reporter.update(Stage.TRANSCRIBING, 50)
        output = buf.getvalue()
        assert "Transcrevendo" in output

    def test_explicit_file_parameter(self):
        """ProgressReporter should write to the explicit file object."""
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter.update(Stage.LOADING, 25)
        output = buf.getvalue()
        assert "Loading model" in output
        assert "25%" in output

    def test_default_file_is_stdout(self, capsys):
        """Without explicit file, should write to sys.stdout."""
        reporter = ProgressReporter(total_stages=4, lang="en")
        reporter.update(Stage.LOADING, 10)
        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    def test_error_method_outputs_error_message(self, capsys):
        """error() should print error message to stderr."""
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter.error("something went wrong")
        captured = capsys.readouterr()
        assert "Error" in captured.err
        assert "something went wrong" in captured.err

    def test_error_method_portuguese(self, capsys):
        """error() should use Portuguese label when lang=pt."""
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="pt", file=buf)
        reporter.error("algo deu errado")
        captured = capsys.readouterr()
        assert "Erro" in captured.err
        assert "algo deu errado" in captured.err

    def test_error_stops_ticker(self):
        """error() should stop the ticker thread."""
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter.update(Stage.LOADING, 50)
        reporter.error("test")
        # Ticker thread should be stopped (None or not alive)
        assert (
            reporter._ticker_thread is None
            or not reporter._ticker_thread.is_alive()
        )


class TestRenderThrottle:
    """Tests for render throttle behavior."""

    def test_first_update_always_renders(self):
        """First call to update() should always render."""
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter.update(Stage.LOADING, 10)
        assert buf.getvalue() != ""

    def test_rapid_updates_throttled(self):
        """Rapid successive updates should be throttled to ~100ms intervals."""
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)

        # First update renders
        reporter.update(Stage.LOADING, 10)
        first_output_len = len(buf.getvalue())
        assert first_output_len > 0

        # Rapid second update (within 100ms) should be throttled
        reporter.update(Stage.LOADING, 11)
        second_output_len = len(buf.getvalue())
        # The output length should not have increased (throttled)
        assert second_output_len == first_output_len

    def test_update_after_throttle_window_renders(self):
        """Update after 100ms throttle window should render."""
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)

        reporter.update(Stage.LOADING, 10)
        first_len = len(buf.getvalue())

        # Simulate time passing beyond throttle window
        reporter._last_render_time = time.monotonic() - 0.2
        reporter.update(Stage.LOADING, 50)
        second_len = len(buf.getvalue())
        assert second_len > first_len


class TestStagnationDetection:
    """Tests for ETA stagnation detection."""

    def test_not_stagnant_at_low_percent(self):
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter._current_percent = 50
        # Even with old change time, not stagnant below 95%
        reporter._last_pct_change_time = time.monotonic() - 20
        assert reporter._is_stagnant() is False

    def test_not_stagnant_when_recently_changed(self):
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter._current_percent = 99
        reporter._last_pct_change_time = time.monotonic()
        assert reporter._is_stagnant() is False

    def test_stagnant_at_high_percent_with_no_change(self):
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter._current_percent = 99
        reporter._last_pct_change_time = time.monotonic() - 15
        assert reporter._is_stagnant() is True

    def test_stagnant_at_95_percent(self):
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter._current_percent = 95
        reporter._last_pct_change_time = time.monotonic() - 11
        assert reporter._is_stagnant() is True

    def test_stagnant_shows_finalizing_english(self):
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter._current_percent = 99
        reporter._last_pct_value = 99  # Same as update value so no reset
        reporter._last_pct_change_time = time.monotonic() - 15
        reporter.update(Stage.DIARIZING, 99)
        output = buf.getvalue()
        assert "finalizing..." in output

    def test_stagnant_shows_finalizando_portuguese(self):
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="pt", file=buf)
        reporter._current_percent = 99
        reporter._last_pct_value = 99  # Same as update value so no reset
        reporter._last_pct_change_time = time.monotonic() - 15
        reporter.update(Stage.DIARIZING, 99)
        output = buf.getvalue()
        assert "finalizando..." in output

    def test_update_resets_stagnation_on_pct_change(self):
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter._current_percent = 95
        reporter._last_pct_change_time = time.monotonic() - 15
        reporter._last_pct_value = 95
        # Update with new percentage (>0.5 diff)
        reporter.update(Stage.DIARIZING, 97)
        assert reporter._is_stagnant() is False

    def test_advance_resets_stagnation_tracking(self):
        buf = StringIO()
        reporter = ProgressReporter(total_stages=4, lang="en", file=buf)
        reporter._last_pct_change_time = time.monotonic() - 100
        reporter._last_pct_value = 99
        reporter.update(Stage.LOADING, 100)
        reporter.advance()
        assert reporter._last_pct_value == 0
        # Should not be stagnant after advance
        reporter._current_percent = 0
        assert reporter._is_stagnant() is False
