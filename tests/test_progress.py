"""Tests for progress reporting."""
import time
from unittest.mock import patch

import pytest
from io import StringIO
from src.progress import ProgressReporter, Stage


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


class TestStagnationDetection:
    """Tests for ETA stagnation detection."""

    def test_not_stagnant_at_low_percent(self):
        reporter = ProgressReporter(total_stages=4, lang="en")
        reporter._current_percent = 50
        # Even with old change time, not stagnant below 95%
        reporter._last_pct_change_time = time.monotonic() - 20
        assert reporter._is_stagnant() is False

    def test_not_stagnant_when_recently_changed(self):
        reporter = ProgressReporter(total_stages=4, lang="en")
        reporter._current_percent = 99
        reporter._last_pct_change_time = time.monotonic()
        assert reporter._is_stagnant() is False

    def test_stagnant_at_high_percent_with_no_change(self):
        reporter = ProgressReporter(total_stages=4, lang="en")
        reporter._current_percent = 99
        reporter._last_pct_change_time = time.monotonic() - 15
        assert reporter._is_stagnant() is True

    def test_stagnant_at_95_percent(self):
        reporter = ProgressReporter(total_stages=4, lang="en")
        reporter._current_percent = 95
        reporter._last_pct_change_time = time.monotonic() - 11
        assert reporter._is_stagnant() is True

    def test_stagnant_shows_finalizing_english(self, capsys):
        reporter = ProgressReporter(total_stages=4, lang="en")
        reporter._current_percent = 99
        reporter._last_pct_value = 99  # Same as update value so no reset
        reporter._last_pct_change_time = time.monotonic() - 15
        reporter.update(Stage.DIARIZING, 99)
        captured = capsys.readouterr()
        assert "finalizing..." in captured.out

    def test_stagnant_shows_finalizando_portuguese(self, capsys):
        reporter = ProgressReporter(total_stages=4, lang="pt")
        reporter._current_percent = 99
        reporter._last_pct_value = 99  # Same as update value so no reset
        reporter._last_pct_change_time = time.monotonic() - 15
        reporter.update(Stage.DIARIZING, 99)
        captured = capsys.readouterr()
        assert "finalizando..." in captured.out

    def test_update_resets_stagnation_on_pct_change(self):
        reporter = ProgressReporter(total_stages=4, lang="en")
        reporter._current_percent = 95
        reporter._last_pct_change_time = time.monotonic() - 15
        reporter._last_pct_value = 95
        # Update with new percentage (>0.5 diff)
        reporter.update(Stage.DIARIZING, 97)
        assert reporter._is_stagnant() is False

    def test_advance_resets_stagnation_tracking(self):
        reporter = ProgressReporter(total_stages=4, lang="en")
        reporter._last_pct_change_time = time.monotonic() - 100
        reporter._last_pct_value = 99
        reporter.update(Stage.LOADING, 100)
        reporter.advance()
        assert reporter._last_pct_value == 0
        # Should not be stagnant after advance
        reporter._current_percent = 0
        assert reporter._is_stagnant() is False
