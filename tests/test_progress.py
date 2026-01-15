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
