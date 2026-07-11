import pytest

from polybot import research
from polybot.journal import Journal


def test_fit_rejects_training_holdout_overlap_before_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(research, "_season_games",
                        lambda *args, **kwargs: pytest.fail("must reject before fetching"))
    with pytest.raises(SystemExit, match="disjoint"):
        research.fit_state_artifact([2024, 2025], 2025, str(tmp_path / "model.json"))


def test_diagnose_empty_day_is_explicit(tmp_path, capsys):
    path = tmp_path / "journal.db"
    Journal(str(path)).close()
    research.diagnose_day(str(path), "2026-07-08", "America/Los_Angeles", 0.06)
    assert "No completed signal counterfactuals" in capsys.readouterr().out
