import json
from datetime import datetime

import pytest

from polybot.config import Config
from polybot.journal import Journal
from polybot import walkforward


def test_build_folds_are_28_7_7_and_roll_weekly():
    folds = walkforward.build_folds("2026-05-01", 2, "America/Los_Angeles")
    first = folds[0]
    second = folds[1]
    assert (datetime.fromisoformat(first["train"][1])
            - datetime.fromisoformat(first["train"][0])).days == 28
    assert (datetime.fromisoformat(first["validate"][1])
            - datetime.fromisoformat(first["validate"][0])).days == 7
    assert (datetime.fromisoformat(first["test"][1])
            - datetime.fromisoformat(first["test"][0])).days == 7
    assert (datetime.fromisoformat(second["train"][0])
            - datetime.fromisoformat(first["train"][0])).days == 7


def test_manifest_is_write_once_and_detects_tampering(tmp_path):
    db = tmp_path / "tape.db"
    Journal(str(db)).close()
    cfg = Config()
    cfg.ai.enabled = False
    cfg.engine.db_path = str(db)
    path = tmp_path / "prereg.json"
    manifest = walkforward.prepare_manifest(
        cfg, str(db), "2026-05-01", 1,
        "frozen strategy has positive expectancy", str(path),
    )
    assert manifest["windows"] == {
        "train_days": 28, "validate_days": 7,
        "locked_test_days": 7, "roll_days": 7,
    }
    with pytest.raises(FileExistsError, match="overwrite"):
        walkforward.prepare_manifest(
            cfg, str(db), "2026-05-01", 1, "again", str(path)
        )

    data = json.loads(path.read_text())
    data["hypothesis"] = "edited after reveal"
    path.write_text(json.dumps(data))
    journal = Journal(str(db))
    with pytest.raises(ValueError, match="checksum"):
        walkforward.load_manifest(str(path), cfg, journal)
    journal.close()


def test_manifest_detects_tape_change(tmp_path):
    db = tmp_path / "tape.db"
    cfg = Config()
    cfg.ai.enabled = False
    cfg.engine.db_path = str(db)
    path = tmp_path / "prereg.json"
    walkforward.prepare_manifest(
        cfg, str(db), "2026-05-01", 1, "registered", str(path)
    )
    journal = Journal(str(db))
    journal.conn.execute(
        """INSERT INTO game_states VALUES
           (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (1778000000.0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
         "Live", "run", 1778000000.0),
    )
    journal.conn.commit()
    with pytest.raises(ValueError, match="event tape changed"):
        walkforward.load_manifest(str(path), cfg, journal)
    journal.close()


def test_cluster_ci_resamples_clusters_not_ticks():
    low, high = walkforward._cluster_ci({"game-1": 1.0, "game-2": 3.0}, "seed")
    assert low == 1.0
    assert high == 3.0


def test_empty_tape_evaluation_writes_a_locked_failure(tmp_path):
    db = tmp_path / "empty.db"
    cfg = Config()
    cfg.ai.enabled = False
    cfg.engine.db_path = str(db)
    manifest = tmp_path / "prereg.json"
    result_path = tmp_path / "result.json"
    walkforward.prepare_manifest(
        cfg, str(db), "2026-05-01", 1, "registered", str(manifest)
    )
    result = walkforward.evaluate_manifest(
        cfg, str(db), str(manifest), str(result_path)
    )
    assert result_path.exists()
    assert result["folds"][0]["selected_strategy"] is None
    assert result["selected_champion_aggregate"]["passes_preregistered_rules"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        walkforward.evaluate_manifest(
            cfg, str(db), str(manifest), str(result_path)
        )


def test_locked_test_cannot_change_validation_selection(monkeypatch, tmp_path):
    db = tmp_path / "empty.db"
    Journal(str(db)).close()
    cfg = Config()
    cfg.ai.enabled = False
    metrics = lambda pnl: {
        "round_trips": 1, "net_pnl": pnl, "trading_days": 1, "games": 1,
        "fees": 0.1, "max_drawdown": 1.0, "top_day_profit_share": 0.2,
        "game_clustered_mean_pnl_ci95": [pnl, pnl],
    }

    manifest = {
        "manifest_sha256": "locked",
        "folds": [{"fold": 1, "train": ["a", "b"],
                   "validate": ["b", "c"], "test": ["c", "d"]}],
        "promotion_rules": {
            "min_round_trips": 0, "min_trading_days": 0, "min_games": 0,
            "require_positive_net_pnl": False,
            "require_positive_game_cluster_ci_low": False,
            "max_top_day_profit_share": 1.0,
        },
    }
    monkeypatch.setattr(walkforward, "load_manifest",
                        lambda path, current_cfg, journal: manifest)

    def fake_window(current_cfg, journal, window, label):
        if "validate" in label:
            values = {"alpha": metrics(1.0), "beta": metrics(2.0)}
        elif "locked" in label:
            values = {"alpha": metrics(100.0), "beta": metrics(-100.0)}
        else:
            values = {"alpha": metrics(0.0), "beta": metrics(0.0)}
        return {"strategies": values}

    monkeypatch.setattr(walkforward, "_run_window", fake_window)
    result = walkforward.evaluate_manifest(
        cfg, str(db), "ignored.json", str(tmp_path / "result.json")
    )
    assert result["folds"][0]["selected_strategy"] == "beta"
