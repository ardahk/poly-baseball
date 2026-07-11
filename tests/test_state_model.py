from polybot.models import GameState
import json

import pytest

from polybot.state_model import EmpiricalStateModel, empirical_state_key, score_model


def _game(game_pk, states, home_won):
    return ([(float(i), GameState(game_pk, status="Live", **state))
             for i, state in enumerate(states)], home_won)


def test_empirical_fit_counts_each_distinct_state_once():
    state = {"inning": 5, "home_score": 2, "away_score": 1}
    games = [_game(1, [state, state], True), _game(2, [state], False)]
    model = EmpiricalStateModel.fit(games, prior_strength=0)
    cell = model.cells[empirical_state_key(GameState(9, status="Live", **state))]
    assert cell == {"count": 2, "home_wins": 1}


def test_empirical_artifact_round_trip(tmp_path):
    games = [_game(1, [{"inning": 9, "home_score": 2, "away_score": 1}], True)]
    model = EmpiricalStateModel.fit(games, prior_strength=10)
    path = tmp_path / "state.json"
    digest = model.save(path)
    loaded = EmpiricalStateModel.load(path)
    gs = games[0][0][0][1]
    assert len(digest) == 64
    assert loaded.predict(gs) == model.predict(gs)
    assert score_model(games, loaded.predict).games == 1


def test_artifact_checksum_detects_manual_edits(tmp_path):
    games = [_game(1, [{"inning": 9, "home_score": 2, "away_score": 1}], True)]
    path = tmp_path / "state.json"
    EmpiricalStateModel.fit(games).save(path)
    data = json.loads(path.read_text())
    data["prior_strength"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        EmpiricalStateModel.load(path)
