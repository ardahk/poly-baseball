from polybot.backtest import _CalibResult, _brier, _logloss, _naive_leader, _state_at
from polybot.models import GameState


def test_brier_perfect_and_worst():
    perfect = [_CalibResult(1.0, 1), _CalibResult(0.0, 0)]
    worst = [_CalibResult(0.0, 1), _CalibResult(1.0, 0)]
    assert _brier(perfect) == 0.0
    assert _brier(worst) == 1.0


def test_brier_middle():
    rows = [_CalibResult(0.5, 1), _CalibResult(0.5, 0)]
    assert _brier(rows) == 0.25


def test_logloss_rewards_confidence_when_right():
    confident = [_CalibResult(0.9, 1)]
    unsure = [_CalibResult(0.6, 1)]
    assert _logloss(confident) < _logloss(unsure)


def test_naive_leader():
    assert _naive_leader(GameState(1, home_score=3, away_score=1)) == 0.75
    assert _naive_leader(GameState(1, home_score=1, away_score=3)) == 0.25
    assert _naive_leader(GameState(1, home_score=2, away_score=2)) == 0.54


def test_state_at_returns_most_recent():
    tl = [(100.0, GameState(1, inning=1)), (200.0, GameState(1, inning=3)),
          (300.0, GameState(1, inning=5))]
    assert _state_at(tl, 50.0) is None
    assert _state_at(tl, 150.0).inning == 1
    assert _state_at(tl, 250.0).inning == 3
    assert _state_at(tl, 999.0).inning == 5
