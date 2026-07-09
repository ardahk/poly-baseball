from polybot.backtest import (
    _CalibResult,
    _brier,
    _gamma_home_token,
    _gamma_slug,
    _logloss,
    _naive_leader,
    _state_at,
)
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


def test_gamma_slug_uses_lowercase_mlb_abbreviations():
    assert _gamma_slug("Arizona Diamondbacks", "San Diego Padres",
                       "2026-07-07") == "mlb-ari-sd-2026-07-07"
    assert _gamma_slug("New York Yankees", "Boston Red Sox",
                       "2026-07-08") == "mlb-nyy-bos-2026-07-08"


def test_gamma_slug_unknown_team_returns_none():
    assert _gamma_slug("Springfield Isotopes", "San Diego Padres",
                       "2026-07-07") is None


_GAMMA_EVENT = {
    "title": "Diamondbacks vs. Padres",
    "markets": [
        {
            "question": "Will the Diamondbacks vs. Padres game go to extra innings?",
            "clobTokenIds": '["901", "902"]',
            "outcomes": '["Yes", "No"]',
        },
        {
            "question": "Diamondbacks vs. Padres",
            "clobTokenIds": '["111", "222"]',
            "outcomes": '["Diamondbacks", "Padres"]',
        },
    ],
}


def test_gamma_home_token_picks_moneyline_home_outcome():
    assert _gamma_home_token(_GAMMA_EVENT, "San Diego Padres") == "222"
    assert _gamma_home_token(_GAMMA_EVENT, "Arizona Diamondbacks") == "111"


def test_gamma_home_token_missing_moneyline_returns_none():
    event = {"title": "Diamondbacks vs. Padres",
             "markets": [_GAMMA_EVENT["markets"][0]]}
    assert _gamma_home_token(event, "San Diego Padres") is None


def test_gamma_home_token_malformed_json_returns_none():
    event = {
        "title": "Diamondbacks vs. Padres",
        "markets": [{
            "question": "Diamondbacks vs. Padres",
            "clobTokenIds": "not json",
            "outcomes": '["Diamondbacks", "Padres"]',
        }],
    }
    assert _gamma_home_token(event, "San Diego Padres") is None


def test_state_at_returns_most_recent():
    tl = [(100.0, GameState(1, inning=1)), (200.0, GameState(1, inning=3)),
          (300.0, GameState(1, inning=5))]
    assert _state_at(tl, 50.0) is None
    assert _state_at(tl, 150.0).inning == 1
    assert _state_at(tl, 250.0).inning == 3
    assert _state_at(tl, 999.0).inning == 5
