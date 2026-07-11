import pytest

from polybot.model_features import ModelHistory, transfer_model_delta
from polybot.models import GameState


def probability(gs):
    return 0.70 if gs.home_score > gs.away_score else 0.50


def test_state_transition_uses_only_price_known_before_state():
    history = ModelHistory(probability)
    history.add_price(0.55, 1.0)
    first = GameState(1, home_score=0, away_score=0, status="Live")
    changed = GameState(1, home_score=1, away_score=0, status="Live")
    assert history.observe_state(first, 2.0)
    history.add_price(0.56, 3.0)
    assert history.observe_state(changed, 4.0)
    # A post-state quote cannot become its own anchor.
    history.add_price(0.66, 4.0)
    view = history.state_view(0.66, 4.0)
    assert view.anchor_price == 0.56
    assert view.anchor_model == 0.50
    assert view.current_model == 0.70


def test_duplicate_state_does_not_reanchor():
    history = ModelHistory(probability)
    gs = GameState(1, status="Live")
    history.add_price(0.50, 1.0)
    assert history.observe_state(gs, 2.0)
    history.add_price(0.60, 3.0)
    assert not history.observe_state(gs, 4.0)
    assert history.state_view(0.60, 4.0) is None


def test_pregame_anchor_freezes_at_first_live_state():
    history = ModelHistory(probability, pregame_model_home=0.54)
    history.add_price(0.42, 1.0)
    history.add_price(0.44, 2.0)
    history.observe_state(GameState(1, status="Live"), 3.0)
    history.add_price(0.60, 4.0)
    view = history.market_view(0.60, 4.0)
    assert view.anchor_price == 0.44
    assert view.anchor_model == 0.54


def test_price_after_scheduled_start_cannot_replace_pregame_anchor():
    history = ModelHistory(probability, pregame_model_home=0.54)
    history.add_price(0.42, 1.0)
    history.add_price(0.80, 2.0, pregame_eligible=False)
    history.observe_state(GameState(1, status="Live"), 3.0)
    assert history.market_view(0.80, 3.0).anchor_price == 0.42


def test_pregame_model_baseline_comes_from_active_predictor():
    history = ModelHistory(lambda gs: 0.61)
    history.add_price(0.45, 1.0)
    history.observe_state(GameState(1, status="Live"), 2.0)
    assert history.market_view(0.45, 2.0).anchor_model == 0.61


def test_reset_rolling_preserves_frozen_pregame_anchor():
    history = ModelHistory(probability, pregame_model_home=0.54)
    history.add_price(0.44, 1.0)
    history.observe_state(GameState(1, status="Live"), 2.0)
    history.add_price(0.50, 3.0)

    history.reset_rolling()  # mid-game data gap

    assert history.state_view(0.50, 4.0) is None  # rolling anchors discarded
    gs = GameState(1, home_score=1, away_score=0, status="Live")
    history.observe_state(gs, 5.0)
    view = history.market_view(0.60, 5.0)
    assert view is not None and view.anchor_price == 0.44  # prior survived
    # A live post-gap price still cannot become the pregame anchor.
    history.add_price(0.70, 6.0, pregame_eligible=False)
    assert history.market_view(0.70, 6.0).anchor_price == 0.44


def test_log_odds_transfer_is_symmetric_and_finite_near_boundaries():
    home = transfer_model_delta(0.40, 0.50, 0.70)
    away = transfer_model_delta(0.60, 0.50, 0.30)
    assert home == pytest.approx(1.0 - away)
    assert 0 < transfer_model_delta(0.001, 0.001, 0.999) < 1
