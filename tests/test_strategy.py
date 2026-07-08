import time

from polybot.config import StrategyConfig
from polybot.models import GameState, Market, Position
from polybot.strategy import check_entry, check_exit
from polybot.volatility import PriceHistory


def make_market():
    return Market(condition_id="c1", question="A vs. B",
                  home_team="Homers", away_team="Awayers",
                  home_token="HT", away_token="AT", game_pk=1)


def live_gs(**kw):
    return GameState(game_pk=1, status="Live", **kw)


def playful_history(prices, step=30.0):
    h = PriceHistory(flip_band=0.03)
    for i, p in enumerate(prices):
        h.add(p, ts=i * step)
    return h


CFG = StrategyConfig(move_lookback_secs=60, move_threshold=0.08, min_edge=0.05,
                     min_flips=2, min_volatility=99.0)


def test_entry_fades_drop_when_model_disagrees():
    # playful (2 flips), then a sharp drop to 0.40 while home leads big -> buy home
    h = playful_history([0.60, 0.40, 0.60, 0.60, 0.40])
    gs = live_gs(inning=7, is_top=True, home_score=4, away_score=1)  # fair ~high
    sig = check_entry(make_market(), h, gs, CFG)
    assert sig is not None
    assert sig.token == "HT"
    assert sig.fair > sig.price


def test_no_entry_when_model_agrees_with_move():
    # price dropped and the model also says home is weak -> no fade
    h = playful_history([0.60, 0.40, 0.60, 0.60, 0.40])
    gs = live_gs(inning=7, is_top=True, home_score=1, away_score=4)
    sig = check_entry(make_market(), h, gs, CFG)
    assert sig is None


def test_entry_buys_away_token_on_spike():
    # home token spiked to 0.62 but home is losing -> away side undervalued
    h = playful_history([0.42, 0.60, 0.42, 0.42, 0.62])
    gs = live_gs(inning=7, is_top=True, home_score=1, away_score=4)
    sig = check_entry(make_market(), h, gs, CFG)
    assert sig is not None
    assert sig.token == "AT"


def test_no_entry_without_playfulness():
    h = PriceHistory(flip_band=0.03)
    for i, p in enumerate([0.60, 0.60, 0.60, 0.48]):
        h.add(p, ts=i * 30.0)
    assert h.flips < 2
    gs = live_gs(inning=7, is_top=True, home_score=4, away_score=1)
    assert check_entry(make_market(), h, gs, CFG) is None


def test_no_entry_when_game_not_live():
    h = playful_history([0.60, 0.40, 0.60, 0.60, 0.40])
    gs = GameState(game_pk=1, status="Scheduled")
    assert check_entry(make_market(), h, gs, CFG) is None
    assert check_entry(make_market(), h, None, CFG) is None


def pos(entry=0.50, opened_offset=0.0):
    return Position(strategy="math", market_key="c1", token="HT", team="Homers",
                    qty=20.0, entry_price=entry,
                    opened_at=time.time() - opened_offset)


def test_exit_take_profit():
    cfg = StrategyConfig(take_profit=0.12, stop_loss=0.10)
    assert "take profit" in check_exit(pos(0.50), 0.57, None, False, cfg)


def test_exit_stop_loss():
    cfg = StrategyConfig(take_profit=0.12, stop_loss=0.10)
    assert "stop loss" in check_exit(pos(0.50), 0.44, None, False, cfg)


def test_exit_time_stop():
    cfg = StrategyConfig(max_hold_secs=900)
    assert "time stop" in check_exit(pos(0.50, opened_offset=1000), 0.51, None, False, cfg)


def test_exit_on_game_final():
    cfg = StrategyConfig()
    assert "game final" in check_exit(pos(0.50), 0.99, None, True, cfg)


def test_exit_edge_gone():
    cfg = StrategyConfig(edge_exit=0.03)
    assert "edge gone" in check_exit(pos(0.50), 0.52, 0.45, False, cfg)


def test_hold_otherwise():
    cfg = StrategyConfig()
    assert check_exit(pos(0.50), 0.52, 0.55, False, cfg) is None
