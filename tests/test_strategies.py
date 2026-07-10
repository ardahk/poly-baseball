import pytest

from polybot.config import Config, StrategyConfig
from polybot.models import GameState, Market, MarketQuote, Position
from polybot.strategies import (
    AIShadowStrategy,
    FadeStrategy,
    Intent,
    StratContext,
    build_strategies,
)
from polybot.ai_judge import Judgment
from polybot.volatility import PriceHistory


def market():
    return Market(slug="m1", question="A vs B", home_team="Homers",
                  away_team="Awayers", long_team="Homers", game_pk=1)


def ctx(quote):
    return StratContext(market=market(), history=PriceHistory(),
                        game_state=GameState(1, status="Live"),
                        quote=quote, now=1000.0)


# --------------------------------------------------------------- Task 1: context

def test_context_entry_price_is_executable_ask_per_side():
    q = MarketQuote("m1", home_bid=0.50, home_ask=0.52, long_bid=0.50, long_ask=0.52)
    c = ctx(q)
    assert c.entry_price(c.market.home_token) == 0.52       # buy home at ask
    assert c.entry_price(c.market.away_token) == 0.50       # buy away at 1-home_bid


def test_context_exit_price_is_executable_bid_per_side():
    q = MarketQuote("m1", home_bid=0.50, home_ask=0.52, long_bid=0.50, long_ask=0.52)
    c = ctx(q)
    assert c.exit_price(c.market.home_token) == 0.50        # sell home at bid
    assert c.exit_price(c.market.away_token) == pytest.approx(0.48)  # 1-home_ask


# --------------------------------------------------------------- Task 2: FadeStrategy

CFG = StrategyConfig(move_lookback_secs=60, move_threshold=0.08, min_edge=0.05,
                     min_flips=2, min_volatility=99.0, max_price=0.99)


def playful(prices, step=30.0):
    h = PriceHistory(flip_band=0.03)
    for i, p in enumerate(prices):
        h.add(p, ts=i * step)
    return h


def fade_ctx(history, gs, bid, ask):
    return StratContext(market=market(), history=history, game_state=gs,
                        quote=MarketQuote("m1", bid, ask, bid, ask), now=1000.0)


def test_fade_emits_intent_when_edge_survives_execution():
    s = FadeStrategy("fade_v1_frozen", "v1", CFG)
    h = playful([0.60, 0.40, 0.60, 0.60, 0.40])
    gs = GameState(1, status="Live", inning=7, is_top=True, home_score=4, away_score=1)
    d = s.evaluate(fade_ctx(h, gs, bid=0.39, ask=0.41))
    assert d.outcome == "signal"
    assert d.signal_candidate is True
    assert d.intent is not None
    assert d.intent.token == "m1:LONG"
    assert d.intent.edge == pytest.approx(d.intent.fair - 0.41)


def test_fade_rejects_when_execution_cost_eats_edge():
    s = FadeStrategy("fade_v1_frozen", "v1", CFG)
    gs = GameState(1, status="Live", inning=7, is_top=True, home_score=4, away_score=1)
    # First read the model fair with a tight book, then price the ask just below
    # fair so the executable edge (0.02) falls under min_edge (0.05).
    signal = s.evaluate(fade_ctx(playful([0.60, 0.40, 0.60, 0.60, 0.40]), gs,
                                 bid=0.39, ask=0.41))
    fair = signal.intent.fair
    d = s.evaluate(fade_ctx(playful([0.60, 0.40, 0.60, 0.60, 0.40]), gs,
                            bid=fair - 0.03, ask=fair - 0.02))
    assert d.outcome == "execution_cost"
    assert d.intent is None
    assert d.signal_candidate is True


def test_fade_edge_is_net_of_round_trip_fees():
    s = FadeStrategy("fade_v1_frozen", "v1", CFG)
    gs = GameState(1, status="Live", inning=7, is_top=True, home_score=4, away_score=1)
    h = playful([0.60, 0.40, 0.60, 0.60, 0.40])
    c = StratContext(market=market(), history=h, game_state=gs,
                     quote=MarketQuote("m1", 0.39, 0.41, 0.39, 0.41), now=1000.0,
                     fee_theta=0.06)
    d = s.evaluate(c)
    assert d.outcome == "signal"
    fee = c.round_trip_fee(0.41, d.intent.fair)
    assert fee > 0
    assert d.intent.edge == pytest.approx(d.intent.fair - 0.41 - fee)


def test_fade_wide_spread_tracks_signal_without_trading():
    s = FadeStrategy("fade_v1_frozen", "v1", CFG)
    gs = GameState(1, status="Live", inning=7, is_top=True, home_score=4, away_score=1)
    h = playful([0.60, 0.40, 0.60, 0.60, 0.40])
    # Fresh but very wide book (spread 0.30 >> max_spread) — exactly when fades fire.
    c = StratContext(market=market(), history=h, game_state=gs,
                     quote=MarketQuote("m1", 0.25, 0.55, 0.25, 0.55), now=1000.0)
    d = s.evaluate(c)
    assert d.outcome == "wide_spread"
    assert d.intent is None
    assert d.signal_candidate is True


def test_fade_no_quote_still_yields_signal_candidate():
    s = FadeStrategy("fade_v1_frozen", "v1", CFG)
    gs = GameState(1, status="Live", inning=7, is_top=True, home_score=4, away_score=1)
    h = playful([0.60, 0.40, 0.60, 0.60, 0.40])
    c = StratContext(market=market(), history=h, game_state=gs, quote=None, now=1000.0)
    d = s.evaluate(c)
    assert d.outcome == "no_quote"
    assert d.intent is None
    assert d.signal_candidate is True


def test_fade_no_signal_passes_through_outcome():
    s = FadeStrategy("fade_v1_frozen", "v1", CFG)
    h = playful([0.60, 0.60, 0.60, 0.48])   # not playful
    gs = GameState(1, status="Live", inning=7, is_top=True, home_score=4, away_score=1)
    d = s.evaluate(fade_ctx(h, gs, bid=0.47, ask=0.49))
    assert d.intent is None
    assert d.signal_candidate is False
    assert d.evaluation is not None


def test_fade_manage_returns_exit_intents():
    s = FadeStrategy("fade_v1_frozen", "v1", CFG)
    gs = GameState(1, status="Live")
    c = fade_ctx(PriceHistory(), gs, bid=0.57, ask=0.58)
    p = Position(strategy="fade_v1_frozen", market_key="m1", token="m1:LONG",
                 team="Homers", qty=20.0, entry_price=0.50)
    exits = s.manage(c, [p])
    assert len(exits) == 1
    assert "take profit" in exits[0].reason      # sells at bid 0.57 -> +14%


# --------------------------------------------------------------- Task 3: registry

def test_default_registry_builds_two_fade_variants():
    cfg = Config()
    cfg.ai.enabled = False
    strats = build_strategies(cfg)
    names = [s.name for s in strats]
    assert names == ["fade_v1_frozen", "fade_tight"]
    assert isinstance(strats[0], FadeStrategy)


def test_frozen_variant_overrides_base_config():
    cfg = Config()
    cfg.ai.enabled = False
    cfg.strategies = [
        {"name": "fade_v1_frozen", "kind": "fade"},
        {"name": "fade_tight", "kind": "fade",
         "overrides": {"move_threshold": 0.15, "min_edge": 0.09}},
    ]
    strats = {s.name: s for s in build_strategies(cfg)}
    assert strats["fade_v1_frozen"].config.move_threshold == cfg.strategy.move_threshold
    assert strats["fade_tight"].config.move_threshold == 0.15
    assert strats["fade_tight"].config.min_edge == 0.09


def test_unknown_kind_raises():
    cfg = Config()
    cfg.strategies = [{"name": "x", "kind": "bogus"}]
    with pytest.raises(ValueError, match="unknown strategy kind"):
        build_strategies(cfg)


# --------------------------------------------------------------- Task 4: AI shadow

class FakeJudge:
    def __init__(self, verdict, available=True):
        self.verdict = verdict
        self.available = available
        self.calls = 0

    def judge(self, signal, gs):
        self.calls += 1
        return self.verdict


def ai_ctx(bid=0.39, ask=0.41):
    h = playful([0.60, 0.40, 0.60, 0.60, 0.40])
    gs = GameState(1, status="Live", inning=7, is_top=True, home_score=4, away_score=1)
    return StratContext(market=market(), history=h, game_state=gs,
                        quote=MarketQuote("m1", bid, ask, bid, ask), now=1000.0)


def test_ai_shadow_first_tick_submits_and_holds():
    base = FadeStrategy("fade_v1_frozen", "v1", CFG)
    ai = AIShadowStrategy("ai_shadow", "v1", base,
                          judge=FakeJudge(Judgment(True, 0.9, "ok")))
    d = ai.evaluate(ai_ctx())
    assert d.intent is None
    assert d.outcome == "ai_pending"
    ai.wait_idle()
    d2 = ai.evaluate(ai_ctx())
    assert d2.intent is not None
    assert d2.outcome == "ai_opened"
    assert d2.intent.token == "m1:LONG"
    ai.close()


def test_ai_shadow_rejection_opens_nothing():
    base = FadeStrategy("fade_v1_frozen", "v1", CFG)
    ai = AIShadowStrategy("ai_shadow", "v1", base,
                          judge=FakeJudge(Judgment(False, 0.2, "no")))
    ai.evaluate(ai_ctx())
    ai.wait_idle()
    d = ai.evaluate(ai_ctx())
    assert d.intent is None
    assert d.outcome in {"ai_rejected", "ai_pending"}
    ai.close()


def test_ai_shadow_does_not_resubmit_while_pending():
    base = FadeStrategy("fade_v1_frozen", "v1", CFG)
    judge = FakeJudge(Judgment(True, 0.9, "ok"))
    ai = AIShadowStrategy("ai_shadow", "v1", base, judge=judge)
    ai.evaluate(ai_ctx())
    ai.evaluate(ai_ctx())
    ai.wait_idle()
    assert judge.calls == 1
    ai.close()
