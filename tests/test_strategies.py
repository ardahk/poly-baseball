import dataclasses

import pytest

from polybot.config import Config, StrategyConfig
from polybot.models import GameState, Market, MarketQuote, Position
from polybot.strategies import (
    AIShadowStrategy,
    FadeStrategy,
    Intent,
    MarketAnchoredStrategy,
    StateResidualStrategy,
    StratContext,
    build_strategies,
)
from polybot.ai_judge import Judgment
from polybot.volatility import PriceHistory
from polybot.model_features import ModelHistory


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


def test_sampling_stable_fade_ignores_lifetime_only_flips():
    cfg = dataclasses.replace(CFG, sampling_stable_features=True,
                              flip_window_secs=20, min_volatility=99.0)
    strategy = FadeStrategy("time", "v1", cfg)
    history = PriceHistory(flip_band=0.03)
    for ts, price in [(0, 0.40), (10, 0.60), (100, 0.40)]:
        history.add(price, ts)
    gs = GameState(1, status="Live", inning=7, home_score=4, away_score=1)
    decision = strategy.evaluate(fade_ctx(history, gs, bid=0.39, ask=0.41))
    assert history.flips == 2
    assert decision.outcome == "not_playful"


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

def test_default_registry_builds_controls_and_phase3_variants():
    cfg = Config()
    cfg.ai.enabled = False
    strats = build_strategies(cfg)
    names = [s.name for s in strats]
    assert names == [
        "fade_v1_frozen", "fade_tight", "liquidity_fade_v2",
        "state_residual_v1", "market_anchor_v1",
    ]
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


def _anchored_context(current_mid, *, pregame=True, changed=True):
    model = ModelHistory(
        lambda gs: 0.70 if gs.home_score > gs.away_score else 0.50,
        pregame_model_home=0.50,
    )
    if pregame:
        model.add_price(0.50, 1.0)
    first = GameState(1, status="Live", home_score=0, away_score=0)
    model.observe_state(first, 2.0)
    model.add_price(0.50, 3.0)
    gs = first
    if changed:
        gs = GameState(1, status="Live", home_score=1, away_score=0)
        model.observe_state(gs, 4.0)
    history = PriceHistory()
    history.add(current_mid, 5.0)
    return StratContext(
        market(), history, gs,
        MarketQuote("m1", current_mid - 0.01, current_mid + 0.01,
                    current_mid - 0.01, current_mid + 0.01, ts=5.0),
        now=5.0, model_history=model,
    )


def test_state_residual_trades_toward_causal_anchored_fair():
    cfg = StrategyConfig(min_edge=0.02, residual_threshold=0.04,
                         residual_min_model_delta=0.01, max_price=0.99)
    strat = StateResidualStrategy("residual", "v1", cfg)
    decision = strat.evaluate(_anchored_context(0.60))
    assert decision.outcome == "signal"
    assert decision.intent.token == market().home_token
    assert decision.evaluation.anchor_price == 0.50
    assert decision.evaluation.model_delta == pytest.approx(0.20)
    assert decision.evaluation.residual < 0


def test_market_anchor_requires_a_recorded_pregame_price():
    cfg = StrategyConfig(min_edge=0.02, residual_threshold=0.04,
                         residual_min_model_delta=0.01, max_price=0.99)
    strat = MarketAnchoredStrategy("anchor", "v1", cfg)
    decision = strat.evaluate(_anchored_context(0.60, pregame=False))
    assert decision.outcome == "no_pregame_anchor"
    assert decision.intent is None


def test_anchored_manage_ignores_stale_anchor_fair():
    cfg = StrategyConfig(residual_response_secs=45, edge_exit=0.03,
                         take_profit=99.0, stop_loss=0.99, max_hold_secs=9999)
    strat = StateResidualStrategy("residual", "v1", cfg)
    model = ModelHistory(lambda gs: 0.70 if gs.home_score > gs.away_score else 0.50)
    model.observe_state(GameState(1, status="Live"), 2.0)
    model.add_price(0.50, 3.0)
    gs = GameState(1, home_score=1, status="Live")
    model.observe_state(gs, 4.0)  # transition anchor at 3.0
    history = PriceHistory()
    history.add(0.60, 120.0)
    ctx = StratContext(
        market(), history, gs,
        MarketQuote("m1", 0.59, 0.61, 0.59, 0.61, ts=120.0),
        now=120.0, model_history=model,
    )
    pos = Position(strategy="residual", market_key="m1", token="m1:SHORT",
                   team="Away", qty=10.0, entry_price=0.40, opened_at=100.0)
    # The stale anchored fair (home 0.70 -> away 0.30 vs bid 0.39) would fire
    # "edge gone", but the anchor is 116s old — far past residual_response_secs
    # — so exits must not act on a fair the entry gates consider invalid.
    assert strat.manage(ctx, [pos]) == []


def test_anchored_price_band_reject_is_not_a_signal_candidate():
    cfg = StrategyConfig(min_edge=0.02, residual_threshold=0.04,
                         residual_min_model_delta=0.01, max_price=0.55)
    strat = StateResidualStrategy("residual", "v1", cfg)
    decision = strat.evaluate(_anchored_context(0.60))
    assert decision.outcome == "price_band"
    assert decision.signal_candidate is False  # matches the fade control


def test_registry_builds_phase3_strategy_kinds():
    cfg = Config()
    cfg.ai.enabled = False
    cfg.strategies = [
        {"name": "r", "kind": "state_residual"},
        {"name": "a", "kind": "market_anchored"},
    ]
    built = build_strategies(cfg)
    assert isinstance(built[0], StateResidualStrategy)
    assert isinstance(built[1], MarketAnchoredStrategy)


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


# ----------------------------------------------- Phase 5 fleet: new kinds

from polybot.strategies import (
    CalibrationCellStrategy,
    EventReactionStrategy,
    ExtremeHoldStrategy,
    MicrostructureStrategy,
    MomentumStrategy,
    SettlementHoldStrategy,
    STRATEGY_KINDS,
)

NOEDGE = dataclasses.replace(CFG, min_edge=-1.0, min_price=0.01, max_price=0.99)
HOLD = dataclasses.replace(NOEDGE, take_profit=10.0, stop_loss=10.0,
                           max_hold_secs=1e9, edge_exit=10.0)


def live_gs(**kw):
    defaults = dict(status="Live", inning=7, is_top=True, home_score=4, away_score=1)
    defaults.update(kw)
    return GameState(1, **defaults)


def trending(prices, step=30.0):
    h = PriceHistory(flip_band=0.03)
    for i, p in enumerate(prices):
        h.add(p, ts=i * step)
    return h


def kctx(history, gs, bid, ask, model_history=None, now=1000.0):
    return StratContext(market=market(), history=history, game_state=gs,
                        quote=MarketQuote("m1", bid, ask, bid, ask, ts=now),
                        now=now, model_history=model_history)


def test_momentum_buys_the_moving_side():
    s = MomentumStrategy("momentum_fast_v1", "v1", NOEDGE)
    h = trending([0.40, 0.42, 0.55, 0.60])       # +0.18 over the last 60s
    d = s.evaluate(kctx(h, live_gs(), bid=0.59, ask=0.61))
    assert d.outcome == "signal"
    assert d.intent.token == "m1:LONG"           # rising home side


def test_momentum_small_move_rejected_with_margin():
    s = MomentumStrategy("momentum_fast_v1", "v1", NOEDGE)
    h = trending([0.50, 0.51, 0.52, 0.53])
    d = s.evaluate(kctx(h, live_gs(), bid=0.52, ask=0.54))
    assert d.outcome == "small_move"
    assert d.evaluation.margin < 0


def test_momentum_model_agree_filter():
    cfg = dataclasses.replace(NOEDGE, momentum_require_model_agree=True)
    s = MomentumStrategy("momentum_confirmed_v1", "v1", cfg)
    # Home trailing badly (fair low) but price spiking up -> model disagrees.
    h = trending([0.40, 0.42, 0.55, 0.60])
    gs = live_gs(inning=8, home_score=0, away_score=9)
    d = s.evaluate(kctx(h, gs, bid=0.59, ask=0.61))
    assert d.outcome == "model_disagrees"


def test_event_reaction_trades_underreaction_in_model_direction():
    s = EventReactionStrategy("news_underreact_v1", "v1", NOEDGE)
    mh = ModelHistory(anchor_lookback_secs=30.0)
    mh.add_price(0.50, ts=900.0)
    gs0 = live_gs(inning=6, home_score=2, away_score=2)
    gs0.received_at = 950.0
    mh.observe_state(gs0, ts=950.0)
    h = trending([0.50, 0.50, 0.50, 0.51])
    ctx0 = kctx(h, gs0, bid=0.50, ask=0.52, model_history=mh, now=960.0)
    s.evaluate(ctx0)                              # prime prev-state tracking
    # Home scores 3: model jumps, market barely moves within 30s.
    gs1 = live_gs(inning=6, home_score=5, away_score=2)
    gs1.received_at = 970.0
    mh.observe_state(gs1, ts=970.0)
    ctx1 = kctx(h, gs1, bid=0.50, ask=0.52, model_history=mh, now=975.0)
    d = s.evaluate(ctx1)
    assert d.outcome == "signal"
    assert d.intent.token == "m1:LONG"            # with the news, home side


def test_event_reaction_class_filter_blocks_other_events():
    cfg = dataclasses.replace(NOEDGE, event_class="score_change")
    s = EventReactionStrategy("news_underreact_score_v1", "v1", cfg)
    mh = ModelHistory(anchor_lookback_secs=30.0)
    mh.add_price(0.50, ts=900.0)
    gs0 = live_gs(inning=6, home_score=2, away_score=2, outs=0)
    gs0.received_at = 950.0
    mh.observe_state(gs0, ts=950.0)
    h = trending([0.50, 0.50, 0.50, 0.51])
    s.evaluate(kctx(h, gs0, bid=0.50, ask=0.52, model_history=mh, now=960.0))
    gs1 = live_gs(inning=6, home_score=2, away_score=2, outs=2)   # outs only
    gs1.received_at = 970.0
    mh.observe_state(gs1, ts=970.0)
    d = s.evaluate(kctx(h, gs1, bid=0.50, ask=0.52, model_history=mh, now=975.0))
    assert d.outcome == "wrong_event_class"


def test_extreme_hold_enters_band_and_holds_to_settlement():
    cfg = dataclasses.replace(HOLD, extreme_min_price=0.90, extreme_max_price=0.97,
                              extreme_min_inning=7)
    s = ExtremeHoldStrategy("favorite_late_v1", "v1", cfg)
    h = trending([0.90, 0.91, 0.92, 0.92])
    d = s.evaluate(kctx(h, live_gs(inning=8, home_score=9, away_score=1),
                        bid=0.91, ask=0.93))
    assert d.outcome == "signal"
    assert d.intent.token == "m1:LONG"
    # Held position: no exit while live even at a big loss...
    pos = Position(strategy="favorite_late_v1", market_key="m1", token="m1:LONG",
                   team="Homers", qty=10.0, entry_price=0.93, opened_at=990.0)
    exits = s.manage(kctx(h, live_gs(inning=9), bid=0.50, ask=0.52), [pos])
    assert exits == []
    # ...but settles on game final.
    exits = s.manage(kctx(h, GameState(1, status="Final"), bid=0.99, ask=1.0), [pos])
    assert len(exits) == 1 and "game final" in exits[0].reason


def test_extreme_hold_outside_band_rejected():
    cfg = dataclasses.replace(HOLD, extreme_min_price=0.90, extreme_min_inning=1)
    s = ExtremeHoldStrategy("favorite_late_v1", "v1", cfg)
    h = trending([0.60, 0.61, 0.60, 0.62])
    d = s.evaluate(kctx(h, live_gs(), bid=0.61, ask=0.63))
    assert d.outcome == "outside_price_band"


def test_settlement_hold_buys_model_side_on_gap():
    cfg = dataclasses.replace(HOLD, hold_min_edge=0.10, hold_max_inning=6)
    s = SettlementHoldStrategy("settle_gap10_v1", "v1", cfg)
    # Home leads 5-0 in the 6th: model fair >> 0.55 market price.
    h = trending([0.54, 0.55, 0.55, 0.55])
    d = s.evaluate(kctx(h, live_gs(inning=6, home_score=5, away_score=0),
                        bid=0.54, ask=0.56))
    assert d.outcome == "signal"
    assert d.intent.token == "m1:LONG"


def test_settlement_hold_side_filter():
    cfg = dataclasses.replace(HOLD, hold_min_edge=0.10, hold_max_inning=6,
                              hold_side_filter="away")
    s = SettlementHoldStrategy("settle_away_v1", "v1", cfg)
    h = trending([0.54, 0.55, 0.55, 0.55])
    d = s.evaluate(kctx(h, live_gs(inning=6, home_score=5, away_score=0),
                        bid=0.54, ask=0.56))
    assert d.outcome == "side_filtered"


def test_calibration_cell_leader_side():
    cfg = dataclasses.replace(HOLD, cell_side="leader", cell_price_min=0.50,
                              cell_price_max=0.62, cell_inning_min=4)
    s = CalibrationCellStrategy("cell_leader_coinflip_v1", "v1", cfg)
    h = trending([0.55, 0.56, 0.55, 0.56])
    d = s.evaluate(kctx(h, live_gs(inning=5, home_score=3, away_score=2),
                        bid=0.55, ask=0.57))
    assert d.outcome == "signal"
    assert d.intent.token == "m1:LONG"            # home leads and is in-band
    tie = s.evaluate(kctx(h, live_gs(inning=5, home_score=3, away_score=3),
                          bid=0.55, ask=0.57))
    assert tie.outcome == "no_cell_side"


def test_microstructure_stale_reprice():
    cfg = dataclasses.replace(NOEDGE, micro_mode="stale_reprice",
                              micro_window_secs=60.0, micro_min_reprice=0.03)
    s = MicrostructureStrategy("stale_reprice_v1", "v1", cfg)
    h = trending([0.50, 0.50, 0.50, 0.58])
    gs = live_gs()
    c1 = StratContext(market=market(), history=h, game_state=gs,
                      quote=MarketQuote("m1", 0.49, 0.51, 0.49, 0.51, ts=100.0),
                      now=100.0)
    assert s.evaluate(c1).outcome in {"no_gap", "small_reprice"}
    # 90s book gap, then a two-sided quote 8 cents higher.
    c2 = StratContext(market=market(), history=h, game_state=gs,
                      quote=MarketQuote("m1", 0.57, 0.59, 0.57, 0.59, ts=190.0),
                      now=190.0)
    d = s.evaluate(c2)
    assert d.outcome == "signal"
    assert d.intent.token == "m1:LONG"


def test_microstructure_pregame_drift_fires_once():
    cfg = dataclasses.replace(NOEDGE, micro_mode="pregame_drift",
                              micro_min_reprice=0.03)
    s = MicrostructureStrategy("pregame_drift_v1", "v1", cfg)
    h = PriceHistory(flip_band=0.03)
    h.add(0.50, ts=0.0)          # 30+ minutes pregame
    h.add(0.56, ts=1900.0)       # first live tick, +6c drift
    gs = live_gs(inning=1, home_score=0, away_score=0)
    c = StratContext(market=market(), history=h, game_state=gs,
                     quote=MarketQuote("m1", 0.55, 0.57, 0.55, 0.57, ts=1900.0),
                     now=1900.0)
    d = s.evaluate(c)
    assert d.outcome == "signal"
    assert d.intent.token == "m1:LONG"
    assert s.evaluate(c).outcome == "already_fired"
    s.reset_market("m1")
    assert s.evaluate(c).outcome == "signal"      # replay boundary resets state


def test_registry_kind_table_covers_all_deterministic_kinds():
    assert set(STRATEGY_KINDS) == {
        "fade", "state_residual", "market_anchored", "momentum",
        "event_reaction", "extreme_hold", "settlement_hold",
        "calibration_cell", "microstructure",
    }


def test_yaml_registry_builds_deterministic_strategies():
    # 5 frozen controls + 25 v1 hypothesis fleet + 25 v2 retune fleet (2026-07-17),
    # minus liquidity_fade_v2 (retired 2026-07-19, cash below min stake).
    from polybot.config import load_config
    cfg = load_config("config.yaml")
    strats = [s for s in build_strategies(cfg) if s.kind != "ai_shadow"]
    assert len(strats) == 54
    assert len({s.name for s in strats}) == 54
    # Every v2 pairs with a frozen v1 of the same kind (v1 left untouched).
    names = {s.name for s in strats}
    for s in strats:
        if s.name.endswith("_v2") and s.name != "liquidity_fade_v2":
            assert s.name[:-1] + "1" in names, s.name
