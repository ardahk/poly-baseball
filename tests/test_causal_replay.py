from datetime import datetime
from zoneinfo import ZoneInfo

from polybot import causal_replay
from polybot.config import Config
from polybot.journal import Journal
from polybot.models import GameState, Market, MarketQuote
from polybot.strategies import Decision, Intent, Strategy


class AlwaysHomeStrategy(Strategy):
    def evaluate(self, ctx):
        price = ctx.entry_price(ctx.market.home_token)
        return Decision(
            outcome="signal",
            intent=Intent(
                token=ctx.market.home_token, side_team=ctx.market.home_team,
                signal_price=ctx.history.last, fair=0.99, move=-0.10,
                edge=0.99 - price, reason="test",
            ),
        )

    def manage(self, ctx, positions):
        return []


def _base():
    return datetime(2026, 7, 8, 12, tzinfo=ZoneInfo("America/Los_Angeles")).timestamp()


def _market(slug="m1", game_pk=1):
    return Market(slug, "Away vs Home", "Home", "Away", "Home", game_pk=game_pk)


def _state(game_pk, ts, status="Live", home=0, away=0):
    return GameState(game_pk, home_score=home, away_score=away,
                     status=status, received_at=ts)


def _quote(market, ts, bid, ask):
    return MarketQuote(market.key, bid, ask, bid, ask, ts=ts)


def _cfg(db_path):
    cfg = Config()
    cfg.ai.enabled = False
    cfg.engine.db_path = str(db_path)
    cfg.engine.causal_replay_latency_secs = 0.5
    cfg.risk.strong_stake_min_edge = 99.0
    cfg.strategies = [{"name": "always", "kind": "fade"}]
    return cfg


def test_order_fills_only_on_later_bbo_then_settles(monkeypatch, tmp_path):
    path = tmp_path / "causal.db"
    cfg = _cfg(path)
    base = _base()
    market = _market()
    journal = Journal(str(path))
    journal.start_run("paper", "cfg")
    journal.record_market(market, ts=base)
    journal.record_game_state(_state(1, base - 1), ts=base - 1)
    journal.record_price(market, _quote(market, base, 0.49, 0.51))
    journal.record_price(market, _quote(market, base + 1, 0.54, 0.56))
    journal.record_game_state(_state(1, base + 2, "Final", home=1), ts=base + 2)
    monkeypatch.setattr(causal_replay, "build_strategies",
                        lambda cfg: [AlwaysHomeStrategy("always", "v1", cfg.strategy)])

    report = causal_replay.CausalReplay(cfg, journal, base - 10, base + 10, "day").run()

    assert len(report.trades) == 1
    assert report.trades[0].entry_ts == base + 1
    assert report.trades[0].entry_price == 0.56
    assert report.trades[0].exit_price == 1.0
    journal.close()


def test_final_state_cancels_unfilled_order(monkeypatch, tmp_path):
    path = tmp_path / "final.db"
    cfg = _cfg(path)
    base = _base()
    market = _market()
    journal = Journal(str(path))
    journal.start_run("paper", "cfg")
    journal.record_market(market, ts=base)
    journal.record_game_state(_state(1, base - 1), ts=base - 1)
    journal.record_price(market, _quote(market, base, 0.49, 0.51))
    journal.record_game_state(_state(1, base + 0.25, "Final", home=1), ts=base + 0.25)
    journal.record_price(market, _quote(market, base + 1, 0.99, 1.0))
    monkeypatch.setattr(causal_replay, "build_strategies",
                        lambda cfg: [AlwaysHomeStrategy("always", "v1", cfg.strategy)])

    report = causal_replay.CausalReplay(cfg, journal, base - 10, base + 10, "day").run()

    assert report.trades == []
    assert report.results[0].open_positions == 0
    journal.close()


def test_overlapping_markets_share_portfolio_limit(monkeypatch, tmp_path):
    path = tmp_path / "overlap.db"
    cfg = _cfg(path)
    cfg.risk.max_positions = 1
    base = _base()
    markets = [_market("m1", 1), _market("m2", 2)]
    journal = Journal(str(path))
    journal.start_run("paper", "cfg")
    for market in markets:
        journal.record_market(market, ts=base)
        journal.record_game_state(_state(market.game_pk, base - 1), ts=base - 1)
        journal.record_price(market, _quote(market, base, 0.49, 0.51))
        journal.record_price(market, _quote(market, base + 1, 0.49, 0.51))
        journal.record_game_state(
            _state(market.game_pk, base + 2, "Final", home=1), ts=base + 2)
    monkeypatch.setattr(causal_replay, "build_strategies",
                        lambda cfg: [AlwaysHomeStrategy("always", "v1", cfg.strategy)])

    report = causal_replay.CausalReplay(cfg, journal, base - 10, base + 10, "day").run()

    assert len(report.trades) == 1
    assert report.results[0].rejected_orders >= 1
    journal.close()


def test_run_boundary_clears_pending_orders(monkeypatch, tmp_path):
    path = tmp_path / "restart.db"
    cfg = _cfg(path)
    base = _base()
    market = _market()
    journal = Journal(str(path))
    journal.start_run("paper", "one")
    journal.record_market(market, ts=base)
    journal.record_game_state(_state(1, base - 1), ts=base - 1)
    journal.record_price(market, _quote(market, base, 0.49, 0.51))
    journal.start_run("paper", "two")
    journal.record_game_state(_state(1, base + 0.5), ts=base + 0.5)
    journal.record_price(market, _quote(market, base + 1, 0.49, 0.51))
    monkeypatch.setattr(causal_replay, "build_strategies",
                        lambda cfg: [AlwaysHomeStrategy("always", "v1", cfg.strategy)])

    report = causal_replay.CausalReplay(cfg, journal, base - 10, base + 10, "day").run()

    assert report.run_boundaries == 1
    assert report.trades == []
    assert report.results[0].open_positions == 0
    journal.close()

