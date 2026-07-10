import time

import pytest

from polybot.config import Config
from polybot.engine import Engine
from polybot.models import GameState, Market, MarketQuote
from polybot.pmus import BookQuote


def make_engine(tmp_path) -> Engine:
    cfg = Config()
    cfg.ai.enabled = False
    cfg.engine.db_path = str(tmp_path / "test.db")
    return Engine(cfg)


def tracked_market(engine: Engine, live: bool = True) -> Market:
    market = Market(slug="aec-mlb-a-b-2026-07-08", question="A vs. B",
                    home_team="Homers", away_team="Awayers",
                    long_team="Homers", game_pk=1)
    engine.markets[market.key] = market
    from polybot.volatility import PriceHistory
    engine.histories[market.key] = PriceHistory()
    engine.game_states[1] = GameState(game_pk=1, status="Live" if live else "Final")
    return market


def test_two_sided_book_records_quote_and_history(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)

    engine._poll_prices({market.slug: BookQuote(long_bid=0.50, long_ask=0.52,
                                                long_last=0.51)})

    assert market.key in engine.latest_quotes
    assert engine.latest_quotes[market.key].home_mid == 0.51
    assert engine.histories[market.key].last == 0.51
    assert engine.latest_prices[market.away_token] == 0.48  # executable away-side bid
    assert engine.journal.ticks_for_market(market.key)[0]["source"] == "bbo"


def test_one_sided_book_keeps_history_alive_without_quote(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)

    engine._poll_prices({market.slug: BookQuote(long_bid=None, long_ask=0.01,
                                                long_last=0.01)})

    assert market.key not in engine.latest_quotes  # untradeable, no fresh BBO
    assert engine.histories[market.key].last == 0.01  # but tracking continues
    tick = engine.journal.ticks_for_market(market.key)[0]
    assert tick["source"] == "mark"
    assert tick["two_sided"] == 0


def test_one_sided_mark_invalidates_previous_executable_quote(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    engine._poll_prices({market.slug: BookQuote(long_bid=0.50, long_ask=0.52, long_last=0.51)})
    assert market.key in engine.latest_quotes

    engine._poll_prices({market.slug: BookQuote(long_bid=None, long_ask=0.01, long_last=0.01)})

    assert market.key not in engine.latest_quotes


def test_quote_keeps_gateway_receipt_and_source_times(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    engine._poll_prices({market.slug: BookQuote(
        long_bid=0.50, long_ask=0.52, long_last=0.51,
        received_at=1234.0, source_ts=1233.0,
    )})

    quote = engine.latest_quotes[market.key]
    assert quote.ts == 1234.0
    assert quote.source_ts == 1233.0
    tick = engine.journal.ticks_for_market(market.key)[0]
    assert tick["received_at"] == 1234.0
    assert tick["source_ts"] == 1233.0


def test_final_game_settles_paper_position_at_official_outcome(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    pos = engine.broker.open("math", market.key, market.home_token, market.home_team, 0.50, 10.0)
    assert pos is not None

    engine._settle_final_game(GameState(1, home_score=4, away_score=3, status="Final"))

    assert engine.broker.open_positions("math") == []
    assert engine.broker.realized["math"] == pytest.approx(9.70)
    close = engine.journal.recent_trades(1)[0]
    assert close[2] == "CLOSE"
    assert "official game settlement" in close[-1]


def test_engine_starts_a_provenance_run(tmp_path):
    engine = make_engine(tmp_path)
    run = engine.journal.conn.execute("SELECT * FROM runs WHERE id = ?", (engine.run_id,)).fetchone()
    assert run is not None
    assert run["mode"] == "paper"


def test_entry_uses_executable_ask_for_each_token_side(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    quote = MarketQuote(market.key, 0.50, 0.52, 0.50, 0.52)

    assert engine._entry_price(market, quote, market.home_token) == 0.52
    assert engine._entry_price(market, quote, market.away_token) == 0.50


def test_engine_refuses_live_mode(tmp_path):
    cfg = Config()
    cfg.ai.enabled = False
    cfg.engine.live = True
    cfg.engine.db_path = str(tmp_path / "live.db")

    with pytest.raises(RuntimeError, match="Live trading is disabled"):
        Engine(cfg)


def test_short_home_side_inverts_book(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    market.long_team = market.away_team

    engine._poll_prices({market.slug: BookQuote(long_bid=0.40, long_ask=0.44,
                                                long_last=0.42)})

    quote = engine.latest_quotes[market.key]
    assert quote.home_bid == 0.56
    assert quote.home_ask == 0.60


def test_stale_quote_blocks_entries_and_is_counted(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    engine._poll_prices({market.slug: BookQuote(long_bid=0.50, long_ask=0.52,
                                                long_last=0.51)})
    engine.latest_quotes[market.key].ts = time.time() - 120  # stale

    engine._look_for_entries()

    assert engine.funnel.get("stale_quote") == 1


def test_discovery_only_tracks_matched_markets(tmp_path):
    engine = make_engine(tmp_path)
    engine.mlb.todays_games = lambda: [
        {"game_pk": 7, "home": "Homers", "away": "Awayers",
         "status": "Live", "game_date": None},
    ]
    matched = Market(slug="m-yes", question="", home_team="Homers",
                     away_team="Awayers", long_team="Homers")
    unmatched = Market(slug="m-no", question="", home_team="Others",
                       away_team="Nobodies", long_team="Others")

    engine._maybe_discover([matched, unmatched])

    assert "m-yes" in engine.markets
    assert engine.markets["m-yes"].game_pk == 7
    assert "m-no" not in engine.markets
    assert engine.journal.markets_between(0, time.time() + 1)[0].slug == "m-yes"


def test_paper_account_restores_after_engine_restart(tmp_path):
    engine = make_engine(tmp_path)
    first = engine.broker.open("math", "m1", "m1:LONG", "Homers", 0.50, 10.0)
    assert first is not None
    closed = engine.broker.close("math", first.token, 0.60)
    assert closed is not None
    second = engine.broker.open("math", "m2", "m2:SHORT", "Awayers", 0.25, 5.0)
    assert second is not None
    engine._save_paper_account()
    engine.journal.close()

    restarted = make_engine(tmp_path)
    try:
        assert restarted.broker.cash["math"] == pytest.approx(96.19)
        assert restarted.broker.realized["math"] == pytest.approx(1.41)
        assert restarted.broker.closes["math"] == 1
        restored = restarted.broker.open_positions("math")
        assert len(restored) == 1
        assert restored[0].token == "m2:SHORT"
        assert "restored paper account" in restarted.events[0]
    finally:
        restarted.journal.close()
