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


def test_price_history_resets_after_data_gap(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    engine.histories[market.key].add(0.60, ts=1000.0)

    engine._poll_prices({market.slug: BookQuote(
        long_bid=0.39, long_ask=0.41, long_last=0.40, received_at=1200.0,
    )})

    assert list(engine.histories[market.key].samples) == [(1200.0, 0.40)]
    assert any("history reset after data gap" in event for event in engine.events)


def test_final_game_settles_paper_position_at_official_outcome(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    pos = engine.broker.open("fade_v1_frozen", market.key, market.home_token,
                             market.home_team, 0.50, 10.0)
    assert pos is not None

    engine._settle_final_game(GameState(1, home_score=4, away_score=3, status="Final"))

    assert engine.broker.open_positions("fade_v1_frozen") == []
    assert engine.broker.realized["fade_v1_frozen"] == pytest.approx(9.70)
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


def _seed_fade_signal(engine, market, now):
    """Seed a playful, sharply-down home price so the fade wants the home token."""
    engine.game_states[market.game_pk] = GameState(
        market.game_pk, status="Live", inning=7, is_top=True,
        home_score=4, away_score=1, received_at=now - 200)
    history = engine.histories[market.key]
    for offset, price in [(-100, 0.62), (-75, 0.38), (-50, 0.62), (-25, 0.55), (0, 0.40)]:
        history.add(price, ts=now + offset)


def test_stale_quote_blocks_trade_but_still_records_signal(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    now = time.time()
    _seed_fade_signal(engine, market, now)
    engine.latest_quotes[market.key] = MarketQuote(
        market.key, 0.39, 0.41, 0.39, 0.41, ts=now - 120)  # fresh signal, stale book

    engine._look_for_entries()

    assert engine.funnel.get("stale_quote", 0) >= 1
    assert engine.broker.open_positions("fade_v1_frozen") == []  # not tradeable
    outcome = engine.journal.conn.execute(
        "SELECT outcome FROM signals WHERE strategy='fade_v1_frozen'").fetchone()
    assert outcome["outcome"] == "stale_quote"  # signal captured despite stale book


def test_quote_received_before_state_cannot_fill_under_new_state(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    now = time.time()
    _seed_fade_signal(engine, market, now)
    engine.game_states[market.game_pk].received_at = now
    engine.latest_quotes[market.key] = MarketQuote(
        market.key, 0.39, 0.41, 0.39, 0.41, ts=now - 1,
    )

    engine._look_for_entries()

    assert engine.broker.open_positions("fade_v1_frozen") == []
    outcome = engine.journal.conn.execute(
        "SELECT outcome FROM decisions WHERE strategy='fade_v1_frozen' "
        "ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    assert outcome["outcome"] == "no_quote"


def test_pre_state_candidate_does_not_consume_executable_signal_episode(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    now = time.time()
    _seed_fade_signal(engine, market, now)
    engine.game_states[market.game_pk].received_at = now
    engine.latest_quotes[market.key] = MarketQuote(
        market.key, 0.39, 0.41, 0.39, 0.41, ts=now - 1,
    )
    engine._look_for_entries()
    assert engine.journal.conn.execute(
        "SELECT COUNT(*) FROM signals WHERE strategy='fade_v1_frozen'"
    ).fetchone()[0] == 0

    engine.latest_quotes[market.key] = MarketQuote(
        market.key, 0.39, 0.41, 0.39, 0.41, ts=time.time(),
    )
    engine._look_for_entries()
    signal = engine.journal.conn.execute(
        "SELECT entry_price FROM signals WHERE strategy='fade_v1_frozen'"
    ).fetchone()
    assert signal["entry_price"] == 0.41


def test_unchanged_game_poll_preserves_distinct_state_receipt_time(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    previous = GameState(1, inning=3, status="Live", received_at=100.0)
    engine.game_states[market.game_pk] = previous
    engine.mlb.game_state = lambda game_pk: GameState(
        1, inning=3, status="Live", received_at=999.0,
    )

    engine._maybe_poll_games()

    assert engine.game_states[market.game_pk] is previous
    assert engine.game_states[market.game_pk].received_at == 100.0


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
    first = engine.broker.open("fade_v1_frozen", "m1", "m1:LONG", "Homers", 0.50, 10.0)
    assert first is not None
    closed = engine.broker.close("fade_v1_frozen", first.token, 0.60)
    assert closed is not None
    second = engine.broker.open("fade_v1_frozen", "m2", "m2:SHORT", "Awayers", 0.25, 5.0)
    assert second is not None
    engine._save_paper_account()
    engine.journal.close()

    restarted = make_engine(tmp_path)
    try:
        assert restarted.broker.cash["fade_v1_frozen"] == pytest.approx(96.19)
        assert restarted.broker.realized["fade_v1_frozen"] == pytest.approx(1.41)
        assert restarted.broker.closes["fade_v1_frozen"] == 1
        restored = restarted.broker.open_positions("fade_v1_frozen")
        assert len(restored) == 1
        assert restored[0].token == "m2:SHORT"
        assert "restored paper account" in restarted.events[0]
    finally:
        restarted.journal.close()


def test_engine_runs_multiple_frozen_strategies(tmp_path):
    engine = make_engine(tmp_path)
    assert "fade_v1_frozen" in engine.strategies
    assert "fade_tight" in engine.strategies
    assert set(engine.broker.cash) == set(engine.strategies)


def test_wide_spread_records_signal_and_queues_counterfactual(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    now = time.time()
    _seed_fade_signal(engine, market, now)
    engine.latest_quotes[market.key] = MarketQuote(
        market.key, 0.30, 0.50, 0.30, 0.50, ts=now)  # fresh but wide

    engine._look_for_entries()

    assert engine.broker.open_positions("fade_v1_frozen") == []
    row = engine.journal.conn.execute(
        "SELECT outcome FROM signals WHERE strategy='fade_v1_frozen'").fetchone()
    assert row["outcome"] == "wide_spread"
    assert any(p["market_key"] == market.key for p in engine.pending_cf)


def test_persistent_signal_registers_one_episode(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    now = time.time()
    _seed_fade_signal(engine, market, now)
    # Stale book keeps it a signal-grade candidate (never opens) across ticks.
    engine.latest_quotes[market.key] = MarketQuote(
        market.key, 0.39, 0.41, 0.39, 0.41, ts=now - 120)

    engine._look_for_entries()
    engine._look_for_entries()
    engine._look_for_entries()

    count = engine.journal.conn.execute(
        "SELECT COUNT(*) c FROM signals WHERE strategy='fade_v1_frozen'"
    ).fetchone()["c"]
    assert count == 1  # one episode -> one row, not one-per-tick


def test_orphaned_position_rehydrates_market_and_settles(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    engine.journal.record_market(market)
    engine.broker.open("fade_v1_frozen", market.key, market.home_token,
                       market.home_team, 0.50, 10.0)
    engine._save_paper_account()
    engine.journal.close()

    restarted = make_engine(tmp_path)
    try:
        assert market.key in restarted.markets  # rehydrated despite closed discovery
        restarted._settle_final_game(
            GameState(1, home_score=5, away_score=2, status="Final"))
        assert restarted.broker.open_positions("fade_v1_frozen") == []
    finally:
        restarted.journal.close()


def test_pending_counterfactuals_survive_restart(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    engine.journal.record_market(market)
    sid = engine.journal.record_signal(
        strategy="fade_v1_frozen", market=market.key, token=market.home_token,
        side_team="Homers", entry_price=0.52, fair=0.6, edge=0.08, move=-0.1,
        spread=0.02, inning=7, is_top=1, home_score=4, away_score=1)
    engine.journal.record_pending_cf(sid, market.home_token, market.key, time.time() - 10)
    engine.journal.record_counterfactual(sid, 5, exec_bid=0.50, exec_ask=0.52,
                                         mid=0.51, two_sided=1, spread=0.02)
    engine.journal.close()

    restarted = make_engine(tmp_path)
    try:
        pend = [p for p in restarted.pending_cf if p["signal_id"] == sid]
        assert len(pend) == 1
        assert 5 not in pend[0]["remaining"]   # already captured before restart
        assert 300 in pend[0]["remaining"]     # still pending
    finally:
        restarted.journal.close()


def test_run_freezes_strategy_registry(tmp_path):
    engine = make_engine(tmp_path)
    rows = engine.journal.conn.execute(
        "SELECT strategy, version, kind, config_hash FROM strategy_registry WHERE run_id=?",
        (engine.run_id,)).fetchall()
    by_name = {r["strategy"]: r for r in rows}
    assert {"fade_v1_frozen", "fade_tight"} <= set(by_name)
    assert by_name["fade_v1_frozen"]["version"] == "v1"
    assert by_name["fade_v1_frozen"]["kind"] == "fade"
    assert by_name["fade_v1_frozen"]["config_hash"]
    # different frozen configs must hash differently
    assert by_name["fade_v1_frozen"]["config_hash"] != by_name["fade_tight"]["config_hash"]


def test_counterfactuals_recorded_after_horizon(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    engine.latest_quotes[market.key] = MarketQuote(market.key, 0.50, 0.52, 0.50, 0.52)
    engine.histories[market.key].add(0.51)
    sid = engine.journal.record_signal(
        strategy="fade_v1_frozen", market=market.key, token=market.home_token,
        side_team="Homers", entry_price=0.52, fair=0.6, edge=0.08, move=-0.1,
        spread=0.02, inning=7, is_top=1, home_score=4, away_score=1)
    engine.pending_cf.append({"signal_id": sid, "token": market.home_token,
                              "market_key": market.key, "born": time.time() - 31,
                              "remaining": {30}})

    engine._flush_counterfactuals()

    row = engine.journal.conn.execute(
        "SELECT * FROM signal_counterfactuals WHERE signal_id=?", (sid,)).fetchone()
    assert row["horizon_secs"] == 30
    assert row["exec_ask"] == 0.52
    assert row["two_sided"] == 1
    assert engine.pending_cf == []


def test_overdue_counterfactual_is_unavailable_not_backfilled(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    engine.latest_quotes[market.key] = MarketQuote(
        market.key, 0.70, 0.72, 0.70, 0.72, ts=time.time()
    )
    sid = engine.journal.record_signal(
        strategy="fade_v1_frozen", market=market.key, token=market.home_token,
        side_team="Homers", entry_price=0.52, fair=0.6, edge=0.08, move=-0.1,
        spread=0.02, inning=7, is_top=1, home_score=4, away_score=1,
    )
    engine.pending_cf.append({
        "signal_id": sid, "token": market.home_token,
        "market_key": market.key, "born": time.time() - 100,
        "remaining": {30},
    })

    engine._flush_counterfactuals()

    row = engine.journal.conn.execute(
        "SELECT * FROM signal_counterfactuals WHERE signal_id=?", (sid,)
    ).fetchone()
    assert row["two_sided"] == 0
    assert row["exec_bid"] is None
    assert row["mid"] is None


def test_counterfactual_waits_for_first_post_target_observation(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    now = time.time()
    engine.latest_quotes[market.key] = MarketQuote(
        market.key, 0.50, 0.52, 0.50, 0.52, ts=now - 0.2,
    )
    engine.histories[market.key].add(0.51, ts=now - 0.2)
    sid = engine.journal.record_signal(
        strategy="fade_v1_frozen", market=market.key, token=market.home_token,
        side_team="Homers", entry_price=0.52, fair=0.6, edge=0.08, move=-0.1,
        spread=0.02, inning=7, is_top=1, home_score=4, away_score=1,
    )
    engine.pending_cf.append({
        "signal_id": sid, "token": market.home_token,
        "market_key": market.key, "born": now - 30.1, "remaining": {30},
    })

    engine._flush_counterfactuals()
    assert engine.pending_cf[0]["remaining"] == {30}
    assert engine.journal.conn.execute(
        "SELECT COUNT(*) FROM signal_counterfactuals WHERE signal_id=?", (sid,)
    ).fetchone()[0] == 0

    observed = time.time()
    engine.latest_quotes[market.key] = MarketQuote(
        market.key, 0.53, 0.55, 0.53, 0.55, ts=observed,
    )
    engine.histories[market.key].add(0.54, ts=observed)
    engine._flush_counterfactuals()
    row = engine.journal.conn.execute(
        "SELECT * FROM signal_counterfactuals WHERE signal_id=?", (sid,)
    ).fetchone()
    assert row["exec_bid"] == 0.53
    assert engine.pending_cf == []
