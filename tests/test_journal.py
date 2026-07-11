import sqlite3
import time

from polybot.journal import Journal
from polybot.models import GameState, Market, MarketQuote


def _legacy_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            strategy TEXT NOT NULL,
            action TEXT NOT NULL,
            market TEXT NOT NULL,
            team TEXT NOT NULL,
            token TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            pnl_usd REAL,
            pnl_pct REAL,
            reason TEXT
        );
        CREATE TABLE equity (
            ts REAL NOT NULL,
            strategy TEXT NOT NULL,
            equity REAL NOT NULL,
            cash REAL NOT NULL,
            open_positions INTEGER NOT NULL
        );
        CREATE TABLE price_ticks (
            ts REAL NOT NULL,
            market TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_bid REAL NOT NULL,
            home_ask REAL NOT NULL,
            home_mid REAL NOT NULL,
            home_spread REAL NOT NULL,
            long_bid REAL NOT NULL,
            long_ask REAL NOT NULL
        );
        INSERT INTO trades
            (ts, strategy, action, market, team, token, qty, price, reason)
            VALUES (1, 'math', 'OPEN', 'legacy', 'Homers', 'legacy:LONG', 10, 0.50, 'old');
        INSERT INTO price_ticks VALUES
            (2, 'legacy', 'Homers', 'Awayers', 0.49, 0.51, 0.50, 0.02, 0.49, 0.51);
        """
    )
    conn.commit()
    conn.close()


def test_migrates_v1_schema_and_preserves_legacy_rows(tmp_path):
    db_path = tmp_path / "legacy.db"
    _legacy_db(db_path)

    journal = Journal(str(db_path))
    try:
        trade_cols = journal._table_columns("trades")
        assert "trade_id" in trade_cols
        assert "exit_kind" in trade_cols
        price_cols = journal._table_columns("price_ticks")
        assert "two_sided" in price_cols
        assert "source" in price_cols
        assert price_cols["home_bid"]["notnull"] == 0

        tick = journal.ticks_for_market("legacy")[0]
        assert tick["two_sided"] == 1
        assert tick["source"] == "bbo"

        market = Market("legacy", "Q", "Homers", "Awayers", "Homers", game_pk=1)
        journal.record_mark(market, 0.44, 0.44)
        mark = journal.ticks_for_market("legacy")[-1]
        assert mark["two_sided"] == 0
        assert mark["home_bid"] is None
    finally:
        journal.close()


def test_record_and_query_round_trips_decisions_states_and_markets(tmp_path):
    journal = Journal(str(tmp_path / "journal.db"))
    try:
        market = Market("m1", "Homers vs Awayers", "Homers", "Awayers",
                        "Homers", game_pk=7, start_time=1000)
        quote = MarketQuote("m1", 0.48, 0.50, 0.48, 0.50, ts=1100)
        journal.record_market(market, ts=1000)
        journal.record_game_state(
            GameState(7, inning=6, is_top=False, home_score=4, away_score=2, status="Live"),
            ts=1100,
        )
        journal.record_price(market, quote)
        journal.record_decisions([
            {
                "ts": 1100,
                "market": "m1",
                "stage": "strategy",
                "outcome": "no_edge",
                "margin": -0.01,
            },
            {
                "ts": 1101,
                "market": "m1",
                "stage": "strategy",
                "outcome": "small_move",
                "margin": -0.05,
            },
        ])
        journal.record_open(
            "math", "m1", "Homers", "m1:LONG", 20, 0.50, "open",
            trade_id="abc123", fair=0.62, edge=0.12, move=-0.10,
            spread=0.02, intended_price=0.49, slippage=0.01,
        )
        journal.record_close(
            "math", "m1", "Homers", "m1:LONG", 20, 0.58, 1.6, 0.16, "take profit",
            trade_id="abc123", fair=0.57, intended_price=0.59,
            slippage=-0.01, exit_kind="take_profit",
        )
        journal.record_open("math", "m1", "Homers", "m1:LONG", 5, 0.51, "open",
                            trade_id="stillopen")

        trips = journal.round_trips(0, time.time() + 1)
        linked = [t for t in trips if t["trade_id"] == "abc123"][0]
        assert linked["exit_kind"] == "take_profit"
        assert linked["hold_secs"] is not None
        unclosed = [t for t in trips if t["trade_id"] == "stillopen"][0]
        assert unclosed["exit_ts"] is None

        assert journal.near_misses(0, 2000, within=0.02)[0]["outcome"] == "no_edge"
        assert journal.decisions_summary(0, 2000)[0]["count"] == 1
        assert journal.game_state_timeline(7)[0][1].inning == 6
        assert journal.markets_between(0, 2000)[0].slug == "m1"
    finally:
        journal.close()


def test_records_signal_and_counterfactuals(tmp_path):
    j = Journal(str(tmp_path / "j.db"))
    j.start_run("paper", "hash")
    sid = j.record_signal(strategy="fade_v1_frozen", market="m1", token="m1:LONG",
                          side_team="Homers", entry_price=0.41, fair=0.55, edge=0.14,
                          move=-0.10, spread=0.02, inning=7, is_top=1,
                          home_score=4, away_score=1, anchor_price=0.50,
                          anchor_model=0.54, model_delta=0.10, residual=-0.05,
                          anchor_age=12.0)
    assert isinstance(sid, int)
    j.record_counterfactual(sid, horizon_secs=30, exec_bid=0.44, exec_ask=0.46,
                            mid=0.45, two_sided=1, spread=0.02)
    rows = j.conn.execute(
        "SELECT * FROM signal_counterfactuals WHERE signal_id=?", (sid,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["horizon_secs"] == 30
    assert rows[0]["exec_ask"] == 0.46
    sig = j.conn.execute("SELECT * FROM signals WHERE id=?", (sid,)).fetchone()
    assert sig["strategy"] == "fade_v1_frozen"
    assert sig["run_id"] is not None
    assert sig["anchor_price"] == 0.50
    assert sig["residual"] == -0.05
    j.close()


def test_records_one_model_observation_per_distinct_state(tmp_path):
    j = Journal(str(tmp_path / "model.db"))
    j.start_run("paper", "hash")
    gs = GameState(7, inning=3, outs=1, home_score=1, away_score=0, status="Live")
    kwargs = dict(
        model="analytic_v1", market="m1", game_state=gs,
        state_signature="state-1", model_home=0.62, pregame_anchor=0.55,
        anchored_fair=0.64, home_mid=0.60, spread=0.02, ts=100.0,
    )
    j.record_model_observation(**kwargs)
    j.record_model_observation(**kwargs)
    rows = j.conn.execute("SELECT * FROM model_observations").fetchall()
    assert len(rows) == 1
    assert rows[0]["pregame_anchor"] == 0.55
    assert rows[0]["anchored_fair"] == 0.64
    j.close()
