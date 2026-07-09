from datetime import datetime

import pytest

from polybot.journal import Journal
from polybot.models import Market
from polybot.review import _annotate_trips, _token_price, print_review


def _ts():
    return datetime(2026, 7, 8, 12, 0, 0).timestamp()


def _insert_trade(journal, **kw):
    fields = {
        "ts": kw["ts"],
        "trade_id": kw.get("trade_id", "tid"),
        "strategy": kw.get("strategy", "math"),
        "action": kw["action"],
        "market": kw.get("market", "m1"),
        "team": kw.get("team", "Awayers"),
        "token": kw["token"],
        "qty": kw.get("qty", 20.0),
        "price": kw["price"],
        "fair": kw.get("fair"),
        "edge": kw.get("edge"),
        "move": kw.get("move"),
        "spread": kw.get("spread"),
        "intended_price": kw.get("intended_price"),
        "slippage": kw.get("slippage"),
        "exit_kind": kw.get("exit_kind"),
        "pnl_usd": kw.get("pnl_usd"),
        "pnl_pct": kw.get("pnl_pct"),
        "reason": kw.get("reason", ""),
    }
    cols = ",".join(fields)
    placeholders = ",".join("?" for _ in fields)
    journal.conn.execute(
        f"INSERT INTO trades ({cols}) VALUES ({placeholders})",
        tuple(fields.values()),
    )


def _insert_tick(journal, ts, market, home_mid):
    journal.conn.execute(
        """INSERT INTO price_ticks
           (ts, market, home_team, away_team, home_mid, two_sided, source)
           VALUES (?, ?, ?, ?, ?, 1, 'bbo')""",
        (ts, market.slug, market.home_team, market.away_team, home_mid),
    )


def test_review_excursions_exit_grouping_hint_and_short_token_flip(tmp_path, capsys):
    journal = Journal(str(tmp_path / "review.db"))
    base = _ts()
    market = Market("m1", "Homers vs Awayers", "Homers", "Awayers",
                    "Awayers", game_pk=1)
    home_long = Market("m2", "Homers vs Awayers", "Homers", "Awayers",
                       "Homers", game_pk=2)
    try:
        journal.record_market(market, ts=base)
        journal.conn.executemany(
            "INSERT INTO equity (ts, strategy, equity, cash, open_positions) VALUES (?, 'math', ?, ?, ?)",
            [(base + 1, 100.0, 90.0, 1), (base + 500, 98.0, 98.0, 0)],
        )
        _insert_trade(
            journal, ts=base + 100, action="OPEN", token=market.away_token,
            price=0.55, fair=0.65, edge=0.10, move=0.09, spread=0.02,
            intended_price=0.54, slippage=0.01, reason="open",
        )
        _insert_trade(
            journal, ts=base + 220, action="CLOSE", token=market.away_token,
            price=0.45, exit_kind="stop_loss", pnl_usd=-2.0, pnl_pct=-0.18,
            reason="stop loss",
        )
        for offset, home_mid in [
            (100, 0.45),  # away token 0.55
            (140, 0.35),  # away token 0.65 => MFE +0.10
            (200, 0.58),  # away token 0.42 => MAE -0.13
            (260, 0.40),  # away token 0.60 => stop recovered after exit
        ]:
            _insert_tick(journal, base + offset, market, home_mid)
        journal.record_decisions([
            {
                "ts": base + 110,
                "market": market.key,
                "stage": "strategy",
                "outcome": "no_edge",
                "margin": -0.01,
            },
            {
                "ts": base + 120,
                "market": market.key,
                "stage": "post_signal",
                "outcome": "opened",
                "strategy": "math",
            },
        ])
        journal.conn.commit()

        trips = _annotate_trips(journal, journal.round_trips(base, base + 1000),
                                {market.key: market}, base + 1000)
        assert trips[0]["mfe"] == pytest.approx(0.10)
        assert trips[0]["mae"] == pytest.approx(-0.13)
        assert _token_price(home_long, home_long.away_token, 0.42) == pytest.approx(0.58)
    finally:
        journal.close()

    print_review(str(tmp_path / "review.db"), day="2026-07-08", near=0.02)
    output = capsys.readouterr().out
    assert "EXIT BREAKDOWN" in output
    assert "stop_loss" in output
    assert "stop_loss may be tight" in output
    assert "no_edge" in output
