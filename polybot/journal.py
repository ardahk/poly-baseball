"""SQLite trade journal + performance queries."""
from __future__ import annotations

import sqlite3
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    strategy TEXT NOT NULL,
    action TEXT NOT NULL,          -- OPEN | CLOSE
    market TEXT NOT NULL,
    team TEXT NOT NULL,
    token TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    pnl_usd REAL,                  -- CLOSE only
    pnl_pct REAL,                  -- CLOSE only
    reason TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    ts REAL NOT NULL,
    strategy TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    open_positions INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS price_ticks (
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
CREATE INDEX IF NOT EXISTS idx_price_ticks_market_ts
    ON price_ticks (market, ts);
"""


class Journal:
    def __init__(self, path: str = "polybot.db"):
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def record_open(self, strategy, market, team, token, qty, price, reason=""):
        self.conn.execute(
            "INSERT INTO trades (ts, strategy, action, market, team, token, qty, price, reason)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (time.time(), strategy, "OPEN", market, team, token, qty, price, reason),
        )
        self.conn.commit()

    def record_close(self, strategy, market, team, token, qty, price,
                     pnl_usd, pnl_pct, reason=""):
        self.conn.execute(
            "INSERT INTO trades (ts, strategy, action, market, team, token, qty, price,"
            " pnl_usd, pnl_pct, reason) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), strategy, "CLOSE", market, team, token, qty, price,
             pnl_usd, pnl_pct, reason),
        )
        self.conn.commit()

    def record_equity(self, strategy, equity, cash, open_positions):
        self.conn.execute(
            "INSERT INTO equity (ts, strategy, equity, cash, open_positions) VALUES (?,?,?,?,?)",
            (time.time(), strategy, equity, cash, open_positions),
        )
        self.conn.commit()

    def record_price(self, market, quote):
        self.conn.execute(
            "INSERT INTO price_ticks (ts, market, home_team, away_team, home_bid, home_ask,"
            " home_mid, home_spread, long_bid, long_ask) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                quote.ts, market.key, market.home_team, market.away_team,
                quote.home_bid, quote.home_ask, quote.home_mid, quote.home_spread,
                quote.long_bid, quote.long_ask,
            ),
        )
        self.conn.commit()

    def strategy_stats(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT strategy,
                      COUNT(*)                                   AS trades,
                      SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins,
                      COALESCE(SUM(pnl_usd), 0)                  AS pnl_usd,
                      COALESCE(AVG(pnl_pct), 0)                  AS avg_pnl_pct,
                      COALESCE(MAX(pnl_pct), 0)                  AS best_pct,
                      COALESCE(MIN(pnl_pct), 0)                  AS worst_pct
               FROM trades WHERE action = 'CLOSE'
               GROUP BY strategy ORDER BY strategy"""
        ).fetchall()
        return [
            {"strategy": r[0], "trades": r[1], "wins": r[2] or 0, "pnl_usd": r[3],
             "avg_pnl_pct": r[4], "best_pct": r[5], "worst_pct": r[6]}
            for r in rows
        ]

    def latest_equity(self) -> list[tuple[str, float]]:
        rows = self.conn.execute(
            """SELECT strategy, equity FROM equity e
               WHERE ts = (SELECT MAX(ts) FROM equity WHERE strategy = e.strategy)
               GROUP BY strategy"""
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def recent_trades(self, limit: int = 20) -> list[tuple]:
        return self.conn.execute(
            "SELECT ts, strategy, action, team, price, pnl_pct, reason FROM trades"
            " ORDER BY ts DESC LIMIT ?", (limit,),
        ).fetchall()

    def db_status(self) -> dict:
        ticks = self.conn.execute(
            "SELECT COUNT(*), MAX(ts) FROM price_ticks"
        ).fetchone()
        trades = self.conn.execute(
            "SELECT action, COUNT(*) FROM trades GROUP BY action"
        ).fetchall()
        equity = self.conn.execute(
            "SELECT COUNT(*), MAX(ts) FROM equity"
        ).fetchone()
        return {
            "price_ticks": ticks[0] or 0,
            "latest_price_ts": ticks[1],
            "trades": {action: count for action, count in trades},
            "equity_snapshots": equity[0] or 0,
            "latest_equity_ts": equity[1],
        }

    def recent_price_markets(self, limit: int = 10) -> list[tuple]:
        return self.conn.execute(
            """SELECT market, home_team, away_team, COUNT(*) AS ticks,
                      MAX(ts) AS latest_ts, AVG(home_mid) AS avg_mid,
                      AVG(home_spread) AS avg_spread
               FROM price_ticks
               GROUP BY market, home_team, away_team
               ORDER BY latest_ts DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()

    def close(self):
        self.conn.close()
