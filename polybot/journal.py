"""SQLite trade journal + observability/replay queries."""
from __future__ import annotations

import sqlite3
import time
import uuid

from .models import GameState, Market

_PRICE_TICKS_DDL = """
CREATE TABLE IF NOT EXISTS price_ticks (
    ts REAL NOT NULL,
    market TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_bid REAL,
    home_ask REAL,
    home_mid REAL NOT NULL,
    home_spread REAL,
    long_bid REAL,
    long_ask REAL,
    two_sided INTEGER NOT NULL DEFAULT 1,
    source TEXT NOT NULL DEFAULT 'bbo',
    run_id TEXT,
    received_at REAL,
    source_ts REAL
);
"""

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    trade_id TEXT,
    strategy TEXT NOT NULL,
    action TEXT NOT NULL,          -- OPEN | CLOSE
    market TEXT NOT NULL,
    team TEXT NOT NULL,
    token TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    fair REAL,
    edge REAL,
    move REAL,
    spread REAL,
    intended_price REAL,
    slippage REAL,
    exit_kind TEXT,
    pnl_usd REAL,                  -- CLOSE only
    pnl_pct REAL,                  -- CLOSE only
    reason TEXT,
    run_id TEXT,
    fee_usd REAL
);
CREATE TABLE IF NOT EXISTS equity (
    ts REAL NOT NULL,
    strategy TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    open_positions INTEGER NOT NULL,
    run_id TEXT
);
{_PRICE_TICKS_DDL}
CREATE INDEX IF NOT EXISTS idx_price_ticks_market_ts
    ON price_ticks (market, ts);
CREATE TABLE IF NOT EXISTS decisions (
    ts REAL NOT NULL,
    market TEXT NOT NULL,
    strategy TEXT,
    stage TEXT NOT NULL,
    outcome TEXT NOT NULL,
    mid REAL,
    move REAL,
    flips INTEGER,
    realized_vol REAL,
    fair_home REAL,
    side TEXT,
    price REAL,
    fair REAL,
    edge REAL,
    spread REAL,
    quote_age REAL,
    margin REAL,
    inning INTEGER,
    is_top INTEGER,
    home_score INTEGER,
    away_score INTEGER,
    run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_market_ts
    ON decisions (market, ts);
CREATE INDEX IF NOT EXISTS idx_decisions_outcome_ts
    ON decisions (outcome, ts);
CREATE TABLE IF NOT EXISTS game_states (
    ts REAL NOT NULL,
    game_pk INTEGER NOT NULL,
    inning INTEGER NOT NULL,
    is_top INTEGER NOT NULL,
    outs INTEGER NOT NULL,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    on_first INTEGER NOT NULL,
    on_second INTEGER NOT NULL,
    on_third INTEGER NOT NULL,
    status TEXT NOT NULL,
    run_id TEXT,
    received_at REAL
);
CREATE INDEX IF NOT EXISTS idx_game_states_game_ts
    ON game_states (game_pk, ts);
CREATE TABLE IF NOT EXISTS markets (
    slug TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    long_team TEXT NOT NULL,
    game_pk INTEGER,
    start_time REAL,
    first_seen_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_accounts (
    strategy TEXT PRIMARY KEY,
    cash REAL NOT NULL,
    realized REAL NOT NULL,
    closes INTEGER NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_positions (
    strategy TEXT NOT NULL,
    token TEXT NOT NULL,
    market TEXT NOT NULL,
    team TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    entry_fee REAL NOT NULL DEFAULT 0,
    opened_at REAL NOT NULL,
    trade_id TEXT NOT NULL,
    PRIMARY KEY (strategy, token)
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    mode TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    code_revision TEXT NOT NULL
);
"""

_TRADE_V2_COLUMNS = {
    "trade_id": "TEXT",
    "fair": "REAL",
    "edge": "REAL",
    "move": "REAL",
    "spread": "REAL",
    "intended_price": "REAL",
    "slippage": "REAL",
    "exit_kind": "TEXT",
    "run_id": "TEXT",
    "fee_usd": "REAL",
}

_COLUMN_MIGRATIONS = {
    "equity": {"run_id": "TEXT"},
    "price_ticks": {"run_id": "TEXT", "received_at": "REAL", "source_ts": "REAL"},
    "game_states": {"run_id": "TEXT", "received_at": "REAL"},
    "decisions": {"run_id": "TEXT"},
    "paper_positions": {"entry_fee": "REAL NOT NULL DEFAULT 0"},
}

_DECISION_COLUMNS = (
    "ts", "market", "strategy", "stage", "outcome", "mid", "move", "flips",
    "realized_vol", "fair_home", "side", "price", "fair", "edge", "spread",
    "quote_age", "margin", "inning", "is_top", "home_score", "away_score", "run_id",
)


class Journal:
    def __init__(self, path: str = "polybot.db"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.active_run_id: str | None = None
        self._migrate()
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # --------------------------------------------------------------- migration

    def _table_columns(self, table: str) -> dict[str, sqlite3.Row]:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r["name"]: r for r in rows}

    def _migrate(self) -> None:
        trade_cols = self._table_columns("trades")
        if trade_cols:
            for name, ddl in _TRADE_V2_COLUMNS.items():
                if name not in trade_cols:
                    self.conn.execute(f"ALTER TABLE trades ADD COLUMN {name} {ddl}")

        price_cols = self._table_columns("price_ticks")
        if price_cols and self._price_ticks_needs_rebuild(price_cols):
            self._rebuild_price_ticks(price_cols)
        for table, columns in _COLUMN_MIGRATIONS.items():
            existing = self._table_columns(table)
            for name, ddl in columns.items():
                if existing and name not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
        self.conn.commit()

    def start_run(self, mode: str, config_hash: str, code_revision: str = "unknown") -> str:
        run_id = uuid.uuid4().hex
        self.conn.execute(
            "INSERT INTO runs (id, started_at, mode, config_hash, code_revision) VALUES (?,?,?,?,?)",
            (run_id, time.time(), mode, config_hash, code_revision),
        )
        self.conn.commit()
        self.active_run_id = run_id
        return run_id

    @staticmethod
    def _price_ticks_needs_rebuild(cols: dict[str, sqlite3.Row]) -> bool:
        if "two_sided" not in cols or "source" not in cols:
            return True
        for name in ("home_bid", "home_ask", "home_spread", "long_bid", "long_ask"):
            if name in cols and cols[name]["notnull"]:
                return True
        return False

    def _rebuild_price_ticks(self, old_cols: dict[str, sqlite3.Row]) -> None:
        self.conn.execute("DROP TABLE IF EXISTS price_ticks_new")
        self.conn.execute(_PRICE_TICKS_DDL.replace("price_ticks", "price_ticks_new", 1))
        new_cols = (
            "ts", "market", "home_team", "away_team", "home_bid", "home_ask",
            "home_mid", "home_spread", "long_bid", "long_ask", "two_sided", "source",
        )
        defaults = {
            "two_sided": "1",
            "source": "'bbo'",
            "home_bid": "NULL",
            "home_ask": "NULL",
            "home_mid": "0.0",
            "home_spread": "NULL",
            "long_bid": "NULL",
            "long_ask": "NULL",
        }
        select_exprs = [name if name in old_cols else defaults.get(name, "NULL")
                        for name in new_cols]
        self.conn.execute(
            f"INSERT INTO price_ticks_new ({','.join(new_cols)}) "
            f"SELECT {','.join(select_exprs)} FROM price_ticks"
        )
        self.conn.execute("DROP TABLE price_ticks")
        self.conn.execute("ALTER TABLE price_ticks_new RENAME TO price_ticks")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_price_ticks_market_ts "
            "ON price_ticks (market, ts)"
        )

    # ---------------------------------------------------------------- records

    def record_open(self, strategy, market, team, token, qty, price, reason="",
                    *, trade_id: str = "", fair: float | None = None,
                    edge: float | None = None, move: float | None = None,
                    spread: float | None = None, intended_price: float | None = None,
                    slippage: float | None = None, fee_usd: float = 0.0,
                    commit: bool = True):
        self.conn.execute(
            """INSERT INTO trades
               (ts, trade_id, strategy, action, market, team, token, qty, price,
                fair, edge, move, spread, intended_price, slippage, reason, run_id, fee_usd)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), trade_id, strategy, "OPEN", market, team, token, qty, price,
                fair, edge, move, spread, intended_price, slippage, reason,
                self.active_run_id, fee_usd,
            ),
        )
        if commit:
            self.conn.commit()

    def record_close(self, strategy, market, team, token, qty, price,
                     pnl_usd, pnl_pct, reason="", *, trade_id: str = "",
                     fair: float | None = None, edge: float | None = None,
                     move: float | None = None, spread: float | None = None,
                     intended_price: float | None = None,
                     slippage: float | None = None,
                     exit_kind: str | None = None, fee_usd: float = 0.0,
                     commit: bool = True):
        self.conn.execute(
            """INSERT INTO trades
               (ts, trade_id, strategy, action, market, team, token, qty, price,
                fair, edge, move, spread, intended_price, slippage, exit_kind,
                pnl_usd, pnl_pct, reason, run_id, fee_usd)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(), trade_id, strategy, "CLOSE", market, team, token, qty, price,
                fair, edge, move, spread, intended_price, slippage, exit_kind,
                pnl_usd, pnl_pct, reason, self.active_run_id, fee_usd,
            ),
        )
        if commit:
            self.conn.commit()

    def record_equity(self, strategy, equity, cash, open_positions):
        self.conn.execute(
            "INSERT INTO equity (ts, strategy, equity, cash, open_positions, run_id) VALUES (?,?,?,?,?,?)",
            (time.time(), strategy, equity, cash, open_positions, self.active_run_id),
        )
        self.conn.commit()

    def record_price(self, market, quote):
        self.conn.execute(
            """INSERT INTO price_ticks
               (ts, market, home_team, away_team, home_bid, home_ask, home_mid,
                home_spread, long_bid, long_ask, two_sided, source, run_id, received_at, source_ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                quote.ts, market.key, market.home_team, market.away_team,
                quote.home_bid, quote.home_ask, quote.home_mid, quote.home_spread,
                quote.long_bid, quote.long_ask, 1, "bbo", self.active_run_id,
                quote.ts, quote.source_ts,
            ),
        )
        self.conn.commit()

    def record_mark(self, market: Market, home_mid: float, long_last: float | None):
        self.conn.execute(
            """INSERT INTO price_ticks
               (ts, market, home_team, away_team, home_mid, two_sided, source, run_id, received_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (time.time(), market.key, market.home_team, market.away_team,
             home_mid, 0, "mark", self.active_run_id, time.time()),
        )
        self.conn.commit()

    def record_game_state(self, gs: GameState, ts: float | None = None):
        self.conn.execute(
            """INSERT INTO game_states
               (ts, game_pk, inning, is_top, outs, home_score, away_score,
                on_first, on_second, on_third, status, run_id, received_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time() if ts is None else ts,
                gs.game_pk, gs.inning, int(gs.is_top), gs.outs,
                gs.home_score, gs.away_score, int(gs.on_first), int(gs.on_second),
                int(gs.on_third), gs.status, self.active_run_id, gs.received_at,
            ),
        )
        self.conn.commit()

    def record_market(self, market: Market, ts: float | None = None):
        first_seen = self.conn.execute(
            "SELECT first_seen_ts FROM markets WHERE slug = ?", (market.slug,)
        ).fetchone()
        self.conn.execute(
            """INSERT OR REPLACE INTO markets
               (slug, question, home_team, away_team, long_team, game_pk, start_time, first_seen_ts)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                market.slug, market.question, market.home_team, market.away_team,
                market.long_team, market.game_pk, market.start_time,
                first_seen["first_seen_ts"] if first_seen else (time.time() if ts is None else ts),
            ),
        )
        self.conn.commit()

    def record_decisions(self, rows: list[dict]):
        if not rows:
            return
        now = time.time()
        values = []
        for row in rows:
            item = {col: row.get(col) for col in _DECISION_COLUMNS}
            item["ts"] = item["ts"] if item["ts"] is not None else now
            item["run_id"] = item["run_id"] or self.active_run_id
            values.append(tuple(item[col] for col in _DECISION_COLUMNS))
        placeholders = ",".join("?" for _ in _DECISION_COLUMNS)
        self.conn.executemany(
            f"INSERT INTO decisions ({','.join(_DECISION_COLUMNS)}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()

    def paper_state(self, strategies: list[str]) -> tuple[dict[str, dict], list[dict]]:
        """Return the latest persisted paper account state for active strategies."""
        if not strategies:
            return {}, []
        placeholders = ",".join("?" for _ in strategies)
        accounts = self.conn.execute(
            f"SELECT strategy, cash, realized, closes FROM paper_accounts "
            f"WHERE strategy IN ({placeholders})", strategies,
        ).fetchall()
        positions = self.conn.execute(
            f"SELECT strategy, token, market, team, qty, entry_price, entry_fee, opened_at, trade_id "
            f"FROM paper_positions WHERE strategy IN ({placeholders})", strategies,
        ).fetchall()
        return (
            {r["strategy"]: dict(r) for r in accounts},
            [dict(r) for r in positions],
        )

    def save_paper_state(self, broker, commit: bool = True) -> None:
        """Atomically checkpoint a PaperBroker ledger after every fill."""
        strategies = list(broker.cash)
        now = time.time()
        def save() -> None:
            for strategy in strategies:
                self.conn.execute(
                    """INSERT INTO paper_accounts (strategy, cash, realized, closes, updated_at)
                       VALUES (?,?,?,?,?)
                       ON CONFLICT(strategy) DO UPDATE SET cash=excluded.cash,
                           realized=excluded.realized, closes=excluded.closes,
                           updated_at=excluded.updated_at""",
                    (strategy, broker.cash[strategy], broker.realized[strategy],
                     broker.closes[strategy], now),
                )
                self.conn.execute("DELETE FROM paper_positions WHERE strategy = ?", (strategy,))
                self.conn.executemany(
                    """INSERT INTO paper_positions
                       (strategy, token, market, team, qty, entry_price, entry_fee, opened_at, trade_id)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    [
                        (position.strategy, position.token, position.market_key, position.team,
                         position.qty, position.entry_price, position.entry_fee,
                         position.opened_at, position.trade_id)
                        for position in broker.open_positions(strategy)
                    ],
                )
        if commit:
            with self.conn:
                save()
        else:
            save()

    def realized_pnl(self, strategy: str, start: float, end: float) -> float:
        row = self.conn.execute(
            """SELECT COALESCE(SUM(pnl_usd), 0) AS pnl FROM trades
               WHERE strategy = ? AND action = 'CLOSE' AND ts >= ? AND ts < ?""",
            (strategy, start, end),
        ).fetchone()
        return float(row["pnl"])

    # --------------------------------------------------------------- summaries

    def strategy_stats(self) -> list[dict]:
        rows = self.conn.execute(
            """SELECT strategy,
                      COUNT(*)                                     AS trades,
                      SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins,
                      COALESCE(SUM(pnl_usd), 0)                    AS pnl_usd,
                      COALESCE(AVG(pnl_pct), 0)                    AS avg_pnl_pct,
                      COALESCE(MAX(pnl_pct), 0)                    AS best_pct,
                      COALESCE(MIN(pnl_pct), 0)                    AS worst_pct
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
        decisions = self.conn.execute(
            "SELECT COUNT(*), MAX(ts) FROM decisions"
        ).fetchone()
        game_states = self.conn.execute(
            "SELECT COUNT(*), MAX(ts) FROM game_states"
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
            "decisions": decisions[0] or 0,
            "latest_decision_ts": decisions[1],
            "game_states": game_states[0] or 0,
            "latest_game_state_ts": game_states[1],
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

    # -------------------------------------------------------------- review API

    def round_trips(self, start: float | None = None,
                    end: float | None = None) -> list[dict]:
        where = ["o.action = 'OPEN'"]
        params: list[float] = []
        if start is not None:
            where.append("o.ts >= ?")
            params.append(start)
        if end is not None:
            where.append("o.ts < ?")
            params.append(end)
        rows = self.conn.execute(
            f"""SELECT
                    o.id AS open_id, o.ts AS entry_ts, o.trade_id, o.strategy,
                    o.market, o.team, o.token, o.qty, o.price AS entry_price,
                    o.fair AS entry_fair, o.edge AS entry_edge, o.move,
                    o.spread, o.intended_price AS entry_intended_price,
                    o.slippage AS entry_slippage, o.reason AS open_reason,
                    c.id AS close_id, c.ts AS exit_ts, c.price AS exit_price,
                    c.fair AS exit_fair, c.intended_price AS exit_intended_price,
                    c.slippage AS exit_slippage, c.exit_kind, c.pnl_usd,
                    c.pnl_pct, c.reason AS close_reason
                FROM trades o
                LEFT JOIN trades c ON c.id = (
                    SELECT c2.id FROM trades c2
                    WHERE c2.action = 'CLOSE'
                      AND c2.trade_id IS NOT NULL AND c2.trade_id != ''
                      AND c2.trade_id = o.trade_id
                      AND c2.strategy = o.strategy
                      AND c2.ts >= o.ts
                    ORDER BY c2.ts
                    LIMIT 1
                )
                WHERE {' AND '.join(where)}
                ORDER BY o.ts""",
            params,
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["hold_secs"] = (
                item["exit_ts"] - item["entry_ts"]
                if item.get("exit_ts") is not None else None
            )
            result.append(item)
        return result

    def ticks_for_market(self, market: str, start: float | None = None,
                         end: float | None = None) -> list[dict]:
        where = ["market = ?"]
        params: list[float | str] = [market]
        if start is not None:
            where.append("ts >= ?")
            params.append(start)
        if end is not None:
            where.append("ts <= ?")
            params.append(end)
        rows = self.conn.execute(
            f"""SELECT ts, market, home_team, away_team, home_bid, home_ask,
                       home_mid, home_spread, long_bid, long_ask, two_sided, source,
                       run_id, received_at, source_ts
                FROM price_ticks
                WHERE {' AND '.join(where)}
                ORDER BY ts""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def decisions_summary(self, start: float | None = None,
                          end: float | None = None) -> list[dict]:
        where = ["1 = 1"]
        params: list[float] = []
        if start is not None:
            where.append("ts >= ?")
            params.append(start)
        if end is not None:
            where.append("ts < ?")
            params.append(end)
        rows = self.conn.execute(
            f"""SELECT stage, outcome, strategy, COUNT(*) AS count
                FROM decisions
                WHERE {' AND '.join(where)}
                GROUP BY stage, outcome, strategy
                ORDER BY stage, count DESC, outcome""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def near_misses(self, start: float | None = None, end: float | None = None,
                    within: float = 0.02) -> list[dict]:
        where = [
            "margin IS NOT NULL",
            "margin < 0",
            "margin >= ?",
            "outcome NOT IN ('signal', 'opened')",
        ]
        params: list[float] = [-abs(within)]
        if start is not None:
            where.append("ts >= ?")
            params.append(start)
        if end is not None:
            where.append("ts < ?")
            params.append(end)
        rows = self.conn.execute(
            f"""SELECT *
                FROM decisions
                WHERE {' AND '.join(where)}
                ORDER BY margin DESC, ts
                LIMIT 200""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def game_state_timeline(self, game_pk: int, start: float | None = None,
                            end: float | None = None) -> list[tuple[float, GameState]]:
        where = ["game_pk = ?"]
        params: list[int | float] = [game_pk]
        if start is not None:
            where.append("ts >= ?")
            params.append(start)
        if end is not None:
            where.append("ts <= ?")
            params.append(end)
        rows = self.conn.execute(
            f"""SELECT *
                FROM game_states
                WHERE {' AND '.join(where)}
                ORDER BY ts""",
            params,
        ).fetchall()
        return [
            (
                r["ts"],
                GameState(
                    game_pk=r["game_pk"],
                    inning=r["inning"],
                    is_top=bool(r["is_top"]),
                    outs=r["outs"],
                    home_score=r["home_score"],
                    away_score=r["away_score"],
                    on_first=bool(r["on_first"]),
                    on_second=bool(r["on_second"]),
                    on_third=bool(r["on_third"]),
                    status=r["status"],
                ),
            )
            for r in rows
        ]

    def markets_between(self, start: float, end: float) -> list[Market]:
        rows = self.conn.execute(
            """SELECT DISTINCT m.*
               FROM markets m
               WHERE (m.first_seen_ts >= ? AND m.first_seen_ts < ?)
                  OR m.slug IN (SELECT market FROM price_ticks WHERE ts >= ? AND ts < ?)
                  OR m.slug IN (SELECT market FROM decisions WHERE ts >= ? AND ts < ?)
                  OR m.slug IN (SELECT market FROM trades WHERE ts >= ? AND ts < ?)
               ORDER BY m.start_time, m.slug""",
            (start, end, start, end, start, end, start, end),
        ).fetchall()
        return [
            Market(
                slug=r["slug"],
                question=r["question"],
                home_team=r["home_team"],
                away_team=r["away_team"],
                long_team=r["long_team"],
                game_pk=r["game_pk"],
                start_time=r["start_time"],
            )
            for r in rows
        ]

    def close(self):
        self.conn.close()
