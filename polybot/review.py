"""End-of-day review for recorded polybot paper-trading sessions."""
from __future__ import annotations

import statistics
from collections import defaultdict
from .journal import Journal
from .models import Market
from .timeframe import day_bounds

_KNOBS = {
    "no_edge": "strategy.min_edge",
    "small_move": "strategy.move_threshold",
    "not_playful": "strategy.min_volatility",
    "wide_spread": "strategy.max_spread",
    "stale_quote": "strategy.max_quote_age_secs",
    "price_band": "strategy.min_price/max_price",
    "early_game": "strategy.early_game_min_edge/fair_extreme",
    "small_residual": "strategy.residual_threshold",
    "small_model_move": "strategy.residual_min_model_delta",
    "anchor_stale": "strategy.residual_response_secs/market_anchor_max_age_secs",
    "execution_cost": "strategy.min_edge (net of fees + spread)",
    "cost_floor": "strategy.cost_floor_multiple (upside at target vs all-in cost)",
}


def _fmt_money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:+.2f}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def _token_price(market: Market | None, token: str, home_mid: float) -> float | None:
    if market is None:
        if token.endswith(":LONG"):
            return home_mid
        if token.endswith(":SHORT"):
            return 1.0 - home_mid
        return None
    if token == market.home_token:
        return home_mid
    if token == market.away_token:
        return 1.0 - home_mid
    return None


def _prices_for_trip(journal: Journal, trip: dict, market: Market | None,
                     start: float, end: float) -> list[tuple[float, float]]:
    prices = []
    for tick in journal.ticks_for_market(trip["market"], start, end):
        px = _token_price(market, trip["token"], tick["home_mid"])
        if px is not None:
            prices.append((tick["ts"], px))
    return prices


def _annotate_trips(journal: Journal, trips: list[dict],
                    markets: dict[str, Market], day_end: float) -> list[dict]:
    annotated = []
    for trip in trips:
        market = markets.get(trip["market"])
        exit_ts = trip["exit_ts"] or day_end
        during = _prices_for_trip(journal, trip, market, trip["entry_ts"], exit_ts)
        entry = trip["entry_price"]
        if during:
            trip["mfe"] = max(px - entry for _, px in during)
            trip["mae"] = min(px - entry for _, px in during)
        else:
            trip["mfe"] = None
            trip["mae"] = None
        if trip["exit_ts"] is not None:
            post = _prices_for_trip(
                journal, trip, market, trip["exit_ts"], trip["exit_ts"] + 600
            )
            if post:
                trip["post_max"] = max(px for _, px in post)
                trip["post_min"] = min(px for _, px in post)
            else:
                trip["post_max"] = None
                trip["post_min"] = None
        else:
            trip["post_max"] = None
            trip["post_min"] = None
        annotated.append(trip)
    return annotated


def _equity_summary(journal: Journal, start: float, end: float) -> list[dict]:
    rows = journal.conn.execute(
        """SELECT strategy, ts, equity
           FROM equity
           WHERE ts >= ? AND ts < ?
           ORDER BY strategy, ts""",
        (start, end),
    ).fetchall()
    by_strategy: dict[str, list] = defaultdict(list)
    for row in rows:
        by_strategy[row["strategy"]].append(row)
    return [
        {
            "strategy": strat,
            "first": values[0]["equity"],
            "last": values[-1]["equity"],
            "change": values[-1]["equity"] - values[0]["equity"],
        }
        for strat, values in sorted(by_strategy.items())
    ]


def _closed_trade_summary(journal: Journal, start: float, end: float) -> list[dict]:
    rows = journal.conn.execute(
        """SELECT strategy, COUNT(*) AS trades,
                  SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS wins,
                  COALESCE(SUM(pnl_usd), 0) AS pnl_usd
           FROM trades
           WHERE action = 'CLOSE' AND ts >= ? AND ts < ?
           GROUP BY strategy
           ORDER BY strategy""",
        (start, end),
    ).fetchall()
    return [dict(r) for r in rows]


def _print_day_summary(journal: Journal, start: float, end: float) -> None:
    equity = {row["strategy"]: row for row in _equity_summary(journal, start, end)}
    trades = {row["strategy"]: row for row in _closed_trade_summary(journal, start, end)}
    strategies = sorted(set(equity) | set(trades))
    print("DAY SUMMARY")
    print("-" * 72)
    if not strategies:
        print("  No equity snapshots or closed trades in this window.")
        return
    print(f"{'strategy':<10}{'equity':>19}{'change':>12}{'trades':>9}{'win %':>8}{'P&L':>10}")
    for strat in strategies:
        eq = equity.get(strat)
        tr = trades.get(strat, {})
        count = tr.get("trades", 0) or 0
        wins = tr.get("wins", 0) or 0
        win_rate = 100.0 * wins / count if count else 0.0
        equity_text = "n/a"
        change = None
        if eq:
            equity_text = f"${eq['first']:.2f} -> ${eq['last']:.2f}"
            change = eq["change"]
        print(f"{strat:<10}{equity_text:>19}{_fmt_money(change):>12}"
              f"{count:>9}{win_rate:>7.1f}%{_fmt_money(tr.get('pnl_usd')):>10}")


def _print_trades(trips: list[dict]) -> None:
    print("\nTRADES")
    print("-" * 108)
    if not trips:
        print("  No opened trades in this window.")
        return
    header = (
        f"{'strategy':<8}{'team':<18}{'entry':>8}{'fair':>8}{'edge':>8}"
        f"{'move':>8}{'spr':>7}{'slip':>8}{'exit':>8}{'kind':>12}"
        f"{'hold':>8}{'pnl':>9}{'MFE':>8}{'MAE':>8}{'post':>8}"
    )
    print(header)
    for t in trips:
        hold = "open" if t["hold_secs"] is None else f"{t['hold_secs'] / 60:.1f}m"
        post = None
        if t.get("post_max") is not None and t.get("exit_price") is not None:
            post = t["post_max"] - t["exit_price"]
        print(
            f"{t['strategy']:<8}{t['team'][:17]:<18}{t['entry_price']:>8.3f}"
            f"{(t['entry_fair'] if t['entry_fair'] is not None else 0):>8.3f}"
            f"{(t['entry_edge'] if t['entry_edge'] is not None else 0):>8.3f}"
            f"{(t['move'] if t['move'] is not None else 0):>8.3f}"
            f"{(t['spread'] if t['spread'] is not None else 0):>7.3f}"
            f"{(t['entry_slippage'] if t['entry_slippage'] is not None else 0):>8.3f}"
            f"{(t['exit_price'] if t['exit_price'] is not None else 0):>8.3f}"
            f"{(t['exit_kind'] or 'open'):>12}{hold:>8}"
            f"{_fmt_money(t.get('pnl_usd')):>9}{_fmt_pct(t.get('mfe')):>8}"
            f"{_fmt_pct(t.get('mae')):>8}{_fmt_pct(post):>8}"
        )


def _print_exit_breakdown(trips: list[dict]) -> None:
    print("\nEXIT BREAKDOWN")
    print("-" * 72)
    closed = [t for t in trips if t["exit_ts"] is not None]
    if not closed:
        print("  No closed trades with linked trade IDs.")
        return
    groups: dict[str, list[dict]] = defaultdict(list)
    for trip in closed:
        groups[trip["exit_kind"] or "other"].append(trip)
    print(f"{'exit kind':<14}{'count':>7}{'total pnl':>13}{'avg pnl':>11}{'avg hold':>11}")
    for kind, rows in sorted(groups.items()):
        total = sum(r["pnl_usd"] or 0.0 for r in rows)
        avg_hold = statistics.mean((r["hold_secs"] or 0.0) / 60.0 for r in rows)
        print(f"{kind:<14}{len(rows):>7}{total:>13.2f}{total / len(rows):>11.2f}"
              f"{avg_hold:>10.1f}m")


def _print_funnel(journal: Journal, start: float, end: float) -> int:
    print("\nENTRY FUNNEL")
    print("-" * 72)
    rows = journal.decisions_summary(start, end)
    total = sum(r["count"] for r in rows)
    if not rows:
        print("  No decision rows recorded.")
        return 0
    print(f"{'stage':<14}{'outcome':<18}{'strategy':<10}{'count':>8}{'%':>8}")
    for row in rows:
        pct = 100.0 * row["count"] / total if total else 0.0
        print(f"{row['stage']:<14}{row['outcome']:<18}{(row['strategy'] or '-'): <10}"
              f"{row['count']:>8}{pct:>7.1f}%")
    return sum(r["count"] for r in rows if r["outcome"] == "opened")


def _print_near_misses(journal: Journal, start: float, end: float, near: float) -> list[dict]:
    print("\nNEAR MISSES")
    print("-" * 72)
    rows = journal.near_misses(start, end, within=near)
    if not rows:
        print(f"  No rejected gates within {near:.3f} of their threshold.")
        return []
    print(f"{'outcome':<16}{'knob':<34}{'margin':>9}{'market':>12}")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(row["outcome"], _KNOBS.get(row["outcome"], "config"))] += 1
    for (outcome, knob), count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        sample = next(r for r in rows if r["outcome"] == outcome)
        print(f"{outcome:<16}{knob:<34}{sample['margin']:>9.3f}{count:>12}")
    return rows


def _print_hints(trips: list[dict], near_rows: list[dict], opened_count: int) -> None:
    print("\nTHRESHOLD HINTS")
    print("-" * 72)
    hints = []
    stop_recovers = [
        t for t in trips
        if t.get("exit_kind") == "stop_loss"
        and t.get("post_max") is not None
        and t["post_max"] >= t["entry_price"]
    ]
    if stop_recovers:
        hints.append(f"{len(stop_recovers)} stop-loss exits recovered within 10 min; stop_loss may be tight.")
    tp_runs = [
        t for t in trips
        if t.get("exit_kind") == "take_profit"
        and t.get("post_max") is not None
        and t.get("exit_price") is not None
        and t["post_max"] - t["exit_price"] >= 0.03
    ]
    if tp_runs:
        hints.append(f"{len(tp_runs)} take-profits kept running by 3c+; take_profit may be low.")
    time_positive = [
        t for t in trips
        if t.get("exit_kind") == "time_stop" and t.get("mfe") is not None and t["mfe"] > 0
    ]
    if time_positive:
        hints.append(f"{len(time_positive)} time-stops had positive excursion; max_hold_secs may need a look.")
    if near_rows and len(near_rows) > max(5, opened_count * 2):
        hints.append(
            f"{len(near_rows)} near-miss rejections versus {opened_count} opens; sweep the listed knobs."
        )
    slippage_ratios = [
        abs(t["entry_slippage"]) / abs(t["entry_edge"])
        for t in trips
        if t.get("entry_slippage") is not None and t.get("entry_edge")
    ]
    if slippage_ratios and statistics.mean(slippage_ratios) > 0.25:
        hints.append("Average entry slippage consumed more than 25% of edge; tighten spread/slippage assumptions.")
    if not hints:
        print("  No obvious threshold hint fired.")
    for hint in hints:
        print(f"  - {hint}")


def print_review(db_path: str = "polybot.db", day: str | None = None,
                 near: float = 0.02, timezone: str = "America/Los_Angeles") -> None:
    start, end, label = day_bounds(day, timezone)
    journal = Journal(db_path)
    try:
        markets = {m.key: m for m in journal.markets_between(start, end)}
        trips = _annotate_trips(journal, journal.round_trips(start, end), markets, end)
        print("=" * 72)
        print(f"POLYBOT REVIEW {label} ({timezone})")
        print("=" * 72)
        _print_day_summary(journal, start, end)
        _print_trades(trips)
        _print_exit_breakdown(trips)
        opened_count = _print_funnel(journal, start, end)
        near_rows = _print_near_misses(journal, start, end, near)
        _print_hints(trips, near_rows, opened_count)
    finally:
        journal.close()
