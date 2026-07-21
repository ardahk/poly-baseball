#!/usr/bin/env python3
"""Would maker execution rescue any strategy? Measured, not assumed.

The fleet is 100% taker: it crosses the spread and pays theta*p*(1-p) on both
legs. Fees alone are 46% of every dollar the fleet has lost. A maker pays no fee
and collects a rebate, which looks like a free halving of the cost floor.

It is not free. Resting an order means you are filled precisely when someone
wants to trade against you, so the mid walks away afterwards. Session 1 measured
that adverse selection at -1.12c/contract on 16 games, against a +0.22c rebate:
naked liquidity provision was NET NEGATIVE. This re-measures it on the current
tape and then applies the result to each strategy's real trade history.

Two parts:

  A. Adverse selection now, via the Part C markout method from cost_floor.py --
     mark a simulated resting bid against the future MID, not against
     settlement. Marking against settlement is contaminated by the base-rate
     luck of who won; marking against the mid is unbiased and far lower
     variance.

  B. Per-strategy P&L restated under maker execution: entry at the bid instead
     of the ask, rebate instead of fee, minus the measured adverse selection.

Both parts are OPTIMISTIC by construction: front-of-queue is assumed and every
touch fills. Real queue position means you miss the benign fills and still take
every toxic one, so true adverse selection is worse than what prints here.

    ./.venv/bin/python scripts/maker_vs_taker.py polybot.db
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from polybot.broker import (  # noqa: E402
    MAKER_THETA, TAKER_THETA, maker_rebate, taker_fee,
)

REST_SECS = 30          # how long a simulated order rests before being pulled
MARKOUT_SECS = 60       # horizon the fill is marked out against


def measure_adverse_selection(con: sqlite3.Connection) -> tuple[float, int]:
    """Mean per-contract markout of a resting bid, vs the mid `MARKOUT_SECS` later.

    Negative return value = adverse selection: you were picked off.
    """
    book: dict[str, list] = defaultdict(list)
    for r in con.execute(
        "SELECT market, ts, home_bid, home_ask, home_mid FROM price_ticks "
        "WHERE home_bid IS NOT NULL AND home_ask IS NOT NULL AND two_sided = 1 "
        "ORDER BY market, ts"
    ):
        book[r["market"]].append((r["ts"], r["home_bid"], r["home_ask"], r["home_mid"]))

    total, fills = 0.0, 0
    for ticks in book.values():
        n = len(ticks)
        for i, (ts, bid, _ask, _mid) in enumerate(ticks):
            # Filled when the best ask trades down to our resting bid.
            k, filled = i + 1, False
            while k < n and ticks[k][0] <= ts + REST_SECS:
                if ticks[k][2] <= bid:
                    filled = True
                    break
                k += 1
            if not filled:
                continue
            fill_ts = ticks[k][0]
            m = k
            while m < n and ticks[m][0] <= fill_ts + MARKOUT_SECS:
                m += 1
            if m >= n:
                continue                      # no future mid: drop, don't guess
            total += ticks[m][3] - bid
            fills += 1
    return (total / fills if fills else 0.0), fills


def main() -> int:
    db = sys.argv[1] if len(sys.argv) > 1 else "polybot.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    print("=" * 78)
    print("PART A - adverse selection on the current tape")
    print("=" * 78)
    markout, fills = measure_adverse_selection(con)
    adverse = -markout                        # positive number = a cost
    rebate = maker_rebate(0.50)
    print(f"  simulated resting-bid fills : {fills:,}")
    print(f"  markout vs mid(+{MARKOUT_SECS}s)      : {markout:+.4f} /contract")
    print(f"  => adverse selection        : {adverse:+.4f} /contract (a cost)")
    print(f"  maker rebate at p=0.50      : {rebate:+.4f} /contract (a credit)")
    print(f"  naked market-making         : {rebate - adverse:+.4f} /contract "
          f"({'PROFITABLE' if rebate > adverse else 'NEGATIVE'})")
    print()
    print(f"  taker cost at p=0.50        : "
          f"{0.0025 + taker_fee(TAKER_THETA, 0.50):.4f} /contract")
    print(f"  maker cost at p=0.50        : {adverse - rebate:.4f} /contract")
    print(f"  maker advantage             : "
          f"{(0.0025 + taker_fee(TAKER_THETA, 0.50)) - (adverse - rebate):+.4f} /contract")

    print()
    print("=" * 78)
    print("PART B - per-strategy P&L restated under maker execution")
    print("=" * 78)
    rows = con.execute(
        """SELECT o.strategy AS s, COUNT(*) AS n,
                  SUM(c.pnl_usd)                        AS taker_pnl,
                  SUM(o.qty)                            AS qty,
                  SUM(o.fee_usd + COALESCE(c.fee_usd,0)) AS fees,
                  AVG(o.price)                          AS px,
                  AVG(COALESCE(o.spread, 0))            AS spread,
                  SUM(CASE WHEN c.exit_kind = 'game_final' THEN 1 ELSE 0 END) AS holds
           FROM trades o
           JOIN trades c ON c.trade_id = o.trade_id AND c.action = 'CLOSE'
           WHERE o.action = 'OPEN'
           GROUP BY o.strategy HAVING n >= 10"""
    ).fetchall()

    print(f"{'strategy':26s}{'n':>5}{'taker P&L':>11}{'maker P&L':>11}"
          f"{'delta':>9}  verdict")
    out = []
    for r in rows:
        qty, px = r["qty"], r["px"]
        legs = 1.0 + (1.0 - r["holds"] / r["n"])      # holds pay one leg, not two
        # Maker saves the fees and the spread crossing it would have paid, then
        # gives back adverse selection on every entry (and every intraday exit).
        saved = r["fees"] + qty * (r["spread"] / 2.0) * legs
        cost = qty * (adverse - maker_rebate(px)) * legs
        maker_pnl = r["taker_pnl"] + saved - cost
        out.append((r["s"], r["n"], r["taker_pnl"], maker_pnl))
    for s, n, tp, mp in sorted(out, key=lambda x: x[3] - x[2], reverse=True):
        verdict = "rescued" if mp > 0 > tp else ("still losing" if mp < 0 else "")
        print(f"{s:26s}{n:>5}{tp:>+11.2f}{mp:>+11.2f}{mp - tp:>+9.2f}  {verdict}")

    rescued = sum(1 for _s, _n, tp, mp in out if mp > 0 > tp)
    still = sum(1 for _s, _n, _tp, mp in out if mp < 0)
    print()
    print(f"  {rescued} of {len(out)} strategies cross into profit under maker "
          f"execution; {still} still lose.")
    print("  OPTIMISTIC: assumes front-of-queue and a fill on every touch. Real")
    print("  queue position misses benign fills and takes every toxic one.")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
