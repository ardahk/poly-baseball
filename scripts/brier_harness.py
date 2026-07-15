"""Score any set of predictors against the market's Brier, clustered by game.

Why this exists rather than extending model_vs_market.py: that script pools
every tick as an independent observation (~6,000 ticks x 16 games). A pooled
Brier difference of 0.002 across 16 games is pseudo-replication, not evidence.
Here each GAME is one observation, and the model-vs-market difference is
bootstrapped paired within game.

All candidates are scored on the IDENTICAL tick set (any tick where some
candidate is undefined is dropped from all), so the comparison is clean.

Run: ./.venv/bin/python scripts/brier_harness.py /tmp/snap.db \
         analytic empirical anchored:empirical:1.0 anchored:empirical:0
"""
from __future__ import annotations

import os
import sqlite3
import statistics
import sys
from bisect import bisect_right

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predictors as P
from polybot.models import GameState
from polybot.walkforward import _cluster_ci

MARKET_BAR = 0.1716   # the bar: market Brier on collected ticks


def main() -> None:
    db = sys.argv[1] if len(sys.argv) > 1 else "/tmp/snap.db"
    specs = sys.argv[2:] or ["analytic", "empirical", "anchored:empirical:1.0"]

    built = [P.build(s, db) for s in specs]

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    slug_pk = {r["slug"]: r["game_pk"]
               for r in con.execute("SELECT slug, game_pk FROM markets")
               if r["game_pk"] is not None}
    home_won = {r["game_pk"]: int(r["home_score"] > r["away_score"])
                for r in con.execute("SELECT game_pk, home_score, away_score "
                                     "FROM game_states WHERE status='Final'")}
    states: dict[int, list] = {}
    for r in con.execute("SELECT * FROM game_states WHERE status='Live' "
                         "ORDER BY game_pk, ts"):
        gs = GameState(
            game_pk=r["game_pk"], inning=r["inning"], is_top=bool(r["is_top"]),
            outs=r["outs"], home_score=r["home_score"], away_score=r["away_score"],
            on_first=bool(r["on_first"]), on_second=bool(r["on_second"]),
            on_third=bool(r["on_third"]), status="Live",
        )
        states.setdefault(r["game_pk"], []).append((r["ts"], gs))
    state_ts = {pk: [t for t, _ in lst] for pk, lst in states.items()}

    # game_pk -> [n, sum_brier_market, sum_brier_model_0, sum_brier_model_1, ...]
    per_game: dict[int, list[float]] = {}
    skipped = 0
    for r in con.execute("SELECT market, ts, home_mid FROM price_ticks "
                         "ORDER BY market, ts"):
        pk = slug_pk.get(r["market"])
        if pk is None or pk not in home_won or pk not in state_ts:
            continue
        idx = bisect_right(state_ts[pk], r["ts"]) - 1
        if idx < 0:
            continue
        gs = states[pk][idx][1]
        y = home_won[pk]
        mkt = r["home_mid"]

        preds = [fn(gs, pk) for fn, _ in built]
        if any(p is None for p in preds):     # undefined for some candidate
            skipped += 1
            continue

        acc = per_game.setdefault(pk, [0.0] * (2 + len(built)))
        acc[0] += 1
        acc[1] += (mkt - y) ** 2
        for i, p in enumerate(preds):
            acc[2 + i] += (p - y) ** 2
    con.close()

    if not per_game:
        print("No scorable ticks.")
        return

    n_ticks = int(sum(a[0] for a in per_game.values()))
    games = len(per_game)
    mkt_by_game = {str(pk): a[1] / a[0] for pk, a in per_game.items()}
    mkt_brier = statistics.mean(mkt_by_game.values())

    print(f"db={db}  games={games}  scored ticks={n_ticks:,}"
          f"  (dropped {skipped:,} ticks undefined for some candidate)")
    print(f"Bar: beat market Brier {MARKET_BAR}. Each GAME is one observation.\n")
    print(f"{'predictor':28}{'Brier':>9}{'vs market':>11}"
          f"{'ci95 (paired, per-game)':>28}{'beats bar?':>12}")
    print(f"{'market (live mid)':28}{mkt_brier:>9.4f}{'':>11}{'':>28}{'--':>12}")

    for i, (_, label) in enumerate(built):
        by_game = {str(pk): a[2 + i] / a[0] for pk, a in per_game.items()}
        brier = statistics.mean(by_game.values())
        # paired within game: model minus market. negative = model better.
        diff = {pk: by_game[pk] - mkt_by_game[pk] for pk in by_game}
        lo, hi = _cluster_ci(diff, seed=f"brier:{label}")
        mean_diff = statistics.mean(diff.values())
        # beats the bar only if better than market AND the CI excludes zero
        beats = "YES" if (brier < MARKET_BAR and hi < 0) else "no"
        print(f"{label:28}{brier:>9.4f}{mean_diff:>+11.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>28}{beats:>12}")

    print("\n'vs market' < 0 means the model is closer to the outcome than the live mid.")
    print("A candidate only beats the bar if its paired CI upper bound < 0 "
          "(i.e. better than the market, significantly).")


if __name__ == "__main__":
    main()
