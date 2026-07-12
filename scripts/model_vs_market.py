"""Does the win-probability model beat the market's own price?

The fade strategy bets the model over the market whenever they disagree by
>= min_edge. This script tests that premise directly on collected data:
for every price tick, as-of join the game state in effect, then compare the
model prediction and the market mid to the ACTUAL final outcome (Brier score).

If the market's Brier is <= the model's, the model has no informational edge
and fading the market is structurally a losing bet — no exit/fee tuning fixes it.

Run:  ./.venv/bin/python scripts/model_vs_market.py [db_path] [min_edge]
"""
from __future__ import annotations

import os
import sqlite3
import sys
from bisect import bisect_right

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polybot.models import GameState
from polybot.winprob import home_win_probability

DB = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pb_check.db"
MIN_EDGE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05


def brier(p: float, y: int) -> float:
    return (p - y) ** 2


def main() -> None:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row

    # market slug -> game_pk
    slug_pk = {r["slug"]: r["game_pk"]
               for r in con.execute("SELECT slug, game_pk FROM markets")
               if r["game_pk"] is not None}

    # final outcome per game_pk (needs a Final state row)
    home_won: dict[int, int] = {}
    for r in con.execute(
        "SELECT game_pk, home_score, away_score FROM game_states "
        "WHERE status='Final'"
    ):
        home_won[r["game_pk"]] = int(r["home_score"] > r["away_score"])

    # states per game_pk, time-ordered, as (ts, GameState)
    states: dict[int, list[tuple[float, GameState]]] = {}
    for r in con.execute(
        "SELECT * FROM game_states WHERE status='Live' ORDER BY game_pk, ts"
    ):
        gs = GameState(
            game_pk=r["game_pk"], inning=r["inning"], is_top=bool(r["is_top"]),
            outs=r["outs"], home_score=r["home_score"], away_score=r["away_score"],
            on_first=bool(r["on_first"]), on_second=bool(r["on_second"]),
            on_third=bool(r["on_third"]), status="Live",
        )
        states.setdefault(r["game_pk"], []).append((r["ts"], gs))
    state_ts = {pk: [t for t, _ in lst] for pk, lst in states.items()}

    # accumulators
    n = 0
    bm = bmod = 0.0            # total Brier market / model
    by_inning: dict[str, list] = {}     # bucket -> [n, bm, bmod]
    by_gap: dict[str, list] = {}        # disagreement bucket -> [n, bm, bmod]
    fade_n = fade_market_closer = 0     # at trade-worthy disagreements

    for r in con.execute(
        "SELECT market, ts, home_mid FROM price_ticks ORDER BY market, ts"
    ):
        pk = slug_pk.get(r["market"])
        if pk is None or pk not in home_won or pk not in state_ts:
            continue
        idx = bisect_right(state_ts[pk], r["ts"]) - 1
        if idx < 0:
            continue
        gs = states[pk][idx][1]
        y = home_won[pk]
        mkt = r["home_mid"]
        mdl = home_win_probability(gs)

        n += 1
        bm += brier(mkt, y)
        bmod += brier(mdl, y)

        ib = "1-3" if gs.inning <= 3 else "4-6" if gs.inning <= 6 else "7+"
        b = by_inning.setdefault(ib, [0, 0.0, 0.0])
        b[0] += 1; b[1] += brier(mkt, y); b[2] += brier(mdl, y)

        gap = abs(mdl - mkt)
        gb = "<.03" if gap < .03 else ".03-.06" if gap < .06 \
            else ".06-.10" if gap < .10 else ">=.10"
        g = by_gap.setdefault(gb, [0, 0.0, 0.0])
        g[0] += 1; g[1] += brier(mkt, y); g[2] += brier(mdl, y)

        # the fade's actual bet: model disagrees with market by >= min_edge
        if gap >= MIN_EDGE:
            fade_n += 1
            if abs(mkt - y) < abs(mdl - y):
                fade_market_closer += 1

    con.close()
    if not n:
        print("No scorable ticks (need markets->game_pk, Final states, Live states).")
        return

    print(f"db={DB}  scored ticks={n:,}  min_edge={MIN_EDGE}")
    print(f"{'':16}{'Brier':>10}   (lower is better)")
    print(f"{'market':16}{bm/n:>10.4f}")
    print(f"{'model':16}{bmod/n:>10.4f}")
    verdict = "MODEL beats market" if bmod < bm else "MARKET beats model"
    print(f"  --> {verdict} by {abs(bm-bmod)/n:.4f} Brier\n")

    print(f"{'inning':10}{'n':>8}{'Brier_mkt':>12}{'Brier_mdl':>12}{'winner':>10}")
    for k in ("1-3", "4-6", "7+"):
        if k in by_inning:
            c, m, d = by_inning[k]
            print(f"{k:10}{c:>8}{m/c:>12.4f}{d/c:>12.4f}"
                  f"{'model' if d < m else 'market':>10}")

    print(f"\n{'|mdl-mkt|':10}{'n':>8}{'Brier_mkt':>12}{'Brier_mdl':>12}{'winner':>10}"
          "   <- fade trades in the wide buckets")
    for k in ("<.03", ".03-.06", ".06-.10", ">=.10"):
        if k in by_gap:
            c, m, d = by_gap[k]
            print(f"{k:10}{c:>8}{m/c:>12.4f}{d/c:>12.4f}"
                  f"{'model' if d < m else 'market':>10}")

    if fade_n:
        pct = 100 * fade_market_closer / fade_n
        print(f"\nAt disagreements >= {MIN_EDGE} (where the fade fires), the MARKET "
              f"was closer to the outcome {pct:.1f}% of the time ({fade_n:,} moments).")
        print("  >50% means fading the market is a losing bet by construction.")


if __name__ == "__main__":
    main()
