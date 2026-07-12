"""Is there a tradeable corner? Hold-to-settlement EV of betting the model.

Whenever |model - market| >= threshold, buy the side the model favors at the
market's executable ask, hold to settlement, collect $1 if that side wins else
$0. Net of the entry ask premium + taker fee (settlement pays no exit fee/spread
in this venue). This is the PUREST test of whether the model's disagreements
carry real directional edge -- if even hold-to-settlement loses, there is none.

Run: ./.venv/bin/python scripts/model_markout.py [db] [min_gap] [artifact]
"""
from __future__ import annotations

import os
import sqlite3
import statistics
import sys
from bisect import bisect_right

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polybot.broker import taker_fee
from polybot.models import GameState
from polybot.winprob import home_win_probability

DB = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pb_check.db"
MIN_GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
ARTIFACT = sys.argv[3] if len(sys.argv) > 3 else None
THETA = 0.06

if ARTIFACT:
    from polybot.state_model import EmpiricalStateModel
    predict = EmpiricalStateModel.load(ARTIFACT, require_accepted=False).predict
    LABEL = f"empirical:{os.path.basename(ARTIFACT)}"
else:
    predict = home_win_probability
    LABEL = "analytic"


def main() -> None:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    slug_pk = {r["slug"]: r["game_pk"]
               for r in con.execute("SELECT slug, game_pk FROM markets")
               if r["game_pk"] is not None}
    home_won = {r["game_pk"]: int(r["home_score"] > r["away_score"])
                for r in con.execute(
                    "SELECT game_pk, home_score, away_score FROM game_states "
                    "WHERE status='Final'")}
    states: dict[int, list] = {}
    for r in con.execute(
            "SELECT * FROM game_states WHERE status='Live' ORDER BY game_pk, ts"):
        gs = GameState(game_pk=r["game_pk"], inning=r["inning"], is_top=bool(r["is_top"]),
                       outs=r["outs"], home_score=r["home_score"], away_score=r["away_score"],
                       on_first=bool(r["on_first"]), on_second=bool(r["on_second"]),
                       on_third=bool(r["on_third"]), status="Live")
        states.setdefault(r["game_pk"], []).append((r["ts"], gs))
    state_ts = {pk: [t for t, _ in lst] for pk, lst in states.items()}

    # bucket -> {game_pk: [n_ticks, sum_pnl, sum_win]}  (cluster by game)
    cells: dict[tuple, dict] = {}
    tot: dict = {}
    for r in con.execute(
            "SELECT market, ts, home_mid, home_bid, home_ask FROM price_ticks "
            "WHERE home_bid IS NOT NULL AND home_ask IS NOT NULL ORDER BY market, ts"):
        pk = slug_pk.get(r["market"])
        if pk is None or pk not in home_won or pk not in state_ts:
            continue
        idx = bisect_right(state_ts[pk], r["ts"]) - 1
        if idx < 0:
            continue
        gs = states[pk][idx][1]
        y = home_won[pk]
        mid = r["home_mid"]
        mdl = predict(gs)
        gap = mdl - mid
        if abs(gap) < MIN_GAP:
            continue
        if gap > 0:                       # model: home undervalued -> BUY HOME
            ask = r["home_ask"]
            won = (y == 1)
        else:                             # model: away undervalued -> BUY AWAY
            ask = 1.0 - r["home_bid"]
            won = (y == 0)
        if not (0.02 < ask < 0.98):       # skip untradeable extremes
            continue
        payout = 1.0 if won else 0.0
        pnl = payout - ask - taker_fee(THETA, ask)   # settle pays no exit fee
        ib = "1-3" if gs.inning <= 3 else "4-6" if gs.inning <= 6 else "7+"
        gb = ".05-.10" if abs(gap) < .10 else ">=.10"
        for bucket in (cells.setdefault((ib, gb), {}), tot):
            g = bucket.setdefault(pk, [0, 0.0, 0.0])
            g[0] += 1; g[1] += pnl; g[2] += payout

    con.close()
    if not tot:
        print("No qualifying moments.")
        return

    def summarize(by_game: dict) -> tuple[int, int, float, float, float]:
        games = len(by_game)
        ticks = sum(v[0] for v in by_game.values())
        pnl_sum = sum(v[1] for v in by_game.values())
        win_sum = sum(v[2] for v in by_game.values())
        # per-game mean pnl/contract: each game votes once (honest clustering)
        per_game = statistics.mean(v[1] / v[0] for v in by_game.values())
        return games, ticks, 100 * win_sum / ticks, pnl_sum / ticks, per_game

    print(f"db={DB}  model={LABEL}  min_gap={MIN_GAP}  theta={THETA}")
    print("Buy the model's favored side at market ask, hold to settlement.\n")
    print(f"{'inning':8}{'gap':10}{'games':>7}{'ticks':>8}{'win%':>8}"
          f"{'pnl/tick':>11}{'pnl/game':>11}")
    for key in sorted(cells):
        gm, tk, wp, pt, pg = summarize(cells[key])
        print(f"{key[0]:8}{key[1]:10}{gm:>7}{tk:>8}{wp:>7.1f}%{pt:>11.4f}{pg:>11.4f}")
    gm, tk, wp, pt, pg = summarize(tot)
    print(f"\n{'ALL':8}{'':10}{gm:>7}{tk:>8}{wp:>7.1f}%{pt:>11.4f}{pg:>11.4f}")
    print("\npnl/game clusters by game (each game = one vote); trust it over "
          "pnl/tick.\nA cell needs many games AND pnl/game > +0.02 to be real.")


if __name__ == "__main__":
    main()
