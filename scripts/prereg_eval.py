"""Gated evaluator for the preregistered model-disagreement hypothesis.

Locked rule (docs/plans/2026-07-12-model-disagreement-hypothesis.md):
  predictor : artifacts/state_v1.json (frozen)
  entry     : live game, inning <= 6 AND |model P(home) - market mid| >= 0.10
              -> buy the side the model favors at the executable ask
  exit      : hold to settlement
  criteria  : per-game-clustered mean pnl/contract > +0.02
              AND bootstrap 95% CI lower bound > 0
              on >= 60 games NOT among the original 16.

The gate is STRUCTURAL: below 60 new games this prints progress and exits
without computing pnl, so the result cannot be peeked at and cherry-picked.

KNOWN POWER PROBLEM (see docs/research-log.md): per-game pnl SD is ~0.41, so at
n=60 the bootstrap lower bound only clears 0 if the observed mean exceeds
~+0.088 -- 4.4x the +0.02 threshold the test asks for. The two locked criteria
are mutually inconsistent at n=60. Detecting a true +0.02 edge needs ~1,150
games. This script does not alter the criteria; it reports them honestly.

Run: ./.venv/bin/python scripts/prereg_eval.py /tmp/snap.db
"""
from __future__ import annotations

import json
import os
import sqlite3
import statistics
import sys
from bisect import bisect_right

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polybot.broker import taker_fee
from polybot.models import GameState
from polybot.state_model import EmpiricalStateModel
from polybot.walkforward import _cluster_ci

REQUIRED_GAMES = 60
MEAN_BAR = 0.02
THETA = 0.06
ARTIFACT = "artifacts/state_v1.json"
EXCLUDED = "artifacts/prereg_excluded_games.json"


def main() -> None:
    db = sys.argv[1] if len(sys.argv) > 1 else "/tmp/snap.db"
    excluded = set(json.load(open(EXCLUDED))["game_pks"])
    predict = EmpiricalStateModel.load(ARTIFACT, require_accepted=False).predict

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    slug_pk = {r["slug"]: r["game_pk"]
               for r in con.execute("SELECT slug, game_pk FROM markets")
               if r["game_pk"] is not None}
    finals = {r["game_pk"]: int(r["home_score"] > r["away_score"])
              for r in con.execute("SELECT game_pk, home_score, away_score "
                                   "FROM game_states WHERE status='Final'")}

    # A game counts only if it is final, not in the frozen 16, and we observed it
    # from the start (a first Live state in inning 1 -- games we caught mid-flight
    # after an outage have biased coverage and are not eligible).
    first_inning = {r["game_pk"]: r["mi"] for r in con.execute(
        "SELECT game_pk, MIN(inning) AS mi FROM game_states "
        "WHERE status='Live' GROUP BY game_pk")}
    eligible = {pk for pk in finals
                if pk not in excluded and first_inning.get(pk, 99) == 1}

    print(f"db={db}")
    print(f"preregistered OOS progress: {len(eligible)} / {REQUIRED_GAMES} new games")
    print(f"  (excluded in-sample: {len(excluded)}; "
          f"final games in db: {len(finals)})")

    if len(eligible) < REQUIRED_GAMES:
        print(f"\nGATE: below {REQUIRED_GAMES} games. Not evaluating -- "
              "no pnl computed, by design.")
        print("Keep collecting. Do not evaluate early.")
        con.close()
        return

    states: dict[int, list] = {}
    for r in con.execute("SELECT * FROM game_states WHERE status='Live' "
                         "ORDER BY game_pk, ts"):
        if r["game_pk"] not in eligible:
            continue
        gs = GameState(
            game_pk=r["game_pk"], inning=r["inning"], is_top=bool(r["is_top"]),
            outs=r["outs"], home_score=r["home_score"], away_score=r["away_score"],
            on_first=bool(r["on_first"]), on_second=bool(r["on_second"]),
            on_third=bool(r["on_third"]), status="Live",
        )
        states.setdefault(r["game_pk"], []).append((r["ts"], gs))
    state_ts = {pk: [t for t, _ in lst] for pk, lst in states.items()}

    by_game: dict[int, list[float]] = {}
    for r in con.execute("SELECT market, ts, home_mid, home_bid, home_ask "
                         "FROM price_ticks WHERE home_bid IS NOT NULL "
                         "AND home_ask IS NOT NULL ORDER BY market, ts"):
        pk = slug_pk.get(r["market"])
        if pk not in eligible or pk not in state_ts:
            continue
        idx = bisect_right(state_ts[pk], r["ts"]) - 1
        if idx < 0:
            continue
        gs = states[pk][idx][1]
        if gs.inning > 6:                       # LOCKED: innings <= 6 only
            continue
        y = finals[pk]
        gap = predict(gs) - r["home_mid"]
        if abs(gap) < 0.10:                     # LOCKED: |disagreement| >= 0.10
            continue
        if gap > 0:
            ask, won = r["home_ask"], (y == 1)
        else:
            ask, won = 1.0 - r["home_bid"], (y == 0)
        if not (0.02 < ask < 0.98):
            continue
        pnl = (1.0 if won else 0.0) - ask - taker_fee(THETA, ask)
        a = by_game.setdefault(pk, [0.0, 0.0])
        a[0] += 1
        a[1] += pnl
    con.close()

    per_game = {str(pk): v[1] / v[0] for pk, v in by_game.items()}
    mean = statistics.mean(per_game.values())
    lo, hi = _cluster_ci(per_game, seed="prereg")

    print(f"\ngames traded      : {len(per_game)}")
    print(f"mean pnl/contract : {mean:+.4f}   (bar: > {MEAN_BAR:+.2f})")
    print(f"bootstrap 95% CI  : [{lo:+.4f}, {hi:+.4f}]   (bar: lower bound > 0)")

    passed = mean > MEAN_BAR and lo > 0
    print(f"\nRESULT: {'PASS' if passed else 'REJECT'}")
    if not passed:
        print("Hypothesis rejected under its own locked criteria. "
              "Pivot per the plan; do not retune and retest.")


if __name__ == "__main__":
    main()
