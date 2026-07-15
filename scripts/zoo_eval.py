"""Fit the whole model zoo on TRAIN, compare on VALID, confirm survivors on TEST.

The protocol exists to defeat the multiple-comparisons trap. With ~40 candidates,
whichever looks best on a small sample is probably just the luckiest. So:

  TRAIN 2022+2023  fit
  VALID 2024       rank all candidates; pick survivors (this is where selection
                   happens, on ~2,700 games -- real statistical power)
  TEST  2025       touched ONCE, only for survivors that beat the baselines

Ranking is by log-loss (a proper scoring rule that punishes overconfidence),
with Brier and ECE reported alongside. Brier CIs are clustered by GAME.

Run: ./.venv/bin/python scripts/zoo_eval.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_zoo
import zoo_data as Z

# state_v1's published holdout (2025) numbers -- the bar to beat on TEST
STATE_V1_TEST_BRIER = 0.15485
ANALYTIC_TEST_BRIER = 0.15786


def main() -> None:
    t0 = time.time()
    print("loading history...", flush=True)
    Xtr, ytr, gtr = Z.load(Z.TRAIN)
    Xva, yva, gva = Z.load(Z.VALID)
    print(f"TRAIN {Z.TRAIN}: {len(ytr):,} states / {len(set(gtr)):,} games")
    print(f"VALID {Z.VALID}: {len(yva):,} states / {len(set(gva)):,} games")

    print("precomputing analytic baseline (used as a stacking feature)...", flush=True)
    atr, ava = Z.analytic_probs(Xtr), Z.analytic_probs(Xva)

    zoo = model_zoo.build_zoo()
    print(f"\nfitting {len(zoo)} models...\n", flush=True)

    rows = []
    for m in zoo:
        t = time.time()
        try:
            m.fit(Xtr, ytr, atr)
            p = m.predict(Xva, ava)
        except Exception as exc:                      # a broken model must not
            print(f"  {m.name:44} FAILED: {exc}")     # kill the whole run
            continue
        ll = Z.log_loss(p, yva)
        br, lo, hi = Z.game_clustered_brier_ci(p, yva, gva)
        rows.append((ll, br, lo, hi, Z.ece(p, yva), m))
        print(f"  {m.name:44} logloss={ll:.5f}  brier={br:.5f}  "
              f"ece={Z.ece(p, yva):.4f}  ({time.time()-t:.0f}s)", flush=True)

    rows.sort(key=lambda r: r[0])
    print("\n" + "=" * 96)
    print(f"VALID {Z.VALID} LEADERBOARD (ranked by log-loss; selection happens HERE)")
    print("=" * 96)
    print(f"{'#':>3} {'model':44}{'logloss':>10}{'brier':>9}"
          f"{'brier ci95 (per-game)':>26}{'ece':>8}")
    for i, (ll, br, lo, hi, e, m) in enumerate(rows, 1):
        print(f"{i:>3} {m.name:44}{ll:>10.5f}{br:>9.5f}"
              f"{f'[{lo:.5f}, {hi:.5f}]':>26}{e:>8.4f}")

    base = next((r for r in rows if r[5].name == "analytic"), None)
    print(f"\nanalytic baseline on VALID: logloss={base[0]:.5f} brier={base[1]:.5f}"
          if base else "")

    # ---- TEST: touched once, only for the survivors ----
    survivors = [r for r in rows[:5]]
    print("\n" + "=" * 96)
    print(f"TEST {Z.TEST} -- top 5 by VALID log-loss, scored ONCE")
    print(f"bars: state_v1 holdout brier={STATE_V1_TEST_BRIER}, "
          f"analytic={ANALYTIC_TEST_BRIER}")
    print("=" * 96)
    Xte, yte, gte = Z.load(Z.TEST)
    ate = Z.analytic_probs(Xte)
    print(f"TEST states={len(yte):,} games={len(set(gte)):,}\n")
    print(f"{'model':44}{'logloss':>10}{'brier':>9}"
          f"{'brier ci95 (per-game)':>26}{'ece':>8}{'beats v1?':>11}")
    for _, _, _, _, _, m in survivors:
        p = m.predict(Xte, ate)
        ll = Z.log_loss(p, yte)
        br, lo, hi = Z.game_clustered_brier_ci(p, yte, gte)
        beats = "YES" if br < STATE_V1_TEST_BRIER else "no"
        print(f"{m.name:44}{ll:>10.5f}{br:>9.5f}"
              f"{f'[{lo:.5f}, {hi:.5f}]':>26}{Z.ece(p, yte):>8.4f}{beats:>11}")

    print(f"\ndone in {time.time()-t0:.0f}s")
    print("\nNOTE: beating state_v1 on TEST means a better STATE model. It does NOT")
    print("mean it beats the MARKET -- that is a separate bar (0.1716) and requires")
    print("scoring against market ticks via scripts/brier_harness.py.")


if __name__ == "__main__":
    main()
