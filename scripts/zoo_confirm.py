"""Re-score the zoo winners against state_v1 ON THE SAME DATA, SAME STATISTIC.

zoo_eval.py compared its TEST numbers to state_v1's *published* holdout Brier
(0.15485). That is not a fair comparison: that number came from a different
pipeline, a different tick set, and (crucially) a pooled per-STATE mean, while
zoo_eval reports a per-GAME clustered mean. Two different statistics on two
different samples can differ by more than the effect we are hunting.

So here state_v1 is loaded and run through the identical code path as every
candidate, on the identical TEST states. Both statistics are printed:

  brier_pooled  -- mean over states (what state_v1 published)
  brier_game    -- mean over per-game means (what we make decisions on)

The paired per-game difference vs state_v1 is bootstrapped, so "better" has to
survive a CI rather than just win a point estimate.

Only the models that need no grid search are rebuilt here (NegBin is
reconstructed at its already-fitted params), which keeps this cheap to rerun.

Run: ./.venv/bin/python scripts/zoo_confirm.py
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
from polybot.state_model import EmpiricalStateModel
from polybot.walkforward import _cluster_ci

STATE_V1 = "artifacts/state_v1.json"


class StateV1(model_zoo.Model):
    """The frozen artifact, unmodified, wrapped in the zoo's predict API."""
    name = "state_v1 (frozen artifact)"

    def __init__(self):
        self.m = EmpiricalStateModel.load(STATE_V1, require_accepted=False)

    def fit(self, states, y, analytic=None):
        return self                                   # already fitted; never refit

    def predict(self, states, analytic=None):
        return np.asarray([self.m.predict(Z.to_gamestate(r)) for r in states])


def per_game(sq: np.ndarray, gid: np.ndarray) -> dict[str, float]:
    order = np.argsort(gid, kind="stable")
    g, s = gid[order], sq[order]
    _, starts = np.unique(g, return_index=True)
    ends = list(starts[1:]) + [len(g)]
    return {str(g[a]): float(s[a:b].mean()) for a, b in zip(starts, ends)}


def main() -> None:
    print("loading TRAIN + TEST...", flush=True)
    Xtr, ytr, _ = Z.load(Z.TRAIN)
    Xte, yte, gte = Z.load(Z.TEST)
    atr, ate = Z.analytic_probs(Xtr), Z.analytic_probs(Xte)
    print(f"TRAIN {Z.TRAIN}: {len(ytr):,} states")
    print(f"TEST  {Z.TEST}: {len(yte):,} states / {len(set(gte)):,} games\n")

    # NegBin at its VALID-selected params -- reconstructed, NOT re-searched.
    negbin = model_zoo.NegBin(lam=0.52, k=4, home_edge=0.22, tie_home=0.50)
    negbin.name = "negbin(lam=0.52,k=4,he=0.22,tie=0.50)"
    emp30 = model_zoo.Empirical(prior=30)
    ens = model_zoo.LogOddsEnsemble([negbin, model_zoo.Empirical(prior=30)],
                                    "ens(negbin+empirical)")

    cands = [model_zoo.Analytic(), StateV1(), emp30, negbin, ens]

    scores: dict[str, dict[str, float]] = {}
    print(f"{'model':44}{'brier_pooled':>14}{'brier_game':>12}{'logloss':>10}{'ece':>8}")
    for m in cands:
        t = time.time()
        m.fit(Xtr, ytr, atr)
        p = m.predict(Xte, ate)
        scores[m.name] = per_game((p - yte) ** 2, gte)
        print(f"{m.name:44}{Z.brier(p, yte):>14.5f}"
              f"{np.mean(list(scores[m.name].values())):>12.5f}"
              f"{Z.log_loss(p, yte):>10.5f}{Z.ece(p, yte):>8.4f}"
              f"   ({time.time()-t:.0f}s)", flush=True)

    print("\n" + "=" * 84)
    print("PAIRED vs state_v1 on TEST 2025 (per-game clustered; negative = better)")
    print("=" * 84)
    base = scores[StateV1.name]
    print(f"{'model':44}{'mean diff':>12}{'ci95 (per-game)':>26}{'better?':>9}")
    for m in cands:
        if m.name == StateV1.name:
            continue
        diff = {k: scores[m.name][k] - base[k] for k in base}
        lo, hi = _cluster_ci(diff, seed=f"v1:{m.name}")
        mean = float(np.mean(list(diff.values())))
        print(f"{m.name:44}{mean:>+12.5f}"
              f"{f'[{lo:+.5f}, {hi:+.5f}]':>26}{('YES' if hi < 0 else 'no'):>9}")


if __name__ == "__main__":
    main()
