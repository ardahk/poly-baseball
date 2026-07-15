"""Refit chosen zoo models on ALL cached history and freeze them to artifacts/zoo/.

Why refit rather than reuse the TRAIN-fitted object: selection happened on
VALID 2024 and confirmation on TEST 2025, so those seasons are already spent
as evidence. The live market ticks we ultimately score against are from 2026,
which no historical season overlaps -- so for the *deployment* fit it is
correct (and strictly better) to use every season we have.

The frozen baselines (artifacts/state_v1.json, polybot/winprob.py) are not
touched. Each model lands in its own new file.

Selectors match a zoo model by exact name, else by unique prefix. Because
NegBin/AnalyticRecal only learn their name at fit time, the pre-fit name is
the bare family ("negbin"), which is what you pass here.

Run: ./.venv/bin/python scripts/zoo_export.py negbin "empirical(prior=30," ...
"""
from __future__ import annotations

import os
import pickle
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model_zoo
import zoo_data as Z

OUT_DIR = "artifacts/zoo"
ALL_YEARS = Z.TRAIN + Z.VALID + Z.TEST


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _pick(zoo, sel: str):
    exact = [m for m in zoo if m.name == sel]
    if exact:
        return exact[0]
    pre = [m for m in zoo if m.name.startswith(sel)]
    if len(pre) == 1:
        return pre[0]
    if not pre:
        raise SystemExit(f"no zoo model matches {sel!r}")
    raise SystemExit(f"{sel!r} is ambiguous: {[m.name for m in pre]}")


def main() -> None:
    sels = sys.argv[1:]
    if not sels:
        raise SystemExit("usage: zoo_export.py <name-or-prefix> [...]")

    zoo = model_zoo.build_zoo()
    chosen = [_pick(zoo, s) for s in sels]     # resolve BEFORE the slow load

    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, g = Z.load(ALL_YEARS)
    print(f"refit set {ALL_YEARS}: {len(y):,} states / {len(set(g)):,} games",
          flush=True)
    a = Z.analytic_probs(X)

    for sel, m in zip(sels, chosen):
        m.fit(X, y, a)
        path = os.path.join(OUT_DIR, f"{_slug(sel)}.pkl")
        with open(path, "wb") as fh:
            pickle.dump(m, fh)
        print(f"  {m.name:44} -> {path}", flush=True)


if __name__ == "__main__":
    main()
