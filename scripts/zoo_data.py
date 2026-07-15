"""Load cached historical MLB state timelines and build features.

Splits are chosen so model SELECTION never touches the final test set:

  TRAIN 2022+2023   fit every model here
  VALID 2024        compare all models here, pick survivors
  TEST  2025        touched ONCE, only for survivors

2025 is deliberately the same holdout season state_v1 used, so its reported
holdout Brier (0.15485) is directly comparable to anything we build.
Note state_v1 trained on 2022-2024, so it SAW our VALID year -- its valid score
is optimistic and is flagged, but its TEST score is clean.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polybot.models import GameState
from polybot.winprob import _RE24, _remaining_halves, home_win_probability

HIST_DIR = "artifacts/history"
TRAIN, VALID, TEST = [2022, 2023], [2024], [2025]

# columns of the raw state matrix
INNING, IS_TOP, OUTS, HS, AS_, B1, B2, B3 = range(8)


def load(years: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (states[n,8], y[n], game_id[n]) over all cached games in `years`."""
    states, ys, gids = [], [], []
    for year in years:
        path = os.path.join(HIST_DIR, f"{year}.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run scripts/fetch_history.py {year}")
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                for s in rec["s"]:
                    states.append(s)
                    ys.append(rec["y"])
                    gids.append(rec["pk"])
    return (np.asarray(states, dtype=np.int16),
            np.asarray(ys, dtype=np.int8),
            np.asarray(gids, dtype=np.int64))


def to_gamestate(row) -> GameState:
    return GameState(
        game_pk=0, inning=int(row[INNING]), is_top=bool(row[IS_TOP]),
        outs=int(row[OUTS]), home_score=int(row[HS]), away_score=int(row[AS_]),
        on_first=bool(row[B1]), on_second=bool(row[B2]), on_third=bool(row[B3]),
        status="Live",
    )


def analytic_probs(states: np.ndarray) -> np.ndarray:
    """The frozen analytic model, evaluated per state (baseline + stacking feature)."""
    return np.asarray([home_win_probability(to_gamestate(r)) for r in states])


_RE24_FLAT = np.asarray(_RE24)            # [outs, base_bitmask]


def features(states: np.ndarray, analytic: np.ndarray | None = None) -> np.ndarray:
    """Engineered features for the ML models.

    Includes the analytic model's own output as a feature (stacking), so a
    learner only has to fit the RESIDUAL of the physics model rather than
    rediscover baseball from scratch.
    """
    inning = states[:, INNING].astype(float)
    is_top = states[:, IS_TOP].astype(float)
    outs = np.clip(states[:, OUTS], 0, 2).astype(int)
    diff = (states[:, HS].astype(float) - states[:, AS_].astype(float))
    bases = (states[:, B1] | (states[:, B2] << 1) | (states[:, B3] << 2)).astype(int)
    n_on = (states[:, B1] + states[:, B2] + states[:, B3]).astype(float)

    # remaining half-innings of offense for each side (reuses the analytic model's rule)
    rem = np.asarray([_remaining_halves(to_gamestate(r)) for r in states], dtype=float)
    away_rem, home_rem = rem[:, 0], rem[:, 1]
    total_rem = away_rem + home_rem

    re_now = _RE24_FLAT[outs, bases]                 # run expectancy of current half
    re_signed = np.where(is_top > 0, -re_now, re_now)

    # the single most informative construct: score diff scaled by time left
    z = diff / np.sqrt(np.maximum(total_rem, 0.25))

    cols = [
        diff, z, inning, is_top, outs.astype(float), n_on, re_now, re_signed,
        away_rem, home_rem, total_rem,
        diff * total_rem, diff ** 2, np.sign(diff),
        (inning >= 7).astype(float), (np.abs(diff) <= 1).astype(float),
    ]
    if analytic is not None:
        a = np.clip(analytic, 1e-6, 1 - 1e-6)
        cols += [a, np.log(a / (1 - a))]             # stacking: level + log-odds
    return np.column_stack(cols)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Expected calibration error."""
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def game_clustered_brier_ci(p: np.ndarray, y: np.ndarray, gid: np.ndarray,
                            seed: int = 0, n_boot: int = 2000) -> tuple[float, float, float]:
    """Per-game mean Brier + bootstrap CI. Games, not states, are the unit."""
    order = np.argsort(gid, kind="stable")
    g, sq = gid[order], ((p - y) ** 2)[order]
    _, starts = np.unique(g, return_index=True)
    per_game = np.asarray([sq[a:b].mean() for a, b in
                           zip(starts, list(starts[1:]) + [len(g)])])
    rng = np.random.default_rng(seed)
    means = np.sort(rng.choice(per_game, size=(n_boot, len(per_game))).mean(axis=1))
    return (float(per_game.mean()),
            float(means[int(0.025 * n_boot)]),
            float(means[int(0.975 * n_boot)]))
