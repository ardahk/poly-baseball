"""Pluggable win-probability predictors for the Brier harness.

A predictor is `(GameState, game_pk) -> float | None`. Returning None means
"undefined for this game" (e.g. no pregame anchor); the harness drops those
ticks from *every* candidate so all models are scored on identical data.

The market-anchored predictor is the interesting one. The analytic and
empirical models are team-AGNOSTIC: both print ~0.52 at every pregame state,
while the market's last pregame mid across our games ranges 0.357 -> 0.732 --
the market knows who is pitching and we do not. Anchoring transfers the
model's log-odds *change* onto the market's pregame price, inheriting team
strength from the market and adding only the in-game state evolution.

It therefore cannot beat the market by knowing more. It can only beat the LIVE
market if the live market adds noise/overshoot after first pitch. That is the
claim under test.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polybot.model_features import transfer_model_delta
from polybot.models import GameState
from polybot.state_model import EmpiricalStateModel
from polybot.winprob import home_win_probability

Predictor = Callable[[GameState, int], "float | None"]

DEFAULT_ARTIFACT = "artifacts/state_v1.json"
ZOO_DIR = "artifacts/zoo"


def _load_zoo(slug: str):
    """Wrap a pickled zoo model (vectorised, states[n,8]) as fn(GameState)->float.

    Scoring 100k live ticks one state at a time is fine here: the harness makes
    one call per tick and the zoo models are cheap per row.
    """
    import pickle

    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import zoo_data as ZD                      # noqa: F401  (unpickling needs it)

    with open(os.path.join(ZOO_DIR, f"{slug}.pkl"), "rb") as fh:
        model = pickle.load(fh)

    def fn(gs: GameState) -> float:
        row = np.asarray([[gs.inning, int(gs.is_top), gs.outs,
                           gs.home_score, gs.away_score,
                           int(gs.on_first), int(gs.on_second), int(gs.on_third)]],
                         dtype=np.int16)
        a = np.asarray([home_win_probability(gs)])
        return float(model.predict(row, a)[0])

    return fn, f"zoo@{slug}"


def load_base(name: str, artifact: str = DEFAULT_ARTIFACT):
    """Return (fn(GameState) -> float, label) for a base model."""
    if name == "analytic":
        return home_win_probability, "analytic"
    if name == "empirical":
        model = EmpiricalStateModel.load(artifact, require_accepted=False)
        return model.predict, f"empirical:{os.path.basename(artifact)}"
    if name.startswith("zoo@"):
        return _load_zoo(name[4:])
    raise ValueError(f"unknown base model: {name}")


def pregame_anchors(db: str) -> dict[int, float]:
    """game_pk -> last home_mid strictly before scheduled first pitch.

    This is the offline equivalent of the anchor `ModelHistory.add_price`
    freezes live (polybot/model_features.py:122-131).
    """
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    anchors: dict[int, float] = {}
    for r in con.execute(
        "SELECT m.game_pk AS pk, "
        "       (SELECT p.home_mid FROM price_ticks p "
        "         WHERE p.market = m.slug AND p.ts < m.start_time "
        "         ORDER BY p.ts DESC LIMIT 1) AS anchor "
        "  FROM markets m "
        " WHERE m.game_pk IS NOT NULL AND m.start_time IS NOT NULL"
    ):
        if r["anchor"] is not None:
            anchors[r["pk"]] = r["anchor"]
    con.close()
    return anchors


def make_anchored(base_fn, base_label: str, anchors: dict[int, float],
                  beta: float) -> tuple[Predictor, str]:
    """pregame market price, moved by the model's log-odds change since pregame.

    beta=0 degenerates to "hold the pregame price all game, ignore the game" --
    the control that says how much the in-game market adds over the open.
    """
    anchor_model = base_fn(GameState(0, status="Live"))   # generic pregame state

    def predict(gs: GameState, pk: int) -> float | None:
        anchor_price = anchors.get(pk)
        if anchor_price is None:
            return None
        return transfer_model_delta(anchor_price, anchor_model, base_fn(gs), beta)

    return predict, f"anchored({base_label.split(':')[0]},b={beta:g})"


def build(spec: str, db: str, artifact: str = DEFAULT_ARTIFACT) -> tuple[Predictor, str]:
    """Build a predictor from a spec string.

    "analytic" | "empirical" | "anchored:<base>:<beta>"
    """
    parts = spec.split(":")
    if parts[0] != "anchored":
        base_fn, label = load_base(spec, artifact)
        return (lambda gs, pk: base_fn(gs)), label
    base_fn, base_label = load_base(parts[1], artifact)
    return make_anchored(base_fn, base_label, pregame_anchors(db), float(parts[2]))
