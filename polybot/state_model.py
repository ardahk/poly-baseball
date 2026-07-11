"""Versioned empirical correction for the analytic live win-probability model."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from .model_features import state_signature
from .models import GameState
from .winprob import home_win_probability

ARTIFACT_VERSION = 1


def clamp_probability(value: float, eps: float = 1e-6) -> float:
    return min(max(float(value), eps), 1.0 - eps)


def brier_loss(p: float, y: int) -> float:
    return (p - y) ** 2


def log_loss(p: float, y: int) -> float:
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def holdout_regressed(analytic, empirical) -> bool:
    """One acceptance rule shared by artifact fitting and artifact loading."""
    a = analytic if isinstance(analytic, dict) else vars(analytic)
    e = empirical if isinstance(empirical, dict) else vars(empirical)
    return e["brier"] > a["brier"] or e["log_loss"] > a["log_loss"]


def empirical_state_key(gs: GameState) -> str:
    """Sampling-stable baseball state cell used by the empirical lookup."""
    inning = min(max(gs.inning, 1), 10)
    diff = min(max(gs.home_score - gs.away_score, -6), 6)
    bases = (1 if gs.on_first else 0) | (2 if gs.on_second else 0) \
        | (4 if gs.on_third else 0)
    return f"{inning}:{int(gs.is_top)}:{min(max(gs.outs, 0), 2)}:{diff}:{bases}"


@dataclass(frozen=True)
class Score:
    games: int
    states: int
    brier: float
    log_loss: float
    calibration_error: float


class EmpiricalStateModel:
    """Beta-smoothed state lookup with the analytic formula as its prior."""

    def __init__(self, cells: dict[str, dict], prior_strength: float = 30.0,
                 metadata: dict | None = None):
        self.cells = cells
        self.prior_strength = float(prior_strength)
        self.metadata = metadata or {}

    def predict(self, gs: GameState) -> float:
        analytic = home_win_probability(gs)
        cell = self.cells.get(empirical_state_key(gs))
        if not cell:
            return analytic
        count = float(cell["count"])
        wins = float(cell["home_wins"])
        return (wins + self.prior_strength * analytic) / (count + self.prior_strength)

    __call__ = predict

    @classmethod
    def fit(cls, games: list[tuple[list[tuple[float, GameState]], bool]],
            prior_strength: float = 30.0) -> "EmpiricalStateModel":
        cells: dict[str, dict] = {}
        for timeline, home_won in games:
            outcome = int(home_won)
            seen: set[tuple] = set()
            for _, gs in timeline:
                # One row per distinct received state, never per price tick.
                signature = state_signature(gs)
                if signature in seen:
                    continue
                seen.add(signature)
                key = empirical_state_key(gs)
                cell = cells.setdefault(key, {"count": 0, "home_wins": 0})
                cell["count"] += 1
                cell["home_wins"] += outcome
        return cls(cells, prior_strength)

    def to_dict(self) -> dict:
        return {
            "artifact_version": ARTIFACT_VERSION,
            "model": "empirical_state_v1",
            "prior_strength": self.prior_strength,
            "metadata": self.metadata,
            "cells": self.cells,
        }

    def save(self, path: str | Path) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({**self.to_dict(), "sha256": digest}, indent=2) + "\n")
        return digest

    @classmethod
    def load(cls, path: str | Path, *, require_accepted: bool = False) -> "EmpiricalStateModel":
        data = json.loads(Path(path).read_text())
        if data.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError(f"unsupported state-model artifact version: {data.get('artifact_version')}")
        if data.get("model") != "empirical_state_v1" or not isinstance(data.get("cells"), dict):
            raise ValueError("invalid empirical state-model artifact")
        supplied = data.pop("sha256", None)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        actual = hashlib.sha256(canonical.encode()).hexdigest()
        if supplied != actual:
            raise ValueError("state-model artifact checksum mismatch")
        for key, cell in data["cells"].items():
            if not isinstance(key, str) or not isinstance(cell, dict) \
                    or not {"count", "home_wins"} <= set(cell):
                raise ValueError("invalid empirical state-model cells")
        metadata = data.get("metadata") or {}
        if require_accepted:
            required = {
                "train_seasons", "holdout_season", "analytic_holdout",
                "empirical_holdout",
            }
            if not required <= set(metadata):
                raise ValueError("state-model artifact has no holdout acceptance evidence")
            if metadata["holdout_season"] in metadata["train_seasons"]:
                raise ValueError("state-model artifact leaks its holdout season")
            analytic = metadata["analytic_holdout"]
            empirical = metadata["empirical_holdout"]
            try:
                regressed = holdout_regressed(analytic, empirical)
            except (KeyError, TypeError):
                raise ValueError("state-model artifact has invalid holdout metrics") from None
            if regressed:
                raise ValueError("state-model artifact failed its holdout acceptance gate")
        return cls(data.get("cells") or {}, data.get("prior_strength", 30.0),
                   metadata)


def score_model(games: list[tuple[list[tuple[float, GameState]], bool]],
                predict) -> Score:
    rows: list[tuple[float, int]] = []
    for timeline, home_won in games:
        outcome = int(home_won)
        seen: set[tuple] = set()
        for _, gs in timeline:
            signature = state_signature(gs)
            if signature in seen:
                continue
            seen.add(signature)
            rows.append((clamp_probability(predict(gs)), outcome))
    if not rows:
        raise ValueError("no game states to score")
    brier = statistics.mean(brier_loss(p, y) for p, y in rows)
    mean_log_loss = statistics.mean(log_loss(p, y) for p, y in rows)
    calibration = 0.0
    for bucket in range(10):
        selected = [(p, y) for p, y in rows
                    if bucket / 10 <= p < (bucket + 1) / 10
                    or (bucket == 9 and p == 1.0)]
        if selected:
            calibration += len(selected) / len(rows) * abs(
                statistics.mean(p for p, _ in selected)
                - statistics.mean(y for _, y in selected)
            )
    return Score(len(games), len(rows), brier, mean_log_loss, calibration)


def load_probability_model(path: str | None):
    """Return (predictor, label). Null preserves the analytic control."""
    if not path:
        return home_win_probability, "analytic_v1"
    model = EmpiricalStateModel.load(path, require_accepted=True)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    return model.predict, f"empirical_state_v1:{digest}"
