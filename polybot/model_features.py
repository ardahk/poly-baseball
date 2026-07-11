"""Causal game-state features shared by live trading and recorded replay.

The absolute win-probability formula is deliberately not treated as a price.
Instead, its *change* between two observed states is transferred onto a market
price anchor in log-odds space.  This cancels most team-strength calibration
bias while preserving the useful information in score/inning/base changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .models import GameState
from .winprob import home_win_probability


def state_signature(gs: GameState | None) -> tuple | None:
    if gs is None:
        return None
    return (
        gs.inning, gs.is_top, gs.outs, gs.home_score, gs.away_score,
        gs.on_first, gs.on_second, gs.on_third, gs.status,
    )


def _clip_probability(value: float) -> float:
    return min(max(value, 0.001), 0.999)


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def transfer_model_delta(anchor_price: float, anchor_model: float,
                         current_model: float, beta: float = 1.0) -> float:
    """Apply a model log-odds update to a market price anchor."""
    update = beta * (_logit(current_model) - _logit(anchor_model))
    return _clip_probability(_sigmoid(_logit(anchor_price) + update))


@dataclass(frozen=True)
class Anchor:
    price: float
    model: float
    ts: float


@dataclass(frozen=True)
class AnchoredView:
    anchor_price: float
    anchor_model: float
    current_model: float
    fair_home: float
    model_delta: float
    market_delta: float
    residual: float
    anchor_age: float


class ModelHistory:
    """Receipt-time state transitions plus frozen market anchors for one game.

    `observe_state` must run before any price with the same receipt timestamp.
    Consequently a state transition can only use the last price that was known
    *before* the transition; the quote that triggers a decision can never leak
    backwards into its own anchor.
    """

    def __init__(self, probability: Callable[[GameState], float] = home_win_probability,
                 pregame_model_home: float | None = None):
        self.probability = probability
        self.pregame_model_home = _clip_probability(
            probability(GameState(0, status="Live"))
            if pregame_model_home is None else pregame_model_home
        )
        self.last_price: float | None = None
        self.last_price_ts: float | None = None
        self.pregame_anchor: Anchor | None = None
        self.current_model: float | None = None
        self.current_signature: tuple | None = None
        self.current_state_ts: float | None = None
        self.transition_anchor: Anchor | None = None
        self.live_started = False

    def reset_rolling(self) -> None:
        """Discard rolling state after a data gap, keeping the frozen pregame prior.

        The pregame anchor cannot be rebuilt once the game is live, so a
        mid-game outage must not destroy it; only causal transition state is
        stale after a gap.
        """
        self.last_price = None
        self.last_price_ts = None
        self.current_model = None
        self.current_signature = None
        self.current_state_ts = None
        self.transition_anchor = None

    def add_price(self, price: float, ts: float, *, pregame_eligible: bool = True) -> None:
        self.last_price = price
        self.last_price_ts = ts
        if not self.live_started and pregame_eligible:
            # Continuously refresh before first pitch, then freeze forever.
            self.pregame_anchor = Anchor(price, self.pregame_model_home, ts)

    def observe_state(self, gs: GameState, ts: float) -> bool:
        """Record one distinct received state. Return True only on a change."""
        if not gs.is_live:
            return False
        signature = state_signature(gs)
        if signature == self.current_signature:
            return False
        model = _clip_probability(self.probability(gs))
        if self.current_model is not None and self.last_price is not None:
            self.transition_anchor = Anchor(
                self.last_price, self.current_model,
                self.last_price_ts if self.last_price_ts is not None else ts,
            )
        self.current_model = model
        self.current_signature = signature
        self.current_state_ts = ts
        self.live_started = True
        return True

    def state_view(self, current_price: float, now: float,
                   beta: float = 1.0) -> AnchoredView | None:
        if self.transition_anchor is None or self.current_model is None \
                or self.current_state_ts is None:
            return None
        return self._view(self.transition_anchor, current_price, now, beta,
                          age_from=self.current_state_ts)

    def market_view(self, current_price: float, now: float,
                    beta: float = 1.0) -> AnchoredView | None:
        if self.pregame_anchor is None or self.current_model is None:
            return None
        return self._view(self.pregame_anchor, current_price, now, beta,
                          age_from=self.pregame_anchor.ts)

    def _view(self, anchor: Anchor, current_price: float, now: float,
              beta: float, age_from: float) -> AnchoredView:
        fair = transfer_model_delta(
            anchor.price, anchor.model, self.current_model, beta,
        )
        return AnchoredView(
            anchor_price=anchor.price,
            anchor_model=anchor.model,
            current_model=self.current_model,
            fair_home=fair,
            model_delta=self.current_model - anchor.model,
            market_delta=current_price - anchor.price,
            residual=current_price - fair,
            anchor_age=max(0.0, now - age_from),
        )
