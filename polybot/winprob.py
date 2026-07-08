"""Mathematical win-probability model for a live MLB game.

Model: the final run differential (home - away) is approximately normal.
  mean  = current diff + batting-team run-expectancy adjustment
          + home-field edge spread over remaining half-innings
  sd    = per-half-inning run sd * sqrt(remaining half-innings)

P(home win) = P(diff > 0), with the tie band split ~52/48 to the home team
(extra innings, home bats last). This is a fair-value anchor for detecting
market overreaction — not a sportsbook-grade model, and it doesn't need to be:
signals also require a sharp opposing price move.
"""
from __future__ import annotations

import math

from .models import GameState

# RE24 run-expectancy matrix: expected runs scored in the remainder of the
# current half-inning. Rows = outs, columns indexed by base state bitmask
# (1B=1, 2B=2, 3B=4). Standard published values.
_RE24 = [
    # empty  1B     2B     1B2B   3B     1B3B   2B3B   loaded
    [0.481, 0.859, 1.100, 1.437, 1.350, 1.784, 1.964, 2.292],  # 0 outs
    [0.254, 0.509, 0.664, 0.884, 0.950, 1.130, 1.376, 1.541],  # 1 out
    [0.098, 0.224, 0.319, 0.429, 0.353, 0.478, 0.580, 0.752],  # 2 outs
]
_RE_INNING_START = _RE24[0][0]      # fresh half-inning baseline
_RUNS_SD_PER_HALF = 1.06            # sd of runs in one half-inning
_HOME_EDGE_RUNS = 0.12              # home-field advantage over a full game
_TIE_HOME_WIN = 0.52                # home win prob in extra innings

REGULATION_INNINGS = 9


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _base_state(gs: GameState) -> int:
    return (1 if gs.on_first else 0) | (2 if gs.on_second else 0) | (4 if gs.on_third else 0)


def _remaining_halves(gs: GameState) -> tuple[float, float]:
    """(away_remaining, home_remaining) half-innings of offense, counting the
    current half-inning as a fraction based on outs/bases."""
    innings_after = max(0, REGULATION_INNINGS - gs.inning)
    away = float(innings_after)
    home = float(innings_after)
    if gs.is_top:
        away += 1.0 - gs.outs / 3.0
        home += 1.0
    else:
        home += 1.0 - gs.outs / 3.0
    return away, home


def home_win_probability(gs: GameState) -> float:
    """P(home team wins) given the current game state."""
    if gs.is_final:
        return 1.0 if gs.home_score > gs.away_score else 0.0

    diff = float(gs.home_score - gs.away_score)

    # Walk-off: home leads in the bottom of the 9th or later -> game over.
    if gs.inning >= REGULATION_INNINGS and not gs.is_top and diff > 0:
        return 1.0

    away_rem, home_rem = _remaining_halves(gs)

    # Expected additional runs for the batting team in the current half-inning,
    # relative to a fresh half-inning (which is already priced into *_rem).
    outs = min(max(gs.outs, 0), 2)
    re_adj = _RE24[outs][_base_state(gs)] - _RE_INNING_START
    if gs.is_top:
        diff -= re_adj
    else:
        diff += re_adj

    total_rem = away_rem + home_rem
    diff += _HOME_EDGE_RUNS * (total_rem / (2.0 * REGULATION_INNINGS))

    sd = _RUNS_SD_PER_HALF * math.sqrt(max(total_rem, 0.25))

    p_home_ahead = 1.0 - _phi((0.5 - diff) / sd)
    p_tie = _phi((0.5 - diff) / sd) - _phi((-0.5 - diff) / sd)
    p = p_home_ahead + _TIE_HOME_WIN * p_tie
    return min(max(p, 0.001), 0.999)
