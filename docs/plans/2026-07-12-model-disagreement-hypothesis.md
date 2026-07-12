# Preregistration: model-disagreement hold-to-settlement edge

**Date locked:** 2026-07-12
**Status:** hypothesis — NOT tradeable until out-of-sample criteria below are met.

## Background

The intraday fade family is proven negative-edge (negative counterfactual markout
at every horizon; market beats the model 67% of the time at trade points; win rate
20–35% vs ~60% breakeven). A fitted empirical state model
(`artifacts/state_v1.json`, train 2022–24 / holdout 2025: Brier 0.1549 vs analytic
0.1579, ECE 0.040→0.016) still loses to the live market overall (Brier 0.1739 vs
0.1716) — it is roughly a coin flip, so the ~5¢ round-trip cost floor is the binding
constraint, not forecast quality.

A hold-to-settlement EV probe (`scripts/model_markout.py`) on the initial data
suggested positive cells, consistent across Brier and settlement-EV tests:

| inning | gap    | tick-n | avg pnl/contract |
|--------|--------|--------|------------------|
| 1–3    | ≥.10   | 6973   | **+0.0697**      |
| 4–6    | ≥.10   | 4746   | **+0.0414**      |
| 4–6    | .05–.10| 6014   | +0.0191          |
| 7+     | ≥.10   | 863    | −0.2083 (excluded) |
| ALL    |        | 32175  | −0.0003          |

## Known weakness (why this is only a hypothesis)

Only **16 distinct games** of data. The tick-n counts are pseudo-replication of a
handful of independent game outcomes; the overall EV is flat. The cell pattern has a
plausible mechanism (market under-weights game state early; dominates late) but is
statistically meaningless at this sample size.

## The locked rule (do not edit after 2026-07-12)

- Predictor: `artifacts/state_v1.json` (frozen; sha in artifact).
- Entry: when a **live** game has **inning ≤ 6** AND
  **|model P(home) − market mid| ≥ 0.10**, buy the side the model favors at the
  market's executable ask.
- Exclude innings 7+ entirely (proven catastrophic).
- Hold to settlement (exit only at game final).
- Fixed stake per bet.

## Success criteria (pre-committed, out-of-sample)

Evaluate on **≥ 60 NEW games** whose `game_pk` is NOT among the initial 16.
The rule passes only if, **clustered by game** (each game = one observation, not
per-tick):

1. mean pnl/contract **> +0.02** (clears entry cost + margin), AND
2. bootstrap 95% CI lower bound **> 0**.

If not met → reject; pivot to the cost floor (maker orders) or data-platform mode.
No changing the artifact, thresholds, inning/gap cells, or criteria after this date.

## Method

Keep the pipeline collecting (already running). At ~60+ new games, snapshot
`polybot.db` (copy main file only) and re-run `model_markout.py`, reporting
distinct games per cell and per-game-clustered means. Only then decide whether to
implement it as a frozen paper strategy for a further forward test.
