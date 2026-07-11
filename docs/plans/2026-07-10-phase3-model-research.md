# Phase 3: Causal Model Research

Date: 2026-07-10
Status: implemented

## Objective

Test whether baseball state changes contain tradeable information without
treating the analytic win-probability level as a team-strength-aware price.
Every candidate remains paper-only and competes on the Phase 2 causal simulator.

## Frozen experiments

- `fade_v1_frozen` and `fade_tight` remain the controls.
- `liquidity_fade_v2` isolates unchanged-book shocks with fixed-time-grid
  volatility and trailing-window crossings, avoiding polling-rate artifacts.
- `state_residual_v1` anchors to the last price known before a distinct received
  MLB state and trades only inside a short response window.
- `market_anchor_v1` freezes the last pregame market midpoint and transfers the
  model's cumulative log-odds change onto that prior.
- `ai_shadow` remains asynchronous and is excluded from deterministic replay.

For an anchor price `p0`, anchor model probability `m0`, current model
probability `m1`, and frozen sensitivity `beta`:

```text
fair = sigmoid(logit(p0) + beta * (logit(m1) - logit(m0)))
residual = current_market_mid - fair
```

The executable edge must survive bid/ask spread and estimated round-trip taker
fees. Delayed replay orders are canceled when the game state changes before the
next eligible BBO.

## Empirical artifact gate

`research fit-state` fits a smoothed state-cell correction on explicitly named
training seasons. It evaluates one untouched holdout season and writes the JSON
artifact only when both Brier score and log loss do not regress versus the
analytic control. The runtime stays on `analytic_v1` unless
`engine.state_model_path` names an accepted artifact.

## Diagnostics

- `model_observations`: one row per distinct received game state, never one row
  per price tick.
- `decisions` and `signals`: anchor, model-delta, residual, and anchor-age fields.
- `research diagnose`: executable future-bid markouts net of both taker fees,
  plus two-sided-book coverage.
- `PriceHistory.realized_vol_time` and `flips_within`: receipt-time windows for
  sampling-stable research without changing the frozen legacy control.

## Explicitly out of scope

- Walk-forward selection, confidence intervals, champion promotion, and live
  enablement are Phase 4 or later.
- Pitcher, bullpen, lineup, and park inputs are not admitted without a separate
  untouched-holdout improvement.
- Passive-maker or queue simulation is not claimed because the tape has BBO but
  no depth or queue-position evidence.
