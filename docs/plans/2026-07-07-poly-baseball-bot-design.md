# Polymarket Baseball Trading Bot — Design

Date: 2026-07-07

## Goal

High-frequency, low-margin (5–30% per-trade target) trading on Polymarket MLB
moneyline markets, exploiting the fast win-probability swings baseball produces.
Trade only "playful" games — ones where the market crosses 50% repeatedly —
and measure percentage gains/losses, not absolute dollars.

## Approach chosen

Mean-reversion against overreaction, anchored to a mathematical fair-value model,
with an optional Claude-based judge run as a parallel A/B strategy.

Alternatives considered:

1. **Pure momentum** (ride the move) — rejected: baseball prices overshoot on
   single events (home run, bases loaded escape) then revert; momentum buys tops.
2. **Pure model arbitrage** (trade any model-vs-market gap) — rejected as the
   only signal: the model is approximate, so a standing gap is often model error.
   Requiring *both* a sharp recent move and a model gap in the opposite direction
   filters to genuine overreactions.
3. **AI-only judgment** — rejected as primary: too slow/expensive for
   high-frequency polling. Kept as a *gate* on math signals so AI-vs-no-AI
   performance is directly comparable.

## Architecture

```
Polymarket US sports gateway        MLB Stats API (free, live game state)
        │                                   │
        ▼                                   ▼
  Market registry  ◄── team matching ──►  GameState (inning, outs, bases, score)
        │                                   │
   Market BBO endpoint                winprob.py (fair value model)
        │                                   │
        ▼                                   ▼
  PriceHistory ── volatility.py ──► playfulness filter
        │                                   │
        └────────────► strategy.py ◄────────┘
                       │ signals
        ┌──────────────┴──────────────┐
        ▼                             ▼
   "math" strategy              "ai" strategy (Claude gate)
        │                             │
        └──────► broker (paper/live) ◄┘
                       │
                 journal (SQLite) ──► report.py (A/B comparison)
```

## Key components

- **Fair value model** (`winprob.py`): P(home win) from a normal model of the
  final run differential. Expected diff = current diff + run-expectancy
  adjustment for the batting team's base/out state + home-field edge, spread
  over remaining half-innings; sd = per-half-inning run sd × sqrt(remaining
  half-innings). Ties resolved with ~52% home extra-innings edge.
- **Playfulness filter** (`volatility.py`): count crossings of 0.50 with a
  hysteresis band (must exceed ±band to count a flip) + realized volatility of
  recent price changes. A game is tradeable when flips ≥ N or vol ≥ threshold.
- **Signal** (`strategy.py`): enter when |price move over lookback| ≥ threshold
  AND model fair value disagrees with the move by ≥ min edge. Early innings
  require a larger edge and a more extreme fair value because calibration is
  weaker there. Exit on take profit (default 12%), stop loss (30%), time stop,
  or edge collapse.
- **AI judge** (`ai_judge.py`): Claude (default `claude-opus-4-8`, adaptive
  thinking, low effort, structured JSON output) approves/rejects each math
  signal for the "ai" ledger. Fails closed (reject) on API errors.
- **Brokers** (`broker.py`): `PaperBroker` (default; fills at mid ± slippage,
  per-strategy cash/positions) and `LiveBroker` (requires the official
  `polymarket-us` SDK plus Polymarket US API key id/secret; guarded).
- **Risk** (`risk.py`): max concurrent positions, max stake per market,
  adaptive $5/$10 sizing, spread checks, daily loss kill switch, and longer
  same-market lockout after stop loss — all per strategy.
- **Journal** (`journal.py`): SQLite. Trades, closed round-trips with % return,
  equity snapshots, and every observed BBO tick for future replay. `report.py`
  prints math-vs-AI comparison.

## Error handling

All external calls (Polymarket US gateway/exchange APIs, MLB, Anthropic) are
wrapped; a failed poll skips the tick rather than crashing the loop. AI judge
failure = signal rejected for the AI ledger only. Live orders are never retried
blindly.

## Testing

Unit tests for the pure logic: win-probability model sanity/monotonicity,
flip counting, signal entry/exit rules, paper broker accounting.
