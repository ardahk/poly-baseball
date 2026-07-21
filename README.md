# ⚾ polybot — a Polymarket MLB trading laboratory

A single Python process watches live MLB moneyline markets on Polymarket US and
runs **dozens of frozen strategies side-by-side**, each on its own paper ledger,
to find out which market inefficiencies actually pay after fees. It trades
short-horizon overreactions: when a "playful" game's price swings hard and a
win-probability model (inning / outs / bases / score) disagrees, strategies act
on it — some fade the move, some ride it, some just harvest a calibration bias
and hold to settlement.

Runs continuously on an Oracle Cloud VM; the leaderboard below refreshes daily.

## 🏆 Live leaderboard — top performers

> Paper-trading track record, updated automatically once a day. Ranked by
> **overall account return**; percentages only.

<!-- STATS:START -->
_Updated 2026-07-21 04:34 UTC · paper trading · percentages only · top 5 of qualifying strategies (≥10 closed trades)._

| | Strategy | Trades | Win % | Avg / Trade | Best Trade | Overall Return |
|:--:|---|--:|--:|--:|--:|--:|
| 🥇 | `news_late_v2` | 47 | 53% | +26.0% | +733% | **+66.9%** |
| 🥈 | `settle_gap10_v2` | 40 | 38% | +19.4% | +567% | **+57.8%** |
| 🥉 | `settle_away_v2` | 36 | 36% | -0.8% | +545% | **+28.0%** |
| ④ | `settle_gap05_early_v2` | 38 | 42% | +9.7% | +506% | **+13.1%** |
| ⑤ | `cell_extras_home_v2` | 11 | 64% | +18.8% | +98% | **+7.4%** |

**What each one does**

- 🥇 **`news_late_v2`** — Buys the lag after late-inning, high-leverage events — the slowest to price in.
- 🥈 **`settle_gap10_v2`** — Holds a model-vs-market gap to settlement (one fee leg) — the preregistered rule, live.
- 🥉 **`settle_away_v2`** — Holds away-side gaps to settlement — the model overrates home teams.
- ④ **`settle_gap05_early_v2`** — Holds early-inning model-market gaps to settlement.
- ⑤ **`cell_extras_home_v2`** — Buys the home last-at-bat advantage in extra innings.
<!-- STATS:END -->

<sub>**Avg / Trade** = mean P&L per closed round trip · **Best Trade** = single
best round trip · **Overall Return** = paper account vs. its starting bankroll.
Small samples are noisy — strategies need a minimum number of closed trades to
appear.</sub>

## How it trades

Every strategy is a **frozen variant** competing on one shared market/game
stream with its own paper ledger, so their track records are directly
comparable. The original fade **controls** are:

- **fade_v1_frozen / fade_tight** — original absolute-model fade controls.
- **liquidity_fade_v2** — the unchanged-book shock control with volatility and
  regime crossings measured on fixed receipt-time windows.
- **state_residual_v1** — fades a short-lived market overreaction relative to
  the latest distinct MLB state change.
- **market_anchor_v1** — freezes the pregame market probability, then transfers
  model changes onto it in log-odds space so team-strength bias largely cancels.
- **ai_shadow** — asynchronous, optional judge over the frozen fade control.

On top of these runs a **hypothesis fleet** of ~25 genuinely different
mechanisms grouped by kind — `momentum` (trade *with* a move), `event_reaction`
(buy the market's lag after a game event), `extreme_hold` /
`calibration_cell` (harvest a favorite-longshot or home/leader bias, held to
settlement), `settlement_hold` (a model-vs-market gap held to one fee leg), and
`microstructure` (book-shape and timing signals). Selection happens only through
a preregistered walk-forward gate — the leaderboard is a track record, not a
promotion. Each mechanism's one-line summary shows up next to it in the
leaderboard when it ranks.

## Setup

```bash
cd poly-baseball
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your keys
```

Keys (all optional to start):

| Key | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | the "ai" strategy (bot runs math-only without it) |
| `POLYMARKET_KEY_ID` | Reserved for the later live-execution phase |
| `POLYMARKET_SECRET_KEY` | Reserved for the later live-execution phase |

## Usage

```bash
python main.py scan          # what MLB markets exist right now + model fair values
python main.py run           # paper trade (default, no keys needed)
python main.py run --dashboard # paper trade with a live terminal dashboard
python main.py report        # frozen-strategy performance comparison
python main.py review        # end-of-day trade/funnel/near-miss review
python main.py run --live    # intentionally disabled during Phase 0

python main.py backtest calibrate --days 7   # is the win-prob formula accurate?
python main.py backtest strategy  --days 1   # would the trading logic have profited?
python main.py backtest causal --date 2026-07-08
python main.py research diagnose --date 2026-07-08

# Offline Phase 3 model research; writes only if the untouched holdout improves.
python main.py research fit-state --seasons 2022 2023 2024 --holdout 2025 \
  --output artifacts/state-model-v1.json

# Phase 4 is deliberately two-step. Prepare hashes the config and immutable
# tape without computing results; evaluate verifies that lock before reveal.
python main.py walk-forward prepare --start 2026-05-01 --folds 4 \
  --hypothesis "state-residual fades have positive after-cost expectancy" \
  --output artifacts/walk-forward-prereg.json
python main.py walk-forward evaluate \
  --manifest artifacts/walk-forward-prereg.json \
  --output artifacts/walk-forward-result.json
```

Run it during live MLB games (evenings US time); outside game hours there is
nothing to trade. Trades, equity snapshots, every observed best-bid/best-ask
tick, one-sided mark prices, game-state changes, market metadata, and every
entry decision/rejection are logged to `polybot.db` (SQLite). `report` prints
win rate, average % per trade, best/worst, and account return per strategy;
`review` explains the day with linked round trips, MFE/MAE, exit breakdowns,
the persisted entry funnel, near misses, and threshold hints.
Use `run --dashboard` during a game for a live terminal view of tracked games,
fresh BBO counts, strategy equity, market state, open positions, and recent
engine events.

Phase 0 intentionally disables live orders. Paper results must clear the
execution-realism and reconciliation promotion gates before live trading is
reintroduced.

Paper-account cash, positions, and realized P&L persist in `polybot.db` across
restarts. The dashboard labels its session P&L separately from the persisted
ledger, and the daily-loss limit uses the configured trading-day boundary
(`engine.report_timezone`, default `America/Los_Angeles`). Use
`review --timezone UTC` only when you deliberately want UTC calendar days.

Runtime paper fills use the executable BBO side (buy at ask, sell at bid) and
the configured Polymarket US taker-fee coefficient. One-sided marks are recorded
for research but cannot trigger an entry or exit. New events are linked to an
immutable run ID plus configuration hash in the journal.
Phase 3 also records one `model_observations` row per distinct received game
state, rather than treating thousands of correlated price ticks as independent
calibration samples.

The first command that opens an older `polybot.db` migrates it in place. The
migration is additive, but copy the DB aside first if it contains data you care
about:

```bash
cp polybot.db "polybot.db.backup-$(date +%Y%m%d)"
python main.py status
```

## Tuning

All knobs are in `config.yaml` with comments — entry thresholds, the 5–30%
take-profit band, playfulness definition, stake sizing, and risk limits
(max concurrent positions, per-market cap, and daily-loss kill switch).
`home_fair_shrink` corrects the win-prob formula's home-team bias
(`backtest calibrate` showed homes overpredicted by ~5-9 points mid-range).

If the bot isn't trading, look at the **entry funnel** — a counter of why
each candidate tick was rejected (`not_playful`, `small_move`, `no_edge`,
`early_game`, `stale_quote`, `wide_spread`, ...). It's on the dashboard and
in the periodic status log line. The same gate outcomes are persisted in the
`decisions` table, so `python main.py review --date YYYY-MM-DD` can diagnose a
losing or quiet day after the process exits.

## Backtesting

Two ways to validate the math on **real finished games** before risking money:

- **`backtest calibrate`** — the direct test of the formula. Walks play-by-play
  for finished games, computes the model's P(home win) at every game state, and
  scores it against who actually won. Reports Brier score and log loss vs two
  baselines (bet-the-leader, and a constant), plus a calibration table
  (predicted probability vs actual home-win rate per bucket). Needs only the
  free MLB API, so it works over many days. If the formula's Brier beats the
  baselines and the calibration buckets line up, the formula is decent.
- **`backtest strategy`** — replays minute-level prices for finished games
  through the exact `check_entry` / `check_exit` code the live bot runs, on a
  paper broker, and reports simulated P&L. It tries Polymarket US 1-minute
  trade-stat candles first; when those are unavailable (they need exchange data
  access, and the US gateway drops finished games), it falls back to
  Polymarket.com history (gamma + CLOB minute prices) for the same games as a
  proxy. Treat any strategy P&L as a directional sanity check on thresholds,
  not a promise. Your real strategy track record accumulates in `report` as
  you paper-trade.
- **`backtest causal`** (and the legacy `replay` alias) — merges every recorded
  quote and MLB state onto one receipt-time clock across all games. Decisions
  can fill only on a later executable BBO after configured latency; fees,
  overlapping cash/position limits, run boundaries, data gaps, cooldowns, and
  official settlement are enforced portfolio-wide. Async AI shadows are skipped
  because external model calls are not deterministic historical evidence.

## Phase 3 model research

The state-residual and market-anchored families use causal model deltas. The
state-residual anchor is the last price at least
`residual_anchor_lookback_secs` before a changed MLB state was received — the
market reacts to a play seconds before the polled feed reports it, so the very
last pre-receipt price usually already contains the event's move and would
double-count it. The market anchor is the last valid pregame price and is
frozen when the first live state arrives. A quote observed with or after a
state update can never become that update's prior anchor.

`research fit-state` builds a beta-smoothed empirical state table from declared
training seasons and scores it against both outcomes and the analytic control
on one untouched holdout season. It refuses to write an artifact if either
Brier score or log loss regresses. Set `engine.state_model_path` only after an
artifact is accepted; leaving it null preserves the analytic control.

`research diagnose` reports executable post-signal markouts, BBO coverage, and
residuals by frozen strategy and horizon. It uses the future bid and both taker
fees—not midpoint—to avoid manufacturing paper edge. Parameter selection and
automatic strategy promotion use the preregistered Phase 4 walk-forward gate.

## Phase 4 walk-forward testing

`walk-forward prepare` creates weekly chronological folds with 28 training
days, 7 validation days, and a following locked 7-day test. It stores the
hypothesis, deterministic validation selection rule, promotion thresholds,
config hash, code revision, timezone, exact boundaries, and a hash of every
market/game event in scope. Existing manifests are never overwritten.

`walk-forward evaluate` refuses to run if the manifest, config, or event tape
changed, and each preregistration can be evaluated exactly once: a completed
reveal is recorded in the journal, so re-running with a different output path
is refused rather than silently re-revealing the locked test. Within each fold
it selects the strategy with the best validation *realized* P&L (closed round
trips and settlements, net of fees — unrealized marks at a stale bid cannot
crown a champion) before evaluating the locked test, and records how many
candidates competed so selection multiplicity is visible in the evidence. The
write-once result includes after-cost P&L, return on peak deployed capital,
turnover, drawdown, daily expected shortfall, fill/fee rates, executable-bid
adverse selection at 5, 15, 30, and 60 seconds, game-clustered calibration
losses and confidence intervals, day/game P&L confidence intervals, and
concentration by day, game, inning, entry-price bucket, and spread bucket.
Promotion additionally requires day- and game-concentration limits and — by
default — that every fold selected the same champion, since only one strategy
can go live. Train windows exist to establish liveness; they skip the
execution-quality extras (adverse selection, calibration) that only matter for
validation and the locked test.

## How it works

- **Market discovery**: Polymarket US sports gateway (`/v2/leagues/mlb/events`),
  matched to MLB Stats API games by team name and scheduled start time.
- **Prices**: Polymarket US market BBO midpoints polled every ~2s per market.
  Every two-sided BBO tick is recorded with `two_sided=1, source='bbo'`; when a
  live book is one-sided, the mark/last price is recorded with
  `two_sided=0, source='mark'` so replay sees the same price history the live
  strategy saw. During the configured pregame window, BBOs are recorded so the
  market-anchored strategy can freeze the last valid pre-first-pitch prior;
  trading features themselves begin only once MLB reports the game as live.
- **Fair value**: analytic or accepted empirical state probability. Phase 3
  strategies consume its state-to-state change, transferred onto a causal
  market anchor, rather than pretending the absolute formula knows team strength.
- **Playfulness**: only trades games whose price has crossed 50% repeatedly
  (with hysteresis) or shows high realized volatility — the "crossing lines"
  pattern, not one team cruising.
- **Entry**: |move over 90s| ≥ 8¢ AND model disagrees by ≥ 5¢ in the opposite
  direction AND the spread is not too wide → buy the undervalued side. Through
  inning 5, entries need a stronger edge and a more extreme fair value.
- **Exit**: take profit / stop loss / 15-min time stop / model edge gone /
  game final. Stop-loss exits create a longer same-market lockout.
- **Sizing**: $5 base stake; $10 only for high-edge signals with tight spread.

Design doc: `docs/plans/2026-07-07-poly-baseball-bot-design.md`

## Tests

```bash
python -m pytest
```

## Disclaimers

Paper mode is the default for a reason: the model is an approximation, fills
in live mode depend on current book liquidity, and prediction-market trading may
not be lawful where you are. Start tiny.
