# polybot — Polymarket MLB mean-reversion bot

Trades Polymarket US baseball moneyline markets on short-horizon overreactions:
when a "playful" game's price swings hard and a win-probability model
(inning / outs / bases / score) says the move overshot, the bot buys the
undervalued side, targeting small gains (default +12% take profit, -30% stop).

Two strategies run side-by-side with separate ledgers so you can compare them:

- **math** — pure formula: playfulness filter + sharp move + model edge
- **ai** — the same signals, but each one must also be approved by a Claude
  judge (`claude-opus-4-8`) before it trades

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
python main.py report        # math-vs-AI performance comparison
python main.py review        # end-of-day trade/funnel/near-miss review
python main.py run --live    # intentionally disabled during Phase 0

python main.py backtest calibrate --days 7   # is the win-prob formula accurate?
python main.py backtest strategy  --days 1   # would the trading logic have profited?
python main.py backtest replay --date 2026-07-08 --set strategy.stop_loss=0.15
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
- **`backtest replay`** — replays the bot's own recorded SQLite day using the
  observed tick stream, including one-sided mark prices for move detection and
  a fresh two-sided-BBO gate for entries. Repeat `--set section.key=value` to
  sweep thresholds without editing `config.yaml`, for example
  `python main.py backtest replay --date 2026-07-08 --set strategy.stop_loss=0.15`.

## How it works

- **Market discovery**: Polymarket US sports gateway (`/v2/leagues/mlb/events`),
  matched to MLB Stats API games by team name and scheduled start time.
- **Prices**: Polymarket US market BBO midpoints polled every ~2s per market.
  Every two-sided BBO tick is recorded with `two_sided=1, source='bbo'`; when a
  live book is one-sided, the mark/last price is recorded with
  `two_sided=0, source='mark'` so replay sees the same price history the live
  strategy saw. To avoid noisy API usage, price polling only starts once MLB
  reports a game as live; game-state checks begin shortly before first pitch.
- **Fair value**: normal model of the final run differential using a
  run-expectancy (RE24) adjustment for the current base/out state.
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
