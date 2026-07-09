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
| `POLYMARKET_KEY_ID` | `run --live` only |
| `POLYMARKET_SECRET_KEY` | `run --live` only; paste without surrounding quotes |

## Usage

```bash
python main.py scan          # what MLB markets exist right now + model fair values
python main.py run           # paper trade (default, no keys needed)
python main.py run --dashboard # paper trade with a live terminal dashboard
python main.py report        # math-vs-AI performance comparison
python main.py run --live    # real orders — requires pip install polymarket-us + keys
python main.py run --live --yes-live  # real orders without prompt for systemd

python main.py backtest calibrate --days 7   # is the win-prob formula accurate?
python main.py backtest strategy  --days 1   # would the trading logic have profited?
```

Run it during live MLB games (evenings US time); outside game hours there is
nothing to trade. Trades, equity snapshots, and every observed best-bid /
best-ask price tick are logged to `polybot.db` (SQLite); `report` prints win
rate, average % per trade, best/worst, and account return per strategy.
Use `run --dashboard` during a game for a live terminal view of tracked games,
fresh BBO counts, strategy equity, market state, open positions, and recent
engine events.

For unattended live mode, either use `--yes-live` in the service command or set
`POLYBOT_CONFIRM_LIVE=yes` in `.env`. Manual `--live` runs still prompt by
default.

## Tuning

All knobs are in `config.yaml` with comments — entry thresholds, the 5–30%
take-profit band, playfulness definition, stake sizing, and risk limits
(max concurrent positions, per-market cap, and daily-loss kill switch).
`home_fair_shrink` corrects the win-prob formula's home-team bias
(`backtest calibrate` showed homes overpredicted by ~5-9 points mid-range).

If the bot isn't trading, look at the **entry funnel** — a counter of why
each candidate tick was rejected (`not_playful`, `small_move`, `no_edge`,
`early_game`, `stale_quote`, `wide_spread`, ...). It's on the dashboard and
in the periodic status log line, so "no trades" is always diagnosable.

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

## How it works

- **Market discovery**: Polymarket US sports gateway (`/v2/leagues/mlb/events`),
  matched to MLB Stats API games by team name and scheduled start time.
- **Prices**: Polymarket US market BBO midpoints polled every ~2s per market.
  Every BBO tick is recorded so future backtests can replay the bot's own
  observed market history. To avoid noisy API usage, BBO polling only starts
  once MLB reports a game as live; game-state checks begin shortly before first
  pitch.
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
