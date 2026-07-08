# polybot — Polymarket MLB mean-reversion bot

Trades Polymarket baseball moneyline markets on short-horizon overreactions:
when a "playful" game's price swings hard and a win-probability model
(inning / outs / bases / score) says the move overshot, the bot buys the
undervalued side, targeting small gains (default +12% take profit, -10% stop).

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
| `POLYMARKET_PRIVATE_KEY` | `run --live` only |
| `POLYMARKET_FUNDER_ADDRESS` | `run --live` with a Polymarket proxy wallet |

## Usage

```bash
python main.py scan          # what MLB markets exist right now + model fair values
python main.py run           # paper trade (default, no keys needed)
python main.py report        # math-vs-AI performance comparison
python main.py run --live    # real orders — requires pip install py-clob-client + keys

python main.py backtest calibrate --days 7   # is the win-prob formula accurate?
python main.py backtest strategy  --days 1   # would the trading logic have profited?
```

Run it during live MLB games (evenings US time); outside game hours there is
nothing to trade. Everything is logged to `polybot.db` (SQLite); `report`
prints win rate, average % per trade, best/worst, and account return per
strategy.

## Tuning

All knobs are in `config.yaml` with comments — entry thresholds, the 5–30%
take-profit band, playfulness definition, stake sizing, and risk limits
(max concurrent positions, per-market cap, daily-loss kill switch).

## Backtesting

Two ways to validate the math on **real finished games** before risking money:

- **`backtest calibrate`** — the direct test of the formula. Walks play-by-play
  for finished games, computes the model's P(home win) at every game state, and
  scores it against who actually won. Reports Brier score and log loss vs two
  baselines (bet-the-leader, and a constant), plus a calibration table
  (predicted probability vs actual home-win rate per bucket). Needs only the
  free MLB API, so it works over many days. If the formula's Brier beats the
  baselines and the calibration buckets line up, the formula is decent.
- **`backtest strategy`** — replays real Polymarket 1-minute price history for
  finished games through the exact `check_entry`/`check_exit` code the live bot
  runs, on a paper broker, and reports simulated P&L. Limited to recently-final
  games (Polymarket stops exposing the team-outcome market once a game fully
  settles), and 1-min prices understate the 2-second live edge — treat it as a
  directional sanity check on your thresholds, not a promise. Your real
  strategy track record accumulates in `report` as you paper-trade.

## How it works

- **Market discovery**: Gamma API (`tag_slug=mlb`), matched to MLB Stats API
  games by team name.
- **Prices**: CLOB REST midpoints polled every ~2s per market.
- **Fair value**: normal model of the final run differential using a
  run-expectancy (RE24) adjustment for the current base/out state.
- **Playfulness**: only trades games whose price has crossed 50% repeatedly
  (with hysteresis) or shows high realized volatility — the "crossing lines"
  pattern, not one team cruising.
- **Entry**: |move over 90s| ≥ 8¢ AND model disagrees by ≥ 5¢ in the opposite
  direction → buy the undervalued side.
- **Exit**: take profit / stop loss / 15-min time stop / model edge gone /
  game final.

Design doc: `docs/plans/2026-07-07-poly-baseball-bot-design.md`

## Tests

```bash
python -m pytest
```

## Disclaimers

Paper mode is the default for a reason: the model is an approximation, fills
in live mode are optimistic (GTC limit at ask, no fill confirmation loop yet),
and prediction-market trading may not be lawful where you are. Start tiny.
