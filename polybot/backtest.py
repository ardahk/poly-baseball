"""Backtesting for the mathematical models.

Two modes, both using *real* finished MLB games:

1. calibrate — does the win-probability FORMULA predict actual outcomes?
   Walks play-by-play for finished games, computes the model's P(home win) at
   every game state, and scores it against who actually won (Brier score, log
   loss, calibration table). Needs only the MLB Stats API. This is the direct
   answer to "is the formula decent" — no Polymarket data involved.

2. strategy — would the TRADING logic have made money? Replays real Polymarket
   price history for finished games against the reconstructed game-state
   timeline, through the exact same check_entry/check_exit code the live bot
   uses, on a paper broker. Needs Gamma + CLOB price history + MLB.

Limitation: CLOB price history is ~1-minute resolution, so intraminute moves the
live bot (2s polling) would catch are invisible here. Treat strategy P&L as a
conservative, directional estimate, not a precise forecast.
"""
from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import requests

from . import gamma, mlb, strategy
from .broker import PaperBroker
from .config import Config
from .models import GameState, Market
from .volatility import PriceHistory
from .winprob import home_win_probability

log = logging.getLogger(__name__)

CLOB_HISTORY = "https://clob.polymarket.com/prices-history"


# ----------------------------------------------------------------- timelines

def _ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def build_state_timeline(client: mlb.MLBClient, game_pk: int):
    """Return (timeline, home_won) where timeline is a time-ordered list of
    (ts, GameState) sampled after each completed play, and home_won is a bool
    (None if the game isn't final / can't be determined)."""
    try:
        resp = client.session.get(
            f"{mlb.MLB_LIVE_URL}/game/{game_pk}/feed/live", timeout=20
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.debug("feed fetch failed for %s: %s", game_pk, exc)
        return [], None

    status = data.get("gameData", {}).get("status", {}).get("abstractGameState")
    if status != "Final":
        return [], None
    plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    if not plays:
        return [], None

    timeline: list[tuple[float, GameState]] = []
    bases = {"1B": False, "2B": False, "3B": False}
    cur_half: tuple[int, str] | None = None

    for p in plays:
        about = p.get("about", {})
        inning = int(about.get("inning") or 1)
        half = about.get("halfInning", "top")
        if (inning, half) != cur_half:
            bases = {"1B": False, "2B": False, "3B": False}
            cur_half = (inning, half)
        # Apply runner movements to carry base occupancy forward.
        for r in p.get("runners", []):
            mv = r.get("movement", {})
            start, end = mv.get("start"), mv.get("end")
            if start in bases:
                bases[start] = False
            if end in bases:
                bases[end] = True
        outs = int(p.get("count", {}).get("outs") or 0)
        res = p.get("result", {})
        home = int(res.get("homeScore") or 0)
        away = int(res.get("awayScore") or 0)
        ts = _ts(about.get("endTime") or about.get("startTime"))
        if ts is None or outs >= 3:      # skip inning-ending boundary states
            continue
        timeline.append((ts, GameState(
            game_pk=game_pk, inning=inning, is_top=(half == "top"), outs=outs,
            home_score=home, away_score=away,
            on_first=bases["1B"], on_second=bases["2B"], on_third=bases["3B"],
            status="Live",
        )))

    final = plays[-1].get("result", {})
    fh, fa = final.get("homeScore"), final.get("awayScore")
    if fh is None or fa is None or fh == fa:
        return timeline, None
    return timeline, fh > fa


# --------------------------------------------------------------- calibration

@dataclass
class _CalibResult:
    predicted: float
    outcome: int  # 1 home won, 0 away won


def _brier(rows: list[_CalibResult]) -> float:
    return statistics.mean((r.predicted - r.outcome) ** 2 for r in rows)


def _logloss(rows: list[_CalibResult]) -> float:
    total = 0.0
    for r in rows:
        p = min(max(r.predicted, 1e-6), 1 - 1e-6)
        total += -(r.outcome * math.log(p) + (1 - r.outcome) * math.log(1 - p))
    return total / len(rows)


def _naive_leader(gs: GameState) -> float:
    """Baseline: bet the current leader at fixed confidence."""
    diff = gs.home_score - gs.away_score
    if diff > 0:
        return 0.75
    if diff < 0:
        return 0.25
    return 0.54


def calibrate(days_back: int = 3, max_games: int = 60) -> None:
    client = mlb.MLBClient()
    end = date.today()
    start = end - timedelta(days=days_back)
    games = [g for g in client.schedule(start.isoformat(), end.isoformat())
             if g["status"] == "Final"]
    games = games[:max_games]
    print(f"Calibrating win-prob model on {len(games)} finished games "
          f"({start} to {end})...\n")

    model_rows: list[_CalibResult] = []
    naive_rows: list[_CalibResult] = []
    base_rows: list[_CalibResult] = []
    used = 0
    for g in games:
        timeline, home_won = build_state_timeline(client, g["game_pk"])
        if home_won is None or not timeline:
            continue
        used += 1
        outcome = 1 if home_won else 0
        for _, gs in timeline:
            model_rows.append(_CalibResult(home_win_probability(gs), outcome))
            naive_rows.append(_CalibResult(_naive_leader(gs), outcome))
            base_rows.append(_CalibResult(0.54, outcome))

    if not model_rows:
        print("No usable games found. Try a larger --days window.")
        return

    print(f"Scored {len(model_rows)} game states across {used} games.\n")
    print(f"{'model':<22}{'Brier':>10}{'log loss':>12}   (lower is better)")
    print("-" * 46)
    for name, rows in (("win-prob formula", model_rows),
                       ("naive leader", naive_rows),
                       ("constant 0.54", base_rows)):
        print(f"{name:<22}{_brier(rows):>10.4f}{_logloss(rows):>12.4f}")

    # Calibration table: bucket predictions, show predicted vs actual frequency.
    print("\nCalibration of the win-prob formula (predicted -> actual home-win rate):")
    print(f"{'bucket':<14}{'n':>7}{'avg pred':>10}{'actual':>9}")
    print("-" * 40)
    for lo in [i / 10 for i in range(10)]:
        hi = lo + 0.1
        bucket = [r for r in model_rows if lo <= r.predicted < hi
                  or (hi == 1.0 and r.predicted == 1.0)]
        if not bucket:
            continue
        avg_pred = statistics.mean(r.predicted for r in bucket)
        actual = statistics.mean(r.outcome for r in bucket)
        print(f"{lo:.1f}-{hi:.1f}      {len(bucket):>7}{avg_pred:>10.3f}{actual:>9.3f}")
    print("\nA well-calibrated model has 'avg pred' ~ 'actual' in every row, and a")
    print("lower Brier/log loss than the naive baselines above.")


# ----------------------------------------------------------- strategy replay

def _fetch_price_history(session: requests.Session, token: str) -> list[tuple[float, float]]:
    try:
        resp = session.get(CLOB_HISTORY,
                           params={"market": token, "interval": "max", "fidelity": 1},
                           timeout=20)
        resp.raise_for_status()
        return [(float(pt["t"]), float(pt["p"])) for pt in resp.json().get("history", [])]
    except Exception as exc:
        log.debug("price history failed for %s: %s", token, exc)
        return []


def _state_at(timeline: list[tuple[float, GameState]], ts: float) -> GameState | None:
    """Most recent game state at or before ts (linear scan; timelines are small)."""
    current = None
    for state_ts, gs in timeline:
        if state_ts <= ts:
            current = gs
        else:
            break
    return current


def _replay_game(cfg: Config, market: Market, prices: list[tuple[float, float]],
                 timeline: list[tuple[float, GameState]], broker: PaperBroker) -> None:
    """Run one game's price+state stream through the live strategy logic."""
    scfg = cfg.strategy
    history = PriceHistory(scfg.flip_band)
    key = market.key
    last_trade_ts = -1e18
    if not timeline:
        return
    first_state_ts = timeline[0][0]
    final_state_ts = timeline[-1][0]

    for ts, home_price in prices:
        if ts < first_state_ts:
            continue  # pre-game
        history.add(home_price, ts)
        gs = _state_at(timeline, ts)
        game_final = ts > final_state_ts

        # --- exits first ---
        for pos in list(broker.open_positions("backtest")):
            if pos.market_key != key:
                continue
            price = home_price if pos.token == market.home_token else 1.0 - home_price
            fair = None
            if gs:
                fh = home_win_probability(gs)
                fair = fh if pos.token == market.home_token else 1.0 - fh
            reason = strategy.check_exit(pos, price, fair, game_final, scfg, now=ts)
            if reason:
                broker.close("backtest", pos.token, price)
                last_trade_ts = ts

        if game_final:
            break

        # --- entries ---
        if ts - last_trade_ts < scfg.cooldown_secs:
            continue
        sig = strategy.check_entry(market, history, gs, scfg)
        if sig is None or sig.token in broker.positions["backtest"]:
            continue
        if len(broker.open_positions("backtest")) >= cfg.risk.max_positions:
            continue
        pos = broker.open("backtest", key, sig.token, sig.side_team,
                          sig.price, cfg.risk.stake_usd)
        if pos:
            pos.opened_at = ts       # simulated clock, not wall-clock
            last_trade_ts = ts

    # Force-close anything still open at the final whistle at the settled price.
    for pos in list(broker.open_positions("backtest")):
        if pos.market_key != key:
            continue
        home_won = timeline and home_win_probability(timeline[-1][1])
        settle = prices[-1][1] if prices else pos.entry_price
        price = settle if pos.token == market.home_token else 1.0 - settle
        broker.close("backtest", pos.token, price)


def strategy_backtest(cfg: Config, days_back: int = 2, max_games: int = 30) -> None:
    client = mlb.MLBClient()
    session = requests.Session()
    end = date.today()
    start = end - timedelta(days=days_back)
    finished = [g for g in client.schedule(start.isoformat(), end.isoformat())
                if g["status"] == "Final"]

    # Gamma only exposes the team-outcome moneyline market while the event is
    # still open (recently-final games included); it restructures the event
    # once the game fully settles. So the open feed is what has usable tokens.
    markets = gamma.fetch_mlb_markets(session, include_closed=False, limit=400)
    mlb.match_markets_to_games(markets, finished)
    tradeable = [m for m in markets if m.game_pk][:max_games]
    print(f"Replaying strategy on {len(tradeable)} finished games with live "
          f"markets (mostly today's slate)...\n")
    if not tradeable:
        print("No finished games currently have an open moneyline market to "
              "replay. Run this shortly after games end, or rely on `calibrate`\n"
              "(which validates the formula across many days) plus paper-mode\n"
              "trading to accumulate real strategy results in the journal.")
        return

    broker = PaperBroker(["backtest"], cfg.risk.starting_cash, cfg.engine.slippage)
    games_with_trades = 0
    for market in tradeable:
        prices = _fetch_price_history(session, market.home_token)
        if len(prices) < 5:
            continue
        timeline, _ = build_state_timeline(client, market.game_pk)
        if not timeline:
            continue
        before = broker.realized["backtest"]
        n_before = broker.closes["backtest"]
        _replay_game(cfg, market, prices, timeline, broker)
        if broker.closes["backtest"] > n_before:
            games_with_trades += 1
            pnl = broker.realized["backtest"] - before
            print(f"  {market.question:<46} trades P&L ${pnl:+.2f}")

    realized = broker.realized["backtest"]
    equity = broker.equity("backtest", {})
    trades = broker.closes["backtest"]
    print("\n" + "=" * 60)
    print("STRATEGY BACKTEST RESULTS")
    print("=" * 60)
    print(f"games with trades : {games_with_trades}")
    print(f"round-trip trades : {trades}")
    print(f"realized P&L      : ${realized:+.2f}")
    print(f"account           : ${cfg.risk.starting_cash:.2f} -> ${equity:.2f} "
          f"({100 * (equity - cfg.risk.starting_cash) / cfg.risk.starting_cash:+.1f}%)")
    if trades:
        print(f"avg P&L per trade : ${realized / trades:+.2f} "
              f"({100 * realized / (trades * cfg.risk.stake_usd):+.1f}% of stake)")
    print("\nNote: ~1-min price resolution understates the live bot's edge (or")
    print("its whipsaw). Use this to sanity-check thresholds, not as a promise.")
