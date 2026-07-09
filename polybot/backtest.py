"""Backtesting for the mathematical models.

Two modes, both using *real* finished MLB games:

1. calibrate — does the win-probability FORMULA predict actual outcomes?
   Walks play-by-play for finished games, computes the model's P(home win) at
   every game state, and scores it against who actually won (Brier score, log
   loss, calibration table). Needs only the MLB Stats API. This is the direct
   answer to "is the formula decent" — no Polymarket data involved.

2. strategy — would the TRADING logic have made money? Attempts to replay
   Polymarket US 1-minute trade-stat candles for finished games against the
   reconstructed game-state timeline, through the exact same check_entry /
   check_exit code the live bot uses, on a paper broker. Needs Polymarket US +
   MLB.

Limitation: Polymarket US historical candles are documented in more than one
shape and may require exchange data access, and the US gateway drops finished
games from its events feed. When US candles are unavailable, strategy replay
falls back to Polymarket.com (gamma + CLOB) minute-level price history for the
same games as a proxy; `calibrate` (below) needs no exchange data and is the
reliable way to validate the formula. Even when prices are available,
~1-minute resolution understates the live bot's edge (2s polling) — treat
strategy P&L as directional, not precise.
"""
from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from . import mlb, pmus, strategy
from .broker import PaperBroker
from .config import Config
from .models import GameState, Market
from .volatility import PriceHistory
from .winprob import home_win_probability

log = logging.getLogger(__name__)


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


# ------------------------------------------- Polymarket.com history fallback
#
# Polymarket US does not return historical candles without exchange data
# access, and its gateway drops finished games from the events feed. The same
# games trade on Polymarket.com, whose gamma/CLOB APIs expose minute-level
# price history publicly, so strategy replay uses those prices as a proxy.

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

# MLB Stats API full team name -> abbreviation used in Polymarket.com slugs
# (mlb-{away}-{home}-{YYYY-MM-DD}).
TEAM_ABBREV = {
    "Arizona Diamondbacks": "ari",
    "Atlanta Braves": "atl",
    "Baltimore Orioles": "bal",
    "Boston Red Sox": "bos",
    "Chicago Cubs": "chc",
    "Chicago White Sox": "cws",
    "Cincinnati Reds": "cin",
    "Cleveland Guardians": "cle",
    "Colorado Rockies": "col",
    "Detroit Tigers": "det",
    "Houston Astros": "hou",
    "Kansas City Royals": "kc",
    "Los Angeles Angels": "laa",
    "Los Angeles Dodgers": "lad",
    "Miami Marlins": "mia",
    "Milwaukee Brewers": "mil",
    "Minnesota Twins": "min",
    "New York Mets": "nym",
    "New York Yankees": "nyy",
    "Athletics": "ath",
    "Oakland Athletics": "oak",
    "Philadelphia Phillies": "phi",
    "Pittsburgh Pirates": "pit",
    "San Diego Padres": "sd",
    "San Francisco Giants": "sf",
    "Seattle Mariners": "sea",
    "St. Louis Cardinals": "stl",
    "Tampa Bay Rays": "tb",
    "Texas Rangers": "tex",
    "Toronto Blue Jays": "tor",
    "Washington Nationals": "wsh",
}

_REQUEST_PACING_SECS = 0.3


def _gamma_slug(away_team: str, home_team: str, day: str) -> str | None:
    """Polymarket.com event slug for a game, or None for unmapped team names."""
    away = TEAM_ABBREV.get(away_team)
    home = TEAM_ABBREV.get(home_team)
    if not away or not home:
        return None
    return f"mlb-{away}-{home}-{day}"


def _gamma_home_token(event: dict, home_team: str) -> str | None:
    """CLOB token id of the home team in the event's moneyline market.

    The moneyline is the market whose question equals the event title;
    clobTokenIds and outcomes are parallel JSON-encoded arrays.
    """
    title = event.get("title") or ""
    for m in event.get("markets") or []:
        if m.get("question") != title:
            continue
        try:
            tokens = json.loads(m.get("clobTokenIds") or "[]")
            outcomes = json.loads(m.get("outcomes") or "[]")
        except (TypeError, ValueError):
            return None
        for token, outcome in zip(tokens, outcomes):
            if mlb._team_match(str(outcome), home_team):
                return str(token)
    return None


def _slug_dates(game_date: float | None) -> list[str]:
    """Candidate slug dates for a game. Slugs use the US calendar date, so try
    the Eastern-time date first, then the UTC date if it differs (late games)."""
    if game_date is None:
        return []
    eastern = datetime.fromtimestamp(game_date, tz=ZoneInfo("America/New_York"))
    utc = datetime.fromtimestamp(game_date, tz=timezone.utc)
    days = [eastern.date().isoformat()]
    if utc.date().isoformat() not in days:
        days.append(utc.date().isoformat())
    return days


def _fetch_gamma_home_prices(
    session: requests.Session,
    away_team: str,
    home_team: str,
    game_date: float | None,
    start_ts: float,
    end_ts: float,
) -> tuple[str | None, list[tuple[float, float]]]:
    """(slug, [(ts, home_price)]) from Polymarket.com minute history, ([], None
    slug) when the event/market/history isn't available. Paces requests."""
    for day in _slug_dates(game_date):
        slug = _gamma_slug(away_team, home_team, day)
        if not slug:
            return None, []
        time.sleep(_REQUEST_PACING_SECS)
        try:
            resp = session.get(f"{GAMMA_URL}/events", params={"slug": slug}, timeout=15)
            resp.raise_for_status()
            events = resp.json()
        except Exception as exc:
            log.debug("gamma event fetch failed for %s: %s", slug, exc)
            continue
        if not isinstance(events, list) or not events:
            continue
        token = _gamma_home_token(events[0], home_team)
        if not token:
            log.debug("no moneyline market found in gamma event %s", slug)
            continue
        time.sleep(_REQUEST_PACING_SECS)
        try:
            resp = session.get(
                f"{CLOB_URL}/prices-history",
                params={"market": token, "startTs": int(start_ts),
                        "endTs": int(end_ts), "fidelity": 1},
                timeout=20,
            )
            resp.raise_for_status()
            history = resp.json().get("history") or []
        except Exception as exc:
            log.debug("clob price history failed for %s: %s", slug, exc)
            continue
        rows = [(float(h["t"]), float(h["p"])) for h in history
                if isinstance(h, dict) and h.get("t") is not None and h.get("p") is not None]
        if rows:
            return slug, rows
    return None, []


# ----------------------------------------------------------- strategy replay

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
                 timeline: list[tuple[float, GameState]], home_won: bool | None,
                 broker: PaperBroker) -> None:
    """Run one game's price+state stream through the live strategy logic."""
    scfg = cfg.strategy
    history = PriceHistory(scfg.flip_band)
    key = market.key
    cooldown_until = -1e18
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
                fh = strategy.fair_home_value(gs, scfg)
                fair = fh if pos.token == market.home_token else 1.0 - fh
            reason = strategy.check_exit(pos, price, fair, game_final, scfg, now=ts)
            if reason:
                broker.close("backtest", pos.token, price)
                cooldown = scfg.stop_loss_cooldown_secs \
                    if reason.startswith("stop loss") else scfg.cooldown_secs
                cooldown_until = ts + cooldown

        if game_final:
            break

        # --- entries ---
        if ts < cooldown_until:
            continue
        sig = strategy.check_entry(market, history, gs, scfg)
        if sig is None or sig.token in broker.positions["backtest"]:
            continue
        stake = cfg.risk.stake_usd
        if len(broker.open_positions("backtest")) >= cfg.risk.max_positions:
            continue
        if broker.stake_in_market("backtest", key) + stake > cfg.risk.max_stake_per_market:
            continue
        pos = broker.open("backtest", key, sig.token, sig.side_team,
                          sig.price, stake)
        if pos:
            pos.opened_at = ts       # simulated clock, not wall-clock
            cooldown_until = ts + scfg.cooldown_secs

    # Force-close anything still open at the final whistle at the settled price.
    for pos in list(broker.open_positions("backtest")):
        if pos.market_key != key:
            continue
        if home_won is None:
            settle = prices[-1][1] if prices else pos.entry_price
            price = settle if pos.token == market.home_token else 1.0 - settle
        else:
            token_won = (
                (pos.token == market.home_token and home_won)
                or (pos.token == market.away_token and not home_won)
            )
            price = 1.0 if token_won else 0.0
        broker.settle("backtest", pos.token, price)


def strategy_backtest(cfg: Config, days_back: int = 2, max_games: int = 30) -> None:
    client = mlb.MLBClient()
    end = date.today()
    start = end - timedelta(days=days_back)
    finished = [g for g in client.schedule(start.isoformat(), end.isoformat())
                if g["status"] == "Final"]

    markets = pmus.fetch_mlb_markets(include_closed=True, limit=400)
    mlb.match_markets_to_games(markets, finished)
    us_by_pk = {m.game_pk: m for m in markets if m.game_pk}
    games = finished[:max_games]
    print(f"Replaying strategy on {len(games)} finished games "
          f"(Polymarket US candles, falling back to Polymarket.com history)...\n")
    if not games:
        print("No finished games in the window. Try a larger --days window.")
        return

    session = requests.Session()
    broker = PaperBroker(["backtest"], cfg.risk.starting_cash, cfg.engine.slippage)
    games_with_trades = 0
    games_with_candles = 0
    for game in games:
        timeline, home_won = build_state_timeline(client, game["game_pk"])
        if not timeline:
            continue
        start_ts = timeline[0][0] - 3600
        end_ts = timeline[-1][0] + 3600

        # Preferred source: Polymarket US 1-minute candles for a matched market.
        market = us_by_pk.get(game["game_pk"])
        prices: list[tuple[float, float]] = []
        if market:
            bars = max(1, min(1440, int((end_ts - start_ts) // 60) + 1))
            long_prices = pmus.fetch_long_price_history(
                market.slug, start_ts, end_ts, bars=bars)
            prices = [
                (ts, price if market.home_is_long else 1.0 - price)
                for ts, price in long_prices
            ]

        # Fallback: Polymarket.com gamma/CLOB minute history as a proxy.
        if len(prices) < 5:
            slug, prices = _fetch_gamma_home_prices(
                session, game["away"], game["home"], game.get("game_date"),
                start_ts, end_ts)
            if len(prices) < 5:
                log.debug("no usable price history for %s @ %s",
                          game["away"], game["home"])
                continue
            market = Market(
                slug=slug,
                question=f"{game['away']} @ {game['home']}",
                home_team=game["home"],
                away_team=game["away"],
                long_team=game["home"],
                game_pk=game["game_pk"],
            )
        games_with_candles += 1
        before = broker.realized["backtest"]
        n_before = broker.closes["backtest"]
        _replay_game(cfg, market, prices, timeline, home_won, broker)
        if broker.closes["backtest"] > n_before:
            games_with_trades += 1
            pnl = broker.realized["backtest"] - before
            print(f"  {market.question:<46} trades P&L ${pnl:+.2f}")

    if not games_with_candles:
        print("No usable price history was returned for the finished games — "
              "neither Polymarket US candles nor Polymarket.com gamma/CLOB\n"
              "minute history. That can be an access/API-history limitation, "
              "not necessarily a strategy result. `backtest calibrate` still\n"
              "validates the win-probability formula without exchange data.")
        return

    realized = broker.realized["backtest"]
    equity = broker.equity("backtest", {})
    trades = broker.closes["backtest"]
    print("\n" + "=" * 60)
    print("STRATEGY BACKTEST RESULTS")
    print("=" * 60)
    print(f"games with candles: {games_with_candles}")
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
