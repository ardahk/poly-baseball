"""Live game state from the free MLB Stats API + matching to Polymarket markets."""
from __future__ import annotations

import logging
from datetime import date, datetime

import requests

from .models import GameState, Market

log = logging.getLogger(__name__)

MLB_URL = "https://statsapi.mlb.com/api/v1"
MLB_LIVE_URL = "https://statsapi.mlb.com/api/v1.1"


class MLBClient:
    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def schedule(self, start_date: str, end_date: str | None = None) -> list[dict]:
        """[{game_pk, home, away, status, game_date}] over a date range (ISO dates)."""
        params = {"sportId": 1, "startDate": start_date,
                  "endDate": end_date or start_date}
        try:
            resp = self.session.get(f"{MLB_URL}/schedule", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("MLB schedule fetch failed: %s", exc)
            return []
        games = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                games.append({
                    "game_pk": g["gamePk"],
                    "home": g["teams"]["home"]["team"]["name"],
                    "away": g["teams"]["away"]["team"]["name"],
                    "status": g.get("status", {}).get("abstractGameState", ""),
                    "game_date": _parse_game_date(g.get("gameDate")),
                })
        return games

    def todays_games(self) -> list[dict]:
        """[{game_pk, home, away, status, game_date}] for today's schedule."""
        return self.schedule(date.today().isoformat())

    def game_state(self, game_pk: int) -> GameState | None:
        try:
            resp = self.session.get(
                f"{MLB_LIVE_URL}/game/{game_pk}/feed/live", timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.debug("MLB live feed failed for %s: %s", game_pk, exc)
            return None
        return _parse_live_feed(game_pk, data)


def _parse_live_feed(game_pk: int, data: dict) -> GameState:
    live = data.get("liveData", {})
    line = live.get("linescore", {})
    status = data.get("gameData", {}).get("status", {}).get("abstractGameState", "Scheduled")
    offense = line.get("offense", {})
    return GameState(
        game_pk=game_pk,
        inning=int(line.get("currentInning") or 1),
        is_top=line.get("isTopInning", True),
        outs=int(line.get("outs") or 0),
        home_score=int(line.get("teams", {}).get("home", {}).get("runs") or 0),
        away_score=int(line.get("teams", {}).get("away", {}).get("runs") or 0),
        on_first="first" in offense,
        on_second="second" in offense,
        on_third="third" in offense,
        status=status,
    )


def _parse_game_date(raw) -> float | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


# A market and a game must start within this window to be the same game —
# Polymarket also lists tomorrow's matchup for the same two teams.
_MATCH_WINDOW_SECS = 6 * 3600


def match_markets_to_games(markets: list[Market], games: list[dict]) -> None:
    """Attach MLB game_pk to each market by team names + start time (in place)."""
    for market in markets:
        if market.game_pk:
            continue
        for game in games:
            if not (_team_match(market.home_team, game["home"])
                    and _team_match(market.away_team, game["away"])):
                continue
            if (market.start_time is not None and game.get("game_date") is not None
                    and abs(market.start_time - game["game_date"]) > _MATCH_WINDOW_SECS):
                continue
            market.game_pk = game["game_pk"]
            break
        if not market.game_pk:
            log.debug("no MLB game matched for market %r", market.question)


def _team_match(a: str, b: str) -> bool:
    al, bl = a.lower().strip(), b.lower().strip()
    if al == bl or al in bl or bl in al:
        return True
    # nickname match: last word ("Mariners" vs "Seattle Mariners")
    return al.split()[-1] == bl.split()[-1]
