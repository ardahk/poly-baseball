"""Market discovery via the Polymarket Gamma API."""
from __future__ import annotations

import json
import logging
from datetime import datetime

import requests

from .models import Market

log = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"


def _parse_json_field(value):
    """Gamma returns some list fields as JSON-encoded strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return []
    return value or []


def fetch_mlb_markets(session: requests.Session | None = None) -> list[Market]:
    """Fetch open MLB game (moneyline) markets.

    Each event should hold one binary market whose two outcomes are the teams.
    """
    sess = session or requests.Session()
    markets: list[Market] = []
    try:
        resp = sess.get(
            f"{GAMMA_URL}/events",
            params={"tag_slug": "mlb", "closed": "false", "limit": 200},
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as exc:
        log.warning("Gamma discovery failed: %s", exc)
        return markets

    for event in events:
        for m in event.get("markets", []):
            market = _parse_market(m, event)
            if market:
                markets.append(market)
    log.info("Gamma: %d MLB markets discovered", len(markets))
    return markets


def _parse_market(m: dict, event: dict) -> Market | None:
    outcomes = _parse_json_field(m.get("outcomes"))
    token_ids = _parse_json_field(m.get("clobTokenIds"))
    if len(outcomes) != 2 or len(token_ids) != 2:
        return None
    if m.get("closed") or not m.get("active", True):
        return None
    # Skip prop markets (over/under, "will X hit a home run", etc.):
    # a moneyline market's outcomes are team names, not Yes/No.
    if {o.strip().lower() for o in outcomes} == {"yes", "no"}:
        return None

    question = m.get("question") or event.get("title") or ""
    # Spread/total markets share team-name outcomes with the moneyline;
    # only the plain "Away vs. Home" moneyline is tradeable by our model.
    ql = question.lower()
    if any(word in ql for word in ("spread", "total", "run line", "(-", "(+")):
        return None
    # Convention: "Away vs. Home" / "Away @ Home"; outcomes usually
    # [away, home] but we match by name below, so order only matters as
    # a fallback.
    away_name, home_name = _teams_from_title(event.get("title") or question, outcomes)
    try:
        home_idx = outcomes.index(home_name)
    except ValueError:
        home_idx = 1
    away_idx = 1 - home_idx
    return Market(
        condition_id=m.get("conditionId") or m.get("id") or question,
        question=question,
        home_team=outcomes[home_idx],
        away_team=outcomes[away_idx],
        home_token=str(token_ids[home_idx]),
        away_token=str(token_ids[away_idx]),
        start_time=_parse_start_time(m.get("gameStartTime")),
    )


def _parse_start_time(raw) -> float | None:
    if not raw:
        return None
    try:
        s = str(raw).replace("Z", "+00:00")
        # Gamma sometimes uses a bare "+00" offset, which fromisoformat rejects
        if s.endswith("+00"):
            s += ":00"
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _teams_from_title(title: str, outcomes: list[str]) -> tuple[str, str]:
    """Return (away, home) outcome names. In 'A vs. B' / 'A @ B' titles the
    home team is listed second."""
    lowered = title.lower()
    for sep in (" vs. ", " vs ", " @ ", " at "):
        if sep in lowered:
            idx = lowered.index(sep)
            first = title[:idx].strip()
            second = title[idx + len(sep):].strip()
            away = _match_outcome(first, outcomes)
            home = _match_outcome(second, outcomes)
            if away and home and away != home:
                return away, home
            break
    return outcomes[0], outcomes[1]


def _match_outcome(fragment: str, outcomes: list[str]) -> str | None:
    frag = fragment.lower()
    for o in outcomes:
        ol = o.lower()
        if ol in frag or frag in ol:
            return o
        # match on last word (nickname), e.g. "Seattle Mariners" vs "Mariners"
        if ol.split()[-1] in frag.split() or frag.split()[-1] in ol.split():
            return o
    return None
