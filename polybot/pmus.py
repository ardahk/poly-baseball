"""Polymarket US: market discovery, price feed, and historical candles.

Discovery and BBO use PUBLIC endpoints (no API key needed) on the gateway host.
Historical trade stats are best-effort because Polymarket US documents more
than one exchange report shape and access requirements may vary. Order
authentication is handled separately in broker.py via the official
`polymarket-us` SDK.

Reference: https://docs.polymarket.us (public gateway at gateway.polymarket.us).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from .models import Market, MarketQuote

log = logging.getLogger(__name__)

GATEWAY_URL = "https://gateway.polymarket.us"
EXCHANGE_API_URL = "https://api.prod.polymarketexchange.com"

# A game event carries many markets (spreads, totals, first-five, moneyline).
# This is the one we trade: full-game winner, i.e. the moneyline.
MONEYLINE_TYPE = "baseball_team_full_game_winner"


def _parse_iso(raw) -> float | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


@dataclass
class BookQuote:
    """Long-side book snapshot embedded in the bulk events response."""
    long_bid: float | None
    long_ask: float | None
    long_last: float | None   # last/mark price fallback when the book is one-sided
    received_at: float = field(default_factory=time.time)
    source_ts: float | None = None

    @property
    def two_sided(self) -> bool:
        return self.long_bid is not None and self.long_ask is not None \
            and self.long_bid <= self.long_ask


def fetch_mlb_book(
    session: requests.Session | None = None,
    limit: int = 100,
    include_closed: bool = False,
) -> tuple[list[Market], dict[str, BookQuote]]:
    """Fetch MLB moneyline markets AND their current quotes in ONE request.

    The events payload embeds bestBidQuote/bestAskQuote per market, so the
    engine can refresh every market's BBO with a single gateway call per poll
    tick instead of one /bbo request per market (a big rate-limit win).
    """
    sess = session or requests.Session()
    markets: list[Market] = []
    quotes: dict[str, BookQuote] = {}
    try:
        resp = sess.get(
            f"{GATEWAY_URL}/v2/leagues/mlb/events", params={"limit": limit}, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        received_at = time.time()
    except Exception as exc:
        log.warning("Polymarket US discovery failed: %s", exc)
        return markets, quotes

    for event in data.get("events", []):
        for m in event.get("markets", []):
            if m.get("sportsMarketType") != MONEYLINE_TYPE:
                continue
            market = _parse_market(m, include_closed=include_closed)
            if market:
                markets.append(market)
                quotes[market.slug] = _parse_book_quote(m, received_at=received_at)
    log.debug("Polymarket US: %d MLB moneyline markets discovered", len(markets))
    return markets, quotes


def fetch_mlb_markets(
    session: requests.Session | None = None,
    limit: int = 100,
    include_closed: bool = False,
) -> list[Market]:
    """Fetch today's/upcoming MLB moneyline markets."""
    markets, _ = fetch_mlb_book(session, limit=limit, include_closed=include_closed)
    log.info("Polymarket US: %d MLB moneyline markets discovered", len(markets))
    return markets


def _quote_value(raw) -> float | None:
    if not isinstance(raw, dict):
        return None
    return _price(raw.get("value"))


def _parse_book_quote(m: dict, received_at: float | None = None) -> BookQuote:
    long_last = None
    for side in m.get("marketSides") or []:
        if side.get("long"):
            long_last = _price(side.get("price"))
    return BookQuote(
        long_bid=_quote_value(m.get("bestBidQuote")),
        long_ask=_quote_value(m.get("bestAskQuote")),
        long_last=long_last,
        received_at=time.time() if received_at is None else received_at,
        source_ts=_parse_iso(m.get("transactTime") or m.get("updatedAt")),
    )


def _parse_market(m: dict, include_closed: bool = False) -> Market | None:
    if not include_closed and (m.get("closed") or not m.get("active", True)):
        return None
    sides = m.get("marketSides") or []
    if len(sides) != 2:
        return None

    by_order: dict[str, str] = {}
    long_team = None
    for s in sides:
        team = s.get("team") or {}
        ordering = team.get("ordering")   # "home" | "away"
        name = team.get("name")
        if not ordering or not name:
            return None
        by_order[ordering] = name
        if s.get("long"):
            long_team = name
    if "home" not in by_order or "away" not in by_order or long_team is None:
        return None

    slug = m.get("slug")
    if not slug:
        return None
    return Market(
        slug=slug,
        question=m.get("question") or "",
        home_team=by_order["home"],
        away_team=by_order["away"],
        long_team=long_team,
        tick_size=float(m.get("orderPriceMinTickSize") or 0.01),
        start_time=_parse_iso(m.get("gameStartTime") or m.get("startDate")),
        fee_coefficient=_fee_coefficient(m.get("feeCoefficient")),
    )


def _fee_coefficient(raw) -> float | None:
    """Venue taker fee coefficient, ignoring absent or nonsensical values."""
    try:
        theta = float(raw)
    except (TypeError, ValueError):
        return None
    # Published schedule is 0 (fee-free categories) through 0.07 (crypto).
    # Anything outside that is a payload change we should not silently trust.
    return theta if 0.0 <= theta <= 0.5 else None


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _price(raw) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    # The data guide's report examples use exchange-scaled prices such as
    # "550" for $0.55. Gateway/BBO responses already use decimal dollars.
    if value > 1.0:
        value /= 1000.0
    if 0.0 <= value <= 1.0:
        return value
    return None


def _bar_close(bar: dict) -> float | None:
    for key in ("last", "close", "c"):
        px = _price(bar.get(key))
        if px is not None:
            return px
    return None


def _parse_trade_stats(data: dict) -> list[tuple[float, float]]:
    bars = data.get("bars") or data.get("stats") or []
    if not isinstance(bars, list):
        return []

    starts = data.get("barStartTime") or data.get("bar_start_time") or []
    result: list[tuple[float, float]] = []
    for i, bar in enumerate(bars):
        if not isinstance(bar, dict):
            continue
        raw_ts = (
            bar.get("interval_start")
            or bar.get("startTime")
            or bar.get("start_time")
            or (starts[i] if i < len(starts) else None)
        )
        ts = _parse_iso(raw_ts)
        px = _bar_close(bar)
        if ts is not None and px is not None:
            result.append((ts, px))
    return result


def fetch_long_price_history(
    slug: str,
    start_ts: float,
    end_ts: float,
    session: requests.Session | None = None,
    bars: int = 390,
) -> list[tuple[float, float]]:
    """Fetch historical LONG-side closes for a market, if the exchange exposes them.

    Polymarket US documentation currently shows two report shapes in different
    sections (`/v1/report/trades/stats` and `/v1beta1/report/trades/stats`).
    Try both and return an empty list on auth/shape failures so strategy
    backtests can degrade to a clear message instead of crashing.
    """
    sess = session or requests.Session()
    attempts = [
        (
            f"{EXCHANGE_API_URL}/v1/report/trades/stats",
            {"symbol": slug, "startTime": _iso(start_ts), "endTime": _iso(end_ts), "bars": bars},
        ),
        (
            f"{EXCHANGE_API_URL}/v1beta1/report/trades/stats",
            {
                "symbol": slug,
                "start_time": _iso(start_ts),
                "end_time": _iso(end_ts),
                "interval": "1m",
            },
        ),
    ]
    for url, payload in attempts:
        try:
            resp = sess.post(url, json=payload, timeout=20)
            resp.raise_for_status()
            rows = _parse_trade_stats(resp.json())
            if rows:
                return rows
        except Exception as exc:
            log.debug("trade stats failed for %s via %s: %s", slug, url, exc)
    return []


class PriceFeed:
    """Best-bid/best-offer midpoint per market, public endpoint (no auth)."""

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def home_midpoint(self, market: Market) -> float | None:
        """Midpoint price of the HOME team's win side."""
        quote = self.quote(market)
        if quote is None:
            return None
        return quote.home_mid

    def quote(self, market: Market) -> MarketQuote | None:
        """Best bid/best ask normalized to home-team probability."""
        long_bbo = self._long_bbo(market.slug)
        if long_bbo is None:
            return None
        long_bid, long_ask = long_bbo
        if market.home_is_long:
            home_bid, home_ask = long_bid, long_ask
        else:
            home_bid, home_ask = 1.0 - long_ask, 1.0 - long_bid
        return MarketQuote(
            market_key=market.key,
            home_bid=home_bid,
            home_ask=home_ask,
            long_bid=long_bid,
            long_ask=long_ask,
            ts=time.time(),
        )

    def _long_bbo(self, slug: str) -> tuple[float, float] | None:
        try:
            resp = self.session.get(f"{GATEWAY_URL}/v1/markets/{slug}/bbo", timeout=10)
            resp.raise_for_status()
            data = resp.json().get("marketData", {})
            bid = (data.get("bestBid") or {}).get("value")
            ask = (data.get("bestAsk") or {}).get("value")
            if bid is None or ask is None:
                return None
            long_bid, long_ask = float(bid), float(ask)
            if long_bid > long_ask:
                return None
            return long_bid, long_ask
        except Exception as exc:
            log.debug("bbo fetch failed for %s: %s", slug, exc)
            return None
