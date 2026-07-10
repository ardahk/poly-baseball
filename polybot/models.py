"""Shared data structures."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Market:
    """A Polymarket US MLB moneyline market, matched to an MLB game.

    Polymarket US prices a market in terms of its "long"/"short" sides rather
    than per-team token ids. `long_team` records which team is priced as the
    long side; `home_token`/`away_token` synthesize opaque position-key
    strings ("<slug>:LONG" / "<slug>:SHORT") so the rest of the codebase
    (paper broker, strategy, journal) can keep treating "token" as a plain
    string identifier without knowing about the long/short convention.
    """
    slug: str                # Polymarket US market slug
    question: str
    home_team: str
    away_team: str
    long_team: str            # which team is priced as the long/YES side
    tick_size: float = 0.01
    game_pk: int | None = None   # MLB Stats API game id
    start_time: float | None = None  # scheduled first pitch, epoch seconds
    active: bool = True

    @property
    def key(self) -> str:
        return self.slug

    @property
    def home_is_long(self) -> bool:
        return self.home_team == self.long_team

    @property
    def home_token(self) -> str:
        return f"{self.slug}:{'LONG' if self.home_is_long else 'SHORT'}"

    @property
    def away_token(self) -> str:
        return f"{self.slug}:{'SHORT' if self.home_is_long else 'LONG'}"


@dataclass
class MarketQuote:
    """Best bid/ask normalized to the home-team side for strategy decisions."""
    market_key: str
    home_bid: float
    home_ask: float
    long_bid: float
    long_ask: float
    # Receipt time is the decision-time clock. `source_ts` is optional because
    # the bulk gateway payload does not always expose an exchange timestamp.
    ts: float = field(default_factory=time.time)
    source_ts: float | None = None

    @property
    def home_mid(self) -> float:
        return (self.home_bid + self.home_ask) / 2.0

    @property
    def home_spread(self) -> float:
        return self.home_ask - self.home_bid


@dataclass
class GameState:
    """Live game state from the MLB Stats API."""
    game_pk: int
    inning: int = 1
    is_top: bool = True          # away team bats in the top
    outs: int = 0
    home_score: int = 0
    away_score: int = 0
    on_first: bool = False
    on_second: bool = False
    on_third: bool = False
    status: str = "Scheduled"    # Scheduled | Live | Final
    received_at: float = field(default_factory=time.time, compare=False)

    @property
    def is_live(self) -> bool:
        return self.status == "Live"

    @property
    def is_final(self) -> bool:
        return self.status == "Final"


@dataclass
class Signal:
    market: Market
    token: str               # token to BUY
    side_team: str           # team name we're backing
    price: float             # current market price of that token
    fair: float              # model fair value of that token
    move: float              # recent price move that triggered the fade
    reason: str = ""

    @property
    def edge(self) -> float:
        return self.fair - self.price


@dataclass
class EntryEvaluation:
    """Structured result of running the entry gates for one market tick."""
    outcome: str
    mid: float | None = None
    move: float | None = None
    flips: int = 0
    realized_vol: float = 0.0
    fair_home: float | None = None
    side_team: str = ""
    price: float | None = None
    fair: float | None = None
    edge: float | None = None
    margin: float | None = None
    signal: Signal | None = None


@dataclass
class Position:
    strategy: str
    market_key: str
    token: str
    team: str
    qty: float
    entry_price: float
    entry_fee: float = 0.0
    opened_at: float = field(default_factory=time.time)
    trade_id: str = ""

    @property
    def cost(self) -> float:
        return self.qty * self.entry_price + self.entry_fee

    def pnl_pct(self, price: float) -> float:
        return (price - self.entry_price) / self.entry_price
