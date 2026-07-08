"""Shared data structures."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Market:
    """A Polymarket MLB moneyline market, matched to an MLB game."""
    condition_id: str
    question: str
    home_team: str
    away_team: str
    home_token: str          # CLOB token id for the home-team outcome
    away_token: str
    game_pk: int | None = None   # MLB Stats API game id
    start_time: float | None = None  # scheduled first pitch, epoch seconds
    active: bool = True

    @property
    def key(self) -> str:
        return self.condition_id


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
class Position:
    strategy: str
    market_key: str
    token: str
    team: str
    qty: float
    entry_price: float
    opened_at: float = field(default_factory=time.time)

    @property
    def cost(self) -> float:
        return self.qty * self.entry_price

    def pnl_pct(self, price: float) -> float:
        return (price - self.entry_price) / self.entry_price
