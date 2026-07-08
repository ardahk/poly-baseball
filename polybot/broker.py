"""Order execution: paper broker (default) and live CLOB broker (guarded)."""
from __future__ import annotations

import logging
import os

from .models import Position

log = logging.getLogger(__name__)


class PaperBroker:
    """Simulated fills at the observed price +/- slippage.

    Keeps an independent cash balance and position book per strategy so the
    math and AI ledgers can be compared cleanly.
    """

    def __init__(self, strategies: list[str], starting_cash: float, slippage: float = 0.005):
        self.slippage = slippage
        self.cash: dict[str, float] = {s: starting_cash for s in strategies}
        self.positions: dict[str, dict[str, Position]] = {s: {} for s in strategies}
        self.realized: dict[str, float] = {s: 0.0 for s in strategies}
        self.closes: dict[str, int] = {s: 0 for s in strategies}

    def open(self, strategy: str, market_key: str, token: str, team: str,
             price: float, stake_usd: float) -> Position | None:
        fill = min(price + self.slippage, 0.999)
        qty = stake_usd / fill
        cost = qty * fill
        if cost > self.cash[strategy]:
            log.info("[%s] insufficient cash for %s", strategy, team)
            return None
        if token in self.positions[strategy]:
            return None  # already holding this token
        self.cash[strategy] -= cost
        pos = Position(strategy=strategy, market_key=market_key, token=token,
                       team=team, qty=qty, entry_price=fill)
        self.positions[strategy][token] = pos
        return pos

    def close(self, strategy: str, token: str, price: float) -> tuple[Position, float, float] | None:
        """Returns (position, fill_price, pnl_usd)."""
        pos = self.positions[strategy].pop(token, None)
        if pos is None:
            return None
        fill = max(price - self.slippage, 0.001)
        proceeds = pos.qty * fill
        self.cash[strategy] += proceeds
        pnl = proceeds - pos.cost
        self.realized[strategy] += pnl
        self.closes[strategy] += 1
        return pos, fill, pnl

    def equity(self, strategy: str, prices: dict[str, float]) -> float:
        total = self.cash[strategy]
        for token, pos in self.positions[strategy].items():
            total += pos.qty * prices.get(token, pos.entry_price)
        return total

    def open_positions(self, strategy: str) -> list[Position]:
        return list(self.positions[strategy].values())

    def stake_in_market(self, strategy: str, market_key: str) -> float:
        return sum(p.cost for p in self.positions[strategy].values()
                   if p.market_key == market_key)


class LiveBroker(PaperBroker):
    """Real orders on the Polymarket CLOB. Requires py-clob-client and keys.

    Extends PaperBroker for the local book-keeping; each open/close also
    submits a real marketable limit order. Fills are assumed at our limit
    (fill-or-kill semantics would be the next hardening step).
    """

    def __init__(self, strategies: list[str], starting_cash: float, slippage: float = 0.01):
        super().__init__(strategies, starting_cash, slippage)
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL
        except ImportError as exc:
            raise RuntimeError(
                "Live trading requires py-clob-client: pip install py-clob-client"
            ) from exc
        key = os.environ.get("POLYMARKET_PRIVATE_KEY")
        funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS")
        if not key:
            raise RuntimeError(
                "Set POLYMARKET_PRIVATE_KEY (and optionally POLYMARKET_FUNDER_ADDRESS) "
                "in .env for live trading."
            )
        self._OrderArgs, self._OrderType = OrderArgs, OrderType
        self._BUY, self._SELL = BUY, SELL
        self.client = ClobClient(
            "https://clob.polymarket.com", key=key, chain_id=137,
            signature_type=1 if funder else 0, funder=funder,
        )
        self.client.set_api_creds(self.client.create_or_derive_api_creds())
        log.info("LiveBroker initialised")

    def _submit(self, token: str, side, price: float, qty: float) -> bool:
        try:
            order = self.client.create_order(self._OrderArgs(
                token_id=token, price=round(price, 3), size=round(qty, 2), side=side,
            ))
            resp = self.client.post_order(order, self._OrderType.GTC)
            log.info("live order: %s", resp)
            return bool(resp.get("success"))
        except Exception as exc:
            log.error("live order failed: %s", exc)
            return False

    def open(self, strategy, market_key, token, team, price, stake_usd):
        limit = min(price + self.slippage, 0.999)
        if not self._submit(token, self._BUY, limit, stake_usd / limit):
            return None
        return super().open(strategy, market_key, token, team, price, stake_usd)

    def close(self, strategy, token, price):
        pos = self.positions[strategy].get(token)
        if pos is None:
            return None
        limit = max(price - self.slippage, 0.001)
        if not self._submit(token, self._SELL, limit, pos.qty):
            return None
        return super().close(strategy, token, price)
