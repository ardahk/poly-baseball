"""Order execution: paper broker (default) and live Polymarket US broker (guarded)."""
from __future__ import annotations

import logging
import os
import uuid

from .models import Position

log = logging.getLogger(__name__)


def parse_token(token: str) -> tuple[str, str]:
    """Parse "<market-slug>:LONG" / "<market-slug>:SHORT" into (slug, side)."""
    slug, sep, side = token.rpartition(":")
    if not sep or not slug or side not in {"LONG", "SHORT"}:
        raise ValueError(f"invalid Polymarket US position token: {token!r}")
    return slug, side


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
                       team=team, qty=qty, entry_price=fill,
                       trade_id=uuid.uuid4().hex[:12])
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

    def settle(self, strategy: str, token: str, settlement_price: float) -> tuple[Position, float, float] | None:
        """Close a paper position at an exact settlement price, with no slippage."""
        pos = self.positions[strategy].pop(token, None)
        if pos is None:
            return None
        fill = min(max(settlement_price, 0.0), 1.0)
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
    """Real orders on Polymarket US via the official `polymarket-us` SDK.

    Each open/close submits a marketable limit order (aggressive limit,
    immediate-or-cancel) with synchronous execution, and books the *actual*
    reported fill price/quantity — not the requested price — so local P&L
    tracking reflects real slippage rather than an optimistic assumption.
    """

    def __init__(self, strategies: list[str], starting_cash: float, slippage: float = 0.01):
        super().__init__(strategies, starting_cash, slippage)
        try:
            from polymarket_us import PolymarketUS
        except ImportError as exc:
            raise RuntimeError(
                "Live trading requires the polymarket-us package: pip install polymarket-us"
            ) from exc
        key_id = os.environ.get("POLYMARKET_KEY_ID")
        secret_key = os.environ.get("POLYMARKET_SECRET_KEY")
        if not key_id or not secret_key:
            raise RuntimeError(
                "Set POLYMARKET_KEY_ID and POLYMARKET_SECRET_KEY in .env for live trading "
                "(from polymarket.us/developer — these are Polymarket US API credentials, "
                "not offshore Polymarket.com wallet keys)."
            )
        self.client = PolymarketUS(key_id=key_id, secret_key=secret_key)
        log.info("LiveBroker initialised (Polymarket US)")

    _INTENTS = {
        ("LONG", "BUY"): "ORDER_INTENT_BUY_LONG",
        ("LONG", "SELL"): "ORDER_INTENT_SELL_LONG",
        ("SHORT", "BUY"): "ORDER_INTENT_BUY_SHORT",
        ("SHORT", "SELL"): "ORDER_INTENT_SELL_SHORT",
    }
    _FILL_TYPES = {"EXECUTION_TYPE_FILL", "EXECUTION_TYPE_PARTIAL_FILL"}

    def _submit(self, slug: str, side: str, action: str,
               limit_price: float, qty: float) -> tuple[float, float] | None:
        """Returns (avg_fill_price, filled_qty), or None if nothing filled."""
        intent = self._INTENTS[(side, action)]
        order_qty = max(0.0001, round(qty, 4))
        try:
            resp = self.client.orders.create({
                "marketSlug": slug,
                "intent": intent,
                "type": "ORDER_TYPE_LIMIT",
                "price": {"value": f"{limit_price:.4f}", "currency": "USD"},
                "quantity": order_qty,
                "tif": "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL",
                "synchronousExecution": True,
                "maxBlockTime": "5",
            })
        except Exception as exc:
            log.error("live order failed (%s %s %s): %s", action, side, slug, exc)
            return None
        fills = [e for e in resp.get("executions", []) if e.get("type") in self._FILL_TYPES]
        if not fills:
            log.warning("no fill for order %s (%s %s %s)", resp.get("id"), action, side, slug)
            return None
        filled_qty = sum(float(e["lastShares"]) for e in fills)
        if filled_qty <= 0:
            return None
        notional = sum(float(e["lastShares"]) * float(e["lastPx"]["value"]) for e in fills)
        return notional / filled_qty, filled_qty

    def open(self, strategy, market_key, token, team, price, stake_usd):
        if token in self.positions[strategy]:
            return None
        slug, side = parse_token(token)
        limit = min(price + self.slippage, 0.999)
        result = self._submit(slug, side, "BUY", limit, stake_usd / limit)
        if result is None:
            return None
        fill_price, filled_qty = result
        self.cash[strategy] -= filled_qty * fill_price
        pos = Position(strategy=strategy, market_key=market_key, token=token,
                       team=team, qty=filled_qty, entry_price=fill_price,
                       trade_id=uuid.uuid4().hex[:12])
        self.positions[strategy][token] = pos
        return pos

    def close(self, strategy, token, price):
        pos = self.positions[strategy].get(token)
        if pos is None:
            return None
        slug, side = parse_token(token)
        limit = max(price - self.slippage, 0.001)
        result = self._submit(slug, side, "SELL", limit, pos.qty)
        if result is None:
            return None
        fill_price, filled_qty = result
        proceeds = filled_qty * fill_price
        self.cash[strategy] += proceeds
        closed_cost = min(filled_qty, pos.qty) * pos.entry_price
        pnl = proceeds - closed_cost
        self.realized[strategy] += pnl
        self.closes[strategy] += 1
        closed_pos = Position(strategy=pos.strategy, market_key=pos.market_key, token=pos.token,
                              team=pos.team, qty=min(filled_qty, pos.qty),
                              entry_price=pos.entry_price, opened_at=pos.opened_at,
                              trade_id=pos.trade_id)
        remaining = pos.qty - filled_qty
        if remaining <= 1e-9:
            self.positions[strategy].pop(token, None)
        else:
            pos.qty = remaining
        return closed_pos, fill_price, pnl
