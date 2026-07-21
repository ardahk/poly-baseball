"""Order execution: paper broker (default) and live Polymarket US broker (guarded)."""
from __future__ import annotations

import logging
import os
import time
import uuid
from decimal import Decimal, ROUND_HALF_EVEN

from .models import Position

log = logging.getLogger(__name__)


# Published Polymarket US schedule, verified 2026-07-21 against
# https://docs.polymarket.us/fees AND against the live gateway payload, which
# reports feeCoefficient=0.06 on every MLB market including the moneyline.
# Third-party pages quoting "sports = 0.05" describe the legacy *global*
# Polymarket (help.polymarket.com), a different venue.
TAKER_THETA = 0.06
MAKER_THETA = -0.0125   # negative: makers are PAID


def taker_fee(theta: float, price: float, qty: float = 1.0) -> float:
    """Polymarket US taker fee, UNrounded.

    `theta * qty * price * (1 - price)`. Used for *decisions* (edge/exit math),
    where per-contract cent rounding would collapse to $0.00 and hide the cost.
    `PaperBroker.fee` applies the venue's banker's rounding for actual cash flows.
    """
    return theta * qty * price * (1.0 - price)


def maker_rebate(price: float, qty: float = 1.0,
                 theta: float = MAKER_THETA) -> float:
    """Rebate CREDITED to a maker fill, as a positive number.

    Makers pay no fee and receive `-theta * qty * p * (1-p)`. This is not free
    money: resting an order buys adverse selection, measured at -1.12c/contract
    on this tape versus a +0.22c rebate (docs/research-log.md, Session 1 §6).
    Naked liquidity provision is net NEGATIVE. Use `maker_edge_vs_taker` rather
    than the rebate alone when deciding anything.
    """
    return -theta * qty * price * (1.0 - price)


def maker_edge_vs_taker(price: float, half_spread: float,
                        adverse_selection: float,
                        taker_theta: float = TAKER_THETA) -> float:
    """Per-contract advantage of resting an order over crossing the spread.

    Taker pays the half spread plus the fee. Maker pays neither, but eats
    `adverse_selection` (a positive cost) and collects the rebate. Positive
    result means maker execution is cheaper for the same exposure.
    """
    taker_cost = half_spread + taker_fee(taker_theta, price)
    maker_cost = adverse_selection - maker_rebate(price)
    return taker_cost - maker_cost


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

    def __init__(self, strategies: list[str], starting_cash: float, slippage: float = 0.0,
                 taker_fee_theta: float = 0.0, revival_deposit_usd: float = 0.0,
                 max_revivals: int = 0):
        self.slippage = slippage
        self.taker_fee_theta = taker_fee_theta
        self.revival_deposit_usd = revival_deposit_usd
        self.max_revivals = max_revivals
        self.cash: dict[str, float] = {s: starting_cash for s in strategies}
        self.positions: dict[str, dict[str, Position]] = {s: {} for s in strategies}
        self.realized: dict[str, float] = {s: 0.0 for s in strategies}
        self.closes: dict[str, int] = {s: 0 for s in strategies}
        self.last_fee: dict[str, float] = {s: 0.0 for s in strategies}
        # Lifecycle. `deposited` is every dollar ever put in (starting cash plus
        # revival top-ups) and is the ONLY correct denominator for return %.
        self.deposited: dict[str, float] = {s: starting_cash for s in strategies}
        self.revivals: dict[str, int] = {s: 0 for s in strategies}
        self.last_revival_ts: dict[str, float | None] = {s: None for s in strategies}
        self.retired_at: dict[str, float | None] = {s: None for s in strategies}

    # ------------------------------------------------------------- lifecycle

    def is_retired(self, strategy: str) -> bool:
        return self.retired_at.get(strategy) is not None

    def check_solvency(self, strategy: str, min_stake: float) -> str | None:
        """Revive or retire an account that can no longer open a position.

        An account that cannot fund the minimum stake is dead: it stops trading
        and its return freezes near -100%, which silently poisons every
        cross-strategy comparison. Give it one top-up, then retire it for good.

        Returns "revived", "retired", or None if nothing changed.
        """
        if self.is_retired(strategy) or self.cash[strategy] >= min_stake:
            return None
        if self.revivals[strategy] < self.max_revivals and self.revival_deposit_usd > 0:
            self.cash[strategy] += self.revival_deposit_usd
            self.deposited[strategy] += self.revival_deposit_usd
            self.revivals[strategy] += 1
            self.last_revival_ts[strategy] = time.time()
            log.warning("[%s] account died (cash %.2f < %.2f); second chance: +$%.2f "
                        "(revival %d, total deposited $%.2f)", strategy,
                        self.cash[strategy] - self.revival_deposit_usd, min_stake,
                        self.revival_deposit_usd, self.revivals[strategy],
                        self.deposited[strategy])
            return "revived"
        self.retired_at[strategy] = time.time()
        log.warning("[%s] RETIRED: cash %.2f < %.2f after %d revival(s); "
                    "total deposited $%.2f, ending equity $%.2f",
                    strategy, self.cash[strategy], min_stake,
                    self.revivals[strategy], self.deposited[strategy],
                    self.cash[strategy])
        return "retired"

    def fee(self, price: float, qty: float, theta: float | None = None) -> float:
        """Exchange taker fee on an actual fill, rounded like the venue.

        Banker's rounding to the cent mirrors real cash flows. For pre-trade
        edge/exit decisions use the unrounded `taker_fee` free function instead.
        `theta` overrides the configured coefficient with the venue's own
        per-market value when the gateway reported one.
        """
        th = self.taker_fee_theta if theta is None else theta
        raw = Decimal(str(th)) * Decimal(str(qty)) \
            * Decimal(str(price)) * (Decimal("1") - Decimal(str(price)))
        return float(raw.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))

    def open(self, strategy: str, market_key: str, token: str, team: str,
             price: float, stake_usd: float, theta: float | None = None) -> Position | None:
        if self.is_retired(strategy):
            return None
        fill = min(price + self.slippage, 0.999)
        qty = stake_usd / fill
        fee = self.fee(fill, qty, theta)
        cost = qty * fill + fee
        if cost > self.cash[strategy]:
            log.info("[%s] insufficient cash for %s", strategy, team)
            return None
        if token in self.positions[strategy]:
            return None  # already holding this token
        self.cash[strategy] -= cost
        self.last_fee[strategy] = fee
        pos = Position(strategy=strategy, market_key=market_key, token=token,
                       team=team, qty=qty, entry_price=fill, entry_fee=fee,
                       trade_id=uuid.uuid4().hex[:12])
        self.positions[strategy][token] = pos
        return pos

    def close(self, strategy: str, token: str, price: float,
              theta: float | None = None) -> tuple[Position, float, float] | None:
        """Returns (position, fill_price, pnl_usd)."""
        pos = self.positions[strategy].pop(token, None)
        if pos is None:
            return None
        fill = max(price - self.slippage, 0.001)
        fee = self.fee(fill, pos.qty, theta)
        proceeds = pos.qty * fill - fee
        self.cash[strategy] += proceeds
        pnl = proceeds - pos.cost
        self.realized[strategy] += pnl
        self.closes[strategy] += 1
        self.last_fee[strategy] = fee
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
        self.last_fee[strategy] = 0.0
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

    def restore(self, accounts: dict[str, dict], positions: list[Position]) -> bool:
        """Restore a previously persisted paper ledger.

        Missing strategies deliberately retain their configured starting cash, so
        enabling a new strategy does not inherit another strategy's account.
        """
        restored = False
        for strategy, state in accounts.items():
            if strategy not in self.cash:
                continue
            self.cash[strategy] = float(state["cash"])
            self.realized[strategy] = float(state["realized"])
            self.closes[strategy] = int(state["closes"])
            # Older rows predate the lifecycle columns; fall back to the
            # configured starting cash so `deposited` is never zero.
            deposited = state.get("deposited") if hasattr(state, "get") else None
            self.deposited[strategy] = float(
                deposited if deposited is not None else self.deposited[strategy])
            self.revivals[strategy] = int(state.get("revivals") or 0)
            self.last_revival_ts[strategy] = state.get("last_revival_ts")
            self.retired_at[strategy] = state.get("retired_at")
            restored = True
        for position in positions:
            if position.strategy not in self.positions:
                continue
            self.positions[position.strategy][position.token] = position
            restored = True
        return restored


class LiveBroker(PaperBroker):
    """Real orders on Polymarket US via the official `polymarket-us` SDK.

    Each open/close submits a marketable limit order (aggressive limit,
    immediate-or-cancel) with synchronous execution, and books the *actual*
    reported fill price/quantity — not the requested price — so local P&L
    tracking reflects real slippage rather than an optimistic assumption.
    """

    def __init__(self, strategies: list[str], starting_cash: float, slippage: float = 0.01,
                 taker_fee_theta: float = 0.0):
        super().__init__(strategies, starting_cash, slippage, taker_fee_theta)
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

    @staticmethod
    def _to_long_price(side: str, side_price: float) -> float:
        """Convert a side-relative price to LONG/YES units (its own inverse).

        Polymarket US requires `price.value` in LONG units for every intent,
        including BUY_SHORT/SELL_SHORT, and returns execution prices in LONG
        units too. A SHORT price `s` corresponds to LONG price `1 - s`.
        """
        long_price = side_price if side == "LONG" else 1.0 - side_price
        return min(max(long_price, 0.0001), 0.9999)

    def _submit(self, slug: str, side: str, action: str,
               limit_price: float, qty: float) -> tuple[float, float] | None:
        """Submit an order whose limit is expressed in side-relative units.

        Returns (avg_fill_price, filled_qty) in SIDE-relative units, or None.
        """
        intent = self._INTENTS[(side, action)]
        order_qty = max(0.0001, round(qty, 4))
        long_limit = self._to_long_price(side, limit_price)
        try:
            resp = self.client.orders.create({
                "marketSlug": slug,
                "intent": intent,
                "type": "ORDER_TYPE_LIMIT",
                "price": {"value": f"{long_limit:.4f}", "currency": "USD"},
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
        # lastPx is in LONG units; convert each fill back to side-relative units.
        notional = sum(
            float(e["lastShares"]) * self._to_long_price(side, float(e["lastPx"]["value"]))
            for e in fills
        )
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
