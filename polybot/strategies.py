"""Strategy interface: frozen variants competing on one shared stream.

Each strategy reads the same `StratContext` (one market/game observation) but
decides with its own frozen `StrategyConfig`, so signal sets differ by config
while execution (fills, fees) stays identical across strategies. This keeps
signal quality cleanly separable from execution quality.
"""
from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import strategy as _fade
from .ai_judge import AIJudge
from .broker import taker_fee
from .config import AIConfig, StrategyConfig
from .model_features import AnchoredView, ModelHistory
from .models import EntryEvaluation, GameState, Market, MarketQuote, Position, Signal
from .volatility import PriceHistory

# Post-signal counterfactual horizons (seconds after a candidate signal).
HORIZONS = (5, 15, 30, 60, 120, 300)


def executable_ask(market: Market, quote: MarketQuote, token: str) -> float:
    """Executable taker BUY price for a token side (ask). Single source of truth."""
    if token == market.home_token:
        return quote.home_ask
    if token == market.away_token:
        return 1.0 - quote.home_bid
    raise ValueError(f"token {token!r} not in market {market.key!r}")


def executable_bid(market: Market, quote: MarketQuote, token: str) -> float:
    """Executable taker SELL price for a token side (bid). Single source of truth."""
    if token == market.home_token:
        return quote.home_bid
    if token == market.away_token:
        return 1.0 - quote.home_ask
    raise ValueError(f"token {token!r} not in market {market.key!r}")


@dataclass
class StratContext:
    """The shared market/game observation every strategy decides against.

    `quote` may be None (one-sided/absent book): the price history stays alive
    from mark prices so a signal can still be detected and tracked, but no
    executable price exists, so nothing tradeable is produced.
    """
    market: Market
    history: PriceHistory
    game_state: GameState | None
    quote: MarketQuote | None
    now: float
    fee_theta: float = 0.0
    model_history: ModelHistory | None = None

    def entry_price(self, token: str) -> float:
        return executable_ask(self.market, self.quote, token)

    def exit_price(self, token: str) -> float:
        return executable_bid(self.market, self.quote, token)

    def round_trip_fee(self, entry_price: float, exit_price: float) -> float:
        """Per-contract taker fee paid to open at `entry_price` and later close."""
        return taker_fee(self.fee_theta, entry_price) + taker_fee(self.fee_theta, exit_price)

    @property
    def quote_age(self) -> float | None:
        return None if self.quote is None else self.now - self.quote.ts


@dataclass
class Intent:
    """A strategy's desire to OPEN a position (execution decided by the engine)."""
    token: str
    side_team: str
    signal_price: float      # model/current price of the token
    fair: float
    move: float
    edge: float              # executable edge (fair - executable ask)
    reason: str
    evaluation: EntryEvaluation | None = None


@dataclass
class ExitIntent:
    """A strategy's desire to CLOSE a position."""
    position: Position
    price: float             # executable sell price used for the decision
    reason: str
    fair: float | None = None


@dataclass
class Decision:
    """One strategy's full result for a market tick: log row + optional action."""
    evaluation: EntryEvaluation | None = None
    outcome: str = "n/a"     # decision-row outcome for this strategy
    intent: Intent | None = None
    signal_candidate: bool = False   # true when this tick produced a signal-grade candidate


class Strategy:
    """Base interface. Subclasses decide entries/exits on a shared context."""

    kind = "fade"

    def __init__(self, name: str, version: str, config: StrategyConfig):
        self.name = name
        self.version = version
        self.config = config

    def evaluate(self, ctx: StratContext) -> Decision:
        raise NotImplementedError

    def manage(self, ctx: StratContext, positions: list[Position]) -> list[ExitIntent]:
        raise NotImplementedError

    def close(self) -> None:            # release resources (threads, etc.)
        pass

    def reset_market(self, market_key: str) -> None:
        """Discard strategy-local transient state after a run/data boundary."""
        pass

    def _gate_execution(self, ctx: StratContext, ev: EntryEvaluation,
                        sig: Signal) -> Decision:
        """Shared executability cascade: a signal-grade candidate becomes an
        Intent only if a fresh two-sided book survives spread and fee gates.
        Every path keeps signal_candidate=True so counterfactuals still capture
        the non-executable moments."""
        cfg = self.config
        if ctx.quote is None:
            return Decision(evaluation=ev, outcome="no_quote", signal_candidate=True)
        if ctx.quote_age is not None and ctx.quote_age > cfg.max_quote_age_secs:
            return Decision(evaluation=ev, outcome="stale_quote", signal_candidate=True)
        if ctx.quote.home_spread > cfg.max_spread:
            return Decision(evaluation=ev, outcome="wide_spread", signal_candidate=True)
        exec_price = ctx.entry_price(sig.token)
        fee = ctx.round_trip_fee(exec_price, sig.fair)
        net_edge = sig.fair - exec_price - fee
        if net_edge < cfg.min_edge:
            ev.margin = net_edge - cfg.min_edge
            return Decision(evaluation=ev, outcome="execution_cost",
                            signal_candidate=True)
        intent = Intent(token=sig.token, side_team=sig.side_team,
                        signal_price=sig.price, fair=sig.fair, move=sig.move,
                        edge=net_edge, reason=sig.reason, evaluation=ev)
        return Decision(evaluation=ev, outcome="signal", intent=intent,
                        signal_candidate=True)


class FadeStrategy(Strategy):
    """The current mean-reversion fade, parameterised by a frozen config."""

    def evaluate(self, ctx: StratContext) -> Decision:
        ev = _fade.evaluate_entry(ctx.market, ctx.history, ctx.game_state, self.config)
        if ev.signal is None:
            return Decision(evaluation=ev, outcome=ev.outcome)
        return self._gate_execution(ctx, ev, ev.signal)

    def manage(self, ctx: StratContext, positions: list[Position]) -> list[ExitIntent]:
        gs = ctx.game_state
        out: list[ExitIntent] = []
        for pos in positions:
            price = ctx.exit_price(pos.token)
            fair = None
            if gs:
                fair_home = _fade.fair_home_value(gs, self.config)
                fair = fair_home if pos.token == ctx.market.home_token else 1.0 - fair_home
            reason = _fade.check_exit(pos, price, fair,
                                      bool(gs and gs.is_final), self.config,
                                      now=ctx.now, fee_theta=ctx.fee_theta)
            if reason:
                out.append(ExitIntent(position=pos, price=price, reason=reason, fair=fair))
        return out


class AnchoredStrategy(Strategy):
    """Base for strategies that trade model *changes*, not model level.

    The model update is transferred onto either the immediately preceding
    state price or a frozen pregame market price. All remaining execution and
    fee gates match the frozen fade control.
    """

    view_kind = "state"

    def _view(self, ctx: StratContext) -> AnchoredView | None:
        if ctx.model_history is None or ctx.history.last is None:
            return None
        if self.view_kind == "state":
            return ctx.model_history.state_view(
                ctx.history.last, ctx.now, self.config.residual_beta,
            )
        return ctx.model_history.market_view(
            ctx.history.last, ctx.now, self.config.residual_beta,
        )

    def _max_age(self) -> float:
        return self.config.residual_response_secs if self.view_kind == "state" \
            else self.config.market_anchor_max_age_secs

    def evaluate(self, ctx: StratContext) -> Decision:
        history = ctx.history
        mid = history.last
        base = {
            "mid": mid,
            "flips": history.flips,
            "realized_vol": history.realized_vol(self.config.vol_window),
        }
        if ctx.game_state is None or not ctx.game_state.is_live:
            return Decision(EntryEvaluation(outcome="not_live", **base), "not_live")
        if mid is None:
            return Decision(EntryEvaluation(outcome="no_price", **base), "no_price")
        view = self._view(ctx)
        if view is None:
            outcome = "state_warmup" if self.view_kind == "state" else "no_pregame_anchor"
            return Decision(EntryEvaluation(outcome=outcome, **base), outcome)
        detail = {
            **base,
            "move": view.market_delta,
            "fair_home": view.fair_home,
            "anchor_price": view.anchor_price,
            "anchor_model": view.anchor_model,
            "model_delta": view.model_delta,
            "residual": view.residual,
            "anchor_age": view.anchor_age,
        }
        if view.anchor_age > self._max_age():
            ev = EntryEvaluation(
                outcome="anchor_stale", margin=self._max_age() - view.anchor_age,
                **detail,
            )
            return Decision(ev, ev.outcome)
        if abs(view.model_delta) < self.config.residual_min_model_delta:
            ev = EntryEvaluation(
                outcome="small_model_move",
                margin=abs(view.model_delta) - self.config.residual_min_model_delta,
                **detail,
            )
            return Decision(ev, ev.outcome)
        if abs(view.residual) < self.config.residual_threshold:
            ev = EntryEvaluation(
                outcome="small_residual",
                margin=abs(view.residual) - self.config.residual_threshold,
                **detail,
            )
            return Decision(ev, ev.outcome)

        market = ctx.market
        if view.residual < 0:
            token, team, price, fair = (
                market.home_token, market.home_team, mid, view.fair_home,
            )
        else:
            token, team, price, fair = (
                market.away_token, market.away_team, 1.0 - mid, 1.0 - view.fair_home,
            )
        edge = fair - price
        signal = Signal(
            market=market, token=token, side_team=team, price=price, fair=fair,
            move=view.market_delta,
            reason=(
                f"{self.kind} residual {view.residual:+.3f}; "
                f"model delta {view.model_delta:+.3f}; "
                f"anchored fair {view.fair_home:.3f}"
            ),
        )
        ev = EntryEvaluation(
            outcome="signal", side_team=team, price=price, fair=fair, edge=edge,
            margin=edge - self.config.min_edge, signal=signal, **detail,
        )
        if not (self.config.min_price <= price <= self.config.max_price):
            ev.outcome = "price_band"
            ev.margin = price - self.config.min_price if price < self.config.min_price \
                else self.config.max_price - price
            return Decision(ev, ev.outcome, signal_candidate=True)
        return self._gate_execution(ctx, ev, signal)

    def manage(self, ctx: StratContext, positions: list[Position]) -> list[ExitIntent]:
        view = self._view(ctx)
        out: list[ExitIntent] = []
        for pos in positions:
            price = ctx.exit_price(pos.token)
            fair = None
            if view is not None:
                fair = view.fair_home if pos.token == ctx.market.home_token \
                    else 1.0 - view.fair_home
            reason = _fade.check_exit(
                pos, price, fair,
                bool(ctx.game_state and ctx.game_state.is_final), self.config,
                now=ctx.now, fee_theta=ctx.fee_theta,
            )
            if reason:
                out.append(ExitIntent(pos, price, reason, fair))
        return out


class StateResidualStrategy(AnchoredStrategy):
    """Fade a short-lived market miss after a newly received game state."""

    kind = "state_residual"
    view_kind = "state"


class MarketAnchoredStrategy(AnchoredStrategy):
    """Value strategy anchored to the last pregame market probability."""

    kind = "market_anchored"
    view_kind = "market"


class AIShadowStrategy(Strategy):
    """Async gate over a base fade strategy's candidates; opens in its own book.

    Never blocks the trading loop: a candidate is submitted to a thread pool and
    the verdict is drained on a later tick, then the position (if approved) opens
    at the *then-current* executable price so judge latency and any adverse price
    drift during the think are reflected in the shadow ledger.
    """

    kind = "ai_shadow"

    def __init__(self, name: str, version: str, base: FadeStrategy,
                 cfg: AIConfig | None = None, judge=None):
        super().__init__(name, version, base.config)
        self.base = base
        self.judge = judge if judge is not None else AIJudge(cfg)
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._pending: dict[str, tuple] = {}   # market_key -> (future, intent, submit_ts)

    def evaluate(self, ctx: StratContext) -> Decision:
        mkey = ctx.market.key
        entry = self._pending.get(mkey)
        # 1) drain a resolved verdict for this market
        if entry and entry[0].done():
            future, intent, submit_ts = self._pending.pop(mkey)
            try:
                verdict = future.result()
            except Exception:
                verdict = None
            if verdict and verdict.approve:
                exec_price = ctx.entry_price(intent.token)
                latency = ctx.now - submit_ts
                fresh = Intent(
                    token=intent.token, side_team=intent.side_team,
                    signal_price=intent.signal_price, fair=intent.fair,
                    move=intent.move, edge=intent.fair - exec_price,
                    reason=(f"{intent.reason} | ai {verdict.reason} "
                            f"({verdict.confidence:.2f}, {latency:.1f}s)"),
                    evaluation=intent.evaluation,
                )
                return Decision(outcome="ai_opened", intent=fresh)
            return Decision(outcome="ai_rejected")
        # 2) still in flight -> hold
        if mkey in self._pending:
            return Decision(outcome="ai_pending")
        # 3) ask the base for a fresh candidate; submit it to the judge
        d = self.base.evaluate(ctx)
        if d.intent is None:
            return Decision(evaluation=d.evaluation, outcome=d.outcome,
                            signal_candidate=d.signal_candidate)
        sig = d.evaluation.signal
        fut = self._pool.submit(self.judge.judge, sig, ctx.game_state)
        self._pending[mkey] = (fut, d.intent, ctx.now)
        return Decision(evaluation=d.evaluation, outcome="ai_pending",
                        signal_candidate=True)

    def manage(self, ctx: StratContext, positions: list[Position]) -> list[ExitIntent]:
        return self.base.manage(ctx, positions)

    def wait_idle(self) -> None:
        """Test helper: block until every in-flight verdict has resolved."""
        for future, *_ in list(self._pending.values()):
            future.result()

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


DEFAULT_REGISTRY = [
    {"name": "fade_v1_frozen", "kind": "fade"},
    {"name": "fade_tight", "kind": "fade",
     "overrides": {"move_threshold": 0.15, "min_edge": 0.09}},
    {"name": "liquidity_fade_v2", "kind": "fade",
     "overrides": {"sampling_stable_features": True}},
    {"name": "state_residual_v1", "kind": "state_residual"},
    {"name": "market_anchor_v1", "kind": "market_anchored"},
    {"name": "ai_shadow", "kind": "ai_shadow", "base": "fade_v1_frozen"},
]


def build_strategies(cfg) -> list[Strategy]:
    """Instantiate the strategy registry from config (or the built-in default)."""
    entries = cfg.strategies or DEFAULT_REGISTRY
    strats: list[Strategy] = []
    for entry in entries:
        kind = entry.get("kind", "fade")
        name = entry["name"]
        if kind == "fade":
            sconf = dataclasses.replace(cfg.strategy, **entry.get("overrides", {}))
            strats.append(FadeStrategy(name, entry.get("version", "v1"), sconf))
        elif kind in {"state_residual", "market_anchored"}:
            sconf = dataclasses.replace(cfg.strategy, **entry.get("overrides", {}))
            cls = StateResidualStrategy if kind == "state_residual" \
                else MarketAnchoredStrategy
            strats.append(cls(name, entry.get("version", "v1"), sconf))
        elif kind == "ai_shadow":
            if not cfg.ai.enabled:
                continue
            base = next(s for s in strats if s.name == entry["base"])
            strats.append(AIShadowStrategy(name, entry.get("version", "v1"),
                                           base, cfg.ai))
        else:
            raise ValueError(f"unknown strategy kind: {kind!r}")
    return strats
