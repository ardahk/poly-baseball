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
from .config import AIConfig, StrategyConfig
from .models import EntryEvaluation, GameState, Market, MarketQuote, Position
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
    """The shared market/game observation every strategy decides against."""
    market: Market
    history: PriceHistory
    game_state: GameState | None
    quote: MarketQuote
    now: float

    def entry_price(self, token: str) -> float:
        return executable_ask(self.market, self.quote, token)

    def exit_price(self, token: str) -> float:
        return executable_bid(self.market, self.quote, token)


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


class FadeStrategy(Strategy):
    """The current mean-reversion fade, parameterised by a frozen config."""

    def evaluate(self, ctx: StratContext) -> Decision:
        ev = _fade.evaluate_entry(ctx.market, ctx.history, ctx.game_state, self.config)
        if ev.signal is None:
            return Decision(evaluation=ev, outcome=ev.outcome)
        sig = ev.signal
        exec_price = ctx.entry_price(sig.token)
        edge = sig.fair - exec_price
        if edge < self.config.min_edge:
            return Decision(evaluation=ev, outcome="execution_cost",
                            signal_candidate=True)
        intent = Intent(token=sig.token, side_team=sig.side_team,
                        signal_price=sig.price, fair=sig.fair, move=sig.move,
                        edge=edge, reason=sig.reason, evaluation=ev)
        return Decision(evaluation=ev, outcome="signal", intent=intent,
                        signal_candidate=True)

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
                                      bool(gs and gs.is_final), self.config, now=ctx.now)
            if reason:
                out.append(ExitIntent(position=pos, price=price, reason=reason, fair=fair))
        return out


class AIShadowStrategy(Strategy):
    """Async gate over a base fade strategy's candidates; opens in its own book.

    Never blocks the trading loop: a candidate is submitted to a thread pool and
    the verdict is drained on a later tick, then the position (if approved) opens
    at the *then-current* executable price so judge latency and any adverse price
    drift during the think are reflected in the shadow ledger.
    """

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
        elif kind == "ai_shadow":
            if not cfg.ai.enabled:
                continue
            base = next(s for s in strats if s.name == entry["base"])
            strats.append(AIShadowStrategy(name, entry.get("version", "v1"),
                                           base, cfg.ai))
        else:
            raise ValueError(f"unknown strategy kind: {kind!r}")
    return strats
