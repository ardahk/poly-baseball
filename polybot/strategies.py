"""Strategy interface: frozen variants competing on one shared stream.

Each strategy reads the same `StratContext` (one market/game observation) but
decides with its own frozen `StrategyConfig`, so signal sets differ by config
while execution (fills, fees) stays identical across strategies. This keeps
signal quality cleanly separable from execution quality.
"""
from __future__ import annotations

import dataclasses
from collections import deque
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

    def _manage_positions(self, ctx: StratContext, positions: list[Position],
                          fair_home: float | None) -> list[ExitIntent]:
        """Shared exit management: TP/SL/time/edge/settlement via check_exit.

        `fair_home` None disables the edge-gone exit; price-based stops, the
        time stop, and settlement on game final always apply.
        """
        out: list[ExitIntent] = []
        for pos in positions:
            price = ctx.exit_price(pos.token)
            fair = None
            if fair_home is not None:
                fair = fair_home if pos.token == ctx.market.home_token \
                    else 1.0 - fair_home
            reason = _fade.check_exit(
                pos, price, fair,
                bool(ctx.game_state and ctx.game_state.is_final), self.config,
                now=ctx.now, fee_theta=ctx.fee_theta,
            )
            if reason:
                out.append(ExitIntent(pos, price, reason, fair))
        return out

    def _tick_detail(self, ctx: StratContext) -> tuple[dict, Decision | None]:
        """Common evaluate() prelude: base features + not_live/no_price gates."""
        history = ctx.history
        base = {
            "mid": history.last,
            "flips": history.flips,
            "realized_vol": history.realized_vol(self.config.vol_window),
        }
        if ctx.game_state is None or not ctx.game_state.is_live:
            return base, Decision(EntryEvaluation(outcome="not_live", **base), "not_live")
        if history.last is None:
            return base, Decision(EntryEvaluation(outcome="no_price", **base), "no_price")
        return base, None

    def _side_candidate(self, ctx: StratContext, detail: dict, buy_home: bool,
                        fair_home: float, move: float, reason: str) -> Decision:
        """Build a one-side Signal, apply the config price band, gate execution."""
        market = ctx.market
        mid = ctx.history.last
        if buy_home:
            token, team, price, fair = (
                market.home_token, market.home_team, mid, fair_home,
            )
        else:
            token, team, price, fair = (
                market.away_token, market.away_team, 1.0 - mid, 1.0 - fair_home,
            )
        edge = fair - price
        signal = Signal(market=market, token=token, side_team=team, price=price,
                        fair=fair, move=move, reason=reason)
        ev = EntryEvaluation(
            outcome="signal", side_team=team, price=price, fair=fair, edge=edge,
            margin=edge - self.config.min_edge, signal=signal, **detail,
        )
        if not (self.config.min_price <= price <= self.config.max_price):
            ev.outcome = "price_band"
            ev.margin = price - self.config.min_price if price < self.config.min_price \
                else self.config.max_price - price
            return Decision(ev, ev.outcome)
        return self._gate_execution(ctx, ev, signal)

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
        fair_home = _fade.fair_home_value(gs, self.config) if gs else None
        return self._manage_positions(ctx, positions, fair_home)


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
            # The fade control does not treat price_band rejects as
            # signal-grade candidates; keep episode accounting comparable.
            return Decision(ev, ev.outcome)
        return self._gate_execution(ctx, ev, signal)

    def manage(self, ctx: StratContext, positions: list[Position]) -> list[ExitIntent]:
        view = self._view(ctx)
        if view is not None and view.anchor_age > self._max_age():
            # The entry gates reject a stale anchor; exits must not act on a
            # fair the strategy itself considers invalid. Price-based stops,
            # take profit, time stop, and settlement still apply.
            view = None
        return self._manage_positions(
            ctx, positions, view.fair_home if view is not None else None,
        )


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


class MomentumStrategy(Strategy):
    """Trade WITH a sharp move: in-game moves are information (the refuted-fade
    result inverted), so a large recent move should on average continue or at
    least not revert within the holding window."""

    kind = "momentum"

    def evaluate(self, ctx: StratContext) -> Decision:
        detail, early = self._tick_detail(ctx)
        if early:
            return early
        cfg = self.config
        move = ctx.history.move(cfg.move_lookback_secs)
        detail["move"] = move
        if move is None or abs(move) < cfg.move_threshold:
            margin = None if move is None else abs(move) - cfg.move_threshold
            ev = EntryEvaluation(outcome="small_move", margin=margin, **detail)
            return Decision(ev, ev.outcome)
        if cfg.momentum_min_state_age_secs > 0:
            view = None
            if ctx.model_history is not None and ctx.history.last is not None:
                view = ctx.model_history.state_view(ctx.history.last, ctx.now)
            if view is not None and view.anchor_age < cfg.momentum_min_state_age_secs:
                ev = EntryEvaluation(
                    outcome="state_too_fresh",
                    margin=view.anchor_age - cfg.momentum_min_state_age_secs,
                    anchor_age=view.anchor_age, **detail,
                )
                return Decision(ev, ev.outcome)
        fair_home = _fade.fair_home_value(ctx.game_state, cfg)
        detail["fair_home"] = fair_home
        buy_home = move > 0
        if cfg.momentum_require_model_agree:
            side_fair = fair_home if buy_home else 1.0 - fair_home
            side_price = ctx.history.last if buy_home else 1.0 - ctx.history.last
            if side_fair < side_price:
                ev = EntryEvaluation(outcome="model_disagrees",
                                     margin=side_fair - side_price, **detail)
                return Decision(ev, ev.outcome)
        reason = (f"momentum move {move:+.3f}/{cfg.move_lookback_secs:.0f}s; "
                  f"buying the moving side")
        return self._side_candidate(ctx, detail, buy_home, fair_home, move, reason)

    def manage(self, ctx: StratContext, positions: list[Position]) -> list[ExitIntent]:
        # Momentum exits on price (TP/SL/time/settlement), never on model fair:
        # the mechanism claims the MARKET move is the information.
        return self._manage_positions(ctx, positions, None)


class EventReactionStrategy(Strategy):
    """Buy in the model-delta direction while the market still lags a freshly
    received state change (with-the-news underreaction; the sign-opposite of
    the state_residual fade, isolated from its overreaction half)."""

    kind = "event_reaction"

    def __init__(self, name: str, version: str, config: StrategyConfig):
        super().__init__(name, version, config)
        self._last_state: dict[str, GameState] = {}
        self._events: dict[str, tuple[str, float]] = {}   # market -> (class, received_at)

    def reset_market(self, market_key: str) -> None:
        self._last_state.pop(market_key, None)
        self._events.pop(market_key, None)

    @staticmethod
    def _classify(prev: GameState, cur: GameState) -> str | None:
        if cur.home_score != prev.home_score or cur.away_score != prev.away_score:
            return "score_change"
        if cur.inning != prev.inning or cur.is_top != prev.is_top:
            return "inning_change"
        if (cur.outs != prev.outs or cur.on_first != prev.on_first
                or cur.on_second != prev.on_second or cur.on_third != prev.on_third):
            return "bases_or_outs"
        return None

    def _track_event(self, ctx: StratContext) -> None:
        gs = ctx.game_state
        key = ctx.market.key
        prev = self._last_state.get(key)
        if prev is not None and gs.received_at != prev.received_at:
            event_class = self._classify(prev, gs)
            if event_class is not None:
                self._events[key] = (event_class, gs.received_at)
        self._last_state[key] = gs

    def evaluate(self, ctx: StratContext) -> Decision:
        detail, early = self._tick_detail(ctx)
        if early:
            return early
        cfg = self.config
        self._track_event(ctx)
        gs = ctx.game_state
        if gs.inning < cfg.event_min_inning:
            ev = EntryEvaluation(outcome="early_game",
                                 margin=gs.inning - cfg.event_min_inning, **detail)
            return Decision(ev, ev.outcome)
        event = self._events.get(ctx.market.key)
        if event is None or ctx.now - event[1] > cfg.event_max_age_secs:
            ev = EntryEvaluation(outcome="no_event", **detail)
            return Decision(ev, ev.outcome)
        if cfg.event_class != "any" and event[0] != cfg.event_class:
            ev = EntryEvaluation(outcome="wrong_event_class", **detail)
            return Decision(ev, ev.outcome)
        view = None
        if ctx.model_history is not None:
            view = ctx.model_history.state_view(ctx.history.last, ctx.now)
        if view is None:
            ev = EntryEvaluation(outcome="state_warmup", **detail)
            return Decision(ev, ev.outcome)
        detail.update(fair_home=view.fair_home, anchor_price=view.anchor_price,
                      anchor_model=view.anchor_model, model_delta=view.model_delta,
                      residual=view.residual, anchor_age=view.anchor_age,
                      move=view.market_delta)
        if abs(view.model_delta) < cfg.event_min_model_delta:
            ev = EntryEvaluation(
                outcome="small_model_move",
                margin=abs(view.model_delta) - cfg.event_min_model_delta, **detail,
            )
            return Decision(ev, ev.outcome)
        # Underreaction only: the market must trail the anchored fair in the
        # model-delta direction. (residual = market - anchored fair, so lagging
        # a positive delta means residual < 0 and vice versa.)
        if view.model_delta > 0 and view.residual <= -cfg.event_min_underreaction:
            buy_home = True
        elif view.model_delta < 0 and view.residual >= cfg.event_min_underreaction:
            buy_home = False
        else:
            ev = EntryEvaluation(
                outcome="no_underreaction",
                margin=abs(view.residual) - cfg.event_min_underreaction, **detail,
            )
            return Decision(ev, ev.outcome)
        reason = (f"event_reaction {event[0]} model delta {view.model_delta:+.3f}; "
                  f"market lags anchored fair by {view.residual:+.3f}")
        return self._side_candidate(ctx, detail, buy_home, view.fair_home,
                                    view.market_delta, reason)

    def manage(self, ctx: StratContext, positions: list[Position]) -> list[ExitIntent]:
        # Price-based exits only; the entry edge lives for seconds, so an
        # anchored-fair edge exit would churn every position immediately.
        return self._manage_positions(ctx, positions, None)


class ExtremeHoldStrategy(Strategy):
    """Buy the side priced inside an extreme band and hold to settlement: the
    taker fee is theta*p*(1-p), so the cost floor collapses at the tails while
    any systematic favorite/longshot mispricing persists to the end."""

    kind = "extreme_hold"

    def evaluate(self, ctx: StratContext) -> Decision:
        detail, early = self._tick_detail(ctx)
        if early:
            return early
        cfg = self.config
        gs = ctx.game_state
        if not (cfg.extreme_min_inning <= gs.inning <= cfg.extreme_max_inning):
            ev = EntryEvaluation(outcome="outside_inning_band", **detail)
            return Decision(ev, ev.outcome)
        mid = ctx.history.last
        candidates = [(mid, True), (1.0 - mid, False)]
        in_band = [(p, home) for p, home in candidates
                   if cfg.extreme_min_price <= p <= cfg.extreme_max_price]
        if not in_band:
            nearest = min(abs(p - cfg.extreme_min_price) for p, _ in candidates)
            ev = EntryEvaluation(outcome="outside_price_band", margin=-nearest, **detail)
            return Decision(ev, ev.outcome)
        price, buy_home = max(in_band)
        fair_home = _fade.fair_home_value(gs, cfg)
        detail["fair_home"] = fair_home
        if cfg.extreme_require_model_agree:
            side_fair = fair_home if buy_home else 1.0 - fair_home
            required = price + cfg.extreme_model_agree_margin
            if side_fair < required:
                ev = EntryEvaluation(outcome="model_disagrees",
                                     margin=side_fair - required, **detail)
                return Decision(ev, ev.outcome)
        reason = (f"extreme_hold {price:.3f} in "
                  f"[{cfg.extreme_min_price:.2f}, {cfg.extreme_max_price:.2f}], "
                  f"inning {gs.inning}; hold to settlement")
        move = ctx.history.move(cfg.move_lookback_secs) or 0.0
        return self._side_candidate(ctx, detail, buy_home, fair_home, move, reason)

    def manage(self, ctx: StratContext, positions: list[Position]) -> list[ExitIntent]:
        return self._manage_positions(ctx, positions, None)


class SettlementHoldStrategy(Strategy):
    """Model-vs-market disagreement held to settlement: one fee leg instead of
    two, so the cost floor is ~half the round-trip fade's. Makes the locked
    2026-07-12 preregistered rule a live frozen strategy."""

    kind = "settlement_hold"

    def evaluate(self, ctx: StratContext) -> Decision:
        detail, early = self._tick_detail(ctx)
        if early:
            return early
        cfg = self.config
        gs = ctx.game_state
        if gs.inning > cfg.hold_max_inning:
            ev = EntryEvaluation(outcome="late_inning",
                                 margin=cfg.hold_max_inning - gs.inning, **detail)
            return Decision(ev, ev.outcome)
        if cfg.hold_fair_source == "market_anchored":
            view = None
            if ctx.model_history is not None:
                view = ctx.model_history.market_view(ctx.history.last, ctx.now)
            if view is None:
                ev = EntryEvaluation(outcome="no_pregame_anchor", **detail)
                return Decision(ev, ev.outcome)
            fair_home = view.fair_home
            detail.update(anchor_price=view.anchor_price, anchor_model=view.anchor_model,
                          model_delta=view.model_delta, anchor_age=view.anchor_age)
        else:
            fair_home = _fade.fair_home_value(gs, cfg)
        detail["fair_home"] = fair_home
        gap = fair_home - ctx.history.last
        if gap >= cfg.hold_min_edge:
            buy_home = True
        elif -gap >= cfg.hold_min_edge:
            buy_home = False
        else:
            ev = EntryEvaluation(outcome="small_gap",
                                 margin=abs(gap) - cfg.hold_min_edge, **detail)
            return Decision(ev, ev.outcome)
        if cfg.hold_side_filter != "any" and \
                (cfg.hold_side_filter == "home") != buy_home:
            ev = EntryEvaluation(outcome="side_filtered", **detail)
            return Decision(ev, ev.outcome)
        reason = (f"settlement_hold {cfg.hold_fair_source} gap {gap:+.3f} "
                  f"(fair {fair_home:.3f}), inning {gs.inning}; hold to settlement")
        return self._side_candidate(ctx, detail, buy_home, fair_home, 0.0, reason)

    def manage(self, ctx: StratContext, positions: list[Position]) -> list[ExitIntent]:
        return self._manage_positions(ctx, positions, None)


class CalibrationCellStrategy(Strategy):
    """Model-free bias harvesting: buy a specific side whenever the market
    enters a (price band x inning band x side) cell and hold to settlement.
    The hypothesis is systematic miscalibration of the cell itself
    (favorite-longshot bias, home bias, slow crediting of leads)."""

    kind = "calibration_cell"

    def _cell_side(self, ctx: StratContext) -> bool | None:
        """Which side the cell buys: True=home, False=away, None=undefined."""
        side = self.config.cell_side
        gs = ctx.game_state
        mid = ctx.history.last
        if side == "home":
            return True
        if side == "away":
            return False
        if side == "favorite":
            return mid >= 0.5
        if side == "underdog":
            return mid < 0.5
        if side in ("leader", "trailer"):
            if gs.home_score == gs.away_score:
                return None
            leading_home = gs.home_score > gs.away_score
            return leading_home if side == "leader" else not leading_home
        raise ValueError(f"unknown cell_side {side!r}")

    def evaluate(self, ctx: StratContext) -> Decision:
        detail, early = self._tick_detail(ctx)
        if early:
            return early
        cfg = self.config
        gs = ctx.game_state
        if not (cfg.cell_inning_min <= gs.inning <= cfg.cell_inning_max):
            ev = EntryEvaluation(outcome="outside_cell_inning", **detail)
            return Decision(ev, ev.outcome)
        buy_home = self._cell_side(ctx)
        if buy_home is None:
            ev = EntryEvaluation(outcome="no_cell_side", **detail)
            return Decision(ev, ev.outcome)
        price = ctx.history.last if buy_home else 1.0 - ctx.history.last
        if not (cfg.cell_price_min <= price <= cfg.cell_price_max):
            margin = price - cfg.cell_price_min if price < cfg.cell_price_min \
                else cfg.cell_price_max - price
            ev = EntryEvaluation(outcome="outside_cell_price", margin=margin, **detail)
            return Decision(ev, ev.outcome)
        fair_home = _fade.fair_home_value(gs, cfg)
        detail["fair_home"] = fair_home
        reason = (f"calibration_cell {cfg.cell_side} @ {price:.3f} in "
                  f"[{cfg.cell_price_min:.2f}, {cfg.cell_price_max:.2f}], "
                  f"inning {gs.inning}; hold to settlement")
        return self._side_candidate(ctx, detail, buy_home, fair_home, 0.0, reason)

    def manage(self, ctx: StratContext, positions: list[Position]) -> list[ExitIntent]:
        return self._manage_positions(ctx, positions, None)


class MicrostructureStrategy(Strategy):
    """Book-shape/timing mechanisms selected by `micro_mode`:

    - spread_snap: a spread shock that re-tightens reveals which side's quotes
      improved; the reprice direction carries information.
    - stale_reprice: after a one-sided/absent-book gap, the reprice vs the
      pre-gap mid reflects information accumulated during the outage.
    - pregame_drift: late pregame money is informed (lineups, pitchers); ride
      the last-30-minute pregame move at the first live tick.
    """

    kind = "microstructure"

    def __init__(self, name: str, version: str, config: StrategyConfig):
        super().__init__(name, version, config)
        self._quotes: dict[str, deque] = {}       # market -> (ts, mid, spread)
        self._fired: set[str] = set()             # pregame_drift once per game

    def reset_market(self, market_key: str) -> None:
        self._quotes.pop(market_key, None)
        self._fired.discard(market_key)

    def _reject(self, detail: dict, outcome: str, margin: float | None = None) -> Decision:
        ev = EntryEvaluation(outcome=outcome, margin=margin, **detail)
        return Decision(ev, ev.outcome)

    def evaluate(self, ctx: StratContext) -> Decision:
        detail, early = self._tick_detail(ctx)
        if early:
            return early
        cfg = self.config
        buf = self._quotes.setdefault(ctx.market.key, deque())
        prev_quote = buf[-1] if buf else None
        if ctx.quote is not None:
            buf.append((ctx.quote.ts, ctx.quote.home_mid, ctx.quote.home_spread))
            horizon = max(cfg.micro_window_secs * 3, 300.0)
            while buf and buf[0][0] < ctx.now - horizon:
                buf.popleft()

        if cfg.micro_mode == "pregame_drift":
            return self._pregame_drift(ctx, detail)
        if ctx.quote is None:
            return self._reject(detail, "no_quote_yet")
        if cfg.micro_mode == "spread_snap":
            return self._spread_snap(ctx, detail, buf)
        if cfg.micro_mode == "stale_reprice":
            return self._stale_reprice(ctx, detail, prev_quote)
        raise ValueError(f"unknown micro_mode {cfg.micro_mode!r}")

    def _finish(self, ctx: StratContext, detail: dict, buy_home: bool,
                move: float, reason: str) -> Decision:
        fair_home = _fade.fair_home_value(ctx.game_state, self.config)
        detail["fair_home"] = fair_home
        return self._side_candidate(ctx, detail, buy_home, fair_home, move, reason)

    def _spread_snap(self, ctx: StratContext, detail: dict, buf: deque) -> Decision:
        cfg = self.config
        window = [q for q in buf if q[0] >= ctx.now - cfg.micro_window_secs]
        if len(window) < 2:
            return self._reject(detail, "no_spread_shock")
        peak_ts, peak_mid, peak_spread = max(window[:-1], key=lambda q: q[2])
        cur_spread = ctx.quote.home_spread
        if peak_spread < cfg.micro_spread_shock or cur_spread >= cfg.micro_spread_shock:
            return self._reject(detail, "no_spread_shock",
                                margin=peak_spread - cfg.micro_spread_shock)
        reprice = ctx.quote.home_mid - peak_mid
        detail["move"] = reprice
        if abs(reprice) < cfg.micro_min_reprice:
            return self._reject(detail, "small_reprice",
                                margin=abs(reprice) - cfg.micro_min_reprice)
        reason = (f"spread_snap shock {peak_spread:.3f} -> {cur_spread:.3f}; "
                  f"reprice {reprice:+.3f}")
        return self._finish(ctx, detail, reprice > 0, reprice, reason)

    def _stale_reprice(self, ctx: StratContext, detail: dict,
                       prev_quote: tuple | None) -> Decision:
        cfg = self.config
        if prev_quote is None:
            return self._reject(detail, "no_gap")
        gap = ctx.quote.ts - prev_quote[0]
        if gap < cfg.micro_window_secs:
            return self._reject(detail, "no_gap", margin=gap - cfg.micro_window_secs)
        reprice = ctx.quote.home_mid - prev_quote[1]
        detail["move"] = reprice
        if abs(reprice) < cfg.micro_min_reprice:
            return self._reject(detail, "small_reprice",
                                margin=abs(reprice) - cfg.micro_min_reprice)
        reason = f"stale_reprice after {gap:.0f}s book gap; reprice {reprice:+.3f}"
        return self._finish(ctx, detail, reprice > 0, reprice, reason)

    def _pregame_drift(self, ctx: StratContext, detail: dict) -> Decision:
        cfg = self.config
        key = ctx.market.key
        if key in self._fired:
            return self._reject(detail, "already_fired")
        if ctx.game_state.inning > 1:
            self._fired.add(key)     # missed the window for this game
            return self._reject(detail, "too_late")
        past = ctx.history.price_ago(1800.0)
        if past is None:
            return self._reject(detail, "no_pregame_history")
        drift = ctx.history.last - past
        detail["move"] = drift
        if abs(drift) < cfg.micro_min_reprice:
            return self._reject(detail, "small_reprice",
                                margin=abs(drift) - cfg.micro_min_reprice)
        self._fired.add(key)
        reason = f"pregame_drift {drift:+.3f} over the last 30 pregame minutes"
        return self._finish(ctx, detail, drift > 0, drift, reason)

    def manage(self, ctx: StratContext, positions: list[Position]) -> list[ExitIntent]:
        return self._manage_positions(ctx, positions, None)


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


STRATEGY_KINDS: dict[str, type[Strategy]] = {
    "fade": FadeStrategy,
    "state_residual": StateResidualStrategy,
    "market_anchored": MarketAnchoredStrategy,
    "momentum": MomentumStrategy,
    "event_reaction": EventReactionStrategy,
    "extreme_hold": ExtremeHoldStrategy,
    "settlement_hold": SettlementHoldStrategy,
    "calibration_cell": CalibrationCellStrategy,
    "microstructure": MicrostructureStrategy,
}


def build_strategies(cfg) -> list[Strategy]:
    """Instantiate the strategy registry from config (or the built-in default)."""
    entries = cfg.strategies or DEFAULT_REGISTRY
    strats: list[Strategy] = []
    for entry in entries:
        kind = entry.get("kind", "fade")
        name = entry["name"]
        if kind == "ai_shadow":
            if not cfg.ai.enabled:
                continue
            base = next((s for s in strats if s.name == entry["base"]), None)
            if base is None:
                raise ValueError(
                    f"ai_shadow strategy {name!r} references base "
                    f"{entry['base']!r}, which is not defined earlier in the registry"
                )
            strats.append(AIShadowStrategy(name, entry.get("version", "v1"),
                                           base, cfg.ai))
            continue
        cls = STRATEGY_KINDS.get(kind)
        if cls is None:
            raise ValueError(f"unknown strategy kind: {kind!r}")
        sconf = dataclasses.replace(cfg.strategy, **entry.get("overrides", {}))
        strats.append(cls(name, entry.get("version", "v1"), sconf))
    return strats
