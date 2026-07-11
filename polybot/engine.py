"""Main trading loop."""
from __future__ import annotations

import logging
import time
import hashlib
import json
from collections import deque
from dataclasses import asdict

from . import mlb, pmus, provenance, strategy
from .broker import PaperBroker
from .config import Config
from .dashboard import TerminalDashboard
from .journal import Journal
from .model_features import ModelHistory, state_signature
from .models import GameState, Market, MarketQuote, Position
from .risk import RiskManager
from .strategies import (
    HORIZONS,
    StratContext,
    build_strategies,
    executable_ask,
    executable_bid,
)
from .timeframe import day_bounds
from .state_model import load_probability_model
from .volatility import PriceHistory
log = logging.getLogger(__name__)


class Engine:
    def __init__(self, cfg: Config, dashboard: bool = False):
        if cfg.engine.live:
            raise RuntimeError("Live trading is disabled during Phase 0 validation")
        self.cfg = cfg
        # Frozen strategy registry drives the whole loop; `self.strategies` keeps
        # the plain names that broker / risk / journal are keyed by.
        self.strategy_objs = build_strategies(cfg)
        self.strategy_by_name = {s.name: s for s in self.strategy_objs}
        self.strategies = [s.name for s in self.strategy_objs]
        self.state_probability, self.state_model_label = load_probability_model(
            cfg.engine.state_model_path
        )
        log.info("strategies: %s", self.strategies)

        self.started_at = time.time()
        self.dashboard = TerminalDashboard(enabled=dashboard)
        self.events: deque[str] = deque(maxlen=10)

        self.broker = PaperBroker(
            self.strategies, cfg.risk.starting_cash, slippage=0.0,
            taker_fee_theta=cfg.engine.paper_taker_fee_theta,
        )
        self.journal = Journal(cfg.engine.db_path)
        self.run_id = self.journal.start_run(
            "paper", provenance.config_hash(cfg), provenance.code_revision()
        )
        self._record_strategy_registry()
        self.risk = RiskManager(cfg.risk, self.strategies)
        self.mlb = mlb.MLBClient()

        self.markets: dict[str, Market] = {}
        self.histories: dict[str, PriceHistory] = {}       # market_key -> home-token history
        self.model_histories: dict[str, ModelHistory] = {}
        self.game_states: dict[int, GameState] = {}
        self.cooldowns: dict[tuple[str, str], float] = {}  # (strategy, market) -> reopen ts
        self.latest_prices: dict[str, float] = {}          # token -> price
        self.latest_quotes: dict[str, MarketQuote] = {}    # market_key -> latest BBO
        self.funnel: dict[str, int] = {}                   # entry-gate reject counters
        # Signals awaiting counterfactual snapshots: each is
        # {signal_id, token, market_key, born, remaining:set[int]}.
        self.pending_cf: list[dict] = []
        # Active signal episodes: (strategy, market, token) -> last-fired ts. New
        # episodes register a signal; continuations refresh the timestamp only.
        self.active_signals: dict[tuple[str, str, str], float] = {}

        self._last_discovery = 0.0
        self._last_game_poll = 0.0
        self._last_equity = 0.0
        self._last_status_log = 0.0

        # Restore prior paper ledger + in-flight counterfactuals now that all
        # runtime state exists. Restore also rehydrates the markets of any
        # orphaned positions so their games get polled to settlement.
        self._restore_paper_account()
        self._restore_pending_cf()
        self.journal.save_paper_state(self.broker)
        self.session_realized = dict(self.broker.realized)

    def _restore_paper_account(self) -> None:
        accounts, saved_positions = self.journal.paper_state(self.strategies)
        positions = [
            Position(
                strategy=row["strategy"], market_key=row["market"], token=row["token"],
                team=row["team"], qty=row["qty"], entry_price=row["entry_price"],
                entry_fee=row["entry_fee"],
                opened_at=row["opened_at"], trade_id=row["trade_id"],
            )
            for row in saved_positions
        ]
        if self.broker.restore(accounts, positions):
            restored_positions = sum(len(self.broker.open_positions(s)) for s in self.strategies)
            self._event("restored paper account (%d position%s)", restored_positions,
                        "s" if restored_positions != 1 else "")
            self._rehydrate_markets_for_open_positions()

    def _rehydrate_markets_for_open_positions(self) -> None:
        """Reload markets for restored positions so orphaned games settle.

        Discovery skips closed markets, so a position restored after its game
        ended would never re-enter `self.markets`, never be polled, and never
        settle — locking cash forever. Rebuild those markets from the journal.
        """
        needed = {
            pos.market_key
            for strat in self.strategies
            for pos in self.broker.open_positions(strat)
            if pos.market_key not in self.markets
        }
        for market in self.journal.markets_by_slugs(sorted(needed)):
            self.markets[market.key] = market
            self.histories.setdefault(market.key, PriceHistory(self.cfg.strategy.flip_band))
            self.model_histories.setdefault(market.key, self._new_model_history())
        missing = needed - set(self.markets)
        if missing:
            self._event("warning: %d restored position(s) reference unknown markets: %s",
                        len(missing), ", ".join(sorted(missing)))

    def _restore_pending_cf(self) -> None:
        """Reload signals still inside their counterfactual window after a restart."""
        for row in self.journal.pending_counterfactuals():
            remaining = set(HORIZONS) - row["done"]
            if remaining:
                self.pending_cf.append({
                    "signal_id": row["signal_id"], "token": row["token"],
                    "market_key": row["market_key"], "born": row["born"],
                    "remaining": remaining,
                })
            else:
                self.journal.delete_pending_cf([row["signal_id"]])

    def _record_strategy_registry(self) -> None:
        """Freeze each strategy's version + config hash for this run's provenance."""
        entries = []
        for strat in self.strategy_objs:
            frozen = {"strategy": asdict(strat.config), "state_model": self.state_model_label}
            config_json = json.dumps(frozen, sort_keys=True, separators=(",", ":"))
            payload = f"{strat.kind}:{strat.version}:{config_json}"
            entries.append({
                "strategy": strat.name, "version": strat.version, "kind": strat.kind,
                "config_hash": hashlib.sha256(payload.encode()).hexdigest(),
                "config_json": config_json,
            })
        self.journal.record_strategy_registry(entries)

    def _save_paper_account(self) -> None:
        self.journal.save_paper_state(self.broker)

    # ------------------------------------------------------------------ loop

    def run(self):
        mode = "LIVE" if self.cfg.engine.live else "PAPER"
        self._event("engine starting (%s mode, strategies: %s)", mode, self.strategies)
        self.dashboard.start()
        try:
            while True:
                tick_start = time.time()
                # One bulk request refreshes every market's BBO per tick.
                found, quotes = pmus.fetch_mlb_book()
                self._maybe_discover(found)
                # The bulk book is timestamped before the MLB request below.
                # Apply it first so a pre-state quote cannot be evaluated as if
                # it were observed after that state transition.
                self._poll_prices(quotes)
                self._maybe_poll_games()
                self._manage_exits()
                self._look_for_entries()
                self._flush_counterfactuals()
                self._maybe_snapshot_equity()
                self._maybe_log_status()
                self.dashboard.render(self)
                elapsed = time.time() - tick_start
                time.sleep(max(0.0, self.cfg.engine.poll_interval_secs - elapsed))
        except KeyboardInterrupt:
            self._event("shutting down")
        finally:
            for strat in self.strategy_objs:
                strat.close()
            self.dashboard.close()
            self.journal.close()

    # ------------------------------------------------------------- discovery

    def _maybe_discover(self, found: list[Market]):
        """Match newly seen markets to MLB games (throttled: hits the MLB API)."""
        now = time.time()
        new = [m for m in found if m.key not in self.markets]
        if not new:
            return
        if now - self._last_discovery < self.cfg.engine.discovery_interval_secs \
                and self.markets:
            return
        self._last_discovery = now
        games = self.mlb.todays_games()
        mlb.match_markets_to_games(new, games)
        for m in new:
            if m.game_pk:
                self.markets[m.key] = m
                self.journal.record_market(m)
                self.histories[m.key] = PriceHistory(self.cfg.strategy.flip_band)
                self.model_histories[m.key] = self._new_model_history()
                self._event("tracking: %s (game %s)", m.question, m.game_pk)

    def _maybe_poll_games(self):
        now = time.time()
        if now - self._last_game_poll < self.cfg.engine.game_state_interval_secs:
            return
        self._last_game_poll = now
        seen: set[int] = set()
        for market in self.markets.values():
            if not market.game_pk or market.game_pk in seen:
                continue
            seen.add(market.game_pk)
            if not self._should_poll_game_state(market, now):
                continue
            gs = self.mlb.game_state(market.game_pk)
            if gs:
                previous = self.game_states.get(market.game_pk)
                if gs != previous:
                    for linked in self.markets.values():
                        if linked.game_pk == gs.game_pk:
                            model_history = self.model_histories.setdefault(
                                linked.key, self._new_model_history()
                            )
                            if model_history.observe_state(gs, gs.received_at):
                                anchored_names = {
                                    s.name for s in self.strategy_objs
                                    if s.kind in {"state_residual", "market_anchored"}
                                }
                                self.active_signals = {
                                    key: value for key, value in self.active_signals.items()
                                    if not (key[0] in anchored_names and key[1] == linked.key)
                                }
                                view = None if model_history.last_price is None else \
                                    model_history.market_view(
                                        model_history.last_price, gs.received_at,
                                        beta=self.cfg.strategy.residual_beta,
                                    )
                                quote = self.latest_quotes.get(linked.key)
                                self.journal.record_model_observation(
                                    model=self.state_model_label, market=linked.key,
                                    game_state=gs,
                                    state_signature=repr(state_signature(gs)),
                                    model_home=model_history.current_model,
                                    pregame_anchor=(view.anchor_price if view else None),
                                    anchored_fair=(view.fair_home if view else None),
                                    home_mid=model_history.last_price,
                                    spread=(quote.home_spread if quote else None),
                                    ts=gs.received_at, commit=False,
                                )
                    self.game_states[market.game_pk] = gs
                    self.journal.record_game_state(gs)
                    if gs.is_final and not (previous and previous.is_final):
                        self._settle_final_game(gs)

    def _settle_final_game(self, gs: GameState) -> None:
        """Paper contracts resolve at the official final outcome, never a stale BBO."""
        if gs.home_score == gs.away_score:
            self._event("final game %s is tied; awaiting official resolution", gs.game_pk)
            return
        with self.journal.conn:
            for market in self.markets.values():
                if market.game_pk != gs.game_pk:
                    continue
                for strat in self.strategies:
                    for pos in list(self.broker.open_positions(strat)):
                        if pos.market_key != market.key:
                            continue
                        home_won = gs.home_score > gs.away_score
                        won = (pos.token == market.home_token and home_won) or (
                            pos.token == market.away_token and not home_won
                        )
                        result = self.broker.settle(strat, pos.token, 1.0 if won else 0.0)
                        if result is None:
                            continue
                        position, fill, pnl = result
                        self.journal.record_close(
                            strat, position.market_key, position.team, position.token,
                            position.qty, fill, pnl, position.pnl_pct(fill),
                            "official game settlement", trade_id=position.trade_id,
                            intended_price=fill, slippage=0.0, exit_kind="game_final",
                            fee_usd=0.0, commit=False,
                        )
                        self._event("[%s] SETTLE %s @ %.3f ($%+.2f)", strat, position.team, fill, pnl)
            self.journal.save_paper_state(self.broker, commit=False)

    def _should_poll_game_state(self, market: Market, now: float) -> bool:
        current = self.game_states.get(market.game_pk)
        if current and current.is_final:
            return False
        if current and current.is_live:
            return True
        if market.start_time is None:
            return True
        return now >= market.start_time - self.cfg.engine.pregame_game_state_window_secs

    def _poll_prices(self, quotes: dict[str, pmus.BookQuote]):
        wrote = False
        now = time.time()
        for market in self.markets.values():
            gs = self.game_states.get(market.game_pk)
            live = bool(gs and gs.is_live)
            pregame = not live and market.start_time is not None and \
                now >= market.start_time - self.cfg.engine.pregame_game_state_window_secs
            if not live and not pregame:
                continue
            book = quotes.get(market.slug)
            if book is None:
                continue
            history = self.histories[market.key]
            model_history = self.model_histories.setdefault(
                market.key, self._new_model_history()
            )
            last_observed = model_history.last_price_ts
            if last_observed is None and history.samples:
                last_observed = history.samples[-1][0]
            if last_observed is not None \
                    and book.received_at - last_observed \
                    > self.cfg.engine.history_gap_reset_secs:
                self.histories[market.key] = PriceHistory(self.cfg.strategy.flip_band)
                history = self.histories[market.key]
                model_history.reset_rolling()
                if live:
                    model_history.observe_state(gs, gs.received_at)
                for strat in self.strategy_objs:
                    strat.reset_market(market.key)
                self.latest_quotes.pop(market.key, None)
                self._event("history reset after data gap: %s", market.question)
            if book.two_sided:
                quote = self._to_home_quote(market, book)
                self.journal.record_price(market, quote, commit=False)
                self.latest_quotes[market.key] = quote
                home_mid = quote.home_mid
                # Long positions liquidate at their bid; short positions at
                # the synthetic short bid (1 - long ask), not midpoint.
                self.latest_prices[market.home_token] = quote.home_bid
                self.latest_prices[market.away_token] = 1.0 - quote.home_ask
            elif book.long_last is not None:
                # One-sided book (common mid-play): keep the price history
                # alive from the mark price so move detection doesn't freeze,
                # but leave latest_quotes stale so entries stay blocked.
                home_mid = book.long_last if market.home_is_long \
                    else 1.0 - book.long_last
                self.journal.record_mark(market, home_mid, book.long_last, commit=False)
                # A mark is useful for observability only. It must never be
                # paired with a previous BBO to create an executable order.
                self.latest_quotes.pop(market.key, None)
            else:
                self.latest_quotes.pop(market.key, None)
                continue
            model_history.add_price(
                home_mid, book.received_at,
                pregame_eligible=(
                    market.start_time is not None and book.received_at < market.start_time
                ),
            )
            if live:
                history.add(home_mid, ts=book.received_at)
            wrote = True
        if wrote:
            self.journal.conn.commit()

    @staticmethod
    def _to_home_quote(market: Market, book: pmus.BookQuote) -> MarketQuote:
        if market.home_is_long:
            home_bid, home_ask = book.long_bid, book.long_ask
        else:
            home_bid, home_ask = 1.0 - book.long_ask, 1.0 - book.long_bid
        return MarketQuote(
            market_key=market.key,
            home_bid=home_bid, home_ask=home_ask,
            long_bid=book.long_bid, long_ask=book.long_ask,
            ts=book.received_at,
            source_ts=book.source_ts,
        )

    # ----------------------------------------------------------------- exits

    def _manage_exits(self):
        now = time.time()
        for strat in self.strategy_objs:
            for pos in self.broker.open_positions(strat.name):
                market = self.markets.get(pos.market_key)
                if market is None:
                    continue
                quote = self.latest_quotes.get(market.key)
                if quote is None or now - quote.ts > strat.config.max_quote_age_secs \
                        or (gs := self.game_states.get(market.game_pk)) is None \
                        or quote.ts < gs.received_at:
                    continue
                ctx = StratContext(market, self.histories[market.key], gs, quote, now,
                                   fee_theta=self.cfg.engine.paper_taker_fee_theta,
                                   model_history=self.model_histories.get(market.key))
                for ex in strat.manage(ctx, [pos]):
                    self._close(strat.name, ex.position, ex.price, ex.reason, fair=ex.fair)

    def _close(self, strat: str, pos, price: float, reason: str,
               fair: float | None = None):
        result = self.broker.close(strat, pos.token, price)
        if result is None:
            return
        position, fill, pnl = result
        pnl_pct = position.pnl_pct(fill)
        self._event("[%s] CLOSE %s @ %.3f (%+.1f%%, $%+.2f) - %s",
                    strat, position.team, fill, pnl_pct * 100, pnl, reason)
        with self.journal.conn:
            self.journal.record_close(strat, position.market_key, position.team,
                                      position.token, position.qty, fill, pnl, pnl_pct, reason,
                                      trade_id=position.trade_id, fair=fair,
                                      intended_price=price, slippage=fill - price,
                                      exit_kind=strategy.exit_kind(reason),
                                      fee_usd=self.broker.last_fee[strat], commit=False)
            self.journal.save_paper_state(self.broker, commit=False)
        scfg = self.strategy_by_name[strat].config
        cooldown = scfg.stop_loss_cooldown_secs \
            if reason.startswith("stop loss") else scfg.cooldown_secs
        self.cooldowns[(strat, position.market_key)] = time.time() + cooldown

    # --------------------------------------------------------------- entries

    def _bump(self, reason: str) -> None:
        self.funnel[reason] = self.funnel.get(reason, 0) + 1

    def _decision_row(self, market: Market, gs: GameState, stage: str, outcome: str,
                      *, strategy_name: str | None = None,
                      ev=None, quote: MarketQuote | None = None,
                      quote_age: float | None = None,
                      margin: float | None = None, ts: float | None = None) -> dict:
        history = self.histories.get(market.key)
        return {
            "ts": time.time() if ts is None else ts,
            "market": market.key,
            "strategy": strategy_name,
            "stage": stage,
            "outcome": outcome,
            "mid": ev.mid if ev else (history.last if history else None),
            "move": ev.move if ev else None,
            "flips": ev.flips if ev else (history.flips if history else None),
            "realized_vol": ev.realized_vol if ev else (
                history.realized_vol(self.cfg.strategy.vol_window) if history else None
            ),
            "fair_home": ev.fair_home if ev else None,
            "anchor_price": ev.anchor_price if ev else None,
            "anchor_model": ev.anchor_model if ev else None,
            "model_delta": ev.model_delta if ev else None,
            "residual": ev.residual if ev else None,
            "anchor_age": ev.anchor_age if ev else None,
            "side": ev.side_team if ev else None,
            "price": ev.price if ev else None,
            "fair": ev.fair if ev else None,
            "edge": ev.edge if ev else None,
            "spread": quote.home_spread if quote else None,
            "quote_age": quote_age,
            "margin": ev.margin if margin is None and ev else margin,
            "inning": gs.inning,
            "is_top": int(gs.is_top),
            "home_score": gs.home_score,
            "away_score": gs.away_score,
        }

    def _look_for_entries(self):
        now = time.time()
        day_start, day_end, day_key = day_bounds(timezone=self.cfg.engine.report_timezone)
        rows: list[dict] = []
        for market in self.markets.values():
            gs = self.game_states.get(market.game_pk)
            if not gs or not gs.is_live:
                continue  # not counted: pending/final markets aren't candidates
            # Always show the market to every strategy — even on a stale or
            # wide-spread book. Spreads blow out during exactly the sharp moves
            # a fade wants, so each strategy applies its OWN executability gate
            # (and we still capture the signal + counterfactuals). The quote may
            # be None (one-sided book); the price history is kept alive by marks.
            quote = self.latest_quotes.get(market.key)
            awaiting_post_state_quote = quote is not None and quote.ts < gs.received_at
            if awaiting_post_state_quote:
                quote = None
            ctx = StratContext(market, self.histories[market.key], gs, quote, now,
                               fee_theta=self.cfg.engine.paper_taker_fee_theta,
                               model_history=self.model_histories.get(market.key))
            for strat in self.strategy_objs:
                self._evaluate_strategy(strat, ctx, gs, quote, now,
                                        day_start, day_end, day_key, rows,
                                        register_signal=not awaiting_post_state_quote)
        self.journal.record_decisions(rows)

    def _evaluate_strategy(self, strat, ctx, gs, quote, now,
                           day_start, day_end, day_key, rows,
                           register_signal=True):
        market = ctx.market
        quote_age = None if quote is None else now - quote.ts
        d = strat.evaluate(ctx)
        self._bump(d.outcome)
        ev = d.evaluation or (d.intent.evaluation if d.intent else None)
        rows.append(self._decision_row(
            market, gs, "strategy", d.outcome, strategy_name=strat.name,
            ev=ev, quote=quote, quote_age=quote_age, ts=now,
        ))
        if register_signal and d.signal_candidate and ev is not None and ev.signal is not None:
            self._maybe_register_signal(strat, ctx, d)

        intent = d.intent
        if intent is None:
            return
        if time.time() < self.cooldowns.get((strat.name, market.key), 0):
            self._bump("cooldown")
            rows.append(self._decision_row(
                market, gs, "post_signal", "cooldown", strategy_name=strat.name,
                ev=ev, quote=quote, quote_age=quote_age, ts=now))
            return
        if intent.token in self.broker.positions[strat.name]:
            self._bump("already_open")
            rows.append(self._decision_row(
                market, gs, "post_signal", "already_open", strategy_name=strat.name,
                ev=ev, quote=quote, quote_age=quote_age, ts=now))
            return
        stake = self._stake_for_intent(strat, intent, quote)
        daily_realized = self.journal.realized_pnl(strat.name, day_start, day_end)
        if not self.risk.can_open(
            self.broker, strat.name, market.key, stake,
            daily_realized=daily_realized, day_key=day_key,
        ):
            self._bump("risk_blocked")
            rows.append(self._decision_row(
                market, gs, "post_signal", "risk_blocked", strategy_name=strat.name,
                ev=ev, quote=quote, quote_age=quote_age, ts=now))
            return
        entry_price = ctx.entry_price(intent.token)
        reason = f"{intent.reason}; spread {quote.home_spread:.3f}; stake ${stake:.0f}"
        pos = self.broker.open(strat.name, market.key, intent.token,
                               intent.side_team, entry_price, stake)
        if pos:
            self._bump("opened")
            rows.append(self._decision_row(
                market, gs, "post_signal", "opened", strategy_name=strat.name,
                ev=ev, quote=quote, quote_age=quote_age, ts=now))
            self._event("[%s] OPEN %s @ %.3f - %s",
                        strat.name, intent.side_team, pos.entry_price, reason)
            with self.journal.conn:
                self.journal.record_open(strat.name, market.key, intent.side_team,
                                         intent.token, pos.qty, pos.entry_price, reason,
                                         trade_id=pos.trade_id, fair=intent.fair,
                                         edge=intent.edge, move=intent.move,
                                         spread=quote.home_spread,
                                         intended_price=entry_price,
                                         slippage=pos.entry_price - entry_price,
                                         fee_usd=self.broker.last_fee[strat.name],
                                         commit=False)
                self.journal.save_paper_state(self.broker, commit=False)
        else:
            rows.append(self._decision_row(
                market, gs, "post_signal", "open_failed", strategy_name=strat.name,
                ev=ev, quote=quote, quote_age=quote_age, ts=now))

    def _maybe_register_signal(self, strat, ctx: StratContext, d) -> None:
        """Record ONE signal per episode; refresh the episode on continuations.

        A signal condition persisting across ticks would otherwise write a fresh
        row (and six counterfactuals) every 2s — dozens of dependent duplicates
        per real event. We register only when an episode is new, i.e. the signal
        went quiet for `signal_episode_secs` before firing again.
        """
        sig = d.evaluation.signal
        key = (strat.name, ctx.market.key, sig.token)
        last = self.active_signals.get(key)
        self.active_signals[key] = ctx.now
        if last is not None and ctx.now - last < strat.config.signal_episode_secs:
            return  # same episode, already registered
        self._register_signal(strat.name, ctx, d)

    def _register_signal(self, strat_name: str, ctx: StratContext, d) -> None:
        """Log a signal-grade candidate and queue its counterfactual horizons."""
        ev = d.evaluation
        sig = ev.signal
        if ctx.quote is not None:
            entry_price = ctx.entry_price(sig.token)
            fee = ctx.round_trip_fee(entry_price, ev.fair)
            edge = ev.fair - entry_price          # executable gross edge
            net_edge = edge - fee
            spread = ctx.quote.home_spread
        else:
            entry_price = fee = edge = net_edge = spread = None
        gstate = ctx.game_state
        sid = self.journal.record_signal(
            strategy=strat_name, market=ctx.market.key, token=sig.token,
            side_team=ev.side_team, entry_price=entry_price,
            fair=ev.fair, edge=edge, net_edge=net_edge, fee=fee, outcome=d.outcome,
            move=ev.move, spread=spread,
            anchor_price=ev.anchor_price, anchor_model=ev.anchor_model,
            model_delta=ev.model_delta, residual=ev.residual,
            anchor_age=ev.anchor_age,
            inning=gstate.inning, is_top=int(gstate.is_top),
            home_score=gstate.home_score, away_score=gstate.away_score,
            commit=False,
        )
        self.journal.record_pending_cf(sid, sig.token, ctx.market.key, ctx.now,
                                       commit=False)
        self.pending_cf.append({
            "signal_id": sid, "token": sig.token, "market_key": ctx.market.key,
            "born": ctx.now, "remaining": set(HORIZONS),
        })

    def _flush_counterfactuals(self):
        """Snapshot each pending signal's executable price at every elapsed horizon."""
        now = time.time()
        still: list[dict] = []
        completed: list[int] = []
        dirty = False
        for sig in self.pending_cf:
            due = sorted(h for h in sig["remaining"] if now - sig["born"] >= h)
            handled: list[int] = []
            for horizon in due:
                market = self.markets.get(sig["market_key"])
                quote = self.latest_quotes.get(sig["market_key"])
                target_ts = sig["born"] + horizon
                lag = now - target_ts
                timely = lag <= self.cfg.engine.counterfactual_max_lag_secs
                hist = self.histories.get(sig["market_key"])
                post_target_mark = bool(
                    hist and hist.samples and hist.samples[-1][0] >= target_ts
                )
                fresh = quote is not None and \
                    now - quote.ts <= self.cfg.strategy.max_quote_age_secs \
                    and quote.ts >= target_ts
                if timely and not fresh and not post_target_mark:
                    continue  # wait for the first observation after the target
                if market is not None and fresh and timely:
                    bid = executable_bid(market, quote, sig["token"])
                    ask = executable_ask(market, quote, sig["token"])
                    self.journal.record_counterfactual(
                        sig["signal_id"], horizon, exec_bid=bid, exec_ask=ask,
                        mid=(bid + ask) / 2, two_sided=1, spread=ask - bid,
                        commit=False)
                else:
                    self.journal.record_counterfactual(
                        sig["signal_id"], horizon, exec_bid=None, exec_ask=None,
                        mid=(hist.last if timely and post_target_mark else None),
                        two_sided=0, spread=None,
                        commit=False)
                handled.append(horizon)
                dirty = True
            sig["remaining"].difference_update(handled)
            if sig["remaining"]:
                still.append(sig)
            else:
                completed.append(sig["signal_id"])
        self.pending_cf = still
        if completed:
            self.journal.delete_pending_cf(completed, commit=False)
        if dirty:
            self.journal.conn.commit()

    @staticmethod
    def _entry_price(market: Market, quote: MarketQuote, token: str) -> float:
        """Executable taker buy price for a normalized token side."""
        return executable_ask(market, quote, token)

    def _stake_for_intent(self, strat, intent, quote: MarketQuote) -> float:
        rcfg = self.cfg.risk
        scfg = strat.config
        if intent.edge >= rcfg.strong_stake_min_edge \
                and quote.home_spread <= scfg.strong_stake_max_spread:
            return rcfg.strong_stake_usd
        return rcfg.stake_usd

    def _new_model_history(self) -> ModelHistory:
        return ModelHistory(
            self.state_probability,
            anchor_lookback_secs=self.cfg.strategy.residual_anchor_lookback_secs,
        )

    # ---------------------------------------------------------------- equity

    def _maybe_snapshot_equity(self):
        now = time.time()
        if now - self._last_equity < self.cfg.engine.equity_snapshot_secs:
            return
        self._last_equity = now
        for strat in self.strategies:
            eq = self.broker.equity(strat, self.latest_prices)
            self.journal.record_equity(strat, eq, self.broker.cash[strat],
                                       len(self.broker.open_positions(strat)))

    def _maybe_log_status(self):
        now = time.time()
        if now - self._last_status_log < self.cfg.engine.status_log_interval_secs:
            return
        self._last_status_log = now
        live_games = sum(
            1 for m in self.markets.values()
            if (gs := self.game_states.get(m.game_pk)) and gs.is_live
        )
        recent_quotes = sum(
            1 for q in self.latest_quotes.values()
            if now - q.ts <= max(30.0, self.cfg.engine.poll_interval_secs * 5)
        )
        parts = []
        for strat in self.strategies:
            eq = self.broker.equity(strat, self.latest_prices)
            parts.append(
                f"{strat}: equity=${eq:.2f}, cash=${self.broker.cash[strat]:.2f}, "
                f"open={len(self.broker.open_positions(strat))}"
            )
        funnel = " ".join(f"{k}={v}" for k, v in sorted(self.funnel.items())) or "none"
        log.info(
            "status: tracked=%d live=%d recent_bbo=%d %s | entry funnel: %s",
            len(self.markets), live_games, recent_quotes, "; ".join(parts), funnel,
        )

    def _event(self, message: str, *args):
        text = message % args if args else message
        self.events.append(text)
        self.dashboard.record(text)
        log.info(message, *args)
