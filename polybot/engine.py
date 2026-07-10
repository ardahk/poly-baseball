"""Main trading loop."""
from __future__ import annotations

import logging
import time
import hashlib
import json
import os
from collections import deque
from dataclasses import asdict

from . import mlb, pmus, strategy
from .broker import PaperBroker
from .config import Config
from .dashboard import TerminalDashboard
from .journal import Journal
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
        self.strategies = [s.name for s in self.strategy_objs]
        log.info("strategies: %s", self.strategies)

        self.started_at = time.time()
        self.dashboard = TerminalDashboard(enabled=dashboard)
        self.events: deque[str] = deque(maxlen=10)

        self.broker = PaperBroker(
            self.strategies, cfg.risk.starting_cash, slippage=0.0,
            taker_fee_theta=cfg.engine.paper_taker_fee_theta,
        )
        self.journal = Journal(cfg.engine.db_path)
        config_hash = hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.run_id = self.journal.start_run(
            "paper", config_hash, os.environ.get("POLYBOT_CODE_REVISION", "unknown")
        )
        self._restore_paper_account()
        self.journal.save_paper_state(self.broker)
        self.session_realized = dict(self.broker.realized)
        self.risk = RiskManager(cfg.risk, self.strategies)
        self.mlb = mlb.MLBClient()

        self.markets: dict[str, Market] = {}
        self.histories: dict[str, PriceHistory] = {}       # market_key -> home-token history
        self.game_states: dict[int, GameState] = {}
        self.cooldowns: dict[tuple[str, str], float] = {}  # (strategy, market) -> reopen ts
        self.latest_prices: dict[str, float] = {}          # token -> price
        self.latest_quotes: dict[str, MarketQuote] = {}    # market_key -> latest BBO
        self.funnel: dict[str, int] = {}                   # entry-gate reject counters
        # Signals awaiting counterfactual snapshots: each is
        # {signal_id, token, market_key, born, remaining:set[int]}.
        self.pending_cf: list[dict] = []

        self._last_discovery = 0.0
        self._last_game_poll = 0.0
        self._last_equity = 0.0
        self._last_status_log = 0.0

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
                self._maybe_poll_games()
                self._poll_prices(quotes)
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
                self.game_states[market.game_pk] = gs
                if gs != previous:
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
        for market in self.markets.values():
            gs = self.game_states.get(market.game_pk)
            if not gs or not gs.is_live:
                continue
            book = quotes.get(market.slug)
            if book is None:
                continue
            if book.two_sided:
                quote = self._to_home_quote(market, book)
                self.journal.record_price(market, quote)
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
                self.journal.record_mark(market, home_mid, book.long_last)
                # A mark is useful for observability only. It must never be
                # paired with a previous BBO to create an executable order.
                self.latest_quotes.pop(market.key, None)
            else:
                self.latest_quotes.pop(market.key, None)
                continue
            self.histories[market.key].add(home_mid)

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
                if quote is None or now - quote.ts > self.cfg.strategy.max_quote_age_secs:
                    continue
                gs = self.game_states.get(market.game_pk)
                ctx = StratContext(market, self.histories[market.key], gs, quote, now)
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
        cooldown = self.cfg.strategy.stop_loss_cooldown_secs \
            if reason.startswith("stop loss") else self.cfg.strategy.cooldown_secs
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
        scfg = self.cfg.strategy
        now = time.time()
        day_start, day_end, day_key = day_bounds(timezone=self.cfg.engine.report_timezone)
        rows: list[dict] = []
        for market in self.markets.values():
            gs = self.game_states.get(market.game_pk)
            if not gs or not gs.is_live:
                continue  # not counted: pending/final markets aren't candidates
            quote = self.latest_quotes.get(market.key)
            if quote is None or now - quote.ts > scfg.max_quote_age_secs:
                self._bump("stale_quote")
                quote_age = (now - quote.ts) if quote else None
                rows.append(self._decision_row(
                    market, gs, "engine", "stale_quote", quote=quote,
                    quote_age=quote_age,
                    margin=(scfg.max_quote_age_secs - quote_age)
                    if quote_age is not None else None,
                    ts=now,
                ))
                continue
            if quote.home_spread > scfg.max_spread:
                self._bump("wide_spread")
                rows.append(self._decision_row(
                    market, gs, "engine", "wide_spread", quote=quote,
                    quote_age=now - quote.ts,
                    margin=scfg.max_spread - quote.home_spread,
                    ts=now,
                ))
                continue
            ctx = StratContext(market, self.histories[market.key], gs, quote, now)
            for strat in self.strategy_objs:
                self._evaluate_strategy(strat, ctx, gs, quote, now,
                                        day_start, day_end, day_key, rows)
        self.journal.record_decisions(rows)

    def _evaluate_strategy(self, strat, ctx, gs, quote, now,
                           day_start, day_end, day_key, rows):
        market = ctx.market
        d = strat.evaluate(ctx)
        self._bump(d.outcome)
        ev = d.evaluation or (d.intent.evaluation if d.intent else None)
        rows.append(self._decision_row(
            market, gs, "strategy", d.outcome, strategy_name=strat.name,
            ev=ev, quote=quote, quote_age=now - quote.ts, ts=now,
        ))
        if d.signal_candidate and ev is not None and ev.signal is not None:
            self._register_signal(strat.name, ctx, ev)

        intent = d.intent
        if intent is None:
            return
        if time.time() < self.cooldowns.get((strat.name, market.key), 0):
            self._bump("cooldown")
            rows.append(self._decision_row(
                market, gs, "post_signal", "cooldown", strategy_name=strat.name,
                ev=ev, quote=quote, quote_age=now - quote.ts, ts=now))
            return
        if intent.token in self.broker.positions[strat.name]:
            self._bump("already_open")
            rows.append(self._decision_row(
                market, gs, "post_signal", "already_open", strategy_name=strat.name,
                ev=ev, quote=quote, quote_age=now - quote.ts, ts=now))
            return
        stake = self._stake_for_intent(intent, quote)
        daily_realized = self.journal.realized_pnl(strat.name, day_start, day_end)
        if not self.risk.can_open(
            self.broker, strat.name, market.key, stake,
            daily_realized=daily_realized, day_key=day_key,
        ):
            self._bump("risk_blocked")
            rows.append(self._decision_row(
                market, gs, "post_signal", "risk_blocked", strategy_name=strat.name,
                ev=ev, quote=quote, quote_age=now - quote.ts, ts=now))
            return
        entry_price = ctx.entry_price(intent.token)
        reason = f"{intent.reason}; spread {quote.home_spread:.3f}; stake ${stake:.0f}"
        pos = self.broker.open(strat.name, market.key, intent.token,
                               intent.side_team, entry_price, stake)
        if pos:
            self._bump("opened")
            rows.append(self._decision_row(
                market, gs, "post_signal", "opened", strategy_name=strat.name,
                ev=ev, quote=quote, quote_age=now - quote.ts, ts=now))
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
                ev=ev, quote=quote, quote_age=now - quote.ts, ts=now))

    def _register_signal(self, strat_name: str, ctx: StratContext, ev) -> None:
        """Log a signal-grade candidate and queue its counterfactual horizons."""
        sig = ev.signal
        sid = self.journal.record_signal(
            strategy=strat_name, market=ctx.market.key, token=sig.token,
            side_team=ev.side_team, entry_price=ctx.entry_price(sig.token),
            fair=ev.fair, edge=ev.edge, move=ev.move, spread=ctx.quote.home_spread,
            inning=ctx.game_state.inning, is_top=int(ctx.game_state.is_top),
            home_score=ctx.game_state.home_score, away_score=ctx.game_state.away_score,
            commit=False,
        )
        self.pending_cf.append({
            "signal_id": sid, "token": sig.token, "market_key": ctx.market.key,
            "born": ctx.now, "remaining": set(HORIZONS),
        })

    def _flush_counterfactuals(self):
        """Snapshot each pending signal's executable price at every elapsed horizon."""
        now = time.time()
        still: list[dict] = []
        dirty = False
        for sig in self.pending_cf:
            due = sorted(h for h in sig["remaining"] if now - sig["born"] >= h)
            for horizon in due:
                market = self.markets.get(sig["market_key"])
                quote = self.latest_quotes.get(sig["market_key"])
                fresh = quote is not None and \
                    now - quote.ts <= self.cfg.strategy.max_quote_age_secs
                if market is not None and fresh:
                    bid = executable_bid(market, quote, sig["token"])
                    ask = executable_ask(market, quote, sig["token"])
                    self.journal.record_counterfactual(
                        sig["signal_id"], horizon, exec_bid=bid, exec_ask=ask,
                        mid=(bid + ask) / 2, two_sided=1, spread=ask - bid,
                        commit=False)
                else:
                    hist = self.histories.get(sig["market_key"])
                    self.journal.record_counterfactual(
                        sig["signal_id"], horizon, exec_bid=None, exec_ask=None,
                        mid=hist.last if hist else None, two_sided=0, spread=None,
                        commit=False)
                dirty = True
            sig["remaining"].difference_update(due)
            if sig["remaining"]:
                still.append(sig)
        self.pending_cf = still
        if dirty:
            self.journal.conn.commit()

    @staticmethod
    def _entry_price(market: Market, quote: MarketQuote, token: str) -> float:
        """Executable taker buy price for a normalized token side."""
        return executable_ask(market, quote, token)

    def _stake_for_intent(self, intent, quote: MarketQuote) -> float:
        rcfg = self.cfg.risk
        scfg = self.cfg.strategy
        if intent.edge >= rcfg.strong_stake_min_edge \
                and quote.home_spread <= scfg.strong_stake_max_spread:
            return rcfg.strong_stake_usd
        return rcfg.stake_usd

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
