"""Main trading loop."""
from __future__ import annotations

import logging
import time
from collections import deque

from . import mlb, pmus, strategy
from .ai_judge import AIJudge
from .broker import LiveBroker, PaperBroker
from .config import Config
from .dashboard import TerminalDashboard
from .journal import Journal
from .models import GameState, Market, MarketQuote
from .risk import RiskManager
from .volatility import PriceHistory
log = logging.getLogger(__name__)

MATH = "math"
AI = "ai"


class Engine:
    def __init__(self, cfg: Config, dashboard: bool = False):
        self.cfg = cfg
        self.judge = AIJudge(cfg.ai)
        self.strategies = [MATH] + ([AI] if self.judge.available else [])
        if AI not in self.strategies:
            log.info("AI strategy disabled (no ANTHROPIC_API_KEY or ai.enabled=false)")

        broker_cls = LiveBroker if cfg.engine.live else PaperBroker
        self.broker = broker_cls(self.strategies, cfg.risk.starting_cash,
                                 cfg.engine.slippage)
        self.risk = RiskManager(cfg.risk, self.strategies)
        self.journal = Journal(cfg.engine.db_path)
        self.mlb = mlb.MLBClient()

        self.markets: dict[str, Market] = {}
        self.histories: dict[str, PriceHistory] = {}       # market_key -> home-token history
        self.game_states: dict[int, GameState] = {}
        self.cooldowns: dict[tuple[str, str], float] = {}  # (strategy, market) -> reopen ts
        self.latest_prices: dict[str, float] = {}          # token -> price
        self.latest_quotes: dict[str, MarketQuote] = {}    # market_key -> latest BBO
        self.funnel: dict[str, int] = {}                   # entry-gate reject counters

        self._last_discovery = 0.0
        self._last_game_poll = 0.0
        self._last_equity = 0.0
        self._last_status_log = 0.0
        self.started_at = time.time()
        self.dashboard = TerminalDashboard(enabled=dashboard)
        self.events: deque[str] = deque(maxlen=10)

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
                self._maybe_snapshot_equity()
                self._maybe_log_status()
                self.dashboard.render(self)
                elapsed = time.time() - tick_start
                time.sleep(max(0.0, self.cfg.engine.poll_interval_secs - elapsed))
        except KeyboardInterrupt:
            self._event("shutting down")
        finally:
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
                self.game_states[market.game_pk] = gs

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
            elif book.long_last is not None:
                # One-sided book (common mid-play): keep the price history
                # alive from the mark price so move detection doesn't freeze,
                # but leave latest_quotes stale so entries stay blocked.
                home_mid = book.long_last if market.home_is_long \
                    else 1.0 - book.long_last
            else:
                continue
            self.histories[market.key].add(home_mid)
            self.latest_prices[market.home_token] = home_mid
            self.latest_prices[market.away_token] = 1.0 - home_mid

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
            ts=time.time(),
        )

    # ----------------------------------------------------------------- exits

    def _manage_exits(self):
        for strat in self.strategies:
            for pos in self.broker.open_positions(strat):
                market = self.markets.get(pos.market_key)
                price = self.latest_prices.get(pos.token)
                if market is None or price is None:
                    continue
                gs = self.game_states.get(market.game_pk)
                fair = None
                if gs:
                    fair_home = strategy.fair_home_value(gs, self.cfg.strategy)
                    fair = fair_home if pos.token == market.home_token else 1.0 - fair_home
                reason = strategy.check_exit(
                    pos, price, fair, bool(gs and gs.is_final), self.cfg.strategy
                )
                if reason:
                    self._close(strat, pos, price, reason)

    def _close(self, strat: str, pos, price: float, reason: str):
        result = self.broker.close(strat, pos.token, price)
        if result is None:
            return
        position, fill, pnl = result
        pnl_pct = position.pnl_pct(fill)
        self._event("[%s] CLOSE %s @ %.3f (%+.1f%%, $%+.2f) - %s",
                    strat, position.team, fill, pnl_pct * 100, pnl, reason)
        self.journal.record_close(strat, position.market_key, position.team,
                                  position.token, position.qty, fill, pnl, pnl_pct, reason)
        cooldown = self.cfg.strategy.stop_loss_cooldown_secs \
            if reason.startswith("stop loss") else self.cfg.strategy.cooldown_secs
        self.cooldowns[(strat, position.market_key)] = time.time() + cooldown

    # --------------------------------------------------------------- entries

    def _bump(self, reason: str) -> None:
        self.funnel[reason] = self.funnel.get(reason, 0) + 1

    def _look_for_entries(self):
        scfg = self.cfg.strategy
        now = time.time()
        for market in self.markets.values():
            gs = self.game_states.get(market.game_pk)
            if not gs or not gs.is_live:
                continue  # not counted: pending/final markets aren't candidates
            quote = self.latest_quotes.get(market.key)
            if quote is None or now - quote.ts > scfg.max_quote_age_secs:
                self._bump("stale_quote")
                continue
            if quote.home_spread > scfg.max_spread:
                self._bump("wide_spread")
                continue
            sig = strategy.check_entry(market, self.histories[market.key], gs, scfg,
                                       funnel=self.funnel)
            if sig is None:
                continue
            for strat in self.strategies:
                if time.time() < self.cooldowns.get((strat, market.key), 0):
                    self._bump("cooldown")
                    continue
                if sig.token in self.broker.positions[strat]:
                    self._bump("already_open")
                    continue
                stake = self._stake_for_signal(sig, quote)
                if not self.risk.can_open(self.broker, strat, market.key, stake):
                    self._bump("risk_blocked")
                    continue
                reason = f"{sig.reason}; spread {quote.home_spread:.3f}; stake ${stake:.0f}"
                if strat == AI:
                    verdict = self.judge.judge(sig, gs)
                    if not verdict.approve:
                        self._bump("ai_rejected")
                        self._event("[ai] rejected %s: %s", sig.side_team, verdict.reason)
                        self.cooldowns[(AI, market.key)] = time.time() + scfg.cooldown_secs
                        continue
                    reason += f" | ai: {verdict.reason} ({verdict.confidence:.2f})"
                pos = self.broker.open(strat, market.key, sig.token, sig.side_team,
                                       sig.price, stake)
                if pos:
                    self._bump("opened")
                    self._event("[%s] OPEN %s @ %.3f - %s",
                                strat, sig.side_team, pos.entry_price, reason)
                    self.journal.record_open(strat, market.key, sig.side_team,
                                             sig.token, pos.qty, pos.entry_price, reason)

    def _stake_for_signal(self, sig, quote: MarketQuote) -> float:
        rcfg = self.cfg.risk
        scfg = self.cfg.strategy
        if sig.edge >= rcfg.strong_stake_min_edge \
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
