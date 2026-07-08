"""Main trading loop."""
from __future__ import annotations

import logging
import time

from . import clob, gamma, mlb, strategy
from .ai_judge import AIJudge
from .broker import LiveBroker, PaperBroker
from .config import Config
from .journal import Journal
from .models import GameState, Market
from .risk import RiskManager
from .volatility import PriceHistory
from .winprob import home_win_probability

log = logging.getLogger(__name__)

MATH = "math"
AI = "ai"


class Engine:
    def __init__(self, cfg: Config):
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
        self.feed = clob.PriceFeed()
        self.mlb = mlb.MLBClient()

        self.markets: dict[str, Market] = {}
        self.histories: dict[str, PriceHistory] = {}       # market_key -> home-token history
        self.game_states: dict[int, GameState] = {}
        self.cooldowns: dict[tuple[str, str], float] = {}  # (strategy, market) -> ts
        self.latest_prices: dict[str, float] = {}          # token -> price

        self._last_discovery = 0.0
        self._last_game_poll = 0.0
        self._last_equity = 0.0

    # ------------------------------------------------------------------ loop

    def run(self):
        mode = "LIVE" if self.cfg.engine.live else "PAPER"
        log.info("engine starting (%s mode, strategies: %s)", mode, self.strategies)
        try:
            while True:
                tick_start = time.time()
                self._maybe_discover()
                self._maybe_poll_games()
                self._poll_prices()
                self._manage_exits()
                self._look_for_entries()
                self._maybe_snapshot_equity()
                elapsed = time.time() - tick_start
                time.sleep(max(0.0, self.cfg.engine.poll_interval_secs - elapsed))
        except KeyboardInterrupt:
            log.info("shutting down")
        finally:
            self.journal.close()

    # ------------------------------------------------------------- discovery

    def _maybe_discover(self):
        now = time.time()
        if now - self._last_discovery < self.cfg.engine.discovery_interval_secs \
                and self.markets:
            return
        self._last_discovery = now
        found = gamma.fetch_mlb_markets()
        games = self.mlb.todays_games()
        mlb.match_markets_to_games(found, games)
        for m in found:
            if m.game_pk and m.key not in self.markets:
                self.markets[m.key] = m
                self.histories[m.key] = PriceHistory(self.cfg.strategy.flip_band)
                log.info("tracking: %s (game %s)", m.question, m.game_pk)

    def _maybe_poll_games(self):
        now = time.time()
        if now - self._last_game_poll < self.cfg.engine.game_state_interval_secs:
            return
        self._last_game_poll = now
        for pk in {m.game_pk for m in self.markets.values() if m.game_pk}:
            gs = self.mlb.game_state(pk)
            if gs:
                self.game_states[pk] = gs

    def _poll_prices(self):
        for market in self.markets.values():
            gs = self.game_states.get(market.game_pk)
            if gs and gs.is_final:
                continue
            mid = self.feed.midpoint(market.home_token)
            if mid is None:
                continue
            self.histories[market.key].add(mid)
            self.latest_prices[market.home_token] = mid
            self.latest_prices[market.away_token] = 1.0 - mid

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
                    fair_home = home_win_probability(gs)
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
        log.info("[%s] CLOSE %s @ %.3f (%+.1f%%, $%+.2f) — %s",
                 strat, position.team, fill, pnl_pct * 100, pnl, reason)
        self.journal.record_close(strat, position.market_key, position.team,
                                  position.token, position.qty, fill, pnl, pnl_pct, reason)
        self.cooldowns[(strat, position.market_key)] = time.time()

    # --------------------------------------------------------------- entries

    def _look_for_entries(self):
        scfg = self.cfg.strategy
        for market in self.markets.values():
            gs = self.game_states.get(market.game_pk)
            sig = strategy.check_entry(market, self.histories[market.key], gs, scfg)
            if sig is None:
                continue
            for strat in self.strategies:
                if time.time() - self.cooldowns.get((strat, market.key), 0) < scfg.cooldown_secs:
                    continue
                if sig.token in self.broker.positions[strat]:
                    continue
                if not self.risk.can_open(self.broker, strat, market.key):
                    continue
                reason = sig.reason
                if strat == AI:
                    verdict = self.judge.judge(sig, gs)
                    if not verdict.approve:
                        log.info("[ai] rejected %s: %s", sig.side_team, verdict.reason)
                        self.cooldowns[(AI, market.key)] = time.time()
                        continue
                    reason += f" | ai: {verdict.reason} ({verdict.confidence:.2f})"
                pos = self.broker.open(strat, market.key, sig.token, sig.side_team,
                                       sig.price, self.cfg.risk.stake_usd)
                if pos:
                    log.info("[%s] OPEN %s @ %.3f — %s",
                             strat, sig.side_team, pos.entry_price, reason)
                    self.journal.record_open(strat, market.key, sig.side_team,
                                             sig.token, pos.qty, pos.entry_price, reason)

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
