"""Causal, global-clock replay of the recorded Polymarket US event tape.

Unlike the legacy per-game replay, this module merges every market quote and
MLB state observation by *receipt time*. Strategy decisions schedule orders;
orders can fill only on a later two-sided BBO after configured latency. This
preserves concurrent portfolio constraints and prevents future-state leakage.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from .broker import PaperBroker
from .config import Config
from .journal import Journal
from .model_features import ModelHistory, state_signature
from .models import GameState, Market, MarketQuote
from .risk import RiskManager
from .strategies import (
    AIShadowStrategy, ExitIntent, Intent, StratContext, Strategy,
    build_strategies, executable_ask, executable_bid,
)
from .timeframe import day_bounds
from .state_model import load_probability_model
from .volatility import PriceHistory


@dataclass(order=True)
class TapeEvent:
    ts: float
    priority: int
    seq: int
    kind: str
    run_id: str | None
    payload: dict


@dataclass
class PendingOrder:
    strategy: Strategy
    market_key: str
    action: str
    due: float
    intent: Intent | ExitIntent
    state_signature: tuple | None = None


@dataclass
class ReplayTrade:
    strategy: str
    market: str
    game_pk: int | None
    token: str
    team: str
    entry_ts: float
    exit_ts: float
    entry_price: float
    exit_price: float
    qty: float
    entry_spread: float
    entry_inning: int
    pnl_usd: float
    fees_usd: float
    exit_reason: str


@dataclass
class StrategyResult:
    strategy: str
    trades: int
    wins: int
    realized: float
    fees: float
    equity: float
    max_drawdown: float
    open_positions: int
    rejected_orders: int
    filled_entries: int
    rejected_entries: int
    rejected_exits: int
    deployed_capital: float


@dataclass
class ReplayReport:
    label: str
    timezone: str
    events: int
    run_boundaries: int
    skipped_strategies: list[str]
    trades: list[ReplayTrade]
    results: list[StrategyResult]
    starting_cash: float
    state_model: str = "analytic_v1"


class CausalReplay:
    def __init__(self, cfg: Config, journal: Journal, start: float, end: float, label: str):
        self.cfg = cfg
        self.journal = journal
        self.start = start
        self.end = end
        self.label = label
        self.state_probability, self.state_model_label = load_probability_model(
            cfg.engine.state_model_path
        )
        built = build_strategies(cfg)
        self.strategies = [s for s in built if not isinstance(s, AIShadowStrategy)]
        self.skipped = [s.name for s in built if isinstance(s, AIShadowStrategy)]
        for strat in built:
            if strat not in self.strategies:
                strat.close()
        names = [s.name for s in self.strategies]
        self.broker = PaperBroker(
            names, cfg.risk.starting_cash, slippage=0.0,
            taker_fee_theta=cfg.engine.paper_taker_fee_theta,
        )
        self.risk = {name: RiskManager(cfg.risk, [name]) for name in names}
        self.markets = {m.key: m for m in journal.markets_between(start, end)}
        self.by_game: dict[int, list[Market]] = defaultdict(list)
        for market in self.markets.values():
            if market.game_pk:
                self.by_game[market.game_pk].append(market)
        self.histories: dict[str, PriceHistory] = {}
        self.model_histories: dict[str, ModelHistory] = {}
        self.last_price_ts: dict[str, float] = {}
        self.game_states: dict[int, GameState] = {}
        self.latest_quotes: dict[str, MarketQuote] = {}
        self.latest_bids: dict[str, float] = {}
        self.pending: dict[tuple[str, str, str], PendingOrder] = {}
        self.cooldowns: dict[tuple[str, str], float] = {}
        self.daily_pnl: dict[tuple[str, str], float] = defaultdict(float)
        self.open_meta: dict[str, dict] = {}
        self.trades: list[ReplayTrade] = []
        self.rejected: dict[str, int] = defaultdict(int)
        self.filled_entries: dict[str, int] = defaultdict(int)
        self.rejected_entries: dict[str, int] = defaultdict(int)
        self.rejected_exits: dict[str, int] = defaultdict(int)
        self.deployed_capital: dict[str, float] = defaultdict(float)
        self.peak = {name: cfg.risk.starting_cash for name in names}
        self.max_drawdown = {name: 0.0 for name in names}
        self.current_source_run: str | None = None
        self.run_boundaries = 0

    def run(self) -> ReplayReport:
        events = self._load_events()
        for event in events:
            self._handle_run_boundary(event.run_id)
            if event.kind == "state":
                self._on_state(event)
            else:
                self._on_price(event)
            self._mark_drawdown()
        results = []
        for strat in self.strategies:
            name = strat.name
            rows = [t for t in self.trades if t.strategy == name]
            results.append(StrategyResult(
                strategy=name,
                trades=len(rows),
                wins=sum(t.pnl_usd > 0 for t in rows),
                realized=self.broker.realized[name],
                fees=sum(t.fees_usd for t in rows)
                    + sum(p.entry_fee for p in self.broker.open_positions(name)),
                equity=self.broker.equity(name, self.latest_bids),
                max_drawdown=self.max_drawdown[name],
                open_positions=len(self.broker.open_positions(name)),
                rejected_orders=self.rejected[name],
                filled_entries=self.filled_entries[name],
                rejected_entries=self.rejected_entries[name],
                rejected_exits=self.rejected_exits[name],
                deployed_capital=self.deployed_capital[name],
            ))
            strat.close()
        return ReplayReport(
            label=self.label, timezone=self.cfg.engine.report_timezone,
            events=len(events), run_boundaries=self.run_boundaries,
            skipped_strategies=self.skipped, trades=self.trades, results=results,
            starting_cash=self.cfg.risk.starting_cash,
            state_model=self.state_model_label,
        )

    def _load_events(self) -> list[TapeEvent]:
        events: list[TapeEvent] = []
        seq = 0
        market_keys = set(self.markets)
        price_rows = self.journal.conn.execute(
            """SELECT rowid AS rid, * FROM price_ticks
               WHERE COALESCE(received_at, ts) >= ? AND COALESCE(received_at, ts) < ?
               ORDER BY COALESCE(received_at, ts), rowid""",
            (self.start, self.end),
        ).fetchall()
        for row in price_rows:
            if row["market"] not in market_keys:
                continue
            seq += 1
            events.append(TapeEvent(
                ts=row["received_at"] or row["ts"], priority=1, seq=seq,
                kind="price", run_id=row["run_id"], payload=dict(row),
            ))
        game_ids = set(self.by_game)
        state_rows = self.journal.conn.execute(
            """SELECT rowid AS rid, * FROM game_states
               WHERE COALESCE(received_at, ts) >= ? AND COALESCE(received_at, ts) < ?
               ORDER BY COALESCE(received_at, ts), rowid""",
            (self.start - 6 * 3600, self.end),
        ).fetchall()
        for row in state_rows:
            if row["game_pk"] not in game_ids:
                continue
            seq += 1
            events.append(TapeEvent(
                ts=row["received_at"] or row["ts"], priority=0, seq=seq,
                kind="state", run_id=row["run_id"], payload=dict(row),
            ))
        events.sort()
        return events

    def _handle_run_boundary(self, run_id: str | None) -> None:
        if run_id is None:
            return
        if self.current_source_run is None:
            self.current_source_run = run_id
            return
        if run_id == self.current_source_run:
            return
        self.current_source_run = run_id
        self.run_boundaries += 1
        self.histories.clear()
        self.model_histories.clear()
        self.last_price_ts.clear()
        self.game_states.clear()
        self.latest_quotes.clear()
        self.pending.clear()
        self.cooldowns.clear()
        for strat in self.strategies:
            for market_key in self.markets:
                strat.reset_market(market_key)

    def _on_state(self, event: TapeEvent) -> None:
        row = event.payload
        gs = GameState(
            game_pk=row["game_pk"], inning=row["inning"], is_top=bool(row["is_top"]),
            outs=row["outs"], home_score=row["home_score"], away_score=row["away_score"],
            on_first=bool(row["on_first"]), on_second=bool(row["on_second"]),
            on_third=bool(row["on_third"]), status=row["status"], received_at=event.ts,
        )
        self.game_states[gs.game_pk] = gs
        for market in self.by_game.get(gs.game_pk, []):
            self.model_histories.setdefault(
                market.key, self._new_model_history()
            ).observe_state(gs, event.ts)
        if not gs.is_final:
            return
        for market in self.by_game.get(gs.game_pk, []):
            self._cancel_market_orders(market.key)
        if gs.home_score == gs.away_score:
            return  # suspended/tied final: nothing tradeable, but leave it unsettled
        for market in self.by_game.get(gs.game_pk, []):
            home_won = gs.home_score > gs.away_score
            for strat in self.strategies:
                for pos in list(self.broker.open_positions(strat.name)):
                    if pos.market_key != market.key:
                        continue
                    won = (pos.token == market.home_token and home_won) or (
                        pos.token == market.away_token and not home_won)
                    result = self.broker.settle(strat.name, pos.token, 1.0 if won else 0.0)
                    if result:
                        position, fill, pnl = result
                        self._finish_trade(position, fill, pnl, event.ts,
                                           "official game settlement", 0.0)

    def _on_price(self, event: TapeEvent) -> None:
        row = event.payload
        market = self.markets.get(row["market"])
        if market is None:
            return
        last = self.last_price_ts.get(market.key)
        if last is not None and event.ts - last > self.cfg.engine.history_gap_reset_secs:
            self.histories.pop(market.key, None)
            stale = self.model_histories.get(market.key)
            if stale is not None:
                stale.reset_rolling()
            self.latest_quotes.pop(market.key, None)
            self._cancel_market_orders(market.key)
            for strat in self.strategies:
                strat.reset_market(market.key)
        self.last_price_ts[market.key] = event.ts
        history = self.histories.setdefault(
            market.key, PriceHistory(self.cfg.strategy.flip_band)
        )
        model_history = self.model_histories.setdefault(
            market.key, self._new_model_history()
        )
        if model_history.current_signature is None:
            gs = self.game_states.get(market.game_pk)
            if gs is not None:
                model_history.observe_state(gs, gs.received_at)
        model_history.add_price(
            row["home_mid"], event.ts,
            pregame_eligible=(
                market.start_time is not None and event.ts < market.start_time
            ),
        )
        gs = self.game_states.get(market.game_pk)
        if gs and gs.is_live:
            history.add(row["home_mid"], event.ts)
        if not row["two_sided"]:
            self.latest_quotes.pop(market.key, None)
            return
        quote = MarketQuote(
            market.key, row["home_bid"], row["home_ask"],
            row["long_bid"], row["long_ask"], ts=event.ts, source_ts=row["source_ts"],
        )
        self.latest_quotes[market.key] = quote
        self.latest_bids[market.home_token] = quote.home_bid
        self.latest_bids[market.away_token] = 1.0 - quote.home_ask
        self._fill_due(market, quote, event.ts)
        if not gs or not gs.is_live:
            return
        for strat in self.strategies:
            ctx = StratContext(market, history, gs, quote, event.ts,
                               fee_theta=self.cfg.engine.paper_taker_fee_theta,
                               model_history=model_history)
            positions = [p for p in self.broker.open_positions(strat.name)
                         if p.market_key == market.key]
            for exit_intent in strat.manage(ctx, positions):
                key = (strat.name, market.key, "exit")
                self.pending.setdefault(key, PendingOrder(
                    strat, market.key, "exit",
                    event.ts + self.cfg.engine.causal_replay_latency_secs, exit_intent,
                    state_signature(gs),
                ))
            decision = strat.evaluate(ctx)
            if decision.intent is None:
                continue
            key = (strat.name, market.key, "entry")
            if decision.intent.token in self.broker.positions[strat.name] or key in self.pending:
                continue
            if event.ts < self.cooldowns.get((strat.name, market.key), 0):
                continue
            self.pending[key] = PendingOrder(
                strat, market.key, "entry",
                event.ts + self.cfg.engine.causal_replay_latency_secs, decision.intent,
                state_signature(gs),
            )

    def _fill_due(self, market: Market, quote: MarketQuote, now: float) -> None:
        due = [key for key, order in self.pending.items()
               if order.market_key == market.key and order.due <= now]
        for key in due:
            order = self.pending.pop(key)
            if order.action == "entry":
                self._fill_entry(order, market, quote, now)
            else:
                self._fill_exit(order, market, quote, now)

    def _fill_entry(self, order: PendingOrder, market: Market,
                    quote: MarketQuote, now: float) -> None:
        strat = order.strategy
        intent = order.intent
        gs = self.game_states.get(market.game_pk)
        if not gs or not gs.is_live or state_signature(gs) != order.state_signature \
                or quote.home_spread > strat.config.max_spread:
            self.rejected[strat.name] += 1
            self.rejected_entries[strat.name] += 1
            return
        price = executable_ask(market, quote, intent.token)
        fee_per_contract = StratContext(
            market, self.histories[market.key], gs, quote, now,
            self.cfg.engine.paper_taker_fee_theta,
        ).round_trip_fee(price, intent.fair)
        net_edge = intent.fair - price - fee_per_contract
        if price < strat.config.min_price or price > strat.config.max_price \
                or net_edge < strat.config.min_edge:
            self.rejected[strat.name] += 1
            self.rejected_entries[strat.name] += 1
            return
        day_key = self._day_key(now)
        stake = self.cfg.risk.strong_stake_usd if (
            net_edge >= self.cfg.risk.strong_stake_min_edge
            and quote.home_spread <= strat.config.strong_stake_max_spread
        ) else self.cfg.risk.stake_usd
        if not self.risk[strat.name].can_open(
            self.broker, strat.name, market.key, stake,
            daily_realized=self.daily_pnl[(day_key, strat.name)], day_key=day_key,
        ):
            self.rejected[strat.name] += 1
            self.rejected_entries[strat.name] += 1
            return
        pos = self.broker.open(strat.name, market.key, intent.token,
                               intent.side_team, price, stake)
        if pos is None:
            self.rejected[strat.name] += 1
            self.rejected_entries[strat.name] += 1
            return
        pos.opened_at = now
        self.filled_entries[strat.name] += 1
        self.deployed_capital[strat.name] += pos.cost
        self.open_meta[pos.trade_id] = {
            "entry_ts": now, "entry_fee": pos.entry_fee,
            "entry_spread": quote.home_spread, "entry_inning": gs.inning,
        }

    def _fill_exit(self, order: PendingOrder, market: Market,
                   quote: MarketQuote, now: float) -> None:
        intent = order.intent
        current = self.broker.positions[order.strategy.name].get(intent.position.token)
        if current is None or current.trade_id != intent.position.trade_id:
            return
        price = executable_bid(market, quote, current.token)
        result = self.broker.close(order.strategy.name, current.token, price)
        if result is None:
            self.rejected[order.strategy.name] += 1
            self.rejected_exits[order.strategy.name] += 1
            return
        position, fill, pnl = result
        exit_fee = self.broker.last_fee[order.strategy.name]
        self._finish_trade(position, fill, pnl, now, intent.reason, exit_fee)
        cooldown = order.strategy.config.stop_loss_cooldown_secs \
            if intent.reason.startswith("stop loss") else order.strategy.config.cooldown_secs
        self.cooldowns[(order.strategy.name, market.key)] = now + cooldown

    def _finish_trade(self, position, fill: float, pnl: float, ts: float,
                      reason: str, exit_fee: float) -> None:
        meta = self.open_meta.pop(position.trade_id, {
            "entry_ts": position.opened_at, "entry_fee": position.entry_fee,
            "entry_spread": 0.0, "entry_inning": 0,
        })
        market = self.markets.get(position.market_key)
        self.trades.append(ReplayTrade(
            strategy=position.strategy, market=position.market_key,
            game_pk=market.game_pk if market else None, token=position.token,
            team=position.team,
            entry_ts=meta["entry_ts"], exit_ts=ts, entry_price=position.entry_price,
            exit_price=fill, qty=position.qty,
            entry_spread=meta["entry_spread"], entry_inning=meta["entry_inning"],
            pnl_usd=pnl, fees_usd=meta["entry_fee"] + exit_fee,
            exit_reason=reason,
        ))
        self.daily_pnl[(self._day_key(ts), position.strategy)] += pnl

    def _cancel_market_orders(self, market_key: str) -> None:
        self.pending = {k: v for k, v in self.pending.items() if v.market_key != market_key}

    def _day_key(self, ts: float) -> str:
        return datetime.fromtimestamp(
            ts, ZoneInfo(self.cfg.engine.report_timezone)
        ).date().isoformat()

    def _mark_drawdown(self) -> None:
        for strat in self.strategies:
            name = strat.name
            equity = self.broker.equity(name, self.latest_bids)
            self.peak[name] = max(self.peak[name], equity)
            self.max_drawdown[name] = max(self.max_drawdown[name], self.peak[name] - equity)

    def _new_model_history(self) -> ModelHistory:
        return ModelHistory(
            self.state_probability,
            anchor_lookback_secs=self.cfg.strategy.residual_anchor_lookback_secs,
        )


def replay_recorded_day(cfg: Config, db_path: str, day: str | None = None) -> ReplayReport:
    start, end, label = day_bounds(day, cfg.engine.report_timezone)
    journal = Journal(db_path)
    try:
        return CausalReplay(cfg, journal, start, end, label).run()
    finally:
        journal.close()


def print_report(report: ReplayReport) -> None:
    print("=" * 88)
    print(f"CAUSAL MULTI-MARKET REPLAY {report.label} ({report.timezone})")
    print("=" * 88)
    print(f"events processed : {report.events}")
    print(f"run boundaries   : {report.run_boundaries}")
    print(f"state model      : {report.state_model}")
    if report.skipped_strategies:
        print("skipped          : " + ", ".join(report.skipped_strategies) + " (non-deterministic shadow)")
    print()
    print(f"{'strategy':<22}{'trades':>8}{'win%':>8}{'pnl':>11}{'fees':>10}"
          f"{'equity':>11}{'max DD':>10}{'open':>7}{'reject':>9}")
    print("-" * 96)
    for row in report.results:
        win = 100.0 * row.wins / row.trades if row.trades else 0.0
        print(f"{row.strategy:<22}{row.trades:>8}{win:>7.1f}%{row.realized:>11.2f}"
              f"{row.fees:>10.2f}{row.equity:>11.2f}{row.max_drawdown:>10.2f}"
              f"{row.open_positions:>7}{row.rejected_orders:>9}")
    if not report.results:
        print("No deterministic strategies were configured.")
