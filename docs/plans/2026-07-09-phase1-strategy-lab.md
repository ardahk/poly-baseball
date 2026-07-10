# Phase 1: Strategy Laboratory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace hardcoded `math`/`ai` branches with a strategy interface so several frozen strategies run on one shared stream, each with an independent shadow portfolio, plus live counterfactual price capture and an async shadow-only AI judge.

**Architecture:** A `Strategy` interface (in `polybot/strategies.py`) wraps the existing fade math (`FadeStrategy`) and the async judge (`AIShadowStrategy`). A registry built from a `strategies:` config block replaces the `MATH`/`AI` constants. `PaperBroker`/`RiskManager` are already keyed by strategy name, so seeding them from registry names gives N independent books. The engine loop iterates strategies over a shared `StratContext`; execution price helpers live on the context so all strategies fill identically. A signal/counterfactual scheduler records post-signal executable prices at fixed horizons.

**Tech Stack:** Python 3, dataclasses, SQLite (`sqlite3`), `concurrent.futures.ThreadPoolExecutor`, pytest.

**Design doc:** `docs/plans/2026-07-09-phase1-strategy-lab-design.md`

---

## Conventions

- Run tests with `python -m pytest`.
- Keep the existing free functions in `polybot/strategy.py` (`evaluate_entry`, `check_exit`, `fair_home_value`, `exit_kind`) — `FadeStrategy` wraps them, so `tests/test_strategy.py` keeps passing unchanged.
- Fresh strategy names: default registry is `fade_v1_frozen`, `fade_tight`, `ai_shadow`. The old `math`/`ai` paper ledger does not migrate.
- Commit after every task (test + impl together).

---

## Task 1: Execution-price context + Intent/Exit types

**Files:**
- Create: `polybot/strategies.py`
- Test: `tests/test_strategies.py`

**Step 1: Write failing test**

```python
# tests/test_strategies.py
from polybot.models import Market, MarketQuote, GameState, Position
from polybot.volatility import PriceHistory
from polybot.strategies import StratContext


def market():
    return Market(slug="m1", question="A vs B", home_team="Homers",
                  away_team="Awayers", long_team="Homers", game_pk=1)


def ctx(quote):
    return StratContext(market=market(), history=PriceHistory(),
                        game_state=GameState(1, status="Live"),
                        quote=quote, now=1000.0)


def test_context_entry_price_is_executable_ask_per_side():
    q = MarketQuote("m1", home_bid=0.50, home_ask=0.52, long_bid=0.50, long_ask=0.52)
    c = ctx(q)
    assert c.entry_price(c.market.home_token) == 0.52       # buy home at ask
    assert c.entry_price(c.market.away_token) == 0.50       # buy away at 1-home_bid


def test_context_exit_price_is_executable_bid_per_side():
    q = MarketQuote("m1", home_bid=0.50, home_ask=0.52, long_bid=0.50, long_ask=0.52)
    c = ctx(q)
    assert c.exit_price(c.market.home_token) == 0.50        # sell home at bid
    assert c.exit_price(c.market.away_token) == 0.48        # sell away at 1-home_ask
```

**Step 2: Run — expect ImportError / fail.**

Run: `python -m pytest tests/test_strategies.py -x -q`

**Step 3: Implement**

```python
# polybot/strategies.py
"""Strategy interface: frozen variants competing on one shared stream."""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import EntryEvaluation, GameState, Market, MarketQuote, Position
from .volatility import PriceHistory


@dataclass
class StratContext:
    """The shared market/game observation every strategy decides against."""
    market: Market
    history: PriceHistory
    game_state: GameState
    quote: MarketQuote
    now: float

    def entry_price(self, token: str) -> float:
        """Executable taker BUY price for a token side (ask)."""
        if token == self.market.home_token:
            return self.quote.home_ask
        if token == self.market.away_token:
            return 1.0 - self.quote.home_bid
        raise ValueError(f"token {token!r} not in market {self.market.key!r}")

    def exit_price(self, token: str) -> float:
        """Executable taker SELL price for a token side (bid)."""
        if token == self.market.home_token:
            return self.quote.home_bid
        if token == self.market.away_token:
            return 1.0 - self.quote.home_ask
        raise ValueError(f"token {token!r} not in market {self.market.key!r}")


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
    """One strategy's full result for a market tick, for the decision log + action."""
    evaluation: EntryEvaluation | None = None
    outcome: str = "n/a"     # decision-row outcome for this strategy
    intent: Intent | None = None
    signal_candidate: bool = False   # true when this tick produced a signal-grade candidate
```

**Step 4: Run — expect PASS.**

**Step 5: Commit** `feat: add strategy context and intent types`

---

## Task 2: FadeStrategy wrapping the existing fade math

**Files:**
- Modify: `polybot/strategies.py`
- Test: `tests/test_strategies.py`

**Step 1: Write failing tests**

```python
from polybot.config import StrategyConfig
from polybot.strategies import FadeStrategy

CFG = StrategyConfig(move_lookback_secs=60, move_threshold=0.08, min_edge=0.05,
                     min_flips=2, min_volatility=99.0, max_price=0.99)


def playful(prices, step=30.0):
    h = PriceHistory(flip_band=0.03)
    for i, p in enumerate(prices):
        h.add(p, ts=i * step)
    return h


def fade_ctx(history, gs, bid, ask):
    return StratContext(market=market(), history=history, game_state=gs,
                        quote=MarketQuote("m1", bid, ask, bid, ask), now=1000.0)


def test_fade_emits_intent_when_edge_survives_execution():
    s = FadeStrategy("fade_v1_frozen", "v1", CFG)
    h = playful([0.60, 0.40, 0.60, 0.60, 0.40])
    gs = GameState(1, status="Live", inning=7, is_top=True, home_score=4, away_score=1)
    d = s.evaluate(fade_ctx(h, gs, bid=0.39, ask=0.41))   # tight spread near mid
    assert d.outcome == "signal"
    assert d.signal_candidate is True
    assert d.intent is not None
    assert d.intent.token == "m1:LONG"
    assert d.intent.edge == pytest.approx(d.intent.fair - 0.41)


def test_fade_rejects_when_execution_cost_eats_edge():
    s = FadeStrategy("fade_v1_frozen", "v1", CFG)
    h = playful([0.60, 0.40, 0.60, 0.60, 0.40])
    gs = GameState(1, status="Live", inning=7, is_top=True, home_score=4, away_score=1)
    d = s.evaluate(fade_ctx(h, gs, bid=0.30, ask=0.52))   # wide: ask far above mid
    assert d.outcome == "execution_cost"
    assert d.intent is None
    assert d.signal_candidate is True     # still a signal-grade candidate to track


def test_fade_no_signal_passes_through_outcome():
    s = FadeStrategy("fade_v1_frozen", "v1", CFG)
    h = playful([0.60, 0.60, 0.60, 0.48])   # not playful
    gs = GameState(1, status="Live", inning=7, is_top=True, home_score=4, away_score=1)
    d = s.evaluate(fade_ctx(h, gs, bid=0.47, ask=0.49))
    assert d.intent is None
    assert d.signal_candidate is False
    assert d.evaluation is not None


def test_fade_manage_returns_exit_intents():
    s = FadeStrategy("fade_v1_frozen", "v1", CFG)
    gs = GameState(1, status="Live")
    c = fade_ctx(PriceHistory(), gs, bid=0.57, ask=0.58)
    p = Position(strategy="fade_v1_frozen", market_key="m1", token="m1:LONG",
                 team="Homers", qty=20.0, entry_price=0.50)
    exits = s.manage(c, [p])
    assert len(exits) == 1
    assert "take profit" in exits[0].reason      # sells at bid 0.57 -> +14%
```

**Step 2: Run — expect fail (FadeStrategy missing).**

**Step 3: Implement** (append to `polybot/strategies.py`)

```python
from . import strategy as _fade
from .config import StrategyConfig


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
```

**Step 4: Run — expect PASS.** (Add `import pytest` to the test file.)

**Step 5: Commit** `feat: add FadeStrategy wrapping fade math`

---

## Task 3: Config registry + build_strategies

**Files:**
- Modify: `polybot/config.py`, `polybot/strategies.py`, `config.yaml`
- Test: `tests/test_strategies.py`

**Step 1: Write failing tests**

```python
from polybot.config import Config, StrategyConfig
from polybot.strategies import build_strategies, FadeStrategy


def test_default_registry_builds_two_fade_variants():
    cfg = Config()
    cfg.ai.enabled = False
    strats = build_strategies(cfg)
    names = [s.name for s in strats]
    assert names == ["fade_v1_frozen", "fade_tight"]
    assert isinstance(strats[0], FadeStrategy)


def test_frozen_variant_overrides_base_config():
    cfg = Config()
    cfg.ai.enabled = False
    cfg.strategies = [
        {"name": "fade_v1_frozen", "kind": "fade"},
        {"name": "fade_tight", "kind": "fade",
         "overrides": {"move_threshold": 0.15, "min_edge": 0.09}},
    ]
    strats = {s.name: s for s in build_strategies(cfg)}
    assert strats["fade_v1_frozen"].config.move_threshold == cfg.strategy.move_threshold
    assert strats["fade_tight"].config.move_threshold == 0.15
    assert strats["fade_tight"].config.min_edge == 0.09


def test_unknown_kind_raises():
    cfg = Config()
    cfg.strategies = [{"name": "x", "kind": "bogus"}]
    with pytest.raises(ValueError, match="unknown strategy kind"):
        build_strategies(cfg)
```

**Step 2: Run — expect fail.**

**Step 3: Implement**

In `polybot/config.py`, add to `Config`:
```python
    strategies: list[dict] = field(default_factory=list)
```
and in `load_config`, after loading the four sub-configs:
```python
        if isinstance(raw.get("strategies"), list):
            cfg.strategies = raw["strategies"]
```

Default registry constant + builder in `polybot/strategies.py`:
```python
import dataclasses

DEFAULT_REGISTRY = [
    {"name": "fade_v1_frozen", "kind": "fade"},
    {"name": "fade_tight", "kind": "fade",
     "overrides": {"move_threshold": 0.15, "min_edge": 0.09}},
]


def build_strategies(cfg) -> list["Strategy"]:
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
            base_name = entry["base"]
            base = next(s for s in strats if s.name == base_name)
            strats.append(AIShadowStrategy(name, entry.get("version", "v1"),
                                           base, cfg.ai))
        else:
            raise ValueError(f"unknown strategy kind: {kind!r}")
    return strats
```
(Leave `AIShadowStrategy` referenced — implemented in Task 4. Until then, keep the
default registry ai-free so tests pass; add ai_shadow to `DEFAULT_REGISTRY` only
after Task 4.)

Add a `strategies:` block to `config.yaml` documenting the default registry
(commented example with `fade_v1_frozen`, `fade_tight`, `ai_shadow`).

**Step 4: Run — expect PASS.**

**Step 5: Commit** `feat: add strategy registry and build_strategies`

---

## Task 4: AIShadowStrategy (async, shadow-only)

**Files:**
- Modify: `polybot/strategies.py`
- Test: `tests/test_strategies.py`

**Step 1: Write failing tests** (use a fake judge — no network)

```python
from polybot.strategies import AIShadowStrategy
from polybot.ai_judge import Judgment


class FakeJudge:
    def __init__(self, verdict, available=True):
        self.verdict = verdict
        self.available = available
        self.calls = 0
    def judge(self, signal, gs):
        self.calls += 1
        return self.verdict


def ai_ctx(bid=0.39, ask=0.41):
    h = playful([0.60, 0.40, 0.60, 0.60, 0.40])
    gs = GameState(1, status="Live", inning=7, is_top=True, home_score=4, away_score=1)
    return StratContext(market=market(), history=h, game_state=gs,
                        quote=MarketQuote("m1", bid, ask, bid, ask), now=1000.0)


def test_ai_shadow_first_tick_submits_and_holds(monkeypatch):
    base = FadeStrategy("fade_v1_frozen", "v1", CFG)
    ai = AIShadowStrategy("ai_shadow", "v1", base,
                          judge=FakeJudge(Judgment(True, 0.9, "ok")))
    d = ai.evaluate(ai_ctx())
    assert d.intent is None            # judged asynchronously; nothing yet
    assert d.outcome == "ai_pending"
    ai.wait_idle()                     # test helper: block until futures resolve
    d2 = ai.evaluate(ai_ctx())
    assert d2.intent is not None       # approved verdict drained on a later tick
    assert d2.outcome == "ai_opened"
    assert d2.intent.token == "m1:LONG"
    ai.close()


def test_ai_shadow_rejection_opens_nothing(monkeypatch):
    base = FadeStrategy("fade_v1_frozen", "v1", CFG)
    ai = AIShadowStrategy("ai_shadow", "v1", base,
                          judge=FakeJudge(Judgment(False, 0.2, "no")))
    ai.evaluate(ai_ctx())
    ai.wait_idle()
    d = ai.evaluate(ai_ctx())
    assert d.intent is None
    assert d.outcome in {"ai_rejected", "ai_pending"}
    ai.close()


def test_ai_shadow_does_not_resubmit_while_pending():
    base = FadeStrategy("fade_v1_frozen", "v1", CFG)
    judge = FakeJudge(Judgment(True, 0.9, "ok"))
    ai = AIShadowStrategy("ai_shadow", "v1", base, judge=judge)
    ai.evaluate(ai_ctx())
    ai.evaluate(ai_ctx())              # second tick before resolve
    ai.wait_idle()
    assert judge.calls == 1            # one candidate -> one judge call
    ai.close()
```

**Step 2: Run — expect fail.**

**Step 3: Implement**

```python
from concurrent.futures import ThreadPoolExecutor
import time as _time
from .ai_judge import AIJudge


class AIShadowStrategy(Strategy):
    """Async gate over a base fade strategy's candidates; opens in its own book."""
    def __init__(self, name, version, base: "FadeStrategy", cfg=None, judge=None):
        super().__init__(name, version, base.config)
        self.base = base
        self.judge = judge if judge is not None else AIJudge(cfg)
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._pending: dict[str, tuple] = {}   # market_key -> (future, intent, submit_ts)

    def evaluate(self, ctx: StratContext) -> Decision:
        mkey = ctx.market.key
        # 1) drain a resolved future for this market
        entry = self._pending.get(mkey)
        if entry and entry[0].done():
            future, intent, submit_ts = self._pending.pop(mkey)
            try:
                verdict = future.result()
            except Exception:
                verdict = None
            if verdict and verdict.approve:
                exec_price = ctx.entry_price(intent.token)
                latency = ctx.now - submit_ts
                fresh = Intent(token=intent.token, side_team=intent.side_team,
                               signal_price=intent.signal_price, fair=intent.fair,
                               move=intent.move, edge=intent.fair - exec_price,
                               reason=f"{intent.reason} | ai {verdict.reason} "
                                      f"({verdict.confidence:.2f}, {latency:.1f}s)")
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
        gs = ctx.game_state
        fut = self._pool.submit(self.judge.judge, sig, gs)
        self._pending[mkey] = (fut, d.intent, ctx.now)
        return Decision(evaluation=d.evaluation, outcome="ai_pending",
                        signal_candidate=True)

    def manage(self, ctx, positions):
        return self.base.manage(ctx, positions)

    def wait_idle(self):
        for future, *_ in list(self._pending.values()):
            future.result()

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
```

Then add `{"name": "ai_shadow", "kind": "ai_shadow", "base": "fade_v1_frozen"}` to
`DEFAULT_REGISTRY` (gated by `cfg.ai.enabled` in `build_strategies`).

**Step 4: Run — expect PASS.**

**Step 5: Commit** `feat: add async shadow-only AIShadowStrategy`

---

## Task 5: Journal — signals + counterfactual tables

**Files:**
- Modify: `polybot/journal.py`
- Test: `tests/test_journal.py`

**Step 1: Write failing test**

```python
def test_records_signal_and_counterfactuals(tmp_path):
    j = Journal(str(tmp_path / "j.db"))
    j.start_run("paper", "hash")
    sid = j.record_signal(strategy="fade_v1_frozen", market="m1", token="m1:LONG",
                          side_team="Homers", entry_price=0.41, fair=0.55, edge=0.14,
                          move=-0.10, spread=0.02, inning=7, is_top=1,
                          home_score=4, away_score=1)
    assert isinstance(sid, int)
    j.record_counterfactual(sid, horizon_secs=30, exec_bid=0.44, exec_ask=0.46,
                            mid=0.45, two_sided=1, spread=0.02)
    rows = j.conn.execute(
        "SELECT * FROM signal_counterfactuals WHERE signal_id=?", (sid,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["horizon_secs"] == 30
    assert rows[0]["exec_ask"] == 0.46
    j.close()
```

**Step 2: Run — expect fail.**

**Step 3: Implement**

Add DDL to `_SCHEMA`:
```sql
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, run_id TEXT, strategy TEXT NOT NULL, market TEXT NOT NULL,
    token TEXT NOT NULL, side_team TEXT, entry_price REAL, fair REAL, edge REAL,
    move REAL, spread REAL, inning INTEGER, is_top INTEGER,
    home_score INTEGER, away_score INTEGER
);
CREATE TABLE IF NOT EXISTS signal_counterfactuals (
    signal_id INTEGER NOT NULL, horizon_secs INTEGER NOT NULL, ts REAL NOT NULL,
    exec_bid REAL, exec_ask REAL, mid REAL, two_sided INTEGER, spread REAL,
    PRIMARY KEY (signal_id, horizon_secs)
);
```
Methods:
```python
    def record_signal(self, *, strategy, market, token, side_team, entry_price,
                      fair, edge, move, spread, inning, is_top, home_score,
                      away_score, ts=None, commit=True) -> int:
        cur = self.conn.execute(
            """INSERT INTO signals (ts, run_id, strategy, market, token, side_team,
               entry_price, fair, edge, move, spread, inning, is_top,
               home_score, away_score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time() if ts is None else ts, self.active_run_id, strategy, market,
             token, side_team, entry_price, fair, edge, move, spread, inning, is_top,
             home_score, away_score))
        if commit:
            self.conn.commit()
        return int(cur.lastrowid)

    def record_counterfactual(self, signal_id, horizon_secs, *, exec_bid, exec_ask,
                              mid, two_sided, spread, ts=None, commit=True):
        self.conn.execute(
            """INSERT OR IGNORE INTO signal_counterfactuals
               (signal_id, horizon_secs, ts, exec_bid, exec_ask, mid, two_sided, spread)
               VALUES (?,?,?,?,?,?,?,?)""",
            (signal_id, horizon_secs, time.time() if ts is None else ts,
             exec_bid, exec_ask, mid, two_sided, spread))
        if commit:
            self.conn.commit()
```

**Step 4: Run — expect PASS.**

**Step 5: Commit** `feat: journal signals and counterfactual tables`

---

## Task 6: Engine — drive strategies from the registry

**Files:**
- Modify: `polybot/engine.py`
- Test: `tests/test_engine.py`

This is the largest task. Break into sub-steps, committing once at the end.

**Step 1: Update engine construction.** Replace lines 24-25 (`MATH`/`AI` constants)
and lines 33-37 (strategy list) with:
```python
from .strategies import StratContext, build_strategies, HORIZONS

# in __init__:
self.strategy_objs = build_strategies(cfg)
self.strategies = [s.name for s in self.strategy_objs]   # names, for broker/risk/journal
```
Keep `self.strategies` as the list of *names* so `PaperBroker`, `RiskManager`,
`_restore_paper_account`, `_settle_final_game`, `_maybe_snapshot_equity`, and
`_maybe_log_status` keep working unchanged.

Add counterfactual state to `__init__`:
```python
self.pending_cf: list[dict] = []   # {signal_id, token, market_key, born, remaining:set[int]}
```

**Step 2: Rewrite `_manage_exits`** to iterate strategy objects:
```python
def _manage_exits(self):
    now = time.time()
    for strat in self.strategy_objs:
        positions = self.broker.open_positions(strat.name)
        for pos in positions:
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
```

**Step 3: Rewrite `_look_for_entries`** to loop strategies over a shared context.
For each live, fresh-quoted market build `ctx = StratContext(...)`, then for each
`strat` call `d = strat.evaluate(ctx)`. Log a decision row per strategy using
`d.evaluation`/`d.outcome`. When `d.signal_candidate`, register a counterfactual
(Step 5). When `d.intent`, run cooldown / already-open / stake / risk gates
(unchanged logic, but per `strat.name` and using `d.intent.edge`), then open at
`ctx.entry_price(d.intent.token)` and journal. The `execution_cost`/`ai_*`
outcomes come straight from the Decision — the engine no longer special-cases AI.
`_decision_row` gains a `strategy_name=strat.name` on every row.

**Step 4: Register signal candidates.** Add helper:
```python
def _register_signal(self, strat_name, ctx, intent_or_ev):
    ev = intent_or_ev
    sid = self.journal.record_signal(
        strategy=strat_name, market=ctx.market.key, token=ev.signal.token,
        side_team=ev.side_team, entry_price=ctx.entry_price(ev.signal.token),
        fair=ev.fair, edge=ev.edge, move=ev.move, spread=ctx.quote.home_spread,
        inning=ctx.game_state.inning, is_top=int(ctx.game_state.is_top),
        home_score=ctx.game_state.home_score, away_score=ctx.game_state.away_score,
        commit=False)
    self.pending_cf.append({"signal_id": sid, "token": ev.signal.token,
                            "market_key": ctx.market.key, "born": ctx.now,
                            "remaining": set(HORIZONS)})
```
Call it whenever `d.signal_candidate` is true. (Define `HORIZONS = (5, 15, 30, 60, 120, 300)` in `strategies.py`.)

**Step 5: Add `_flush_counterfactuals`,** called once per tick in `run()` after
`_look_for_entries`:
```python
def _flush_counterfactuals(self):
    now = time.time()
    still: list[dict] = []
    for sig in self.pending_cf:
        due = {h for h in sig["remaining"] if now - sig["born"] >= h}
        for h in sorted(due):
            market = self.markets.get(sig["market_key"])
            quote = self.latest_quotes.get(sig["market_key"])
            fresh = quote is not None and now - quote.ts <= self.cfg.strategy.max_quote_age_secs
            if market is None:
                pass
            elif fresh:
                bid = market and quote and self._exec_bid(market, quote, sig["token"])
                ask = self._exec_ask(market, quote, sig["token"])
                self.journal.record_counterfactual(
                    sig["signal_id"], h, exec_bid=bid, exec_ask=ask,
                    mid=(bid + ask) / 2, two_sided=1, spread=ask - bid, commit=False)
            else:
                self.journal.record_counterfactual(
                    sig["signal_id"], h, exec_bid=None, exec_ask=None,
                    mid=self.histories[sig["market_key"]].last if sig["market_key"] in self.histories else None,
                    two_sided=0, spread=None, commit=False)
        sig["remaining"] -= due
        if sig["remaining"]:
            still.append(sig)
    self.pending_cf = still
    if self.pending_cf is not still or still != []:
        self.journal.conn.commit()
```
Add `_exec_bid`/`_exec_ask` static helpers mirroring `StratContext.exit_price`/`entry_price`
(or reuse `_entry_price` + a new `_exit_price`). Keep the executable-price logic in ONE place — import the context helpers if cleaner.

**Step 6: Call `strat.close()` for every strategy** in `run()`'s `finally` block
(shuts the AI thread pool down).

**Step 7: Migrate `tests/test_engine.py`** — replace `"math"` with `"fade_v1_frozen"`
in `test_final_game_settles_paper_position_at_official_outcome`,
`test_paper_account_restores_after_engine_restart`, and any other place. Update
`test_stale_quote_blocks_entries_and_is_counted` if the funnel key path changed.
Add new tests:
```python
def test_engine_runs_multiple_frozen_strategies(tmp_path):
    engine = make_engine(tmp_path)
    assert "fade_v1_frozen" in engine.strategies
    assert "fade_tight" in engine.strategies
    assert set(engine.broker.cash) == set(engine.strategies)


def test_counterfactuals_recorded_after_horizon(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    engine.latest_quotes[market.key] = MarketQuote(market.key, 0.50, 0.52, 0.50, 0.52)
    engine.histories[market.key].add(0.51)
    sid = engine.journal.record_signal(
        strategy="fade_v1_frozen", market=market.key, token=market.home_token,
        side_team="Homers", entry_price=0.52, fair=0.6, edge=0.08, move=-0.1,
        spread=0.02, inning=7, is_top=1, home_score=4, away_score=1)
    engine.pending_cf.append({"signal_id": sid, "token": market.home_token,
                              "market_key": market.key, "born": time.time() - 31,
                              "remaining": {30}})
    engine._flush_counterfactuals()
    row = engine.journal.conn.execute(
        "SELECT * FROM signal_counterfactuals WHERE signal_id=?", (sid,)).fetchone()
    assert row["horizon_secs"] == 30
    assert row["exec_ask"] == 0.52
    assert engine.pending_cf == []
```

**Step 8: Run the full suite** `python -m pytest -q` — expect PASS.

**Step 9: Commit** `feat: drive engine from strategy registry with counterfactual capture`

---

## Task 7: Cleanup + full verification

**Files:** `polybot/engine.py`, `polybot/dashboard.py`, `README`/docs as needed.

**Step 1:** Grep for stale `MATH`/`AI`/`"math"`/`"ai"` references across `polybot/`
and `tests/`; fix any missed spot (dashboard labels, report.py, review.py).

Run: `grep -rn '"math"\|"ai"\|MATH\|\bAI\b' polybot/ tests/`

**Step 2:** Confirm `dashboard.py` renders per-registry-name strategies (it reads
`engine.strategies` / broker keys — verify it isn't hardcoded to two rows).

**Step 3:** Run full suite + a lint/type pass if configured.

Run: `python -m pytest -q`
Expected: all green.

**Step 4:** Manual smoke — build an engine against a temp db, feed one two-sided
book + one live game state through `_poll_prices` → `_look_for_entries` →
`_flush_counterfactuals`, and confirm `signals` + `signal_counterfactuals` populate
and each strategy has its own account row. Use `@superpowers:verification-before-completion`.

**Step 5: Commit** `chore: phase 1 cleanup and verification`

---

## Done criteria

- `build_strategies` yields `fade_v1_frozen`, `fade_tight`, and (with a key) `ai_shadow`.
- Each strategy has an independent broker account and can hold different positions.
- AI judge runs off the hot path; approving verdicts open at the current executable price.
- Every signal-grade candidate writes a `signals` row and up to six
  `signal_counterfactuals` rows.
- `python -m pytest -q` is green.
