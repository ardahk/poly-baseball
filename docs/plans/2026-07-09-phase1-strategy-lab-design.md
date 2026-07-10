# Phase 1: Strategy Laboratory — Design

Date: 2026-07-09
Status: approved

## Goal

Replace the hardcoded `math` / `ai` branches with a strategy interface so several
**frozen** strategies run simultaneously on one shared market/game stream, each
with its own independent shadow portfolio. Separate signal quality from execution
quality, capture post-signal counterfactual prices, and make the AI judge
asynchronous and shadow-only.

Out of scope (later phases): new model logic (state-residual / market-anchored),
maker-fill simulation (passive_maker), the global-clock causal simulator (Phase 2),
walk-forward promotion (Phase 4).

## Decisions

- **Fresh strategy names.** Registry names replace `math` / `ai`. The old paper
  ledger (`math` / `ai` rows) does not migrate — this is a clean experiment start.
- **Counterfactuals captured live in-engine** (a scheduler tracking pending
  signals), not reconstructed offline.
- **AI async via a background `ThreadPoolExecutor`** inside the engine, opening a
  shadow position at the *then-current* executable price when a verdict returns.

## Components

### 1. Strategy interface + registry (`polybot/strategies.py`)

```python
@dataclass
class StratContext:
    market: Market
    history: PriceHistory
    game_state: GameState
    quote: MarketQuote
    now: float

class Strategy:
    name: str
    version: str
    config: StrategyConfig
    def evaluate(self, ctx) -> list[Intent]: ...       # entries wanted
    def manage(self, ctx, positions) -> list[Exit]: ... # closes wanted
```

Two kinds this pass:
- `FadeStrategy` — wraps existing `evaluate_entry` / `check_exit` with its own
  frozen `StrategyConfig`. One instance per config variant.
- `AIShadowStrategy` — wraps a base `FadeStrategy` for candidate generation; judges
  asynchronously (see §4).

Registry: a `strategies:` list in `config.yaml`, each entry
`{name, kind, base?, overrides:{...}}`. Engine builds the Strategy list from it and
drops the `MATH` / `AI` constants. Default registry:
- `fade_v1_frozen` — today's `strategy:` config verbatim (the control).
- `fade_tight` — one cheap config-only variant (higher `move_threshold`, `min_edge`).
- `ai_shadow` — `kind: ai_shadow`, `base: fade_v1_frozen`.

### 2. Shadow portfolios

`PaperBroker` / `RiskManager` already key everything by strategy string, so they are
already N independent books. Seed them from registry names instead of `[MATH, AI]`.
All strategies fill at the same executable price model, so differences come only
from signal config → clean signal-vs-execution separation.

### 3. Engine loop rewrite

`_look_for_entries`: build one `StratContext` per live, fresh-quoted market, then
`for strat in strategies: strat.evaluate(ctx)`. Each Intent carries that strategy's
own token/price/edge. `_manage_exits` becomes `strat.manage(ctx, positions)`. Engine
keeps ownership of fills, journaling, risk gating, cooldowns, decision rows; the
strategies are pure decision functions. Decision/funnel rows now carry the real
per-strategy outcome.

### 4. Async AI (background thread pool)

`AIShadowStrategy` holds `ThreadPoolExecutor(max_workers=2)` and
`pending: dict[market_key -> Future]`. On `evaluate`:
1. Ask base fade strategy for candidates. New candidate, no in-flight future, no
   open position → submit `judge(snapshot)` (snapshot frozen at submit time),
   return no intent yet.
2. Drain done futures: approved verdict → emit an Intent to open at ctx's current
   executable price. Record judge latency (submit→resolve) and price drift over the
   wait to measure adverse selection during the think.

Fails closed on error. Executor shut down on engine close.

### 5. Counterfactual scheduler + new tables

Engine keeps `pending_cf` for every `outcome == "signal"` (whether or not a
portfolio opened). Tables:

```sql
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, run_id TEXT, strategy TEXT, market TEXT, token TEXT,
    side_team TEXT, entry_price REAL, fair REAL, edge REAL, move REAL,
    spread REAL, inning INTEGER, is_top INTEGER, home_score INTEGER, away_score INTEGER
);
CREATE TABLE signal_counterfactuals (
    signal_id INTEGER, horizon_secs INTEGER, ts REAL,
    exec_bid REAL, exec_ask REAL, mid REAL, two_sided INTEGER, spread REAL
);
```

Each tick: for every pending signal and any elapsed horizon
(5/15/30/60/120/300s), snapshot the current executable bid/ask for that token from
`latest_quotes` → one `signal_counterfactuals` row. One-sided at a horizon →
`two_sided=0`, mid only (an execution finding). Drop the signal after 300s fires.

## Testing

TDD throughout. Key tests:
- Registry builds N strategies from config; unknown kind errors.
- `FadeStrategy` reproduces current `evaluate_entry` behaviour for the frozen config.
- Two fade variants with different configs produce different signal sets on the same
  stream; each opens only in its own portfolio.
- `AIShadowStrategy` submits once per candidate, opens on approve at current price,
  never blocks; fails closed on judge error.
- Counterfactual scheduler writes one row per elapsed horizon; one-sided → two_sided=0.
- Existing engine tests migrated from `math` to registry names.
