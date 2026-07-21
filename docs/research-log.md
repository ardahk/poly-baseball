# Research log

Every entry: the exact command, its output, and whether it beat the bar.

**The bars (stated once):**
- A candidate **model** must beat the market's Brier of **0.1716** out-of-sample.
- A candidate **strategy** must show per-game-clustered mean pnl/contract **> +0.02** with a bootstrap 95% CI lower bound **> 0**.

---

## 2026-07-12 — Session 1

### 0. Collector was DOWN for 12h; restarted

`journalctl` showed a clean manual stop at 06:43 UTC (SIGTERM, "Deactivated successfully", 31.9M peak memory — not a crash or OOM). Restarted:

```
$ sudo systemctl start polybot
Active: active (running) since Sun 2026-07-12 18:26:13 UTC
```

Confirmed writing within 30s — 12 live games captured. Out-of-sample progress on the preregistered hypothesis was **0 / 60 new games**: the 16 games in the DB *are* the original 16, all from a single slate (2026-07-11, ts range 15:35 UTC → 03:56 UTC).

**Recorded, not fixed:** games that started between 16:15 and 18:26 UTC today have **no pregame price ticks**, so they have no pregame anchor. They are unusable for the market-anchored model (WS2) and should be excluded from the preregistered OOS set, which needs games observed from first pitch.

### 1. Baselines reproduce exactly ✅

```
$ cp polybot.db /tmp/snap.db && sqlite3 /tmp/snap.db "PRAGMA integrity_check;"
ok
$ ./.venv/bin/python scripts/model_vs_market.py /tmp/snap.db 0.05 artifacts/state_v1.json
db=/tmp/snap.db  scored ticks=88,088  min_edge=0.05
                     Brier   (lower is better)
market              0.1716
empirical:state_v1.json    0.1739
  --> MARKET beats model by 0.0024 Brier

inning           n   Brier_mkt   Brier_mdl    winner
1-3          36359      0.2368      0.2359     model
4-6          26645      0.1761      0.1802    market
7+           25084      0.0722      0.0775    market
```

Market **0.1716**, empirical **0.1739** — matches the brief. Baseline confirmed.

### 2. 🚩 The "positive cells" that motivated the preregistration are a tick-pooling artifact

`model_markout.py` reports two columns. The hypothesis was built on the wrong one.

```
$ ./.venv/bin/python scripts/model_markout.py /tmp/snap.db 0.05 artifacts/state_v1.json
inning  gap         games   ticks    win%   pnl/tick   pnl/game           ci95_game
1-3     .05-.10        15   10453   38.3%    -0.0548    -0.0830  [-0.2592, +0.1019]
1-3     >=.10          16    6973   47.6%     0.0697    -0.0729  [-0.2641, +0.1400]
4-6     .05-.10        15    6014   51.4%     0.0191    -0.0775  [-0.2742, +0.1173]
4-6     >=.10          15    4746   52.3%     0.0414    -0.0661  [-0.2533, +0.1449]
7+      .05-.10        14    3126   47.2%    -0.0177    -0.0626  [-0.1734, +0.0359]
7+      >=.10          11     863    8.5%    -0.2083    -0.1579  [-0.2389, -0.0814]

ALL                    16   32175   44.9%    -0.0003    -0.0656  [-0.2379, +0.1312]
PREREG  <=6,>=.10      16   11719   49.5%     0.0583    -0.0637  [-0.2501, +0.1421]
```

(`ci95_game` and the `PREREG` row are new — see §3. The pnl math is untouched.)

**The preregistered cell (innings ≤ 6, |gap| ≥ 0.10), in-sample:**

| statistic | value |
|---|---|
| pooled `pnl/tick` (pseudo-replicated) | **+0.0583** ← what motivated the hypothesis |
| per-game clustered mean (**the preregistered metric**) | **−0.0637** |
| median game | **−0.2325** |
| games profitable | **5 / 16** |

The sign flips. A few long games where the model happened to be right (823926: 3,559 ticks @ +0.63; 823356: 1,648 @ +0.52) dominate the pooled average; the typical game loses. **Every one of the six cells is negative per-game.** Verified twice — once via `model_markout.py`, once via an independent re-derivation that reproduces the tick count exactly (11,719 = 6,973 + 4,746).

The script's own footer already warned: *"pnl/game clusters by game; trust it over pnl/tick."* The prior claim of "positive cells (innings 1-6, |model-market|>=0.10)" read the pooled column.

This does **not** formally reject the hypothesis (its test is out-of-sample), but the motivating evidence never existed under the prereg's own clustering rule.

### 3. 🚩 The preregistered test is structurally unpassable at n=60

Criterion #2 (bootstrap CI) was never implemented. I added it by reusing `_cluster_ci()` from `polybot/walkforward.py:209` (2000-sample cluster bootstrap). It reveals the test cannot work at its own sample size.

Per-game pnl/contract SD = **0.4124**. This is irreducible: within one game every qualifying tick is essentially the *same* bet (same game, same side, perfectly correlated), so a game contributes ~one binary outcome.

```
$ ./.venv/bin/python scratchpad/power.py
per-game pnl/contract: mean=-0.0637  sd=0.4124  n=16

 true edge  n games needed  slates (~15/day)
     +0.02            1151                77
     +0.05             185                12
     +0.10              47                 3

What the locked n=60 test can actually detect:
  n=  16: 95% one-sided margin = 0.170  -> observed mean must exceed +0.170 for CI_low > 0
  n=  60: 95% one-sided margin = 0.088  -> observed mean must exceed +0.088 for CI_low > 0
  n=1000: 95% one-sided margin = 0.021  -> observed mean must exceed +0.021 for CI_low > 0
```

**At n=60 the two locked criteria are mutually inconsistent.** Criterion #1 asks for mean > +0.02; criterion #2 (CI low > 0) is only satisfied if the observed mean exceeds **+0.088** — 4.4× higher. Any result that marginally passes #1 (e.g. +0.03) necessarily fails #2. The test can only pass on an edge of ~+0.09 or more.

Detecting the +0.02 edge the prereg actually targets requires **~1,151 games ≈ 2.5 months** of collection.

*The locked criteria are not being changed.* This characterizes their power; it does not move them.

### 4. 🚩 The "~5¢ cost floor" premise is wrong — and the error changes the conclusion

**Spread is 0.5¢, not 2¢.** From 92,229 two-sided ticks:

| home_spread | ticks | % |
|---|---|---|
| **0.005** | **87,371** | **94.7%** |
| 0.010 | 2,187 | 2.4% |
| ≥0.015 | ~2,671 | 2.9% |

mean = 0.0061, min 0.0, max 0.40. Crossing costs ~0.25¢ one-way.

**Taker fee is `θ·p·(1−p)`, θ=0.06** (`polybot/broker.py:14-21`) → **1.5¢ at p=0.5**, per side. Confirmed against https://docs.polymarket.us/fees (taker θ=0.06, max $1.50/100 shares at p=0.50).

**Makers are not charged a fee — they receive a rebate.** Published maker θ = **−0.0125** (≈ +0.31¢ credit at p=0.5).

| Execution | Cost/contract @ p=0.5 |
|---|---|
| Taker round-trip (the fade) | 2×1.5¢ + 0.5¢ ≈ **3.5¢** |
| Taker hold-to-settlement (no exit fee) | 1.5¢ + 0.25¢ ≈ **1.75¢** |
| **Maker hold-to-settlement** | ≈ **0¢ or negative** (rebate) |

The floor is **~86% fee, ~14% spread**. Posting liquidity to "save the spread" saves ~0.5¢ — nearly nothing. What maker execution actually saves is the *fee*. And because the fee is `p(1−p)`-shaped, cost collapses at the tails (~0.5¢ at p=0.9 vs 1.5¢ at p=0.5).

This does not resurrect the fade (dead on win-rate: 20–35%, and it is the family that pays the fee *twice*). But "no edge survives costs" was concluded against a floor **~2–3× too high**.

The real cost of maker execution is **adverse selection**, which no fee schedule can tell us and which has never been measured here. → WS3.

### 5. WS2 — the market-anchored forecast: REFUTED, and we know why ❌

New `scripts/predictors.py` (registry) + `scripts/brier_harness.py` (scores all candidates on the *identical* tick set, clustered by game, paired bootstrap CI vs market). `model_vs_market.py` left untouched so the baseline can't drift.

```
$ ./.venv/bin/python scripts/brier_harness.py /tmp/snap.db analytic empirical \
      anchored:empirical:0 anchored:empirical:0.5 anchored:empirical:1.0 anchored:analytic:1.0
db=/tmp/snap.db  games=16  scored ticks=88,088  (dropped 0 ticks undefined for some candidate)

predictor                       Brier  vs market     ci95 (paired, per-game)  beats bar?
market (live mid)              0.1686                                                 --
analytic                       0.1724    +0.0038          [-0.0236, +0.0256]          no
empirical:state_v1.json        0.1710    +0.0024          [-0.0256, +0.0289]          no
anchored(empirical,b=0)        0.2544    +0.0858          [+0.0251, +0.1384]          no
anchored(empirical,b=0.5)      0.1832    +0.0146          [-0.0148, +0.0378]          no
anchored(empirical,b=1)        0.1729    +0.0043          [-0.0046, +0.0126]          no
anchored(analytic,b=1)         0.1737    +0.0051          [-0.0137, +0.0182]          no
```

Finer sweep: b=0.75 → 0.1749, b=1.0 → 0.1729, b=1.25 → 0.1739, b=1.5 → 0.1761. **β=1 is the optimum.**

**Nothing beats the market.** Anchoring does not help — anchored(empirical, β=1) = 0.1729 is *worse* than plain empirical (0.1710). Failed the 0.1716 bar.

**The β=0 row is the real finding.** β=0 is "hold the pregame price all game, ignore the game entirely" → Brier **0.2544** vs market 0.1686, significantly worse (paired CI [+0.0251, +0.1384] excludes zero).

The *only* mechanism by which an anchored model could beat the live market is if the live market were `pregame price + noise` — i.e. if in-game price moves were overshoot to be faded. β=0 shows the opposite with significance: the in-game market's updates carry **8.6 Brier points** of genuine information. There is no in-game noise to arbitrage. **The premise behind the anchored idea is empirically false.** Do not revisit.

### 6. WS3 — the cost floor and adverse selection

`scripts/cost_floor.py`. Part A, corrected floor per contract:

```
  price  taker fee   taker RT  taker hold  maker hold
   0.50     0.0150     0.0350      0.0175     -0.0056
   0.90     0.0054     0.0158      0.0079     -0.0036
```

Part B (settlement PnL on simulated resting bids) produced an apparent **+0.03 net edge** — **this is a mirage and must not be believed.** Home teams won **9/16 (56.25%)** while the tick-weighted mean home bid was **52.59%**. That 3.7pt base-rate fluke (0.3σ at n=16; SE of a win rate at n=16 is 12.5pt) *is* the entire "unconditional edge to buying home at the bid." Only the *difference* `uncond − cond` is identified (the bias cancels): adverse selection ≈ 0.2¢–1.2¢.

Part C fixes this properly by marking out against the future **mid** instead of settlement — unbiased by the home-win fluke and far lower variance:

```
PART C - maker markout vs future MID (order rests 30s, front-of-queue assumed)
  markout    fills       raw   +rebate  vs taker*
     30s    22623   -0.0112   -0.0090    +0.0085
     60s    22534   -0.0114   -0.0091    +0.0084
    300s    21960   -0.0105   -0.0082    +0.0093
    900s    21071   -0.0075   -0.0052    +0.0123
```

**Adverse selection is real: −1.12¢.** Post a bid, get filled, and 30s later the mid sits 1.1¢ below your fill — you were picked off. The rebate offsets only 0.22¢.

| | cost/contract |
|---|---|
| Taker hold-to-settlement | **1.75¢** (0.25¢ half-spread + 1.5¢ fee) |
| Maker hold (front-of-queue, optimistic) | **0.90¢** (1.12¢ adverse selection − 0.22¢ rebate) |
| **Maker advantage over taker** | **~0.85¢** |

Two conclusions, both important:

1. **Naked market-making is NEGATIVE (−0.90¢/contract).** The rebate does not cover adverse selection. "Pivot to maker orders" is *not* a free lunch, and the ~0¢ floor implied by the fee schedule alone is wrong.
2. **Maker execution is still ~0.85¢ cheaper than taker** — real, but modest. It roughly halves the floor (1.75¢ → 0.90¢); it does not remove it.

Both numbers are **optimistic**: the fill model assumes front-of-queue and fills whenever the level is touched. Real queue position means you miss benign fills but still take every toxic one, so true adverse selection is *worse* than −1.12¢.

### 7. WS4 — preregistered evaluator built and gated

`scripts/prereg_eval.py`; the 16 in-sample game_pks pinned to `artifacts/prereg_excluded_games.json` (they previously existed only inside the DB, so the exclusion set was not tamper-evident). The locked doc and all thresholds are untouched.

```
$ ./.venv/bin/python scripts/prereg_eval.py /tmp/snap2.db
preregistered OOS progress: 0 / 60 new games
GATE: below 60 games. Not evaluating -- no pnl computed, by design.
```

The gate is structural: below 60 new games it computes no pnl at all, so the result cannot be peeked at.

---

## Session 1 summary

**Nothing beat the market baseline. Nothing came close.**

| Candidate | Result vs bar |
|---|---|
| analytic (0.1724) | ❌ worse than market |
| empirical state_v1 (0.1710) | ❌ worse than market (0.1686 clustered) |
| anchored, β=1 (0.1729) | ❌ worse than market *and* worse than plain empirical |
| anchored, β=0 (0.2544) | ❌ significantly worse — but this is the informative one |
| hold-to-settlement disagreement strategy | ❌ −0.0637/contract per-game in-sample |
| naked maker liquidity provision | ❌ −0.0090/contract |

**What we learned that is worth keeping:**

1. The forecast gap is **not** the binding constraint, but neither is the cost floor as previously stated. We have **negative** alpha: our best model is 0.24 Brier points *worse* than the market. Halving the cost of trading a coin flip still loses.
2. The cost floor was overstated ~2–3× (real: 1.75¢ taker-hold, ~0.90¢ maker), but maker execution is not free either — adverse selection eats it.
3. The apparent "positive cells" and the apparent "+3¢ maker edge" were both **small-sample artifacts** (tick pooling; 9/16 home wins). At 16 games, every effect we can measure is smaller than the noise.
4. The preregistered test **cannot pass at n=60** — its two criteria are mutually inconsistent at that sample size.

**Required alpha to break even:** >1.75¢ taker-hold, >0.90¢ maker-hold. Current alpha is negative.

**Recommendation:** keep the collector up and accumulate games. Do not build new strategies against 16 games — at this sample size the noise (per-game SD 0.41) exceeds every effect we are trying to detect by an order of magnitude. The single highest-value action is *more data*, not more models.


---

## 2026-07-21 — Session 2

### 1. 🔴 The preregistered hypothesis has REJECTED. The gate fired unattended.

The OOS counter crossed its threshold and nobody looked. Run today:

```
$ ./.venv/bin/python scripts/prereg_eval.py polybot.db
db=polybot.db
preregistered OOS progress: 66 / 60 new games
  (excluded in-sample: 16; final games in db: 93)

games traded      : 64
mean pnl/contract : -0.0304   (bar: > +0.02)
bootstrap 95% CI  : [-0.1128, +0.0546]   (bar: lower bound > 0)

RESULT: REJECT
Hypothesis rejected under its own locked criteria. Pivot per the plan; do not retune and retest.
```

"State-residual fades have positive after-cost expectancy" is **refuted** on its
own preregistered terms, out of sample, at n=64 games. Note the mean is not
merely below the +0.02 bar — it is **negative**, and the point estimate sits
outside the CI's positive half.

**Action taken:** the model-vs-market-gap family (`fade_*`, `settle_*`,
`state_residual_v1`, `market_anchor_v1`) is retired from active trading via
`retired_strategies`. Configs are left frozen and untouched. Per §3 of Session 1
and the prereg's own instruction, **this family is not retuned and not
retested.** New work must target mechanisms the preregistration did not cover.

### 2. Fleet audit: two thirds of all losses are transaction costs, not bad ideas

Full-fleet audit at 4,087 closed round trips, 55 strategies:

```
closed round trips: 4087   total net P&L: $-2296.68
fleet equity: $3203.32 of $5500 started  (-41.8%)

taker fees          : $ 1058.10   (46.1% of the loss)
spread crossing ~   : $  509.29   (22.2% of the loss)
residual (signal)   : $  729.29   (31.8% of the loss)
```

Per round trip, by exit style:

```
intraday (2 legs)    n=3071  fee=5.64% of notional   grossAvg=-5.78%  -> net -11.42%
settlement (1 leg)   n=1016  fee=2.33% of notional   grossAvg=-5.11%  -> net  -7.45%
```

The mechanisms are roughly break-even *before* costs. Costs are what kill them.

### 3. 🚩 `trades.pnl_pct` is GROSS — the report compared it against a net return

`pnl_pct` equals `(exit-entry)/entry` exactly; both fee legs live only in
`pnl_usd`. Verified on raw rows:

```
ep=0.355 xp=0.260  gross=-26.76%  pnl_pct=-26.76%  fees=$0.35 on $5.00 notional
```

`report.py` and `update_readme_stats.py` printed that gross figure in the "avg %"
column beside a fee-inclusive "return %". **That is the whole -5%/trade vs -95%
overall paradox.** Everything reported between 0% and +6% avg was really losing
money: `news_underreact_score_v1` (+0.04% gross), `settle_gap10_v1` (+0.15%),
`cell_leader_coinflip_v1` (+1.33%), `favorite_mid_v1` (+1.46%).

Fixed: `strategy_stats()` now rebuilds a net per-trade percentage by joining each
CLOSE to its OPEN for the entry notional. `pnl_pct` stays gross by definition and
is documented as such at the DDL.

### 4. 🚩 The leaderboard was selection on variance, not a track record

No strategy is statistically distinguishable from zero on the upside — best
t-stat in the fleet is **+1.11** (`news_late_v2`), against a Bonferroni
requirement of |t| > 3.2 at 55 strategies. Every |t| > 3.2 result is negative.

Remove each top-5 entry's best 3 trades:

| strategy | total | ex-top-3 |
|---|--:|--:|
| news_late_v2 | +73.70 | **-16.39** |
| settle_gap10_v2 | +64.03 | **-46.13** |
| settle_away_v2 | +22.04 | **-66.11** |
| settle_gap05_early_v2 | +13.43 | **-44.73** |
| cell_home_dog_v2 | +8.37 | **-24.79** |
| settle_anchored_v2 | +5.93 | **-52.55** |

All six flip negative. `settle_gap10_v2` and `settle_away_v2` share the *same*
+$54.01 trade — the top of the leaderboard was one Athletics settlement counted
twice.

These are also not 55 independent bets: **14.3 strategies on average** enter the
same (market, side, day); peak 36 on one side of one game.

Fixed: the README block now requires a per-game clustered bootstrap 95% CI lower
bound > 0 (the same bar `prereg_eval.py` uses) and prints "no strategy has
cleared the bar" when nothing qualifies. Today, nothing qualifies.

### 5. Fee schedule re-verified live — θ=0.06 is correct

The gateway payload the bot already parses carries a per-market
`feeCoefficient`. Queried today across all MLB markets:

```
MONEYLINE (baseball_team_full_game_winner)  theta=0.06
baseball_team_full_game_spread              theta=0.06   n=200
baseball_team_first_five_total               theta=0.06   n=150
```

Matches https://docs.polymarket.us/fees (taker θ=0.06, maker θ=-0.0125, banker's
rounding, exchange-wide from 2026-07-01). Third-party pages quoting "sports =
0.05" describe the **legacy global** Polymarket at help.polymarket.com, a
different venue. `broker.settle()` correctly charges zero on redemption.

The bot now reads `feeCoefficient` per market rather than trusting the config
constant, so it self-corrects if the exchange reprices.

### 6. Cost as a share of notional collapses at the price tails

Fee is `θ·p(1−p)`, so the all-in cost of a position **falls monotonically with
entry price**:

```
  price   taker RT %notional   taker hold %notional
   0.10          15.8%                 7.9%
   0.30          10.1%                 5.0%
   0.50           7.0%                 3.5%
   0.80           3.0%                 1.5%
   0.90           1.8%                 0.9%
```

The house style — small % gains, many trades — is **only viable in the high
price tail**. The standing `min_price: 0.10 / max_price: 0.85` band does the
opposite: it excludes the cheapest zone and includes the most expensive one.
This, not signal, is the most promising lever available, and it is the thesis
behind the v3 set.

### 7. WS4 — maker execution halves the cost floor and rescues **nothing**

`scripts/maker_vs_taker.py` (new). Part A re-measures adverse selection on the
current tape — 108,503 simulated resting-bid fills, up from Session 1's 16-game
sample — using the Part C markout-vs-future-mid method:

```
  simulated resting-bid fills : 108,503
  markout vs mid(+60s)        : -0.0110 /contract
  => adverse selection        : +0.0110 /contract (a cost)
  maker rebate at p=0.50      : +0.0031 /contract (a credit)
  naked market-making         : -0.0079 /contract (NEGATIVE)

  taker cost at p=0.50        : 0.0175 /contract
  maker cost at p=0.50        : 0.0079 /contract
  maker advantage             : +0.0096 /contract
```

Session 1's numbers reproduce almost exactly on ~6x more data: adverse selection
−1.10¢ (was −1.12¢), maker advantage +0.96¢ (was +0.85¢). **Naked liquidity
provision remains net negative** — the rebate covers barely a quarter of the
adverse selection.

Part B restates every strategy's real trade history under maker execution
(entry at bid, rebate instead of fee, minus measured adverse selection):

```
  0 of 52 strategies cross into profit under maker execution; 43 still lose.
```

**Maker execution is real but not a rescue.** It roughly halves the cost floor
(1.75¢ → 0.79¢) and hands the worst strategies back $25–48 each, yet not one
crosses zero. The strategies that were already positive gain only $1–7.

The most informative rows are the *smallest* deltas: `favorite_late_v1` +$0.04,
`favorite_model_agree_v1` −$0.10, `anti_longshot_v1` +$0.47. Tail-price
strategies gain nothing from maker execution **because at p≈0.90 there is
almost no fee left to save**. That independently confirms §6: the price tail and
maker execution are substitutes, and the tail is free.

**Conclusion: do not build live maker order placement yet.** It cannot rescue a
strategy with no edge, and the one place it would help most (at the money) is
exactly where the fee is largest and the cost floor highest. Getting to the tail
is the cheaper move and requires no new infrastructure.
