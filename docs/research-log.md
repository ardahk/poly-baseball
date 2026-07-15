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

