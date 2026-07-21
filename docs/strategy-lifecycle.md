# Strategy lifecycle ledger

Append-only. One row per death, revival, or retirement.

- **revived** — the account fell below the minimum stake and received its
  one-time second-chance deposit. `deposited` grows by that amount, so the
  return percentage is measured against total capital in, not the original
  bankroll. A revival is never profit.
- **retired** — the account fell below the minimum stake *again* after its
  second chance. It stops opening positions permanently. Its track record
  stands as the final verdict on that hypothesis.

| UTC | strategy | event | cash | deposited | revivals | closes | realized P&L |
|---|---|---|--:|--:|--:|--:|--:|
| 2026-07-21 06:27 | `momentum_fast_v1` | **revived** | 54.47 | 150.00 | 1 | 150 | -95.53 |
| 2026-07-21 06:27 | `news_underreact_v1` | **revived** | 54.67 | 150.00 | 1 | 189 | -95.33 |
| 2026-07-21 06:27 | `news_underreact_bases_v1` | **revived** | 54.84 | 150.00 | 1 | 170 | -95.16 |
| 2026-07-21 06:27 | `spread_shock_v1` | **revived** | 54.69 | 150.00 | 1 | 146 | -95.31 |
| 2026-07-21 06:27 | `momentum_fast_v2` | **revived** | 54.21 | 150.00 | 1 | 173 | -95.79 |
| 2026-07-21 06:27 | `momentum_slow_v2` | **revived** | 53.33 | 150.00 | 1 | 165 | -96.67 |
| 2026-07-21 06:27 | `momentum_confirmed_v2` | **revived** | 54.92 | 150.00 | 1 | 124 | -95.08 |
| 2026-07-21 06:27 | `news_underreact_v2` | **revived** | 54.41 | 150.00 | 1 | 187 | -95.59 |
| 2026-07-21 06:27 | `news_underreact_bases_v2` | **revived** | 53.39 | 150.00 | 1 | 167 | -96.61 |
| 2026-07-21 06:27 | `longshot_value_v2` | **revived** | 50.12 | 150.00 | 1 | 14 | -99.88 |
| 2026-07-21 06:27 | `spread_shock_v2` | **revived** | 53.75 | 150.00 | 1 | 150 | -96.25 |
