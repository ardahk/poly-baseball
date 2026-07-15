"""The real cost floor, and whether maker execution can escape it.

Part A - the corrected floor. The "~5c round trip" premise was wrong: the
quoted spread is 0.5c on 95% of ticks, and the taker fee is theta*p*(1-p) with
theta=0.06 (=1.5c at p=0.50, per side). So the floor is ~86% FEE, not spread,
and it collapses at the tails because of the p(1-p) shape.

Part B - adverse selection, the question the fee schedule cannot answer.
Makers pay no fee and receive a rebate (theta=-0.0125), so on paper maker entry
is free. The catch is that resting orders fill precisely when the market is
moving against you. We have no L2 or trade prints, so we cannot simulate a
queue -- but we can bound it from BBO alone:

    Post a passive BUY of the home token at the current best bid b.
    Count it FILLED if the best ask trades down to <= b within H seconds
    (someone crossed our level). For those fills, settlement PnL = y - b + rebate.

Decomposition:
    unconditional  E[y] - b          ~ +half-spread  (buying below mid is good)
    conditional    E[y | filled] - b                 (what you actually get)
    adverse selection = unconditional - conditional  (the hidden cost)

If E[y|filled] - b + rebate < 0, passive liquidity provision loses money
regardless of forecast quality, and maker execution is a dead end.

This OVERSTATES fill rates: it assumes we are at the front of the queue and
always filled when our level is touched. So a NEGATIVE result is conclusive; a
positive result is an upper bound needing live paper confirmation.

Run: ./.venv/bin/python scripts/cost_floor.py /tmp/snap.db
"""
from __future__ import annotations

import os
import statistics
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polybot.broker import taker_fee
from polybot.walkforward import _cluster_ci

TAKER_THETA = 0.06     # docs.polymarket.us/fees
MAKER_THETA = -0.0125  # makers are PAID a rebate
HORIZONS = (5, 15, 30, 60, 120)


def maker_rebate(price: float) -> float:
    return -MAKER_THETA * price * (1.0 - price)


def part_a() -> None:
    print("=" * 78)
    print("PART A - cost per contract by price and execution mode")
    print("=" * 78)
    print(f"{'price':>7}{'taker fee':>11}{'taker RT':>11}{'taker hold':>12}"
          f"{'maker hold':>12}")
    print(f"{'':7}{'(1 side)':>11}{'2 fees+sprd':>11}{'1 fee+half':>12}"
          f"{'rebate+half':>12}")
    for p in (0.10, 0.25, 0.50, 0.75, 0.90):
        fee = taker_fee(TAKER_THETA, p)
        rt = 2 * fee + 0.005              # two fees + one full 0.5c spread
        hold = fee + 0.0025               # one fee + half-spread, no exit fee
        mk = -maker_rebate(p) - 0.0025    # rebate credited, and posting earns
        print(f"{p:>7.2f}{fee:>11.4f}{rt:>11.4f}{hold:>12.4f}{mk:>12.4f}")
    print("\n  taker RT   = the fade family: pays the fee TWICE. ~3.5c at p=0.5.")
    print("  taker hold = hold to settlement: one fee, no exit. ~1.75c at p=0.5.")
    print("  maker hold = post and hold: NEGATIVE cost (rebate + earned spread).")
    print("  Fee is theta*p*(1-p), so cost collapses at the tails: "
          "1.5c at p=0.5 vs 0.54c at p=0.9.")


def part_b(db: str) -> None:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    slug_pk = {r["slug"]: r["game_pk"]
               for r in con.execute("SELECT slug, game_pk FROM markets")
               if r["game_pk"] is not None}
    home_won = {r["game_pk"]: int(r["home_score"] > r["away_score"])
                for r in con.execute("SELECT game_pk, home_score, away_score "
                                     "FROM game_states WHERE status='Final'")}

    # per market: time-ordered (ts, bid, ask)
    book: dict[str, list] = {}
    for r in con.execute("SELECT market, ts, home_bid, home_ask FROM price_ticks "
                         "WHERE home_bid IS NOT NULL AND home_ask IS NOT NULL "
                         "ORDER BY market, ts"):
        pk = slug_pk.get(r["market"])
        if pk is None or pk not in home_won:
            continue
        book.setdefault(r["market"], []).append(
            (r["ts"], r["home_bid"], r["home_ask"]))

    print("\n" + "=" * 78)
    print("PART B - adverse selection on a resting bid (passive BUY home @ best bid)")
    print("=" * 78)
    print(f"{'H(s)':>5}{'games':>7}{'posts':>9}{'fill%':>8}"
          f"{'uncond':>9}{'cond':>9}{'advsel':>9}{'+rebate':>9}"
          f"{'ci95 (per-game, net)':>26}")

    for H in HORIZONS:
        # per game: [n_posts, n_fills, sum_uncond_edge, sum_net_pnl_on_fills]
        agg: dict[int, list[float]] = {}
        for market, ticks in book.items():
            pk = slug_pk[market]
            y = home_won[pk]
            n = len(ticks)
            a = agg.setdefault(pk, [0.0, 0.0, 0.0, 0.0])
            for i, (ts, bid, _ask) in enumerate(ticks):
                a[0] += 1
                a[2] += y - bid                     # unconditional: buy at bid
                # did the best ask trade down to our bid within H seconds?
                filled = False
                k = i + 1
                while k < n and ticks[k][0] <= ts + H:
                    if ticks[k][2] <= bid:          # ask <= our bid -> we get lifted
                        filled = True
                        break
                    k += 1
                if filled:
                    a[1] += 1
                    a[3] += (y - bid) + maker_rebate(bid)

        posts = sum(v[0] for v in agg.values())
        fills = sum(v[1] for v in agg.values())
        if not fills:
            print(f"{H:>5}{len(agg):>7}{int(posts):>9}{0.0:>7.1f}%"
                  f"{'':>9}{'':>9}{'':>9}{'':>9}{'no fills':>26}")
            continue
        uncond = sum(v[2] for v in agg.values()) / posts
        cond = sum(v[3] for v in agg.values()) / fills   # incl. rebate, pooled
        net_by_game = {str(pk): v[3] / v[1] for pk, v in agg.items() if v[1]}
        net = statistics.mean(net_by_game.values())
        lo, hi = _cluster_ci(net_by_game, seed=f"advsel:{H}")
        print(f"{H:>5}{len(agg):>7}{int(posts):>9}{100*fills/posts:>7.1f}%"
              f"{uncond:>+9.4f}{cond:>+9.4f}{uncond-cond:>+9.4f}{net:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>26}")

    con.close()
    print("\n  uncond  = E[y] - bid over ALL posts (no fill filter): the naive edge.")
    print("  cond    = E[y] - bid + rebate, on FILLED posts only, pooled.")
    print("  advsel  = uncond - cond: what the fills cost you by being selected.")
    print("  +rebate = per-GAME clustered mean net pnl/contract on fills. "
          "This is the number that matters.")
    print("\n  Fills are OVERSTATED (assumes front-of-queue, always filled when "
          "touched),\n  so a NEGATIVE result here is conclusive.")


def part_c(db: str) -> None:
    """Maker markout against the future MID, not settlement.

    Part B's absolute PnL is contaminated: home teams won 9/16 (56.2%) while the
    mean home bid was 52.6%, so "buy home at the bid" shows a spurious ~+3c edge
    that is pure base-rate luck on 16 games (SE of a win rate at n=16 is 12.5pt).

    Marking out against mid(t+delta) instead of the settlement outcome removes
    the game-outcome variance entirely and is the standard adverse-selection
    measure: if you are filled and the mid then walks away from you, you were
    picked off. This is both unbiased and vastly lower variance.
    """
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    book: dict[str, list] = {}
    for r in con.execute("SELECT market, ts, home_bid, home_ask, home_mid "
                         "FROM price_ticks WHERE home_bid IS NOT NULL "
                         "AND home_ask IS NOT NULL ORDER BY market, ts"):
        book.setdefault(r["market"], []).append(
            (r["ts"], r["home_bid"], r["home_ask"], r["home_mid"]))
    con.close()

    H = 30          # how long the order rests
    print("\n" + "=" * 78)
    print(f"PART C - maker markout vs future MID (order rests {H}s, "
          "front-of-queue assumed)")
    print("=" * 78)
    print(f"{'markout':>9}{'fills':>9}{'raw':>10}{'+rebate':>10}"
          f"{'vs taker*':>11}")

    for delta in (30, 60, 300, 900):
        raw_sum = reb_sum = 0.0
        fills = 0
        for ticks in book.values():
            n = len(ticks)
            for i, (ts, bid, _ask, _mid) in enumerate(ticks):
                # fill: best ask trades down to our resting bid within H
                k, filled = i + 1, False
                while k < n and ticks[k][0] <= ts + H:
                    if ticks[k][2] <= bid:
                        filled = True
                        break
                    k += 1
                if not filled:
                    continue
                fill_ts = ticks[k][0]
                # mid at fill_ts + delta
                m = k
                while m < n and ticks[m][0] <= fill_ts + delta:
                    m += 1
                if m >= n:
                    continue                # no future mid: drop, don't guess
                future_mid = ticks[m][3]
                fills += 1
                raw_sum += future_mid - bid
                reb_sum += (future_mid - bid) + maker_rebate(bid)
        if not fills:
            continue
        raw = raw_sum / fills
        net = reb_sum / fills
        # A taker buying the same exposure crosses to the ask (= mid + half the
        # 0.5c spread) and pays the fee, but suffers no fill selection: if the
        # mid is a martingale its expected markout is exactly -(half spread + fee).
        taker_markout = -(0.0025 + taker_fee(TAKER_THETA, 0.5))
        print(f"{delta:>7}s{fills:>9}{raw:>+10.4f}{net:>+10.4f}"
              f"{net - taker_markout:>+11.4f}")

    print("\n  raw     = mid(fill+delta) - bid. NEGATIVE = adverse selection: the")
    print("            market walked away from you right after filling you.")
    print("  +rebate = raw + maker rebate (-0.0125*p*(1-p), a credit).")
    print("  *vs taker = net advantage over crossing the spread and paying the")
    print("            1.5c taker fee for the same exposure at p=0.5.")
    print("\n  This is unbiased by the 9/16 home-win fluke that contaminates Part B.")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "/tmp/snap.db"
    part_a()
    part_b(db)
    part_c(db)
