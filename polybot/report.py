"""Performance report: frozen-strategy comparison across shadow portfolios."""
from __future__ import annotations

from datetime import datetime

from .journal import Journal


def print_report(db_path: str = "polybot.db", starting_cash: float = 100.0) -> None:
    journal = Journal(db_path)
    stats = journal.strategy_stats()
    equity = dict(journal.latest_equity())
    capital = journal.paper_capital()

    print("=" * 68)
    print("POLYBOT PERFORMANCE — frozen strategies")
    print("=" * 68)
    if not stats and not equity:
        print("No trades recorded yet.")
        journal.close()
        return

    # "gross %" is the raw price move; "net %" subtracts both taker fee legs.
    # The gap between them is the cost of trading, and it is what separates a
    # strategy that looks flat from one that is quietly bleeding.
    header = (f"{'strategy':<26}{'trades':>7}{'win %':>7}{'gross %':>9}{'net %':>8}"
              f"{'best %':>8}{'worst %':>9}{'P&L $':>9}{'return %':>10}  state")
    print(header)
    print("-" * len(header))
    for s in stats:
        name = s["strategy"]
        win_rate = 100.0 * s["wins"] / s["trades"] if s["trades"] else 0.0
        eq = equity.get(name)
        acct = capital.get(name) or {}
        # Return is measured against every dollar ever deposited, so a
        # second-chance top-up can never be mistaken for a gain.
        deposited = acct.get("deposited") or starting_cash
        ret = 100.0 * (eq - deposited) / deposited if eq is not None else None
        net = s["avg_net_pct"]
        state = ""
        if acct.get("retired_at"):
            state = "RETIRED"
        elif acct.get("revivals"):
            state = f"revived x{acct['revivals']}"
        print(f"{name:<26}{s['trades']:>7}{win_rate:>6.1f}%"
              f"{s['avg_pnl_pct'] * 100:>8.1f}%"
              f"{(f'{net * 100:>7.1f}%' if net is not None else '    n/a ')}"
              f"{s['best_pct'] * 100:>7.1f}%"
              f"{s['worst_pct'] * 100:>8.1f}%{s['pnl_usd']:>9.2f}"
              f"{f'{ret:>9.1f}%' if ret is not None else '      n/a'}  {state}")

    print("\ngross % = price move only · net % = after both taker fee legs")
    print("return % = equity vs TOTAL capital deposited (incl. revival top-ups)")

    print("\nRecent trades:")
    for ts, strat, action, team, price, pnl_pct, reason in journal.recent_trades():
        when = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S")
        pnl = f" {pnl_pct * 100:+.1f}%" if pnl_pct is not None else ""
        print(f"  {when} [{strat}] {action:<5} {team:<24} @ {price:.3f}{pnl}  {reason or ''}")
    journal.close()
