"""Performance report: frozen-strategy comparison across shadow portfolios."""
from __future__ import annotations

from datetime import datetime

from .journal import Journal


def print_report(db_path: str = "polybot.db", starting_cash: float = 100.0) -> None:
    journal = Journal(db_path)
    stats = journal.strategy_stats()
    equity = dict(journal.latest_equity())

    print("=" * 68)
    print("POLYBOT PERFORMANCE — frozen strategies")
    print("=" * 68)
    if not stats and not equity:
        print("No trades recorded yet.")
        journal.close()
        return

    header = f"{'strategy':<10}{'trades':>7}{'win %':>8}{'avg %':>8}{'best %':>8}{'worst %':>9}{'P&L $':>9}{'return %':>10}"
    print(header)
    print("-" * len(header))
    for s in stats:
        win_rate = 100.0 * s["wins"] / s["trades"] if s["trades"] else 0.0
        eq = equity.get(s["strategy"])
        ret = 100.0 * (eq - starting_cash) / starting_cash if eq is not None else None
        print(f"{s['strategy']:<10}{s['trades']:>7}{win_rate:>7.1f}%"
              f"{s['avg_pnl_pct'] * 100:>7.1f}%{s['best_pct'] * 100:>7.1f}%"
              f"{s['worst_pct'] * 100:>8.1f}%{s['pnl_usd']:>9.2f}"
              f"{f'{ret:>9.1f}%' if ret is not None else '      n/a'}")

    print("\nRecent trades:")
    for ts, strat, action, team, price, pnl_pct, reason in journal.recent_trades():
        when = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S")
        pnl = f" {pnl_pct * 100:+.1f}%" if pnl_pct is not None else ""
        print(f"  {when} [{strat}] {action:<5} {team:<24} @ {price:.3f}{pnl}  {reason or ''}")
    journal.close()
