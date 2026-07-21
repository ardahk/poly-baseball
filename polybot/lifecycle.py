"""Append-only ledger of strategy deaths, revivals, and retirements.

A paper account that drops below the minimum stake stops trading, but it keeps
appearing in the standings with its return frozen near -100%. That silently
corrupts every cross-strategy comparison, so each transition is written to a
human-readable markdown table as well as the database.

The ledger is intentionally append-only and plain text: it is the audit trail
for which strategies were given a second chance and which were shut down, and
it must stay readable without running the bot.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)

LEDGER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "strategy-lifecycle.md",
)

_HEADER = """# Strategy lifecycle ledger

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
"""


def record(strategy: str, event: str, cash: float, deposited: float,
           revivals: int, closes: int, realized: float,
           path: str | None = None) -> None:
    """Append one lifecycle transition. Never raises into the trading loop."""
    target = path or LEDGER_PATH
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    row = (f"| {stamp} | `{strategy}` | **{event}** | {cash:.2f} | "
           f"{deposited:.2f} | {revivals} | {closes} | {realized:+.2f} |\n")
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        new = not os.path.exists(target) or os.path.getsize(target) == 0
        with open(target, "a", encoding="utf-8") as fh:
            if new:
                fh.write(_HEADER)
            fh.write(row)
    except OSError as exc:
        # Bookkeeping must never take the engine down mid-session.
        log.warning("could not write lifecycle ledger %s: %s", target, exc)
