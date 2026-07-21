#!/usr/bin/env python3
"""Regenerate the auto-updated leaderboard block in README.md.

Pulls frozen-strategy performance straight from the journal (the same numbers
`python main.py report` prints), keeps only the top N by overall return, and
splices a markdown table + one-line strategy descriptions between the
`<!-- STATS:START -->` / `<!-- STATS:END -->` markers in README.md.

Percentages only — no dollar amounts ever leave this file. Run daily from a
systemd timer (see scripts/update_readme.sh).
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from polybot.config import load_config  # noqa: E402
from polybot.journal import Journal  # noqa: E402
from polybot.walkforward import _cluster_ci  # noqa: E402
START = "<!-- STATS:START -->"
END = "<!-- STATS:END -->"
# The block is built with STAMP as a placeholder so two renders can be compared
# ignoring the clock. A real timestamp is substituted only when the standings
# themselves differ — otherwise an unchanged offseason would commit daily noise.
STAMP = "@@STAMP@@"
STAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC")
MEDALS = ["🥇", "🥈", "🥉", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]

# Short, plain-English "what it does" per strategy mechanism. v2 variants inherit
# their v1 base description (they are the same mechanism, retuned). The five
# control strategies have no config `hypothesis`, so they live here too.
DESCRIPTIONS: dict[str, str] = {
    # -- fade controls --
    "fade_v1_frozen": "Original mean-reversion fade — buys the side the market overshot after a sharp swing.",
    "fade_tight": "Fade with stricter move/edge gates — fewer, higher-conviction entries.",
    "liquidity_fade_v2": "Fade variant measuring shock and volatility on fixed receipt-time windows.",
    "state_residual_v1": "Fades a short-lived market overreaction right after a new game state arrives.",
    "market_anchor_v1": "Fair value anchored to the frozen pregame price (log-odds), cancelling team-strength bias.",
    # -- momentum: trade WITH the move --
    "momentum_fast": "Trades with a sharp 90-second move — treats fast swings as real news that continues.",
    "momentum_slow": "Rides slower moves that diffuse over minutes, holding longer.",
    "momentum_confirmed": "Trades a move only when the model agrees on direction — filters dead-cat bounces.",
    "momentum_orderflow": "Buys moves with no recent game-state change — informed flow ahead of the feed.",
    # -- event_reaction: with-the-news underreaction --
    "news_underreact": "Buys in the model's direction while the market still lags a fresh game event.",
    "news_underreact_score": "Buys the market's lag specifically after scoring plays.",
    "news_underreact_bases": "Buys the lag after subtle base/out events the crowd underweights.",
    "news_late": "Buys the lag after late-inning, high-leverage events — the slowest to price in.",
    # -- extreme_hold: buy a price band, hold to settlement --
    "favorite_late": "Buys heavy late-game favorites and holds to settlement (near-zero fee at the tails).",
    "favorite_mid": "Buys mid-game favorites (0.85–0.95) that keep drifting up.",
    "favorite_model_agree": "Buys only model-confirmed favorites, held to settlement.",
    "longshot_value": "Buys model-supported longshots, held to settlement.",
    "anti_longshot": "Fades favorite-longshot bias — buys the cheap complement of an overpriced longshot.",
    # -- settlement_hold: model-vs-market gap held to settlement --
    "settle_gap10": "Holds a model-vs-market gap to settlement (one fee leg) — the preregistered rule, live.",
    "settle_gap05_early": "Holds early-inning model-market gaps to settlement.",
    "settle_anchored": "Holds a team-strength-corrected gap (frozen pregame anchor) to settlement.",
    "settle_away": "Holds away-side gaps to settlement — the model overrates home teams.",
    # -- calibration_cell: model-free bias harvesting --
    "cell_home_dog": "Model-free: buys home underdogs in a fixed price band, held to settlement.",
    "cell_away_fav_late": "Buys late away favorites, fading the crowd's late-home-comeback bias.",
    "cell_leader_coinflip": "Buys actual leaders still priced near a coin flip — the market is slow to credit leads.",
    "cell_trailer_cheap": "Buys early comebacks that are priced too cheaply.",
    "cell_extras_home": "Buys the home last-at-bat advantage in extra innings.",
    # -- microstructure: book-shape / timing --
    "spread_shock": "Reads the informed side from a spread shock that re-tightens.",
    "stale_reprice": "Trades the reprice after a one-sided book gap, which carries the outage's news.",
    "pregame_drift": "Rides late-pregame informed money (lineups/pitchers) into the early game.",
}


def describe(name: str) -> str:
    """Best short description for a strategy name (exact, then v2->v1, then base)."""
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    for suffix in ("_v2", "_v1"):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if base in DESCRIPTIONS:
                return DESCRIPTIONS[base]
            for alt in ("_v1", "_v2"):
                if (base + alt) in DESCRIPTIONS:
                    return DESCRIPTIONS[base + alt]
    return "—"


def per_game_pnl(journal: Journal) -> dict[str, dict[str, float]]:
    """Net P&L per contract per game, per strategy — the project's own metric.

    Round trips within one game are heavily correlated (up to 36 strategies take
    the same side of the same game), so trades are NOT independent samples.
    Clustering by game is what `docs/research-log.md` established as the honest
    unit and what `scripts/prereg_eval.py` tests against.
    """
    rows = journal.conn.execute(
        """SELECT c.strategy AS strategy, m.game_pk AS game_pk,
                  SUM(c.pnl_usd) AS pnl, SUM(c.qty) AS qty
           FROM trades c
           JOIN markets m ON m.slug = c.market
           WHERE c.action = 'CLOSE' AND m.game_pk IS NOT NULL AND c.qty > 0
           GROUP BY c.strategy, m.game_pk"""
    ).fetchall()
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["qty"]:
            out.setdefault(r["strategy"], {})[str(r["game_pk"])] = r["pnl"] / r["qty"]
    return out


def build_block(db_path: str, starting_cash: float, top: int, min_trades: int) -> str:
    journal = Journal(db_path)
    stats = journal.strategy_stats()
    equity = dict(journal.latest_equity())
    capital = journal.paper_capital()
    game_pnl = per_game_pnl(journal)
    journal.close()

    rows = []
    for s in stats:
        name = s["strategy"]
        eq = equity.get(name)
        if eq is None or s["trades"] < min_trades:
            continue
        acct = capital.get(name) or {}
        # Return is against every dollar deposited, so a second-chance top-up
        # can never masquerade as a gain.
        deposited = acct.get("deposited") or starting_cash
        ret = 100.0 * (eq - deposited) / deposited
        win = 100.0 * s["wins"] / s["trades"] if s["trades"] else 0.0
        games = game_pnl.get(name, {})
        lo, _hi = _cluster_ci(games, seed=f"leaderboard:{name}")
        rows.append({
            "name": name, "trades": s["trades"], "win": win,
            # Net of both taker fee legs. The gross number flattered every
            # strategy by 3-7 points and is no longer published.
            "avg": (s["avg_net_pct"] or 0.0) * 100,
            "best": (s["best_net_pct"] or 0.0) * 100, "ret": ret,
            "games": len(games), "ci_lo": lo,
            "proven": lo is not None and lo > 0 and len(games) >= min_trades,
            "retired": bool(acct.get("retired_at")),
        })
    rows.sort(key=lambda r: r["ret"], reverse=True)
    proven = [r for r in rows if r["proven"]]

    lines = [START]
    if not proven:
        # Publishing a podium here would present noise as a track record: the
        # current top entries all turn negative if their best 3 trades are
        # removed, and none clears the project's own bar.
        best = rows[0] if rows else None
        lines += [
            f"_Paper trading · percentages only · net of fees · "
            f"last checked {STAMP}._",
            "",
            "**No strategy has cleared the bar yet.** A strategy is only listed "
            "here once its per-game clustered mean P&L has a bootstrap 95% CI "
            "lower bound above zero — the same test "
            "[`scripts/prereg_eval.py`](scripts/prereg_eval.py) applies. "
            f"{len(rows)} strategies currently qualify on trade count "
            f"(≥{min_trades} closed) and **none** passes.",
            "",
            "Ranking by raw return would put a strategy on top whose entire "
            "edge is two or three lucky settlements, so the standings stay "
            "empty until something is actually distinguishable from luck.",
        ]
        if best is not None:
            lines += [
                "",
                f"_Closest so far: `{best['name']}` at {best['ret']:+.1f}% over "
                f"{best['trades']} trades across {best['games']} games — "
                f"CI lower bound {best['ci_lo']:+.4f}/contract, still below zero._",
            ]
        lines.append(END)
        return "\n".join(lines)

    shown = proven[:top]
    lines += [
        f"_Paper trading · percentages only · net of fees · top {len(shown)} of "
        f"strategies clearing the per-game bootstrap bar · standings last "
        f"changed {STAMP}._",
        "",
        "| | Strategy | Trades | Games | Win % | Net / Trade | Best Trade | Overall Return |",
        "|:--:|---|--:|--:|--:|--:|--:|--:|",
    ]
    for i, r in enumerate(shown):
        medal = MEDALS[i] if i < len(MEDALS) else f"{i + 1}"
        lines.append(
            f"| {medal} | `{r['name']}` | {r['trades']} | {r['games']} | "
            f"{r['win']:.0f}% | {r['avg']:+.1f}% | {r['best']:+.0f}% | "
            f"**{r['ret']:+.1f}%** |"
        )
    lines += ["", "**What each one does**", ""]
    for i, r in enumerate(shown):
        medal = MEDALS[i] if i < len(MEDALS) else f"{i + 1}."
        lines.append(f"- {medal} **`{r['name']}`** — {describe(r['name'])}")
    lines.append(END)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "config.yaml"))
    ap.add_argument("--readme", default=str(REPO / "README.md"))
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--min-trades", type=int, default=10,
                    help="ignore strategies with fewer closed trades (noise guard)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    block = build_block(cfg.engine.db_path, cfg.risk.starting_cash,
                        args.top, args.min_trades)

    readme_path = Path(args.readme)
    text = readme_path.read_text()
    if START not in text or END not in text:
        raise SystemExit(
            f"markers {START} / {END} not found in {readme_path}; add them first."
        )
    pre = text[: text.index(START)]
    post = text[text.index(END) + len(END):]
    old_block = text[text.index(START): text.index(END) + len(END)]

    # Compare with the old block's timestamp normalised back to the placeholder,
    # so only a real change in the standings counts as a change.
    if STAMP_RE.sub(STAMP, old_block) == block:
        print("standings unchanged; README left untouched.")
        return 0

    stamped = block.replace(STAMP, datetime.now(timezone.utc)
                            .strftime("%Y-%m-%d %H:%M UTC"))
    readme_path.write_text(pre + stamped + post)
    print(f"standings changed; README updated ({readme_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
