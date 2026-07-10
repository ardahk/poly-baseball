#!/usr/bin/env python3
"""Polybot CLI.

Usage:
  python main.py run             # paper trade (default)
  python main.py run --dashboard # paper trade with live terminal dashboard
  python main.py run --live      # real orders (needs polymarket-us + keys)
  python main.py run --live --yes-live  # real orders without prompt (systemd)
  python main.py scan            # one-shot: show tradeable markets right now
  python main.py status          # show database activity: ticks/trades/equity
  python main.py report          # frozen-strategy performance comparison
  python main.py review          # end-of-day observability review
  python main.py backtest calibrate --days 3   # is the win-prob formula accurate?
  python main.py backtest strategy  --days 2   # would the trading logic profit?
  python main.py backtest replay --date 2026-07-08 --set strategy.stop_loss=0.15
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from polybot import backtest, mlb, pmus
from polybot.config import load_config
from polybot.engine import Engine
from polybot.journal import Journal
from polybot.report import print_report
from polybot.review import print_review
from polybot.winprob import home_win_probability


def cmd_run(args, cfg):
    if args.live:
        raise SystemExit(
            "Live trading is disabled during Phase 0. Paper results and exchange reconciliation "
            "must pass the promotion gate before real orders can be enabled."
        )
    Engine(cfg, dashboard=args.dashboard).run()


def cmd_scan(args, cfg):
    markets = pmus.fetch_mlb_markets()
    client = mlb.MLBClient()
    games = client.todays_games()
    mlb.match_markets_to_games(markets, games)
    feed = pmus.PriceFeed()
    matched = [m for m in markets if m.game_pk]
    print(f"{len(markets)} MLB markets found, {len(matched)} matched to today's games\n")
    for m in matched:
        gs = client.game_state(m.game_pk)
        mid = feed.home_midpoint(m)
        line = f"  {m.question:<50}"
        if mid is not None:
            line += f" home={mid:.3f}"
        if gs:
            line += f" [{gs.status}"
            if gs.is_live:
                fair = home_win_probability(gs)
                half = "T" if gs.is_top else "B"
                line += (f" {half}{gs.inning} {gs.away_score}-{gs.home_score}"
                         f" fair={fair:.3f}")
            line += "]"
        print(line)


def cmd_report(args, cfg):
    print_report(cfg.engine.db_path, cfg.risk.starting_cash)


def cmd_review(args, cfg):
    print_review(
        cfg.engine.db_path, day=args.date, near=args.near,
        timezone=args.timezone or cfg.engine.report_timezone,
    )


def _fmt_ts(ts):
    if ts is None:
        return "never"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def cmd_status(args, cfg):
    journal = Journal(cfg.engine.db_path)
    try:
        status = journal.db_status()
        print("POLYBOT STATUS")
        print("=" * 60)
        print(f"db path          : {cfg.engine.db_path}")
        print(f"price ticks      : {status['price_ticks']}")
        print(f"latest price tick: {_fmt_ts(status['latest_price_ts'])}")
        print(f"equity snapshots : {status['equity_snapshots']}")
        print(f"latest equity    : {_fmt_ts(status['latest_equity_ts'])}")
        print(f"decisions        : {status['decisions']}")
        print(f"latest decision  : {_fmt_ts(status['latest_decision_ts'])}")
        print(f"game states      : {status['game_states']}")
        print(f"latest game state: {_fmt_ts(status['latest_game_state_ts'])}")
        print(f"trade opens      : {status['trades'].get('OPEN', 0)}")
        print(f"trade closes     : {status['trades'].get('CLOSE', 0)}")
        accounts, positions = journal.paper_state(["math", "ai"])
        if accounts:
            print("persisted paper account:")
            for strategy, account in sorted(accounts.items()):
                open_count = sum(1 for pos in positions if pos["strategy"] == strategy)
                print(
                    f"  {strategy:<13} cash=${account['cash']:.2f} "
                    f"realized=${account['realized']:+.2f} "
                    f"open={open_count} closed={account['closes']}"
                )
        print("\nRecent priced markets:")
        rows = journal.recent_price_markets()
        if not rows:
            print("  none yet")
        for market, home, away, ticks, latest_ts, avg_mid, avg_spread in rows:
            print(
                f"  {home} vs {away:<28} ticks={ticks:<5} "
                f"last={_fmt_ts(latest_ts)} mid~{avg_mid:.3f} spread~{avg_spread:.3f}"
            )
    finally:
        journal.close()


def _cast_override(raw: str, current):
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if current is None:
        return raw
    return type(current)(raw)


def _apply_overrides(cfg, overrides: list[str]) -> None:
    for spec in overrides:
        if "=" not in spec or "." not in spec.split("=", 1)[0]:
            raise SystemExit(f"invalid --set {spec!r}; expected section.key=value")
        path, raw = spec.split("=", 1)
        section, key = path.split(".", 1)
        target = getattr(cfg, section, None)
        if target is None or not hasattr(target, key):
            raise SystemExit(f"unknown config override {path!r}")
        setattr(target, key, _cast_override(raw, getattr(target, key)))


def cmd_backtest(args, cfg):
    _apply_overrides(cfg, args.set or [])
    if args.mode == "calibrate":
        backtest.calibrate(days_back=args.days, max_games=args.max_games)
    elif args.mode == "strategy":
        backtest.strategy_backtest(cfg, days_back=args.days, max_games=args.max_games)
    else:
        backtest.strategy_replay_db(cfg, db_path=cfg.engine.db_path, day=args.date)


def main():
    parser = argparse.ArgumentParser(prog="polybot")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="start the trading loop")
    p_run.add_argument("--live", action="store_true", help="disabled during Phase 0")
    p_run.add_argument(
        "--dashboard",
        action="store_true",
        help="show a live terminal dashboard instead of relying on heartbeat logs",
    )
    p_run.add_argument(
        "--yes-live",
        action="store_true",
        help="skip the live-trading confirmation prompt; use only for unattended services",
    )
    sub.add_parser("scan", help="show current MLB markets and model fair values")
    sub.add_parser("status", help="show database activity and recent price ticks")
    sub.add_parser("report", help="print frozen-strategy performance report")
    p_review = sub.add_parser("review", help="review a recorded paper-trading day")
    p_review.add_argument("--date", help="local date to review, YYYY-MM-DD (default: today)")
    p_review.add_argument(
        "--timezone", help="IANA timezone for the trading-day boundary (default: config)",
    )
    p_review.add_argument("--near", type=float, default=0.02,
                          help="near-miss margin window for rejected gates")
    p_bt = sub.add_parser("backtest", help="validate the models on finished games")
    p_bt.add_argument("mode", choices=["calibrate", "strategy", "replay"])
    p_bt.add_argument("--days", type=int, default=3, help="days back to include")
    p_bt.add_argument("--max-games", type=int, default=40)
    p_bt.add_argument("--date", help="recorded local day for replay, YYYY-MM-DD")
    p_bt.add_argument("--set", action="append", default=[],
                      help="override config for replay/strategy, e.g. strategy.stop_loss=0.15")
    args = parser.parse_args()

    dashboard_mode = args.command == "run" and getattr(args, "dashboard", False)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING if dashboard_mode else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    cfg = load_config(args.config)
    {"run": cmd_run, "scan": cmd_scan, "status": cmd_status, "report": cmd_report,
     "review": cmd_review, "backtest": cmd_backtest}[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
