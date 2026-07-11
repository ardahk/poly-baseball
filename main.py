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
  python main.py backtest causal --date 2026-07-08 --set strategy.stop_loss=0.15
  python main.py research diagnose --date 2026-07-08
  python main.py walk-forward prepare --start 2026-05-01 --folds 4 --hypothesis "..."
  python main.py walk-forward evaluate --manifest artifacts/walk-forward-prereg.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from polybot import backtest, causal_replay, mlb, pmus, research, walkforward
from polybot.config import load_config
from polybot.engine import Engine
from polybot.journal import Journal
from polybot.report import print_report
from polybot.review import print_review
from polybot.strategies import DEFAULT_REGISTRY
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
        registry = cfg.strategies or DEFAULT_REGISTRY
        accounts, positions = journal.paper_state([entry["name"] for entry in registry])
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
        report = causal_replay.replay_recorded_day(
            cfg, db_path=cfg.engine.db_path, day=args.date
        )
        causal_replay.print_report(report)


def cmd_research(args, cfg):
    if args.research_command == "fit-state":
        research.fit_state_artifact(
            args.seasons, args.holdout, args.output,
            prior_strength=args.prior_strength, max_games=args.max_games,
        )
    else:
        research.diagnose_day(
            cfg.engine.db_path, args.date,
            args.timezone or cfg.engine.report_timezone,
            cfg.engine.paper_taker_fee_theta,
        )


def cmd_walk_forward(args, cfg):
    try:
        if args.walk_command == "prepare":
            rules = {
                "min_round_trips": args.min_round_trips,
                "min_trading_days": args.min_trading_days,
                "min_games": args.min_games,
                "min_positive_test_folds": args.min_positive_test_folds,
                "require_positive_game_cluster_ci_low": not args.allow_nonpositive_ci,
                "require_consistent_champion": not args.allow_mixed_champions,
                "max_top_day_profit_share": args.max_top_day_profit_share,
                "max_top_game_profit_share": args.max_top_game_profit_share,
            }
            manifest = walkforward.prepare_manifest(
                cfg, cfg.engine.db_path, args.start, args.folds,
                args.hypothesis, args.output, rules,
            )
            print(f"locked preregistration: {args.output}")
            print(f"manifest sha256      : {manifest['manifest_sha256']}")
            print("No train, validation, or locked-test results were computed.")
        else:
            result = walkforward.evaluate_manifest(
                cfg, cfg.engine.db_path, args.manifest, args.output
            )
            walkforward.print_result(result)
            print(f"full evidence: {args.output}")
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(f"walk-forward: {exc}") from None


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
    p_bt.add_argument("mode", choices=["calibrate", "strategy", "replay", "causal"])
    p_bt.add_argument("--days", type=int, default=3, help="days back to include")
    p_bt.add_argument("--max-games", type=int, default=40)
    p_bt.add_argument("--date", help="recorded local day for replay, YYYY-MM-DD")
    p_bt.add_argument("--set", action="append", default=[],
                      help="override config for replay/strategy, e.g. strategy.stop_loss=0.15")
    p_research = sub.add_parser("research", help="fit and diagnose Phase 3 models")
    research_sub = p_research.add_subparsers(dest="research_command", required=True)
    p_fit = research_sub.add_parser("fit-state", help="fit an empirical state model")
    p_fit.add_argument("--seasons", type=int, nargs="+", required=True,
                       help="training seasons, e.g. 2022 2023 2024")
    p_fit.add_argument("--holdout", type=int, required=True,
                       help="untouched acceptance season")
    p_fit.add_argument("--output", required=True, help="artifact JSON path")
    p_fit.add_argument("--prior-strength", type=float, default=30.0)
    p_fit.add_argument("--max-games", type=int, default=0,
                       help="cap games per season for a smoke run (0 = all)")
    p_diag = research_sub.add_parser("diagnose", help="show executable signal markouts")
    p_diag.add_argument("--date", help="local date, YYYY-MM-DD (default: today)")
    p_diag.add_argument("--timezone", help="IANA trading-day timezone")
    p_walk = sub.add_parser(
        "walk-forward", help="preregister and run chronological Phase 4 evaluation"
    )
    walk_sub = p_walk.add_subparsers(dest="walk_command", required=True)
    p_prepare = walk_sub.add_parser(
        "prepare", help="lock hypothesis, folds, config, tape, and promotion rules"
    )
    p_prepare.add_argument("--start", required=True,
                           help="first training day, YYYY-MM-DD")
    p_prepare.add_argument("--folds", type=int, required=True,
                           help="number of weekly 28/7/7 folds")
    p_prepare.add_argument("--hypothesis", required=True,
                           help="strategy hypothesis declared before test evaluation")
    p_prepare.add_argument("--output", default="artifacts/walk-forward-prereg.json")
    rule_defaults = walkforward.DEFAULT_PROMOTION_RULES
    p_prepare.add_argument("--min-round-trips", type=int,
                           default=rule_defaults["min_round_trips"])
    p_prepare.add_argument("--min-trading-days", type=int,
                           default=rule_defaults["min_trading_days"])
    p_prepare.add_argument("--min-games", type=int,
                           default=rule_defaults["min_games"])
    p_prepare.add_argument("--min-positive-test-folds", type=int,
                           default=rule_defaults["min_positive_test_folds"])
    p_prepare.add_argument("--max-top-day-profit-share", type=float,
                           default=rule_defaults["max_top_day_profit_share"])
    p_prepare.add_argument("--max-top-game-profit-share", type=float,
                           default=rule_defaults["max_top_game_profit_share"])
    p_prepare.add_argument("--allow-nonpositive-ci", action="store_true",
                           help="do not require the game-clustered CI lower bound above zero")
    p_prepare.add_argument("--allow-mixed-champions", action="store_true",
                           help="let folds that select different strategies still pass the gate")
    p_evaluate = walk_sub.add_parser(
        "evaluate", help="verify preregistration and reveal locked tests once"
    )
    p_evaluate.add_argument("--manifest", required=True)
    p_evaluate.add_argument("--output", default="artifacts/walk-forward-result.json")
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
     "review": cmd_review, "backtest": cmd_backtest,
     "research": cmd_research, "walk-forward": cmd_walk_forward}[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
