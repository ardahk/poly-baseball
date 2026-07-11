"""Phase 3 model fitting and causal signal diagnostics."""
from __future__ import annotations

import statistics
from datetime import date

from . import backtest, mlb
from .broker import taker_fee
from .journal import Journal
from .state_model import EmpiricalStateModel, holdout_regressed, score_model
from .timeframe import day_bounds
from .winprob import home_win_probability


def _season_games(year: int, max_games: int = 0):
    client = mlb.MLBClient()
    scheduled = [g for g in client.schedule(f"{year}-03-15", f"{year}-11-15")
                 if g["status"] == "Final"]
    if max_games:
        scheduled = scheduled[:max_games]
    result = []
    for index, game in enumerate(scheduled, 1):
        timeline, home_won = backtest.build_state_timeline(client, game["game_pk"])
        if timeline and home_won is not None:
            result.append((timeline, home_won))
        if index % 100 == 0:
            print(f"  {year}: fetched {index}/{len(scheduled)} games", flush=True)
    return result


def fit_state_artifact(train_seasons: list[int], holdout_season: int, output: str,
                       prior_strength: float = 30.0, max_games: int = 0) -> str:
    if holdout_season in train_seasons:
        raise SystemExit("holdout season must be disjoint from training seasons")
    train = []
    for season in train_seasons:
        train.extend(_season_games(season, max_games))
    holdout = _season_games(holdout_season, max_games)
    if not train or not holdout:
        raise SystemExit("not enough finished games to fit and score the state model")
    model = EmpiricalStateModel.fit(train, prior_strength)
    analytic = score_model(holdout, home_win_probability)
    empirical = score_model(holdout, model.predict)
    print(f"{'model':<24}{'games':>8}{'states':>10}{'Brier':>10}{'log loss':>12}{'ECE':>10}")
    print("-" * 74)
    for name, score in (("analytic control", analytic), ("empirical state v1", empirical)):
        print(f"{name:<24}{score.games:>8}{score.states:>10}{score.brier:>10.4f}"
              f"{score.log_loss:>12.4f}{score.calibration_error:>10.4f}")
    if holdout_regressed(analytic, empirical):
        raise SystemExit(
            "artifact rejected: empirical model regressed on the untouched holdout season"
        )
    model.metadata = {
        "created": date.today().isoformat(),
        "train_seasons": train_seasons,
        "holdout_season": holdout_season,
        "train_games": len(train),
        "holdout_games": len(holdout),
        "analytic_holdout": analytic.__dict__,
        "empirical_holdout": empirical.__dict__,
    }
    digest = model.save(output)
    print(f"accepted artifact: {output}")
    print(f"sha256          : {digest}")
    return digest


def diagnose_day(db_path: str, day: str | None, timezone: str,
                 fee_theta: float) -> None:
    start, end, label = day_bounds(day, timezone)
    journal = Journal(db_path)
    try:
        rows = journal.conn.execute(
            """SELECT s.strategy, COALESCE(r.kind, 'unknown') AS kind,
                      s.entry_price, s.residual, c.horizon_secs, c.exec_bid,
                      c.two_sided
               FROM signals s
               LEFT JOIN strategy_registry r
                 ON r.run_id = s.run_id AND r.strategy = s.strategy
               JOIN signal_counterfactuals c ON c.signal_id = s.id
               WHERE s.ts >= ? AND s.ts < ?
               ORDER BY s.strategy, c.horizon_secs""",
            (start, end),
        ).fetchall()
    finally:
        journal.close()
    grouped: dict[tuple[str, str, int], list] = {}
    for row in rows:
        grouped.setdefault((row["strategy"], row["kind"], row["horizon_secs"]), []).append(row)
    print("=" * 92)
    print(f"PHASE 3 SIGNAL DIAGNOSTICS {label} ({timezone})")
    print("=" * 92)
    print(f"{'strategy':<22}{'kind':<18}{'sec':>6}{'n':>7}{'BBO%':>8}"
          f"{'|resid|':>10}{'net mark':>12}{'positive':>10}")
    print("-" * 94)
    for (strategy, kind, horizon), group in sorted(grouped.items()):
        executable = [r for r in group if r["two_sided"] and r["entry_price"] is not None
                      and r["exec_bid"] is not None]
        marks = [
            r["exec_bid"] - r["entry_price"]
            - taker_fee(fee_theta, r["entry_price"])
            - taker_fee(fee_theta, r["exec_bid"])
            for r in executable
        ]
        residuals = [r["residual"] for r in group if r["residual"] is not None]
        print(f"{strategy:<22}{kind:<18}{horizon:>6}{len(group):>7}"
              f"{100 * len(executable) / len(group):>7.1f}%"
              f"{(statistics.mean(abs(r) for r in residuals) if residuals else 0):>10.3f}"
              f"{(statistics.mean(marks) if marks else 0):>12.3f}"
              f"{(100 * sum(m > 0 for m in marks) / len(marks) if marks else 0):>9.1f}%")
    if not grouped:
        print("No completed signal counterfactuals for this day.")
