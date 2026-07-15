"""Preregistered chronological walk-forward evaluation over the causal tape."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .causal_replay import PREGAME_STATE_LOOKBACK_SECS, CausalReplay
from .config import Config
from .journal import Journal
from .provenance import canonical, code_identity, config_hash, digest
from .state_model import brier_loss, clamp_probability, log_loss

# v2 adds the multiplicity-corrected significance gate. v1 manifests locked
# their gate before the correction existed, so it is computed and REPORTED for
# them but never gates retroactively.
FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = {1, 2}
HORIZONS = (5, 15, 30, 60)
TRAIN_DAYS, VALIDATE_DAYS, TEST_DAYS, ROLL_DAYS = 28, 7, 7, 7

# Single source for promotion defaults; main.py's CLI flags reference these.
DEFAULT_PROMOTION_RULES = {
    "min_round_trips": 300,
    "min_trading_days": 30,
    "min_games": 100,
    "min_positive_test_folds": 2,
    "require_positive_net_pnl": True,
    "require_positive_game_cluster_ci_low": True,
    "require_consistent_champion": True,
    "max_top_day_profit_share": 0.35,
    "max_top_game_profit_share": 0.35,
    # Multiplicity: the champion is the max of N correlated candidates, so its
    # own CI is not deflated for selection. White's Reality Check bootstraps
    # the max over the joint game-resampled fleet, pricing that correlation.
    "multiplicity_correction": "reality_check",   # reality_check | none
    "reality_check_alpha": 0.05,
    "reality_check_samples": 4000,
}

# Every table/column the causal replay or calibration consumes. Pinned
# explicitly so an additive schema migration does not flip data_version and
# spuriously invalidate a locked preregistration.
_TAPE_COLUMNS = {
    "price_ticks": (
        "ts", "market", "home_team", "away_team", "home_bid", "home_ask",
        "home_mid", "home_spread", "long_bid", "long_ask", "two_sided",
        "source", "run_id", "received_at", "source_ts",
    ),
    "game_states": (
        "ts", "game_pk", "inning", "is_top", "outs", "home_score",
        "away_score", "on_first", "on_second", "on_third", "status",
        "run_id", "received_at",
    ),
    "model_observations": (
        "ts", "run_id", "model", "market", "game_pk", "state_signature",
        "model_home", "pregame_anchor", "anchored_fair", "home_mid", "spread",
        "inning", "is_top", "outs", "home_score", "away_score",
    ),
    "markets": (
        "slug", "question", "home_team", "away_team", "long_team",
        "game_pk", "start_time", "first_seen_ts",
    ),
}


def _model_artifact_hash(cfg: Config) -> str | None:
    path = cfg.engine.state_model_path
    if not path:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _midnight(day: str, timezone: str) -> datetime:
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=ZoneInfo(timezone))


def build_folds(start_day: str, count: int, timezone: str) -> list[dict]:
    if count < 1:
        raise ValueError("fold count must be positive")
    start = _midnight(start_day, timezone)
    folds = []
    for index in range(count):
        train_start = start + timedelta(days=ROLL_DAYS * index)
        train_end = train_start + timedelta(days=TRAIN_DAYS)
        validate_end = train_end + timedelta(days=VALIDATE_DAYS)
        test_end = validate_end + timedelta(days=TEST_DAYS)
        folds.append({
            "fold": index + 1,
            "train": [train_start.isoformat(), train_end.isoformat()],
            "validate": [train_end.isoformat(), validate_end.isoformat()],
            "test": [validate_end.isoformat(), test_end.isoformat()],
        })
    return folds


def _bounds(window: list[str]) -> tuple[float, float]:
    return (datetime.fromisoformat(window[0]).timestamp(),
            datetime.fromisoformat(window[1]).timestamp())


def tape_digest(journal: Journal, folds: list[dict]) -> str:
    """Hash every decision-clock input in the declared experiment range."""
    start = min(_bounds(fold["train"])[0] for fold in folds)
    end = max(_bounds(fold["test"])[1] for fold in folds)
    hasher = hashlib.sha256()
    for table, columns in _TAPE_COLUMNS.items():
        hasher.update(table.encode())
        select = ", ".join(columns)
        if table == "markets":
            query = (f"SELECT {select} FROM markets WHERE first_seen_ts<? "
                     "ORDER BY slug")
            params = (end,)
        elif table == "model_observations":
            query = (f"SELECT {select} FROM model_observations WHERE ts>=? "
                     "AND ts<? ORDER BY ts,rowid")
            params = (start, end)
        else:
            query = (f"SELECT {select} FROM {table} "
                     "WHERE COALESCE(received_at,ts)>=? AND COALESCE(received_at,ts)<? "
                     "ORDER BY COALESCE(received_at,ts),rowid")
            if table == "game_states":
                # The replay seeds states from before each window, and final
                # scores up to a day after the last boundary label calibration.
                params = (start - PREGAME_STATE_LOOKBACK_SECS, end + 86400)
            else:
                params = (start, end)
        for row in journal.conn.execute(query, params):
            hasher.update(canonical(dict(row)).encode())
    return hasher.hexdigest()


def prepare_manifest(cfg: Config, db_path: str, start_day: str, fold_count: int,
                     hypothesis: str, output: str, rules: dict | None = None) -> dict:
    if not hypothesis.strip():
        raise ValueError("a non-empty hypothesis is required")
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite preregistration: {target}")
    folds = build_folds(start_day, fold_count, cfg.engine.report_timezone)
    promotion = {**DEFAULT_PROMOTION_RULES, **(rules or {})}
    for key in ("min_round_trips", "min_trading_days", "min_games",
                "min_positive_test_folds"):
        if promotion[key] < 0:
            raise ValueError(f"{key} cannot be negative")
    for key in ("max_top_day_profit_share", "max_top_game_profit_share"):
        if not 0 <= promotion[key] <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if promotion["multiplicity_correction"] not in {"reality_check", "none"}:
        raise ValueError("multiplicity_correction must be 'reality_check' or 'none'")
    if not 0 < promotion["reality_check_alpha"] <= 0.5:
        raise ValueError("reality_check_alpha must be in (0, 0.5]")
    if promotion["reality_check_samples"] < 100:
        raise ValueError("reality_check_samples must be at least 100")
    journal = Journal(db_path)
    try:
        data_hash = tape_digest(journal, folds)
    finally:
        journal.close()
    code_revision, code_hash = code_identity()
    manifest = {
        "format_version": FORMAT_VERSION,
        "kind": "polybot_walk_forward_preregistration",
        "created_at": datetime.now(ZoneInfo("UTC")).isoformat(),
        "hypothesis": hypothesis.strip(),
        "selection_rule": (
            "among strategies with at least one train and validation round trip, "
            "highest validation realized P&L (closed round trips and settlements, "
            "net of fees); ties resolved by strategy name"
        ),
        "windows": {"train_days": TRAIN_DAYS, "validate_days": VALIDATE_DAYS,
                    "locked_test_days": TEST_DAYS, "roll_days": ROLL_DAYS},
        "timezone": cfg.engine.report_timezone,
        "folds": folds,
        "promotion_rules": promotion,
        "config_hash": config_hash(cfg),
        "code_revision": code_revision,
        "code_hash": code_hash,
        "state_model_artifact_hash": _model_artifact_hash(cfg),
        "data_version": data_hash,
    }
    manifest["manifest_sha256"] = digest(manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_manifest(path: str, cfg: Config, journal: Journal) -> dict:
    manifest = json.loads(Path(path).read_text())
    supplied = manifest.pop("manifest_sha256", None)
    actual = digest(manifest)
    manifest["manifest_sha256"] = supplied
    if supplied != actual:
        raise ValueError("preregistration checksum mismatch")
    if manifest.get("format_version") not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError("unsupported walk-forward manifest version")
    if manifest.get("config_hash") != config_hash(cfg):
        raise ValueError("current config does not match the preregistered config")
    if manifest.get("code_hash") != code_identity()[1]:
        raise ValueError("research code changed after preregistration")
    if manifest.get("state_model_artifact_hash") != _model_artifact_hash(cfg):
        raise ValueError("state-model artifact changed after preregistration")
    if manifest.get("data_version") != tape_digest(journal, manifest["folds"]):
        raise ValueError("event tape changed after preregistration")
    return manifest


def _sanitize(value):
    """Replace non-finite floats with None so results digest and serialize
    under one JSON policy instead of crashing after the locked test ran."""
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _cluster_ci(values: dict[str, float], seed: str, alpha: float = 0.05,
                samples: int = 2000) -> list[float | None]:
    points = list(values.values())
    if not points:
        return [None, None]
    if len(points) == 1:
        return [points[0], points[0]]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(points, k=len(points)))
                   for _ in range(samples))
    low = means[int(alpha / 2 * len(means))]
    high = means[min(int((1 - alpha / 2) * len(means)), len(means) - 1)]
    return [low, high]


def _reality_check(game_pnls: dict[str, dict[str, float]],
                   champion_pnls: dict[str, float], seed: str,
                   samples: int = 4000, alpha: float = 0.05) -> dict:
    """White (2000) Reality Check over the candidate fleet, clustered by game.

    Null: no candidate has positive expected per-game P&L. Each candidate's
    per-game series is recentered to zero mean; every bootstrap replicate draws
    ONE game resample shared across all candidates (preserving their
    correlation — they trade the same games) and records the max candidate
    mean. The champion promotes only if its observed mean per-game P&L exceeds
    the (1-alpha) quantile of that max-of-N null distribution. This is what
    makes the promotion bar rise with the number of candidates tried.
    """
    games = sorted(set(champion_pnls).union(*game_pnls.values())) \
        if game_pnls else sorted(champion_pnls)
    result = {
        "candidates": len(game_pnls), "games": len(games),
        "alpha": alpha, "samples": samples,
        "champion_mean_game_pnl": None, "max_null_q95": None,
        "p_value": None, "passes": False,
    }
    if not games or not game_pnls or not champion_pnls:
        return result
    # A game a candidate did not trade contributes 0 — the mean is over the
    # same game universe for every candidate and for the champion statistic.
    t_obs = statistics.mean(champion_pnls.get(game, 0.0) for game in games)
    count = len(games)
    centered: list[list[float]] = []
    for pnls in game_pnls.values():
        series = [pnls.get(game, 0.0) for game in games]
        mean = statistics.mean(series)
        centered.append([value - mean for value in series])
    rng = random.Random(seed)
    max_stats = []
    for _ in range(samples):
        draw = [rng.randrange(count) for _ in range(count)]
        max_stats.append(max(
            sum(series[i] for i in draw) / count for series in centered
        ))
    max_stats.sort()
    q95 = max_stats[min(int((1 - alpha) * samples), samples - 1)]
    result.update(
        champion_mean_game_pnl=t_obs, max_null_q95=q95,
        p_value=(1 + sum(stat >= t_obs for stat in max_stats)) / (samples + 1),
        passes=t_obs > q95,
    )
    return result


def _bucket(value: float, width: float) -> str:
    low = math.floor(value / width) * width
    return f"{low:.2f}-{low + width:.2f}"


def _groups(trades, timezone: str) -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, float]] = {
        "day": {}, "game": {}, "inning": {}, "entry_price": {}, "spread": {},
    }
    tz = ZoneInfo(timezone)
    for trade in trades:
        keys = {
            "day": datetime.fromtimestamp(trade.exit_ts, tz).date().isoformat(),
            "game": str(trade.game_pk or trade.market),
            "inning": str(trade.entry_inning),
            "entry_price": _bucket(trade.entry_price, 0.10),
            "spread": _bucket(trade.entry_spread, 0.02),
        }
        for dimension, key in keys.items():
            groups[dimension][key] = groups[dimension].get(key, 0.0) + trade.pnl_usd
    return groups


def _profit_share(values: dict[str, float]) -> float:
    positive = [value for value in values.values() if value > 0]
    return max(positive, default=0.0) / sum(positive) if positive else 0.0


def _adverse_selection(journal: Journal, trades) -> dict[str, dict]:
    result = {}
    for horizon in HORIZONS:
        marks = []
        for trade in trades:
            target = trade.entry_ts + horizon
            row = journal.conn.execute(
                """SELECT long_bid,long_ask,COALESCE(received_at,ts) AS clock
                   FROM price_ticks WHERE market=? AND two_sided=1
                     AND ts BETWEEN ? AND ?  -- indexed bracket; receipt clock refines
                     AND COALESCE(received_at,ts)>=? AND COALESCE(received_at,ts)<=?
                   ORDER BY COALESCE(received_at,ts),rowid LIMIT 1""",
                (trade.market, target - 3600, target + 3605, target, target + 5),
            ).fetchone()
            if not row or row["long_bid"] is None or row["long_ask"] is None:
                continue
            # Tokens are "<slug>:LONG"/"<slug>:SHORT"; ticks quote the long side.
            bid = row["long_bid"] if trade.token.endswith(":LONG") else 1.0 - row["long_ask"]
            marks.append(bid - trade.entry_price)
        result[str(horizon)] = {
            "observations": len(marks),
            "mean_price_change": statistics.mean(marks) if marks else None,
            "adverse_rate": sum(value < 0 for value in marks) / len(marks) if marks else None,
        }
    return result


def _strategy_metrics(report, strategy: str, journal: Journal, timezone: str,
                      execution_quality: bool = True) -> dict:
    row = next(result for result in report.results if result.strategy == strategy)
    trades = [trade for trade in report.trades if trade.strategy == strategy]
    groups = _groups(trades, timezone)
    day_pnl = list(groups["day"].values())
    tail_count = max(1, math.ceil(0.05 * len(day_pnl))) if day_pnl else 0
    net_pnl = row.equity - report.starting_cash
    turnover = row.deployed_capital
    peak_deployed = row.peak_deployed
    entry_attempts = row.filled_entries + row.rejected_entries
    return {
        "round_trips": len(trades), "wins": row.wins,
        "net_pnl": net_pnl, "realized_pnl": row.realized, "fees": row.fees,
        "return_on_deployed_capital": (
            net_pnl / peak_deployed if peak_deployed else 0.0
        ),
        "peak_deployed_capital": peak_deployed,
        "turnover": turnover,
        "max_drawdown": row.max_drawdown,
        "expected_shortfall_5pct_daily": (
            statistics.mean(sorted(day_pnl)[:tail_count]) if tail_count else None
        ),
        "fill_rate": row.filled_entries / entry_attempts if entry_attempts else 0.0,
        "filled_entries": row.filled_entries,
        "rejected_entries": row.rejected_entries,
        "rejected_exits": row.rejected_exits,
        "fee_share_of_turnover": row.fees / turnover if turnover else 0.0,
        "open_positions_at_boundary": row.open_positions,
        "trading_days": len(groups["day"]), "games": len(groups["game"]),
        "day_clustered_mean_pnl_ci95": _cluster_ci(groups["day"], strategy + ":day"),
        "game_clustered_mean_pnl_ci95": _cluster_ci(groups["game"], strategy + ":game"),
        "top_day_profit_share": _profit_share(groups["day"]),
        "top_game_profit_share": _profit_share(groups["game"]),
        "concentration": groups,
        "adverse_selection": (
            _adverse_selection(journal, trades) if execution_quality else None
        ),
    }


def _calibration(journal: Journal, start: float, end: float) -> dict:
    rows = journal.conn.execute(
        """SELECT o.model,o.game_pk,o.model_home AS p,
                  f.home_score>f.away_score AS outcome
           FROM model_observations o
           JOIN (SELECT game_pk,home_score,away_score,
                        ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY COALESCE(received_at,ts) DESC) n
                 FROM game_states WHERE status='Final'
                   AND COALESCE(received_at,ts) < ?) f
             ON f.game_pk=o.game_pk AND f.n=1
           WHERE o.ts>=? AND o.ts<?""", (end + 86400, start, end)).fetchall()
    by_model: dict[str, list] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    result = {}
    for model, items in by_model.items():
        losses_by_game: dict[str, list[tuple[float, float]]] = {}
        for row in items:
            p = clamp_probability(row["p"])
            y = int(row["outcome"])
            losses_by_game.setdefault(str(row["game_pk"]), []).append(
                (brier_loss(p, y), log_loss(p, y)))
        game_brier = {game: statistics.mean(v[0] for v in values)
                      for game, values in losses_by_game.items()}
        game_log = {game: statistics.mean(v[1] for v in values)
                    for game, values in losses_by_game.items()}
        result[model] = {
            "states": len(items), "games": len(losses_by_game),
            "brier": statistics.mean(game_brier.values()),
            "log_loss": statistics.mean(game_log.values()),
            "game_clustered_brier_ci95": _cluster_ci(game_brier, model + ":brier"),
            "game_clustered_log_loss_ci95": _cluster_ci(game_log, model + ":log"),
        }
    return result


def _run_window(cfg: Config, journal: Journal, window: list[str], label: str,
                execution_quality: bool = True) -> dict:
    start, end = _bounds(window)
    report = CausalReplay(cfg, journal, start, end, label).run()
    return {
        "strategies": {row.strategy: _strategy_metrics(
            report, row.strategy, journal, cfg.engine.report_timezone,
            execution_quality=execution_quality,
        ) for row in report.results},
        # Calibration replays nothing: it scores the model_home probabilities the
        # live engine recorded per distinct state, keyed by the recording model.
        "calibration": (
            _calibration(journal, start, end) if execution_quality else None
        ),
        "calibration_source": "live_recorded_model_observations",
        "events": report.events, "run_boundaries": report.run_boundaries,
        "state_model": report.state_model,
    }


def _multiplicity_evidence(folds: list[dict], rules: dict,
                           candidates_considered: int) -> dict:
    """Reality-Check gate + Bonferroni diagnostic over the locked tests."""
    game_pnls: dict[str, dict[str, float]] = {}
    champion_pnls: dict[str, float] = {}
    for fold in folds:
        strategies = (fold.get("locked_test") or {}).get("strategies") or {}
        for name, metrics in strategies.items():
            games = (metrics.get("concentration") or {}).get("game") or {}
            for game, pnl in games.items():
                game_pnls.setdefault(name, {})[f"f{fold['fold']}:{game}"] = pnl
        selected = fold.get("selected_strategy")
        if selected and selected in strategies:
            games = (strategies[selected].get("concentration") or {}).get("game") or {}
            for game, pnl in games.items():
                champion_pnls[f"f{fold['fold']}:{game}"] = pnl
    candidates = max(candidates_considered, len(game_pnls), 1)
    alpha = rules["reality_check_alpha"]
    bonferroni_alpha = alpha / candidates
    return {
        "reality_check": _reality_check(
            game_pnls, champion_pnls, seed="reality_check",
            samples=rules["reality_check_samples"], alpha=alpha,
        ),
        # Diagnostic only: assumes independent candidates, so it over-penalizes
        # a fleet that trades the same games. Two-sided CI at 2*(alpha/N) makes
        # the LOWER bound the one-sided alpha/N bound.
        "bonferroni": {
            "candidates": candidates,
            "alpha_per_test": bonferroni_alpha,
            "champion_game_ci": _cluster_ci(
                champion_pnls, "champion:bonferroni",
                alpha=2 * bonferroni_alpha, samples=10000,
            ),
        },
    }


def _aggregate_selected(folds: list[dict], rules: dict,
                        manifest_format_version: int = FORMAT_VERSION) -> dict:
    rules = {**DEFAULT_PROMOTION_RULES, **rules}
    if manifest_format_version < 2:
        # v1 preregistrations locked their gate before the multiplicity
        # correction existed; report the numbers, never gate retroactively.
        rules["multiplicity_correction"] = "none"
    chosen = [fold for fold in folds if fold.get("selected_strategy")]
    selected = [fold["locked_test"]["strategies"][fold["selected_strategy"]]
                for fold in chosen]
    strategies_selected = sorted({fold["selected_strategy"] for fold in chosen})
    aggregate = {
        "folds": len(selected),
        "strategies_selected": strategies_selected,
        "candidates_considered": max(
            (fold.get("candidates_considered", 0) for fold in folds), default=0,
        ),
        "round_trips": sum(row["round_trips"] for row in selected),
        "trading_days": sum(row["trading_days"] for row in selected),
        "games": sum(row["games"] for row in selected),
        "net_pnl": sum(row["net_pnl"] for row in selected),
        "fees": sum(row["fees"] for row in selected),
        "max_drawdown": max((row["max_drawdown"] for row in selected), default=0.0),
        "max_top_day_profit_share": max((row["top_day_profit_share"] for row in selected), default=0.0),
        "max_top_game_profit_share": max(
            (row.get("top_game_profit_share", 0.0) for row in selected), default=0.0),
        "positive_test_folds": sum(row["net_pnl"] > 0 for row in selected),
    }
    aggregate["multiplicity"] = _multiplicity_evidence(
        folds, rules, aggregate["candidates_considered"],
    )
    checks = {
        "min_round_trips": aggregate["round_trips"] >= rules["min_round_trips"],
        "min_trading_days": aggregate["trading_days"] >= rules["min_trading_days"],
        "min_games": aggregate["games"] >= rules["min_games"],
        "multiple_positive_folds": aggregate["positive_test_folds"] >=
            rules["min_positive_test_folds"],
        "positive_net_pnl": (not rules["require_positive_net_pnl"] or aggregate["net_pnl"] > 0),
        "day_concentration": aggregate["max_top_day_profit_share"] <= rules["max_top_day_profit_share"],
        "game_concentration": aggregate["max_top_game_profit_share"]
            <= rules["max_top_game_profit_share"],
        # One strategy goes live; evidence stitched from rotating champions
        # must not gate a single-strategy promotion.
        "consistent_champion": (not rules["require_consistent_champion"]
                                or len(strategies_selected) <= 1),
        "positive_game_cluster_ci": (
            not rules["require_positive_game_cluster_ci_low"] or
            all(row["game_clustered_mean_pnl_ci95"][0] is not None and
                row["game_clustered_mean_pnl_ci95"][0] > 0 for row in selected)
        ),
        # The champion is the max of N correlated candidates; it must beat the
        # max-of-N null, so the bar automatically rises with candidate count.
        "multiplicity_corrected_significance": (
            rules["multiplicity_correction"] == "none"
            or aggregate["multiplicity"]["reality_check"]["passes"]
        ),
    }
    aggregate["promotion_checks"] = checks
    aggregate["passes_preregistered_rules"] = bool(selected) and all(checks.values())
    return aggregate


def evaluate_manifest(cfg: Config, db_path: str, manifest_path: str, output: str) -> dict:
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite locked result: {target}")
    journal = Journal(db_path)
    try:
        manifest = load_manifest(manifest_path, cfg, journal)
        prior = journal.walk_forward_evaluation(manifest["manifest_sha256"])
        if prior is not None:
            raise FileExistsError(
                "this preregistration was already evaluated once "
                f"(result {prior['result_sha256'][:12]} at {prior['output_path']}); "
                "locked tests are revealed a single time"
            )
        fold_results = []
        for fold in manifest["folds"]:
            # The train replay exists only to establish liveness; skip the
            # expensive execution-quality extras there.
            train = _run_window(cfg, journal, fold["train"],
                                f"fold {fold['fold']} train",
                                execution_quality=False)
            validate = _run_window(cfg, journal, fold["validate"], f"fold {fold['fold']} validate")
            eligible = [name for name, metrics in validate["strategies"].items()
                        if metrics["round_trips"] > 0 and name in train["strategies"]
                        and train["strategies"][name]["round_trips"] > 0]
            selected = sorted(
                eligible,
                key=lambda name: (-validate["strategies"][name]["realized_pnl"], name),
            )
            # Selection is fixed from train/validation before this call reveals the locked test.
            locked = _run_window(cfg, journal, fold["test"], f"fold {fold['fold']} locked test")
            fold_results.append({
                "fold": fold["fold"], "selected_strategy": selected[0] if selected else None,
                "candidates_considered": len(validate["strategies"]),
                "eligible_ranked": [
                    {"strategy": name,
                     "validation_realized_pnl": validate["strategies"][name]["realized_pnl"]}
                    for name in selected
                ],
                "train": train, "validate": validate, "locked_test": locked,
            })
        result = _sanitize({
            "format_version": FORMAT_VERSION,
            "kind": "polybot_walk_forward_result",
            "evaluated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
            "manifest_path": str(Path(manifest_path)),
            "manifest_sha256": manifest["manifest_sha256"],
            "folds": fold_results,
            "selected_champion_aggregate": _aggregate_selected(
                fold_results, manifest["promotion_rules"],
                manifest_format_version=manifest.get("format_version", 1)),
        })
        result["result_sha256"] = digest(result)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")
        journal.record_walk_forward_evaluation(
            manifest["manifest_sha256"], result["result_sha256"], str(target),
        )
    finally:
        journal.close()
    return result


def print_result(result: dict) -> None:
    print("=" * 88)
    print("PREREGISTERED WALK-FORWARD RESULT")
    print("=" * 88)
    for fold in result["folds"]:
        selected = fold["selected_strategy"] or "none"
        metrics = fold["locked_test"]["strategies"].get(selected)
        suffix = (f" trades={metrics['round_trips']} net=${metrics['net_pnl']:+.2f} "
                  f"DD=${metrics['max_drawdown']:.2f}") if metrics else ""
        print(f"fold {fold['fold']:>2}: selected={selected}{suffix}")
    agg = result["selected_champion_aggregate"]
    print(f"aggregate: trades={agg['round_trips']} days={agg['trading_days']} "
          f"games={agg['games']} net=${agg['net_pnl']:+.2f} fees=${agg['fees']:.2f}")
    if len(agg.get("strategies_selected", [])) > 1:
        print("champions : " + ", ".join(agg["strategies_selected"])
              + " (mixed selection across folds)")
    multiplicity = agg.get("multiplicity")
    if multiplicity:
        rc = multiplicity["reality_check"]
        if rc["p_value"] is not None:
            print(f"multiplicity: reality-check p={rc['p_value']:.4f} over "
                  f"{rc['candidates']} candidates / {rc['games']} games "
                  f"(champion {rc['champion_mean_game_pnl']:+.4f}/game vs "
                  f"max-null q95 {rc['max_null_q95']:+.4f})")
        bon = multiplicity["bonferroni"]
        ci_low = bon["champion_game_ci"][0]
        low = "n/a" if ci_low is None else f"{ci_low:+.4f}"
        print(f"  bonferroni diagnostic: alpha/N={bon['alpha_per_test']:.5f}, "
              f"champion one-sided CI low {low}")
    print("promotion gate: " + ("PASS" if agg["passes_preregistered_rules"] else "FAIL"))
