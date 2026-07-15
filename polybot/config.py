"""Configuration loading: config.yaml + .env overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


@dataclass
class StrategyConfig:
    # Entry
    move_lookback_secs: float = 90.0     # window for "sharp move" detection
    move_threshold: float = 0.08         # min |price change| over lookback
    min_edge: float = 0.05               # min model-vs-market divergence
    home_fair_shrink: float = 0.0        # subtract from model P(home) (bias correction)
    max_quote_age_secs: float = 30.0     # no entries on a BBO staler than this
    early_game_max_inning: int = 5       # through this inning, require a stronger setup
    early_game_min_edge: float = 0.10    # early trades need more edge
    early_game_min_fair_extreme: float = 0.70  # early fair must be >= this or <= 1-this
    min_price: float = 0.10              # don't buy tokens below this
    max_price: float = 0.85              # don't buy tokens above this
    max_spread: float = 0.06             # skip entries if best ask - best bid is wider
    strong_stake_max_spread: float = 0.03  # larger stake only when spread is tight
    # Exit (fractions of entry price)
    take_profit: float = 0.12            # +12% -> close
    stop_loss: float = 0.10              # -10% -> close
    max_hold_secs: float = 900.0         # time stop: 15 minutes
    edge_exit: float = 0.03              # close if model edge flips against us
    cooldown_secs: float = 120.0         # per-market cooldown after a trade
    stop_loss_cooldown_secs: float = 900.0  # longer lockout after getting stopped
    # Playfulness filter
    min_flips: int = 2                   # 0.5-crossings required
    flip_band: float = 0.03              # hysteresis band around 0.5
    min_volatility: float = 0.015        # OR realized vol threshold
    vol_window: int = 30                 # samples for realized vol
    sampling_stable_features: bool = False  # fixed receipt-time grid (Phase 3 variants)
    feature_window_secs: float = 90.0
    feature_bucket_secs: float = 5.0
    flip_window_secs: float = 600.0
    # Signal capture: collapse a continuously-firing signal into ONE episode so
    # the signals/counterfactuals tables aren't flooded with dependent samples.
    # A new episode starts only after the condition goes quiet for this long.
    signal_episode_secs: float = 120.0
    # Phase 3 state-response models. Model probability *changes* are moved onto
    # a causal market anchor in log-odds space; absolute model level is not used
    # as though it were a team-strength-aware market price.
    residual_beta: float = 1.0
    residual_threshold: float = 0.06
    residual_min_model_delta: float = 0.02
    residual_response_secs: float = 45.0
    # The market reacts to a play seconds before the polled MLB feed reports
    # it, so the very last pre-receipt price usually already prices the event.
    # Anchor state transitions to the last price this many seconds before the
    # state was received to avoid double-counting the move.
    residual_anchor_lookback_secs: float = 30.0
    market_anchor_max_age_secs: float = 21600.0
    # --- Phase 5 hypothesis-fleet kinds. Each block belongs to one strategy
    # kind; registry entries in config.yaml override them per hypothesis. ---
    # momentum: trade WITH a sharp move (continuation, not reversion)
    momentum_require_model_agree: bool = False   # model fair must favor the move side
    momentum_min_state_age_secs: float = 0.0     # >0: fire only when the last MLB state
                                                 # change is at least this old (pure orderflow)
    # event_reaction: buy the model-delta direction while the market still lags
    event_max_age_secs: float = 30.0             # act only this soon after state receipt
    event_min_model_delta: float = 0.04          # ignore low-impact state changes
    event_min_underreaction: float = 0.02        # market must trail anchored fair by this
    event_class: str = "any"                     # any|score_change|inning_change|bases_or_outs
    event_min_inning: int = 1
    # extreme_hold: buy an extreme-priced side, hold to settlement (fee ~ p(1-p))
    extreme_min_price: float = 0.90
    extreme_max_price: float = 0.97
    extreme_min_inning: int = 7
    extreme_max_inning: int = 99
    extreme_require_model_agree: bool = False
    extreme_model_agree_margin: float = 0.0      # model fair >= price + this
    # settlement_hold: model-vs-market gap held to settlement (one fee leg)
    hold_min_edge: float = 0.10
    hold_max_inning: int = 6
    hold_fair_source: str = "state"              # state|market_anchored
    hold_side_filter: str = "any"                # any|home|away
    # calibration_cell: model-free bias harvesting in a price/inning/side cell
    cell_price_min: float = 0.0
    cell_price_max: float = 1.0
    cell_inning_min: int = 1
    cell_inning_max: int = 99
    cell_side: str = "home"                      # home|away|favorite|underdog|leader|trailer
    # microstructure: book-shape/timing mechanisms
    micro_mode: str = "spread_snap"              # spread_snap|stale_reprice|pregame_drift
    micro_window_secs: float = 60.0
    micro_min_reprice: float = 0.03
    micro_spread_shock: float = 0.03


@dataclass
class RiskConfig:
    stake_usd: float = 5.0               # base notional per trade
    strong_stake_usd: float = 10.0       # high-edge, tight-spread notional
    strong_stake_min_edge: float = 0.12  # edge needed for strong stake
    max_positions: int = 3               # concurrent positions per strategy
    max_stake_per_market: float = 20.0
    daily_loss_limit_usd: float = 25.0   # kill switch per strategy
    starting_cash: float = 100.0         # paper account size per strategy


@dataclass
class AIConfig:
    enabled: bool = True
    model: str = "claude-opus-4-8"
    effort: str = "low"                  # judge should be fast
    min_confidence: float = 0.55
    timeout_secs: float = 20.0


@dataclass
class EngineConfig:
    poll_interval_secs: float = 2.0
    game_state_interval_secs: float = 10.0
    pregame_game_state_window_secs: float = 1800.0  # start live-state checks 30 min pregame
    discovery_interval_secs: float = 1800.0
    equity_snapshot_secs: float = 60.0
    status_log_interval_secs: float = 300.0
    slippage: float = 0.0                # legacy backtest-only additive fill adjustment
    paper_taker_fee_theta: float = 0.06  # Polymarket US fee coefficient, effective 2026-07
    causal_replay_latency_secs: float = 0.5  # fill on first BBO observed after this delay
    history_gap_reset_secs: float = 180.0    # discard rolling features after data outages
    counterfactual_max_lag_secs: float = 5.0  # late horizons become unavailable, not backfilled
    decision_flush_secs: float = 60.0        # max age of a buffered run-length decision row
    state_model_path: str | None = None      # accepted empirical-state artifact; analytic if null
    report_timezone: str = "America/Los_Angeles"  # trading-day/report boundary
    live: bool = False
    db_path: str = "polybot.db"


@dataclass
class Config:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    strategies: list[dict] = field(default_factory=list)  # frozen-variant registry


def _apply(dc, data: dict):
    for k, v in (data or {}).items():
        if hasattr(dc, k):
            setattr(dc, k, v)


def load_config(path: str | Path = "config.yaml") -> Config:
    if load_dotenv:
        load_dotenv()
    cfg = Config()
    p = Path(path)
    if p.exists():
        raw = yaml.safe_load(p.read_text()) or {}
        _apply(cfg.strategy, raw.get("strategy"))
        _apply(cfg.risk, raw.get("risk"))
        _apply(cfg.ai, raw.get("ai"))
        _apply(cfg.engine, raw.get("engine"))
        if isinstance(raw.get("strategies"), list):
            cfg.strategies = raw["strategies"]
    if not os.environ.get("ANTHROPIC_API_KEY"):
        cfg.ai.enabled = False
    return cfg
