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
    min_price: float = 0.10              # don't buy tokens below this
    max_price: float = 0.85              # don't buy tokens above this
    # Exit (fractions of entry price)
    take_profit: float = 0.12            # +12% -> close
    stop_loss: float = 0.10              # -10% -> close
    max_hold_secs: float = 900.0         # time stop: 15 minutes
    edge_exit: float = 0.03              # close if model edge flips against us
    cooldown_secs: float = 120.0         # per-market cooldown after a trade
    # Playfulness filter
    min_flips: int = 2                   # 0.5-crossings required
    flip_band: float = 0.03              # hysteresis band around 0.5
    min_volatility: float = 0.015        # OR realized vol threshold
    vol_window: int = 30                 # samples for realized vol


@dataclass
class RiskConfig:
    stake_usd: float = 10.0              # notional per trade
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
    discovery_interval_secs: float = 120.0
    equity_snapshot_secs: float = 60.0
    slippage: float = 0.005              # paper fill: mid +/- this
    live: bool = False
    db_path: str = "polybot.db"


@dataclass
class Config:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)


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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        cfg.ai.enabled = False
    return cfg
