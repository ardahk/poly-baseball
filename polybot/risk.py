"""Per-strategy risk limits."""
from __future__ import annotations

import logging

from .broker import PaperBroker
from .config import RiskConfig

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, cfg: RiskConfig, strategies: list[str]):
        self.cfg = cfg
        self.halted: dict[str, bool] = {s: False for s in strategies}

    def can_open(
        self,
        broker: PaperBroker,
        strategy: str,
        market_key: str,
        stake_usd: float | None = None,
    ) -> bool:
        stake = self.cfg.stake_usd if stake_usd is None else stake_usd
        if self.halted[strategy]:
            return False
        if broker.realized[strategy] <= -self.cfg.daily_loss_limit_usd:
            if not self.halted[strategy]:
                log.warning("[%s] daily loss limit hit (%.2f) — halting new entries",
                            strategy, broker.realized[strategy])
            self.halted[strategy] = True
            return False
        if len(broker.open_positions(strategy)) >= self.cfg.max_positions:
            return False
        if (broker.stake_in_market(strategy, market_key) + stake
                > self.cfg.max_stake_per_market):
            return False
        return True
