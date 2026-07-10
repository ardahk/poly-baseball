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
        self.halted_day: dict[str, str | None] = {s: None for s in strategies}

    def can_open(
        self,
        broker: PaperBroker,
        strategy: str,
        market_key: str,
        stake_usd: float | None = None,
        daily_realized: float | None = None,
        day_key: str | None = None,
    ) -> bool:
        stake = self.cfg.stake_usd if stake_usd is None else stake_usd
        if day_key is not None and self.halted_day[strategy] not in {None, day_key}:
            self.halted[strategy] = False
            self.halted_day[strategy] = None
        if self.halted[strategy]:
            return False
        realized = broker.realized[strategy] if daily_realized is None else daily_realized
        if realized <= -self.cfg.daily_loss_limit_usd:
            if not self.halted[strategy]:
                log.warning("[%s] daily loss limit hit (%.2f) — halting new entries",
                            strategy, realized)
            self.halted[strategy] = True
            self.halted_day[strategy] = day_key
            return False
        if len(broker.open_positions(strategy)) >= self.cfg.max_positions:
            return False
        if (broker.stake_in_market(strategy, market_key) + stake
                > self.cfg.max_stake_per_market):
            return False
        return True
