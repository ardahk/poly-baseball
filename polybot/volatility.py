"""Price history per market + "playfulness" detection.

A game is playful when its home-win price has crossed 0.5 several times
(with a hysteresis band so micro-jitter around 0.5 doesn't count) or when
its recent realized volatility is high. That's the screenshot pattern:
lines crossing back and forth vs one team cruising.
"""
from __future__ import annotations

import statistics
import time
from collections import deque


class PriceHistory:
    def __init__(self, flip_band: float = 0.03, max_samples: int = 5000):
        self.samples: deque[tuple[float, float]] = deque(maxlen=max_samples)
        self.flip_band = flip_band
        self.flips = 0
        self._regime: int | None = None  # +1 above band, -1 below band

    def add(self, price: float, ts: float | None = None) -> None:
        ts = ts if ts is not None else time.time()
        self.samples.append((ts, price))
        if price > 0.5 + self.flip_band:
            regime = 1
        elif price < 0.5 - self.flip_band:
            regime = -1
        else:
            return
        if self._regime is not None and regime != self._regime:
            self.flips += 1
        self._regime = regime

    @property
    def last(self) -> float | None:
        return self.samples[-1][1] if self.samples else None

    def price_ago(self, seconds: float) -> float | None:
        """Price at least `seconds` ago (closest sample at or before the cutoff)."""
        if not self.samples:
            return None
        cutoff = self.samples[-1][0] - seconds
        result = None
        for ts, price in self.samples:
            if ts <= cutoff:
                result = price
            else:
                break
        return result

    def move(self, seconds: float) -> float | None:
        """Signed price change over the trailing window."""
        past = self.price_ago(seconds)
        if past is None or self.last is None:
            return None
        return self.last - past

    def realized_vol(self, window: int = 30) -> float:
        """Stdev of successive price changes over the last `window` samples."""
        if len(self.samples) < 3:
            return 0.0
        prices = [p for _, p in list(self.samples)[-window:]]
        deltas = [b - a for a, b in zip(prices, prices[1:])]
        if len(deltas) < 2:
            return 0.0
        return statistics.pstdev(deltas)

    def is_playful(self, min_flips: int, min_volatility: float, vol_window: int = 30) -> bool:
        return self.flips >= min_flips or self.realized_vol(vol_window) >= min_volatility
