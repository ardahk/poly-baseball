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
        self._flip_times: deque[float] = deque(maxlen=max_samples)

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
            self._flip_times.append(ts)
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

    def realized_vol_time(self, window_secs: float = 60.0,
                          bucket_secs: float = 5.0) -> float:
        """Volatility on a fixed time grid, stable across polling frequencies."""
        if len(self.samples) < 3 or window_secs <= 0 or bucket_secs <= 0:
            return 0.0
        samples = list(self.samples)
        end = samples[-1][0]
        start = end - window_secs
        grid: list[float] = []
        cursor = start
        index = 0
        last_price = None
        while cursor <= end + 1e-9:
            while index < len(samples) and samples[index][0] <= cursor:
                last_price = samples[index][1]
                index += 1
            if last_price is not None:
                grid.append(last_price)
            cursor += bucket_secs
        if not grid or grid[-1] != samples[-1][1]:
            grid.append(samples[-1][1])
        deltas = [b - a for a, b in zip(grid, grid[1:])]
        return statistics.pstdev(deltas) if len(deltas) >= 2 else 0.0

    def flips_within(self, seconds: float) -> int:
        """Regime crossings inside a trailing receipt-time window."""
        if not self.samples:
            return 0
        cutoff = self.samples[-1][0] - seconds
        return sum(ts >= cutoff for ts in self._flip_times)
