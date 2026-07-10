"""Trading-day boundaries in the configured reporting timezone."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def day_bounds(day: str | None = None, timezone: str = "America/Los_Angeles") -> tuple[float, float, str]:
    """Return epoch bounds for a local trading day and its display label."""
    zone = ZoneInfo(timezone)
    selected = datetime.now(zone).date() if day is None else datetime.strptime(
        day, "%Y-%m-%d"
    ).date()
    start = datetime.combine(selected, datetime.min.time(), tzinfo=zone)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp(), selected.isoformat()
