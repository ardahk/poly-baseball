from datetime import datetime
from zoneinfo import ZoneInfo

from polybot.timeframe import day_bounds


def test_day_bounds_uses_requested_timezone():
    start, end, label = day_bounds("2026-07-09", "America/Los_Angeles")

    assert label == "2026-07-09"
    assert datetime.fromtimestamp(start, ZoneInfo("America/Los_Angeles")).hour == 0
    assert end - start == 24 * 60 * 60
