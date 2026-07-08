import pytest

from polybot import pmus
from polybot.models import Market


def test_parse_market_synthesizes_home_and_away_tokens():
    raw = {
        "slug": "aec-mlb-sea-oak-2026-07-08",
        "question": "Mariners vs Athletics",
        "active": True,
        "closed": False,
        "sportsMarketType": pmus.MONEYLINE_TYPE,
        "orderPriceMinTickSize": "0.01",
        "marketSides": [
            {"long": True, "team": {"ordering": "away", "name": "Seattle Mariners"}},
            {"long": False, "team": {"ordering": "home", "name": "Athletics"}},
        ],
    }

    market = pmus._parse_market(raw)

    assert market is not None
    assert market.home_team == "Athletics"
    assert market.away_team == "Seattle Mariners"
    assert market.home_token == "aec-mlb-sea-oak-2026-07-08:SHORT"
    assert market.away_token == "aec-mlb-sea-oak-2026-07-08:LONG"


def test_parse_trade_stats_accepts_scaled_and_decimal_prices():
    data = {
        "bars": [
            {"last": "550"},
            {"close": "0.575"},
            {"last": None},
        ],
        "barStartTime": [
            "2026-07-08T00:00:00Z",
            "2026-07-08T00:01:00Z",
            "2026-07-08T00:02:00Z",
        ],
    }

    rows = pmus._parse_trade_stats(data)

    assert rows == pytest.approx([
        (1783468800.0, 0.55),
        (1783468860.0, 0.575),
    ])


def test_quote_normalizes_short_home_side():
    market = Market(slug="m1", question="A vs B",
                    home_team="Home", away_team="Away", long_team="Away")
    feed = pmus.PriceFeed()
    feed._long_bbo = lambda slug: (0.55, 0.59)

    quote = feed.quote(market)

    assert quote.home_bid == pytest.approx(0.41)
    assert quote.home_ask == pytest.approx(0.45)
    assert quote.home_mid == pytest.approx(0.43)
    assert quote.home_spread == pytest.approx(0.04)
