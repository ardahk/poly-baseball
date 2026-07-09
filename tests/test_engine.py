import time

from polybot.config import Config
from polybot.engine import Engine
from polybot.models import GameState, Market
from polybot.pmus import BookQuote


def make_engine(tmp_path) -> Engine:
    cfg = Config()
    cfg.ai.enabled = False
    cfg.engine.db_path = str(tmp_path / "test.db")
    return Engine(cfg)


def tracked_market(engine: Engine, live: bool = True) -> Market:
    market = Market(slug="aec-mlb-a-b-2026-07-08", question="A vs. B",
                    home_team="Homers", away_team="Awayers",
                    long_team="Homers", game_pk=1)
    engine.markets[market.key] = market
    from polybot.volatility import PriceHistory
    engine.histories[market.key] = PriceHistory()
    engine.game_states[1] = GameState(game_pk=1, status="Live" if live else "Final")
    return market


def test_two_sided_book_records_quote_and_history(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)

    engine._poll_prices({market.slug: BookQuote(long_bid=0.50, long_ask=0.52,
                                                long_last=0.51)})

    assert market.key in engine.latest_quotes
    assert engine.latest_quotes[market.key].home_mid == 0.51
    assert engine.histories[market.key].last == 0.51
    assert engine.latest_prices[market.away_token] == 0.49
    assert engine.journal.ticks_for_market(market.key)[0]["source"] == "bbo"


def test_one_sided_book_keeps_history_alive_without_quote(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)

    engine._poll_prices({market.slug: BookQuote(long_bid=None, long_ask=0.01,
                                                long_last=0.01)})

    assert market.key not in engine.latest_quotes  # untradeable, no fresh BBO
    assert engine.histories[market.key].last == 0.01  # but tracking continues
    tick = engine.journal.ticks_for_market(market.key)[0]
    assert tick["source"] == "mark"
    assert tick["two_sided"] == 0


def test_short_home_side_inverts_book(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    market.long_team = market.away_team

    engine._poll_prices({market.slug: BookQuote(long_bid=0.40, long_ask=0.44,
                                                long_last=0.42)})

    quote = engine.latest_quotes[market.key]
    assert quote.home_bid == 0.56
    assert quote.home_ask == 0.60


def test_stale_quote_blocks_entries_and_is_counted(tmp_path):
    engine = make_engine(tmp_path)
    market = tracked_market(engine)
    engine._poll_prices({market.slug: BookQuote(long_bid=0.50, long_ask=0.52,
                                                long_last=0.51)})
    engine.latest_quotes[market.key].ts = time.time() - 120  # stale

    engine._look_for_entries()

    assert engine.funnel.get("stale_quote") == 1


def test_discovery_only_tracks_matched_markets(tmp_path):
    engine = make_engine(tmp_path)
    engine.mlb.todays_games = lambda: [
        {"game_pk": 7, "home": "Homers", "away": "Awayers",
         "status": "Live", "game_date": None},
    ]
    matched = Market(slug="m-yes", question="", home_team="Homers",
                     away_team="Awayers", long_team="Homers")
    unmatched = Market(slug="m-no", question="", home_team="Others",
                       away_team="Nobodies", long_team="Others")

    engine._maybe_discover([matched, unmatched])

    assert "m-yes" in engine.markets
    assert engine.markets["m-yes"].game_pk == 7
    assert "m-no" not in engine.markets
    assert engine.journal.markets_between(0, time.time() + 1)[0].slug == "m-yes"
