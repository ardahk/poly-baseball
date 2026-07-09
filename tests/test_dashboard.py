from io import StringIO

from polybot.broker import PaperBroker
from polybot.config import Config
from polybot.dashboard import TerminalDashboard
from polybot.models import GameState, Market, MarketQuote
from polybot.volatility import PriceHistory


class FakeEngine:
    pass


def test_terminal_dashboard_renders_engine_snapshot():
    cfg = Config()
    broker = PaperBroker(["math"], starting_cash=100.0)
    market = Market(
        slug="dodgers-giants",
        question="Dodgers vs Giants",
        home_team="Giants",
        away_team="Dodgers",
        long_team="Giants",
        game_pk=1,
    )
    quote = MarketQuote(
        market_key=market.key,
        home_bid=0.47,
        home_ask=0.49,
        long_bid=0.47,
        long_ask=0.49,
        ts=1000.0,
    )
    history = PriceHistory()
    history.add(0.55, ts=900.0)
    history.add(0.48, ts=1000.0)

    engine = FakeEngine()
    engine.cfg = cfg
    engine.strategies = ["math"]
    engine.broker = broker
    engine.markets = {market.key: market}
    engine.game_states = {
        1: GameState(game_pk=1, inning=6, is_top=False, outs=1, status="Live")
    }
    engine.latest_quotes = {market.key: quote}
    engine.latest_prices = {market.home_token: quote.home_mid}
    engine.histories = {market.key: history}
    engine._last_discovery = 990.0
    engine._last_game_poll = 995.0
    engine._last_status_log = 980.0
    engine.started_at = 900.0

    stream = StringIO()
    dashboard = TerminalDashboard(enabled=True, stream=stream)
    dashboard.record("tracking: Dodgers vs Giants")
    dashboard.render(engine, force=True)

    output = stream.getvalue()
    assert "POLYBOT PAPER" in output
    assert "Dodgers @ Giants" in output
    assert "STRATEGIES" in output
    assert "tracking: Dodgers vs Giants" in output
