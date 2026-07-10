import pytest

from polybot.broker import PaperBroker, parse_token
from polybot.config import RiskConfig
from polybot.risk import RiskManager


@pytest.fixture
def broker():
    return PaperBroker(["math", "ai"], starting_cash=100.0, slippage=0.0)


def test_open_and_close_round_trip(broker):
    pos = broker.open("math", "m1", "tok1", "Homers", 0.50, 10.0)
    assert pos is not None
    assert pos.trade_id
    assert broker.cash["math"] == pytest.approx(90.0)
    result = broker.close("math", "tok1", 0.60)
    assert result is not None
    closed, fill, pnl = result
    assert closed.trade_id == pos.trade_id
    assert fill == pytest.approx(0.60)
    assert pnl == pytest.approx(2.0)  # 20 shares * 0.10
    assert broker.cash["math"] == pytest.approx(102.0)
    assert broker.realized["math"] == pytest.approx(2.0)


def test_slippage_applied():
    b = PaperBroker(["math"], 100.0, slippage=0.01)
    pos = b.open("math", "m1", "tok1", "T", 0.50, 10.0)
    assert pos.entry_price == pytest.approx(0.51)
    _, fill, _ = b.close("math", "tok1", 0.50)
    assert fill == pytest.approx(0.49)


def test_taker_fees_are_debited_from_entry_and_exit():
    b = PaperBroker(["math"], 100.0, slippage=0.0, taker_fee_theta=0.06)

    pos = b.open("math", "m1", "tok1", "T", 0.50, 10.0)
    assert pos is not None
    assert pos.entry_fee == pytest.approx(0.30)
    assert b.cash["math"] == pytest.approx(89.70)
    _, _, pnl = b.close("math", "tok1", 0.60)
    assert b.last_fee["math"] == pytest.approx(0.29)
    assert pnl == pytest.approx(1.41)
    assert b.cash["math"] == pytest.approx(101.41)


def test_ledgers_are_independent(broker):
    broker.open("math", "m1", "tok1", "T", 0.50, 10.0)
    assert broker.cash["math"] == pytest.approx(90.0)
    assert broker.cash["ai"] == pytest.approx(100.0)
    assert broker.open_positions("ai") == []


def test_no_duplicate_position(broker):
    assert broker.open("math", "m1", "tok1", "T", 0.50, 10.0) is not None
    assert broker.open("math", "m1", "tok1", "T", 0.50, 10.0) is None


def test_trade_ids_are_unique(broker):
    first = broker.open("math", "m1", "tok1", "T", 0.50, 10.0)
    second = broker.open("ai", "m1", "tok1", "T", 0.50, 10.0)
    assert first.trade_id != second.trade_id


def test_insufficient_cash(broker):
    assert broker.open("math", "m1", "tok1", "T", 0.50, 500.0) is None


def test_equity_marks_to_market(broker):
    broker.open("math", "m1", "tok1", "T", 0.50, 10.0)
    assert broker.equity("math", {"tok1": 0.60}) == pytest.approx(102.0)
    assert broker.equity("math", {}) == pytest.approx(100.0)  # falls back to entry


def test_settle_uses_exact_price_without_slippage():
    b = PaperBroker(["math"], 100.0, slippage=0.01)
    b.open("math", "m1", "tok1", "T", 0.50, 10.0)
    _, fill, pnl = b.settle("math", "tok1", 0.0)
    assert fill == 0.0
    assert pnl == pytest.approx(-10.0)


def test_parse_token_validates_shape():
    assert parse_token("market-slug:LONG") == ("market-slug", "LONG")
    with pytest.raises(ValueError):
        parse_token("market-slug:YES")


def test_risk_max_positions(broker):
    risk = RiskManager(RiskConfig(max_positions=1, stake_usd=10), ["math"])
    assert risk.can_open(broker, "math", "m1")
    broker.open("math", "m1", "tok1", "T", 0.50, 10.0)
    assert not risk.can_open(broker, "math", "m2")


def test_risk_per_market_cap(broker):
    risk = RiskManager(RiskConfig(max_positions=10, stake_usd=10,
                                  max_stake_per_market=15), ["math"])
    broker.open("math", "m1", "tok1", "T", 0.50, 10.0)
    assert not risk.can_open(broker, "math", "m1")
    assert risk.can_open(broker, "math", "m2")


def test_risk_daily_loss_halts(broker):
    risk = RiskManager(RiskConfig(daily_loss_limit_usd=5), ["math"])
    broker.realized["math"] = -6.0
    assert not risk.can_open(broker, "math", "m1")
    assert risk.halted["math"]


def test_risk_daily_loss_uses_persisted_day_total_when_given(broker):
    risk = RiskManager(RiskConfig(daily_loss_limit_usd=5), ["math"])
    broker.realized["math"] = 20.0  # lifetime P&L is not the daily guard

    assert not risk.can_open(broker, "math", "m1", daily_realized=-6.0)
    assert risk.halted["math"]


def test_risk_daily_halt_resets_on_next_trading_day(broker):
    risk = RiskManager(RiskConfig(daily_loss_limit_usd=5), ["math"])

    assert not risk.can_open(broker, "math", "m1", daily_realized=-6.0, day_key="2026-07-09")
    assert risk.can_open(broker, "math", "m1", daily_realized=0.0, day_key="2026-07-10")
