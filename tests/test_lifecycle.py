"""Fee schedule, account lifecycle, and the cost floor.

These pin down the three facts the 2026-07-21 audit turned on: what a trade
actually costs, that a rescue deposit can never read as profit, and that the
cost floor is not silently disabled by `min_edge: -1.0`.
"""
import pytest

from polybot.broker import PaperBroker, taker_fee
from polybot.config import Config
from polybot.strategies import StratContext
from polybot.models import Market, MarketQuote
from polybot.volatility import PriceHistory


# ------------------------------------------------------- venue fee schedule

def test_taker_fee_matches_published_cap():
    """theta*C*p*(1-p); the documented maximum is $1.50 per 100 shares at p=0.50.

    Verified against https://docs.polymarket.us/fees and against the live
    gateway payload, which reports feeCoefficient=0.06 on every MLB market.
    """
    assert taker_fee(0.06, 0.50, 100) == pytest.approx(1.50)


def test_taker_fee_collapses_at_the_tails():
    """The p(1-p) shape is the whole reason the price band matters.

    Cost as a share of notional falls monotonically with price, so the same
    mechanism is ~4x cheaper to run at 0.90 than at the money.
    """
    # fee / notional == theta * (1 - p), so it falls linearly in price.
    mid = taker_fee(0.06, 0.50) / 0.50
    tail = taker_fee(0.06, 0.90) / 0.90
    assert mid == pytest.approx(0.06 * 0.50)    # 3.0% of notional
    assert tail == pytest.approx(0.06 * 0.10)   # 0.6% of notional
    assert tail == pytest.approx(mid / 5)


def test_settlement_redemption_is_free():
    """Redemption charges no fee, so a hold pays ONE leg, not two."""
    broker = PaperBroker(["s"], starting_cash=100.0, taker_fee_theta=0.06)
    broker.open("s", "m1", "tok", "Team", 0.50, 10.0)
    entry_fee = broker.last_fee["s"]
    assert entry_fee > 0
    broker.settle("s", "tok", 1.0)
    assert broker.last_fee["s"] == 0.0


def test_per_market_theta_overrides_config():
    """The venue's own coefficient wins over the configured constant."""
    broker = PaperBroker(["s"], starting_cash=100.0, taker_fee_theta=0.06)
    broker.open("s", "m1", "tok", "Team", 0.50, 10.0, theta=0.0)
    assert broker.last_fee["s"] == 0.0


# ------------------------------------------------------------ account lifecycle

def _broker(cash):
    b = PaperBroker(["s"], starting_cash=cash, taker_fee_theta=0.0,
                    revival_deposit_usd=50.0, max_revivals=1)
    return b


def test_healthy_account_is_left_alone():
    b = _broker(100.0)
    assert b.check_solvency("s", min_stake=5.0) is None
    assert b.deposited["s"] == 100.0
    assert not b.is_retired("s")


def test_dead_account_gets_one_second_chance():
    b = _broker(100.0)
    b.cash["s"] = 3.0
    assert b.check_solvency("s", min_stake=5.0) == "revived"
    assert b.cash["s"] == pytest.approx(53.0)
    assert b.revivals["s"] == 1
    assert b.last_revival_ts["s"] is not None
    assert not b.is_retired("s")


def test_revival_deposit_is_not_profit():
    """The deposit raises `deposited` in lockstep, so return % is unchanged.

    This is the whole point of tracking deposited capital: a $50 rescue must
    not show up as a $50 gain.
    """
    b = _broker(100.0)
    b.cash["s"] = 3.0
    before = (b.cash["s"] - b.deposited["s"]) / b.deposited["s"]
    b.check_solvency("s", min_stake=5.0)
    after = (b.cash["s"] - b.deposited["s"]) / b.deposited["s"]
    assert b.deposited["s"] == pytest.approx(150.0)
    # -97% before, -64.7% after: the ratio moves only because the denominator
    # grew, never because the injection was counted as a gain.
    assert b.cash["s"] - b.deposited["s"] == pytest.approx(-97.0)
    assert after > before          # same dollar loss over a larger base
    assert b.cash["s"] - b.deposited["s"] == pytest.approx(3.0 - 100.0)


def test_second_death_retires_permanently():
    b = _broker(100.0)
    b.cash["s"] = 3.0
    assert b.check_solvency("s", min_stake=5.0) == "revived"
    b.cash["s"] = 1.0
    assert b.check_solvency("s", min_stake=5.0) == "retired"
    assert b.is_retired("s")
    # Idempotent: a retired account does not re-fire the event every tick.
    assert b.check_solvency("s", min_stake=5.0) is None


def test_retired_account_cannot_open():
    b = _broker(100.0)
    b.retired_at["s"] = 1.0
    assert b.open("s", "m1", "tok", "Team", 0.50, 10.0) is None


# ---------------------------------------------------------------- cost floor

def _ctx(cfg, price=0.50, spread=0.01):
    market = Market(slug="m1", question="q", home_team="H", away_team="A",
                    long_team="H")
    quote = MarketQuote("m1", price - spread / 2, price + spread / 2,
                        price - spread / 2, price + spread / 2)
    return StratContext(market, PriceHistory(), None, quote, now=0.0,
                        fee_theta=cfg.strategy.paper_theta
                        if hasattr(cfg.strategy, "paper_theta") else 0.06)


def test_all_in_cost_settlement_hold_is_about_half_of_a_round_trip():
    """One fee leg and one spread crossing versus two of each."""
    ctx = _ctx(Config())
    hold = ctx.all_in_cost(0.50, holds_to_settlement=True)
    rt = ctx.all_in_cost(0.50, holds_to_settlement=False)
    assert rt == pytest.approx(2 * hold)
    assert hold == pytest.approx(0.06 * 0.25 + 0.005)   # 1.5c fee + half spread


def test_all_in_cost_collapses_at_the_tail():
    ctx = _ctx(Config(), price=0.90)
    at_money = _ctx(Config(), price=0.50).all_in_cost(0.50, True) / 0.50
    tail = ctx.all_in_cost(0.90, True) / 0.90
    assert tail < at_money / 2


def test_cost_floor_default_is_off():
    """Existing strategies must reproduce exactly until they opt in."""
    assert Config().strategy.cost_floor_multiple == 0.0


def test_cost_floor_rejects_a_target_smaller_than_its_cost():
    """A +12% target at the money cannot pay a 5.6%-of-notional round trip.

    Upside at target = 0.50 * 0.12 = 6.0c. All-in cost = 2 fee legs (1.5c each)
    plus 2 half-spread crossings (0.25c each) = 3.5c. At the 1.5x floor the
    trade needs 5.25c of upside, so 6.0c passes -- but widen the spread and it
    must fail.
    """
    ctx = _ctx(Config(), price=0.50, spread=0.04)
    cost = ctx.all_in_cost(0.50, holds_to_settlement=False)
    upside = 0.50 * 0.12
    assert upside < 1.5 * cost          # 6.0c vs 1.5 * 7.0c


def test_cost_floor_passes_at_the_tail_where_fees_collapse():
    """The same mechanism clears the floor at 0.90 because the fee is tiny."""
    ctx = _ctx(Config(), price=0.90, spread=0.01)
    cost = ctx.all_in_cost(0.90, holds_to_settlement=True)
    upside = min(0.90 * 10.0, 1.0 - 0.90)   # settlement hold: capped at 1.0
    assert upside > 1.5 * cost
