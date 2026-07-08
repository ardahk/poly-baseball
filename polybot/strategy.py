"""Signal generation: fade sharp moves that diverge from model fair value."""
from __future__ import annotations

import logging
import time

from .config import StrategyConfig
from .models import GameState, Market, Position, Signal
from .volatility import PriceHistory
from .winprob import home_win_probability

log = logging.getLogger(__name__)


def check_entry(
    market: Market,
    history: PriceHistory,          # history of the HOME token price
    game_state: GameState | None,
    cfg: StrategyConfig,
) -> Signal | None:
    """Return a buy signal if the market just overreacted vs the model.

    Logic: price of the home token moved sharply over the lookback window;
    the model disagrees with the new price by at least `min_edge` in the
    opposite direction of the move -> buy the undervalued side.
    """
    if game_state is None or not game_state.is_live:
        return None
    mid = history.last
    if mid is None:
        return None
    if not history.is_playful(cfg.min_flips, cfg.min_volatility, cfg.vol_window):
        return None
    move = history.move(cfg.move_lookback_secs)
    if move is None or abs(move) < cfg.move_threshold:
        return None

    fair_home = home_win_probability(game_state)

    if move < 0 and fair_home - mid >= cfg.min_edge:
        token, team, price, fair = market.home_token, market.home_team, mid, fair_home
    elif move > 0 and mid - fair_home >= cfg.min_edge:
        # Home overpriced -> the away token is the undervalued side.
        token, team = market.away_token, market.away_team
        price, fair = 1.0 - mid, 1.0 - fair_home
    else:
        return None

    if not (cfg.min_price <= price <= cfg.max_price):
        return None
    edge = fair - price
    if game_state.inning <= cfg.early_game_max_inning:
        fair_extreme = fair >= cfg.early_game_min_fair_extreme \
            or fair <= 1.0 - cfg.early_game_min_fair_extreme
        if edge < cfg.early_game_min_edge or not fair_extreme:
            return None

    return Signal(
        market=market, token=token, side_team=team,
        price=price, fair=fair, move=move,
        reason=(
            f"fade move {move:+.3f}/{cfg.move_lookback_secs:.0f}s; "
            f"price {price:.3f} vs fair {fair:.3f} (edge {edge:+.3f})"
        ),
    )


def check_exit(
    position: Position,
    current_price: float,
    fair_value: float | None,
    game_final: bool,
    cfg: StrategyConfig,
    now: float | None = None,
) -> str | None:
    """Return a close reason, or None to hold."""
    now = now if now is not None else time.time()
    pnl = position.pnl_pct(current_price)
    if game_final:
        return f"game final (pnl {pnl:+.1%})"
    if pnl >= cfg.take_profit:
        return f"take profit {pnl:+.1%}"
    if pnl <= -cfg.stop_loss:
        return f"stop loss {pnl:+.1%}"
    if now - position.opened_at >= cfg.max_hold_secs:
        return f"time stop {pnl:+.1%}"
    if fair_value is not None and fair_value - current_price <= -cfg.edge_exit:
        return f"edge gone (fair {fair_value:.3f} < price {current_price:.3f}, pnl {pnl:+.1%})"
    return None
