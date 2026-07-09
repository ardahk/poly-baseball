"""Signal generation: fade sharp moves that diverge from model fair value."""
from __future__ import annotations

import logging
import time

from .config import StrategyConfig
from .models import EntryEvaluation, GameState, Market, Position, Signal
from .volatility import PriceHistory
from .winprob import home_win_probability

log = logging.getLogger(__name__)


def fair_home_value(game_state: GameState, cfg: StrategyConfig) -> float:
    """Model P(home win) with the configured home-bias correction applied."""
    fair = home_win_probability(game_state) - cfg.home_fair_shrink
    return min(max(fair, 0.001), 0.999)


def check_entry(
    market: Market,
    history: PriceHistory,          # history of the HOME token price
    game_state: GameState | None,
    cfg: StrategyConfig,
    funnel: dict[str, int] | None = None,   # optional reject-reason counters
) -> Signal | None:
    """Return a buy signal if the market just overreacted vs the model."""
    ev = evaluate_entry(market, history, game_state, cfg)
    if funnel is not None:
        funnel[ev.outcome] = funnel.get(ev.outcome, 0) + 1
    return ev.signal


def evaluate_entry(
    market: Market,
    history: PriceHistory,
    game_state: GameState | None,
    cfg: StrategyConfig,
) -> EntryEvaluation:
    """Run the entry gates and keep every intermediate value for analysis.

    `margin` is the signed distance from the gate that failed. Negative values
    are near-miss rejections; positive values passed that gate.
    """
    mid = history.last
    realized_vol = history.realized_vol(cfg.vol_window)
    base = {
        "mid": mid,
        "flips": history.flips,
        "realized_vol": realized_vol,
    }
    if game_state is None or not game_state.is_live:
        return EntryEvaluation(outcome="not_live", **base)
    if mid is None:
        return EntryEvaluation(outcome="no_price", **base)
    if not history.is_playful(cfg.min_flips, cfg.min_volatility, cfg.vol_window):
        return EntryEvaluation(
            outcome="not_playful",
            margin=realized_vol - cfg.min_volatility,
            **base,
        )
    move = history.move(cfg.move_lookback_secs)
    if move is None or abs(move) < cfg.move_threshold:
        return EntryEvaluation(
            outcome="small_move",
            move=move,
            margin=(abs(move) - cfg.move_threshold) if move is not None else -cfg.move_threshold,
            **base,
        )

    fair_home = fair_home_value(game_state, cfg)

    if move < 0 and fair_home - mid >= cfg.min_edge:
        token, team, price, fair = market.home_token, market.home_team, mid, fair_home
    elif move > 0 and mid - fair_home >= cfg.min_edge:
        # Home overpriced -> the away token is the undervalued side.
        token, team = market.away_token, market.away_team
        price, fair = 1.0 - mid, 1.0 - fair_home
    else:
        directed_edge = (fair_home - mid) if move < 0 else (mid - fair_home)
        return EntryEvaluation(
            outcome="no_edge",
            move=move,
            fair_home=fair_home,
            edge=directed_edge,
            margin=directed_edge - cfg.min_edge,
            **base,
        )

    if not (cfg.min_price <= price <= cfg.max_price):
        margin = price - cfg.min_price if price < cfg.min_price else cfg.max_price - price
        return EntryEvaluation(
            outcome="price_band",
            move=move,
            fair_home=fair_home,
            side_team=team,
            price=price,
            fair=fair,
            edge=fair - price,
            margin=margin,
            **base,
        )
    edge = fair - price
    if game_state.inning <= cfg.early_game_max_inning:
        fair_extreme = fair >= cfg.early_game_min_fair_extreme \
            or fair <= 1.0 - cfg.early_game_min_fair_extreme
        if edge < cfg.early_game_min_edge or not fair_extreme:
            fair_margin = max(
                fair - cfg.early_game_min_fair_extreme,
                (1.0 - cfg.early_game_min_fair_extreme) - fair,
            )
            return EntryEvaluation(
                outcome="early_game",
                move=move,
                fair_home=fair_home,
                side_team=team,
                price=price,
                fair=fair,
                edge=edge,
                margin=min(edge - cfg.early_game_min_edge, fair_margin),
                **base,
            )

    sig = Signal(
        market=market, token=token, side_team=team,
        price=price, fair=fair, move=move,
        reason=(
            f"fade move {move:+.3f}/{cfg.move_lookback_secs:.0f}s; "
            f"price {price:.3f} vs fair {fair:.3f} (edge {edge:+.3f})"
        ),
    )
    return EntryEvaluation(
        outcome="signal",
        move=move,
        fair_home=fair_home,
        side_team=team,
        price=price,
        fair=fair,
        edge=edge,
        margin=edge - cfg.min_edge,
        signal=sig,
        **base,
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


def exit_kind(reason: str) -> str:
    if reason.startswith("take profit"):
        return "take_profit"
    if reason.startswith("stop loss"):
        return "stop_loss"
    if reason.startswith("time stop"):
        return "time_stop"
    if reason.startswith("edge gone"):
        return "edge_gone"
    if reason.startswith("game final"):
        return "game_final"
    return "other"
