"""ANSI terminal dashboard for the live engine."""
from __future__ import annotations

import shutil
import sys
import time
from collections import deque
from typing import Any

from .winprob import home_win_probability


class TerminalDashboard:
    """Small dependency-free dashboard for observing an Engine in a terminal."""

    def __init__(self, enabled: bool = False, refresh_secs: float = 1.0, stream=None):
        self.enabled = enabled
        self.refresh_secs = refresh_secs
        self.stream = stream if stream is not None else sys.stdout
        self.events: deque[tuple[float, str]] = deque(maxlen=10)
        self._last_render = 0.0
        self._started = False

    def start(self) -> None:
        if not self.enabled or self._started:
            return
        self._started = True
        self.stream.write("\033[?25l")
        self.stream.flush()

    def close(self) -> None:
        if not self.enabled or not self._started:
            return
        self.stream.write("\033[?25h\n")
        self.stream.flush()
        self._started = False

    def record(self, message: str) -> None:
        if self.enabled:
            self.events.append((time.time(), message))

    def render(self, engine: Any, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self._last_render < self.refresh_secs:
            return
        self._last_render = now

        width, height = shutil.get_terminal_size((120, 40))
        lines = self._build_lines(engine, width, height, now)
        self.stream.write("\033[H\033[2J")
        self.stream.write("\n".join(line[:width] for line in lines))
        self.stream.write("\n")
        self.stream.flush()

    def _build_lines(self, engine: Any, width: int, height: int, now: float) -> list[str]:
        mode = "LIVE" if engine.cfg.engine.live else "PAPER"
        uptime = _duration(now - getattr(engine, "started_at", now))
        live_games = sum(
            1 for market in engine.markets.values()
            if (gs := engine.game_states.get(market.game_pk)) and gs.is_live
        )
        recent_quotes = sum(
            1 for quote in engine.latest_quotes.values()
            if now - quote.ts <= max(30.0, engine.cfg.engine.poll_interval_secs * 5)
        )

        lines = [
            f"POLYBOT {mode} | up {uptime} | tracked {len(engine.markets)} | "
            f"live {live_games} | fresh BBO {recent_quotes} | "
            f"{time.strftime('%H:%M:%S')}",
            "=" * width,
            self._poll_line(engine, now),
            "",
            "STRATEGIES",
        ]
        lines.extend(_table(
            ["name", "equity", "cash", "open", "realized", "closed"],
            [
                [
                    strat,
                    _money(engine.broker.equity(strat, engine.latest_prices)),
                    _money(engine.broker.cash[strat]),
                    str(len(engine.broker.open_positions(strat))),
                    _signed_money(engine.broker.realized.get(strat, 0.0)),
                    str(engine.broker.closes.get(strat, 0)),
                ]
                for strat in engine.strategies
            ],
            width,
        ))
        lines.extend(["", "MARKETS"])
        market_rows = self._market_rows(engine, now)
        if market_rows:
            lines.extend(_table(
                ["game", "state", "score", "home mid", "spread", "fair", "move", "flips", "age"],
                market_rows,
                width,
            ))
        else:
            lines.append("  no matched live markets yet")

        position_rows = self._position_rows(engine, now)
        lines.extend(["", "OPEN POSITIONS"])
        if position_rows:
            lines.extend(_table(
                ["strat", "team", "entry", "mark", "pnl", "age"],
                position_rows,
                width,
            ))
        else:
            lines.append("  flat")

        lines.extend(["", "EVENTS"])
        if self.events:
            for ts, event in reversed(self.events):
                lines.append(f"  {time.strftime('%H:%M:%S', time.localtime(ts))}  {event}")
        else:
            lines.append("  waiting for engine events")

        footer = "Ctrl-C to stop | logs are quiet in dashboard mode unless --verbose is set"
        available = max(1, height - 1)
        if len(lines) > available:
            lines = lines[:available]
            lines[-1] = footer[:width]
        else:
            lines.append(footer)
        return lines

    def _poll_line(self, engine: Any, now: float) -> str:
        discovery_due = max(
            0.0,
            engine.cfg.engine.discovery_interval_secs - (now - engine._last_discovery),
        )
        game_due = max(
            0.0,
            engine.cfg.engine.game_state_interval_secs - (now - engine._last_game_poll),
        )
        status_due = max(
            0.0,
            engine.cfg.engine.status_log_interval_secs - (now - engine._last_status_log),
        )
        return (
            f"polls: discover {_duration(discovery_due)} | "
            f"game-state {_duration(game_due)} | status-log {_duration(status_due)}"
        )

    def _market_rows(self, engine: Any, now: float) -> list[list[str]]:
        rows = []
        for market in engine.markets.values():
            quote = engine.latest_quotes.get(market.key)
            gs = engine.game_states.get(market.game_pk)
            state = "live" if gs and gs.is_live else "final" if gs and gs.is_final else "pending"
            score = "-"
            fair = "-"
            if gs:
                score = f"{gs.away_score}-{gs.home_score}"
                if gs.is_live:
                    half = "T" if gs.is_top else "B"
                    state = f"{half}{gs.inning}/{gs.outs}o"
                    fair = _prob(home_win_probability(gs))
            history = engine.histories.get(market.key)
            move = history.move(engine.cfg.strategy.move_lookback_secs) if history else None
            rows.append([
                _game_label(market),
                state,
                score,
                _prob(quote.home_mid) if quote else "-",
                _prob(quote.home_spread) if quote else "-",
                fair,
                _signed_prob(move) if move is not None else "-",
                str(history.flips if history else 0),
                _duration(now - quote.ts) if quote else "-",
            ])
        rows.sort(key=lambda row: (row[1] == "pending", row[-1]))
        return rows[:8]

    def _position_rows(self, engine: Any, now: float) -> list[list[str]]:
        rows = []
        for strat in engine.strategies:
            for pos in engine.broker.open_positions(strat):
                mark = engine.latest_prices.get(pos.token, pos.entry_price)
                pnl = pos.pnl_pct(mark)
                rows.append([
                    strat,
                    pos.team,
                    _prob(pos.entry_price),
                    _prob(mark),
                    f"{pnl:+.1%}",
                    _duration(now - pos.opened_at),
                ])
        return rows


def _table(headers: list[str], rows: list[list[str]], width: int) -> list[str]:
    if not rows:
        return ["  none"]
    trimmed_rows = [[_clip(cell, 30) for cell in row] for row in rows]
    widths = [
        min(30, max(len(headers[i]), *(len(row[i]) for row in trimmed_rows)))
        for i in range(len(headers))
    ]
    line = "  " + "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    rule = "  " + "  ".join("-" * widths[i] for i in range(len(headers)))
    rendered = [line[:width], rule[:width]]
    for row in trimmed_rows:
        rendered.append(
            ("  " + "  ".join(row[i].ljust(widths[i]) for i in range(len(row))))[:width]
        )
    return rendered


def _clip(text: object, limit: int) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "."


def _game_label(market: Any) -> str:
    return _clip(f"{market.away_team} @ {market.home_team}", 30)


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _money(value: float) -> str:
    return f"${value:.2f}"


def _signed_money(value: float) -> str:
    return f"{value:+.2f}"


def _prob(value: float) -> str:
    return f"{value:.3f}"


def _signed_prob(value: float) -> str:
    return f"{value:+.3f}"
