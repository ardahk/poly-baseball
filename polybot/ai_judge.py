"""Claude-based trade judge (the "ai" strategy's gate on math signals)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .config import AIConfig
from .models import GameState, Signal

log = logging.getLogger(__name__)

_SYSTEM = """You are a risk gate for a Polymarket baseball trading bot.
The bot trades small, short-horizon mean-reversion positions: it buys a team's
win token after a sharp price move when a win-probability model says the move
overshot. Your job is to approve or reject each proposed trade.

Reject when: the price move is plausibly justified by the game situation
(late innings, large lead, high-leverage state where a single event genuinely
changes the game), the model edge looks like model error rather than market
overreaction, or the token price leaves poor reward-vs-risk for a 5-30% target.
Approve when the move looks like an emotional overreaction likely to revert.
Be decisive; this gate runs on live prices."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "approve": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["approve", "confidence", "reason"],
    "additionalProperties": False,
}


@dataclass
class Judgment:
    approve: bool
    confidence: float
    reason: str


class AIJudge:
    def __init__(self, cfg: AIConfig):
        self.cfg = cfg
        self._client = None
        if cfg.enabled:
            try:
                import anthropic
                self._client = anthropic.Anthropic(timeout=cfg.timeout_secs)
            except Exception as exc:  # missing key / package
                log.warning("AI judge disabled: %s", exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    def judge(self, signal: Signal, game_state: GameState | None) -> Judgment:
        """Fails closed: any error -> reject (only affects the 'ai' ledger)."""
        if not self._client:
            return Judgment(False, 0.0, "AI judge unavailable")
        snapshot = {
            "market": signal.market.question,
            "backing": signal.side_team,
            "token_price": round(signal.price, 3),
            "model_fair_value": round(signal.fair, 3),
            "edge": round(signal.edge, 3),
            "recent_move": round(signal.move, 3),
            "game": None,
        }
        if game_state:
            snapshot["game"] = {
                "inning": game_state.inning,
                "half": "top" if game_state.is_top else "bottom",
                "outs": game_state.outs,
                "score": f"away {game_state.away_score} - home {game_state.home_score}",
                "runners_on": [
                    b for b, on in (
                        ("first", game_state.on_first),
                        ("second", game_state.on_second),
                        ("third", game_state.on_third),
                    ) if on
                ],
            }
        try:
            response = self._client.messages.create(
                model=self.cfg.model,
                max_tokens=1024,
                system=_SYSTEM,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.cfg.effort,
                    "format": {"type": "json_schema", "schema": _SCHEMA},
                },
                messages=[{
                    "role": "user",
                    "content": "Proposed trade:\n" + json.dumps(snapshot, indent=2),
                }],
            )
            if response.stop_reason == "refusal":
                return Judgment(False, 0.0, "model refused")
            text = next(b.text for b in response.content if b.type == "text")
            data = json.loads(text)
            verdict = Judgment(
                approve=bool(data["approve"]),
                confidence=float(data["confidence"]),
                reason=str(data["reason"]),
            )
            if verdict.approve and verdict.confidence < self.cfg.min_confidence:
                return Judgment(False, verdict.confidence,
                                f"below confidence floor: {verdict.reason}")
            return verdict
        except Exception as exc:
            log.warning("AI judge error (rejecting): %s", exc)
            return Judgment(False, 0.0, f"judge error: {exc}")
