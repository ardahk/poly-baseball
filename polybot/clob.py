"""Read-only price data from the Polymarket CLOB REST API."""
from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)

CLOB_URL = "https://clob.polymarket.com"


class PriceFeed:
    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def midpoint(self, token_id: str) -> float | None:
        try:
            resp = self.session.get(
                f"{CLOB_URL}/midpoint", params={"token_id": token_id}, timeout=10
            )
            resp.raise_for_status()
            mid = resp.json().get("mid")
            return float(mid) if mid is not None else None
        except Exception as exc:
            log.debug("midpoint fetch failed for %s: %s", token_id, exc)
            return None

    def best_price(self, token_id: str, side: str) -> float | None:
        """Best executable price. side='buy' -> best ask, side='sell' -> best bid."""
        try:
            resp = self.session.get(
                f"{CLOB_URL}/price",
                params={"token_id": token_id, "side": side.upper()},
                timeout=10,
            )
            resp.raise_for_status()
            price = resp.json().get("price")
            return float(price) if price is not None else None
        except Exception as exc:
            log.debug("price fetch failed for %s/%s: %s", token_id, side, exc)
            return None
