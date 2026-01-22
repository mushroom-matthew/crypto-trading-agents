"""Webhook helpers for Coinbase callbacks."""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Optional

from app.core.errors import CoinbaseAPIError


class WebhookVerifier:
    """Verify Coinbase webhook signatures."""

    def __init__(self, shared_secret: str, *, tolerance_seconds: int = 180) -> None:
        self._secret = shared_secret.encode("utf-8")
        self._tolerance = tolerance_seconds

    def verify(self, *, body: bytes, signature: str, timestamp: str) -> None:
        """Validate signature and timestamp tolerance.

        Raises:
            CoinbaseAPIError: when the signature is invalid or outside tolerance.
        """

        try:
            ts = int(timestamp)
        except (TypeError, ValueError) as exc:  # pragma: no cover - sanity guard
            raise CoinbaseAPIError("Invalid webhook timestamp", 400) from exc

        if abs(time.time() - ts) > self._tolerance:
            raise CoinbaseAPIError("Webhook timestamp outside tolerance", 400)

        payload = f"{timestamp}{body.decode('utf-8')}".encode("utf-8")
        digest = hmac.new(self._secret, payload, hashlib.sha256).digest()
        expected = base64.b64encode(digest).decode("utf-8")
        if not hmac.compare_digest(expected, signature):
            raise CoinbaseAPIError("Webhook signature mismatch", 400)


__all__ = ["WebhookVerifier"]
