"""Helpers for consistent idempotency key generation."""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class IdempotencyKey:
    """Idempotency key with creation timestamp."""

    value: str
    created_at: float

    def headers(self) -> Mapping[str, str]:
        """Return HTTP headers enforcing idempotency."""

        return {"Idempotency-Key": self.value}


def new_key(*, prefix: str | None = None) -> IdempotencyKey:
    """Generate a new deterministic-ish key with optional prefix."""

    salt = uuid.uuid4().hex
    entropy = os.urandom(16)
    digest = hashlib.sha256(entropy + salt.encode("utf-8")).hexdigest()
    value = f"{prefix}-{digest}" if prefix else digest
    return IdempotencyKey(value=value, created_at=time.time())


def from_value(value: str) -> IdempotencyKey:
    """Wrap an existing idempotency string."""

    return IdempotencyKey(value=value, created_at=time.time())


__all__ = ["IdempotencyKey", "new_key", "from_value"]
