"""Shared helpers for data-loader implementations."""

from __future__ import annotations

from datetime import datetime, timezone


def timeframe_to_seconds(granularity: str) -> int:
    """Convert ``1m``/``1h`` style strings to seconds."""

    if not granularity:
        raise ValueError("Granularity string is required")
    unit = granularity[-1]
    value = int(granularity[:-1])
    factors = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if unit not in factors:
        raise ValueError(f"Unsupported timeframe unit: {unit}")
    return value * factors[unit]


def ensure_utc(dt: datetime) -> datetime:
    """Return a timezone-aware datetime in UTC."""

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


__all__ = ["ensure_utc", "timeframe_to_seconds"]
