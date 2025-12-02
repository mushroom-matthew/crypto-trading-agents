"""Normalization helpers for standard OHLCV frames."""

from __future__ import annotations

from datetime import timezone

import pandas as pd

DEFAULT_OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
TIME_INDEX_NAME = "time"


def ensure_datetime_index(
    df: pd.DataFrame,
    index_name: str = TIME_INDEX_NAME,
    tz: timezone = timezone.utc,
) -> pd.DataFrame:
    """Ensure the frame is indexed by a timezone-aware DatetimeIndex."""

    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)
    elif index_name in df.columns:
        df[index_name] = pd.to_datetime(df[index_name], utc=True)
        df = df.set_index(index_name)
    else:
        raise ValueError("DataFrame must be indexed by time or include a time column")
    df.sort_index(inplace=True)
    return df


def ensure_required_columns(df: pd.DataFrame, required: tuple[str, ...]) -> pd.DataFrame:
    """Verify that the expected OHLCV columns are present."""

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return df


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Run lightweight validation for NaNs, duplicates, and ordering."""

    if df.empty:
        raise ValueError("Received empty OHLCV frame")
    if df.index.has_duplicates:
        raise ValueError("OHLCV frame contains duplicate timestamps")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV frame is not time-ordered")
    if df.isna().any().any():
        raise ValueError("OHLCV frame contains NaN values")


__all__ = [
    "DEFAULT_OHLCV_COLUMNS",
    "TIME_INDEX_NAME",
    "ensure_datetime_index",
    "ensure_required_columns",
    "validate_ohlcv",
]
