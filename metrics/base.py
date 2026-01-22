"""Shared utilities and validators for metrics computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: Sequence[str] = ("timestamp", "open", "high", "low", "close", "volume")
FLOAT_COLUMNS: Sequence[str] = ("open", "high", "low", "close", "volume")
EPS = np.finfo(np.float64).eps


class MetricsDataError(ValueError):
    """Raised when provided OHLCV data does not satisfy the contract."""


def prepare_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of the OHLCV dataframe.

    Ensures required columns exist, timestamp is in UTC (if timezone-naive it is
    assumed to be UTC), data is sorted ascending, and numeric columns are float64.
    """

    if df is None:
        raise MetricsDataError("Input dataframe cannot be None")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise MetricsDataError(f"Missing required columns: {missing}")

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    if data["timestamp"].isna().any():
        raise MetricsDataError("All timestamp values must be parseable datetime objects")

    # Enforce ascending chronological order
    data.sort_values("timestamp", inplace=True)
    data.reset_index(drop=True, inplace=True)

    for col in FLOAT_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype("float64")

    return data


def validate_non_negative(series: pd.Series, name: str) -> None:
    """Ensure the provided series contains no negative values after preparation."""

    if (series < 0).any():
        raise MetricsDataError(f"{name} must be non-negative")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return numerator / denominator with zero-denominator protection."""

    numerator = numerator.astype("float64")
    denominator = denominator.astype("float64")
    result = numerator / denominator.replace(0.0, np.nan)
    return result.fillna(0.0)


def wilder_smoothing(series: pd.Series, period: int) -> pd.Series:
    """Return Wilder's exponential moving average."""

    alpha = 1.0 / float(period)
    return series.ewm(alpha=alpha, adjust=False).mean()


def linear_weighted_moving_average(series: pd.Series, period: int) -> pd.Series:
    """Return the linearly weighted moving average (WMA)."""

    weights = np.arange(1, period + 1, dtype=np.float64)
    weight_sum = weights.sum()
    return series.rolling(window=period, min_periods=period).apply(
        lambda arr: np.dot(arr, weights) / weight_sum, raw=True
    )


def rolling_std(series: pd.Series, period: int) -> pd.Series:
    """Return population standard deviation for the rolling window."""

    return series.rolling(window=period, min_periods=period).std(ddof=0)


def ensure_min_period(series: pd.Series, period: int) -> pd.Series:
    """Set initial values to NaN until the rolling window is populated."""

    mask = series.expanding().count() < period
    return series.where(~mask)


def cumulative_by_group(values: pd.Series, group: pd.Series) -> pd.Series:
    """Return cumulative sums partitioned by group boundaries."""

    return values.groupby(group, dropna=False).cumsum()


@dataclass(slots=True)
class MetricSeries:
    """Container describing a single metric time-series."""

    key: str
    column: str
    series: pd.Series


@dataclass(slots=True)
class MetricResult:
    """Structured metric output used by the registry dispatcher."""

    feature: str
    series_list: list[MetricSeries]


def build_single_series(feature: str, key: str, column: str, series: pd.Series) -> MetricResult:
    """Convenience helper for single-value indicators."""

    return MetricResult(feature=feature, series_list=[MetricSeries(key=key, column=column, series=series)])
