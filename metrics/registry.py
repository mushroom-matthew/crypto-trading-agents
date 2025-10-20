"""Registry-driven dispatcher for metric computation."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

import pandas as pd

from . import base
from .technical import (
    adx,
    atr,
    bollinger_bands,
    ema,
    macd,
    obv,
    roc,
    rsi,
    sma,
    vwap,
    wma,
)
from .typing import MetricFunction, MetricParams, MetricRegistry, MetricResult


REGISTRY: MetricRegistry = {
    "SMA": sma,
    "EMA": ema,
    "WMA": wma,
    "MACD": macd,
    "RSI": rsi,
    "BollingerBands": bollinger_bands,
    "ATR": atr,
    "ADX": adx,
    "ROC": roc,
    "OBV": obv,
    "VWAP": vwap,
}


def list_metrics() -> List[str]:
    """Return the list of registered metrics."""

    return sorted(REGISTRY.keys())


def _resolve_features(features: Sequence[str] | None) -> List[str]:
    available = list_metrics()
    if features is None:
        return available

    unknown = [name for name in features if name not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown metrics requested: {unknown}")
    return list(dict.fromkeys(features))  # Preserve order while removing duplicates


def compute_metrics(
    df: pd.DataFrame,
    features: Sequence[str] | None = None,
    params: MetricParams | None = None,
    output: str = "wide",
) -> pd.DataFrame:
    """Compute metrics and return either wide or long dataframe.

    Parameters
    ----------
    df:
        OHLCV dataframe adhering to the data contract.
    features:
        Names of metrics to compute; defaults to all registered metrics.
    params:
        Optional mapping of per-feature keyword arguments.
    output:
        Either ``"wide"`` (append columns to dataframe) or ``"long"`` (tidy format).
    """

    prepared = base.prepare_ohlcv_df(df)
    feature_names = _resolve_features(features)
    params = params or {}

    results: List[MetricResult] = []
    for feature in feature_names:
        fn = REGISTRY[feature]
        feature_params = dict(params.get(feature, {}))
        result = fn(prepared, **feature_params)
        results.append(result)

    if output == "wide":
        return _merge_wide(prepared, results)
    if output == "long":
        return _merge_long(prepared, results)
    raise ValueError("output must be 'wide' or 'long'")


def _merge_wide(df: pd.DataFrame, results: Sequence[MetricResult]) -> pd.DataFrame:
    merged = df.copy()
    for result in results:
        for series in result.series_list:
            merged[series.column] = series.series.values
    return merged


def _merge_long(df: pd.DataFrame, results: Sequence[MetricResult]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for result in results:
        for series in result.series_list:
            frame = pd.DataFrame(
                {
                    "timestamp": df["timestamp"],
                    "feature": result.feature,
                    "key": series.key,
                    "value": series.series.values,
                }
            )
            records.append(frame)

    if not records:
        return pd.DataFrame(columns=["timestamp", "feature", "key", "value"])

    tidy = pd.concat(records, ignore_index=True)
    tidy.sort_values(["timestamp", "feature", "key"], inplace=True)
    tidy.reset_index(drop=True, inplace=True)
    return tidy
