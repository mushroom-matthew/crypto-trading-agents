"""Scaffolding for Tier II/III market context metrics.

These indicators will encapsulate portfolio and market-wide context such as
Sharpe/Sortino, beta, dominance ratios, and regime detection.  They are left
unimplemented for the Tier I milestone to keep scope focused on OHLCV-based
technicals.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def sharpe_ratio(df: pd.DataFrame, **_: Any) -> None:
    """Placeholder for Sharpe ratio computations."""

    raise NotImplementedError(
        "Sharpe ratio will be implemented in Tier II once return series utilities are available."
    )


def sortino_ratio(df: pd.DataFrame, **_: Any) -> None:
    """Placeholder for Sortino ratio computations."""

    raise NotImplementedError(
        "Sortino ratio depends on downside deviation metrics slated for Tier II."
    )


def beta_to_benchmark(df: pd.DataFrame, **_: Any) -> None:
    """Placeholder for beta calculations relative to configurable benchmarks."""

    raise NotImplementedError(
        "Benchmark-aware beta will arrive alongside market data integration in Tier III."
    )
