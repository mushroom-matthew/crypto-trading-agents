"""Sentiment metrics scaffolding for future releases.

Tier II/III roadmap items will populate this module with on-chain and off-chain
signals (Twitter momentum, news sentiment, funding rate deltas, etc.).  The
current milestone only establishes the module boundary so downstream imports are
stable once implementations arrive.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def sentiment_score(df: pd.DataFrame, **_: Any) -> None:
    """Placeholder for blended sentiment score."""

    raise NotImplementedError(
        "Sentiment score requires external NLP pipelines scheduled for Tier III."
    )


def event_volatility_ratio(df: pd.DataFrame, **_: Any) -> None:
    """Placeholder for event-volatility ratio indicator."""

    raise NotImplementedError(
        "Event-volatility analysis will be introduced with event ingestion services."
    )
