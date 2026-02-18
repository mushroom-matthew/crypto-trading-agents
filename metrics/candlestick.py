"""Candlestick morphology features for the indicator pipeline.

All functions operate on standard OHLCV DataFrames with columns:
open, high, low, close, volume.

Boolean features are returned as float (0.0 / 1.0) so they survive the
existing serialization pipeline. Evaluate in trigger rules as:
    is_hammer == 1   or   is_hammer > 0.5
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scalar features
# ---------------------------------------------------------------------------

def body_pct(df: pd.DataFrame) -> pd.Series:
    """Body as fraction of total range. 0 = doji, 1 = marubozu."""
    body = (df["close"] - df["open"]).abs()
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    return (body / range_).fillna(0.0)


def upper_wick_pct(df: pd.DataFrame) -> pd.Series:
    """Upper wick as fraction of total range."""
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    return (upper / range_).fillna(0.0)


def lower_wick_pct(df: pd.DataFrame) -> pd.Series:
    """Lower wick as fraction of total range."""
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    return (lower / range_).fillna(0.0)


def candle_strength(df: pd.DataFrame, atr_series: pd.Series) -> pd.Series:
    """Body size normalized by ATR.

    >1 = impulse candle (body equals or exceeds a full ATR move),
    <0.3 = indecision / doji-like candle.
    """
    body = (df["close"] - df["open"]).abs()
    return (body / atr_series.replace(0, np.nan)).fillna(0.0)


# ---------------------------------------------------------------------------
# Single-bar directional
# ---------------------------------------------------------------------------

def is_bullish(df: pd.DataFrame) -> pd.Series:
    """1.0 if close > open (bullish close)."""
    return (df["close"] > df["open"]).astype(float)


def is_bearish(df: pd.DataFrame) -> pd.Series:
    """1.0 if close < open (bearish close)."""
    return (df["close"] < df["open"]).astype(float)


# ---------------------------------------------------------------------------
# Single-bar reversal patterns
# ---------------------------------------------------------------------------

def is_doji(df: pd.DataFrame) -> pd.Series:
    """Indecision bar: body < 10% of total range."""
    return (body_pct(df) < 0.10).astype(float)


def is_hammer(df: pd.DataFrame) -> pd.Series:
    """Bullish reversal shape at a low: long lower wick, small body near top.

    Conditions: lower_wick > 2 * body, upper_wick < body, body > 0.
    Pattern context matters — a hammer is bullish only at a support level
    or after a downtrend. Combine with structural context in trigger rules.
    """
    body = (df["close"] - df["open"]).abs()
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    return ((lower > 2 * body) & (upper < body) & (body > 0)).astype(float)


def is_shooting_star(df: pd.DataFrame) -> pd.Series:
    """Bearish reversal shape at a high: long upper wick, small body near bottom.

    Conditions: upper_wick > 2 * body, lower_wick < body, body > 0.
    Pattern context matters — bearish only at resistance or after uptrend.
    """
    body = (df["close"] - df["open"]).abs()
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    return ((upper > 2 * body) & (lower < body) & (body > 0)).astype(float)


def is_pin_bar(df: pd.DataFrame) -> pd.Series:
    """Generic pin bar: wick on either side > 60% of total range.

    Covers both hammer and shooting star shapes. Use is_hammer or
    is_shooting_star for direction-specific signals.
    """
    return ((upper_wick_pct(df) > 0.60) | (lower_wick_pct(df) > 0.60)).astype(float)


# ---------------------------------------------------------------------------
# Two-bar patterns (compare current bar to prior bar)
# ---------------------------------------------------------------------------

def is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Current bullish bar fully engulfs prior bearish bar body.

    Conditions: prior close < prior open (bearish), current close > current open
    (bullish), current open <= prior close, current close >= prior open.
    """
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prior_bearish = prev_close < prev_open
    curr_bullish = df["close"] > df["open"]
    engulfs = (df["open"] <= prev_close) & (df["close"] >= prev_open)
    return (prior_bearish & curr_bullish & engulfs).astype(float)


def is_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Current bearish bar fully engulfs prior bullish bar body.

    Conditions: prior close > prior open (bullish), current close < current open
    (bearish), current open >= prior close, current close <= prior open.
    """
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prior_bullish = prev_close > prev_open
    curr_bearish = df["close"] < df["open"]
    engulfs = (df["open"] >= prev_close) & (df["close"] <= prev_open)
    return (prior_bullish & curr_bearish & engulfs).astype(float)


def is_inside_bar(df: pd.DataFrame) -> pd.Series:
    """Current bar's high AND low contained within prior bar's range.

    Indicates compression or indecision within prior bar's range.
    Combine with low Bollinger bandwidth for setup detection.
    """
    return (
        (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))
    ).astype(float)


def is_outside_bar(df: pd.DataFrame) -> pd.Series:
    """Current bar's high exceeds prior high AND low undercuts prior low.

    Indicates volatility expansion (also called 'key reversal bar').
    At a key level, often precedes a strong directional move.
    """
    return (
        (df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))
    ).astype(float)


def is_impulse_candle(
    df: pd.DataFrame,
    atr_series: pd.Series,
    min_strength: float = 1.0,
) -> pd.Series:
    """Body >= min_strength * ATR. Indicates a decisive directional move.

    Default threshold is 1.0 (body equals one full ATR) rather than 0.8,
    to avoid over-triggering in high-volatility crypto regimes. Configurable
    via the IMPULSE_CANDLE_ATR_MULT environment variable.
    """
    return (candle_strength(df, atr_series) >= min_strength).astype(float)


# ---------------------------------------------------------------------------
# Composite entry point
# ---------------------------------------------------------------------------

def compute_candlestick_features(
    df: pd.DataFrame,
    atr_series: pd.Series,
) -> pd.DataFrame:
    """Compute all 15 candlestick features and return as a DataFrame.

    One column per feature, one row per bar in df. Boolean features are
    encoded as float (0.0 / 1.0) for serialization compatibility.

    Args:
        df: OHLCV DataFrame with columns open, high, low, close, volume.
        atr_series: ATR series aligned to df's index (e.g. ATR-14).

    Returns:
        DataFrame with columns:
            candle_body_pct, candle_upper_wick_pct, candle_lower_wick_pct,
            candle_strength, is_bullish, is_bearish, is_doji, is_hammer,
            is_shooting_star, is_pin_bar, is_bullish_engulfing,
            is_bearish_engulfing, is_inside_bar, is_outside_bar,
            is_impulse_candle.
    """
    _min_strength = float(os.environ.get("IMPULSE_CANDLE_ATR_MULT", "1.0"))
    return pd.DataFrame(
        {
            "candle_body_pct": body_pct(df),
            "candle_upper_wick_pct": upper_wick_pct(df),
            "candle_lower_wick_pct": lower_wick_pct(df),
            "candle_strength": candle_strength(df, atr_series),
            "is_bullish": is_bullish(df),
            "is_bearish": is_bearish(df),
            "is_doji": is_doji(df),
            "is_hammer": is_hammer(df),
            "is_shooting_star": is_shooting_star(df),
            "is_pin_bar": is_pin_bar(df),
            "is_bullish_engulfing": is_bullish_engulfing(df),
            "is_bearish_engulfing": is_bearish_engulfing(df),
            "is_inside_bar": is_inside_bar(df),
            "is_outside_bar": is_outside_bar(df),
            "is_impulse_candle": is_impulse_candle(df, atr_series, _min_strength),
        },
        index=df.index,
    )
