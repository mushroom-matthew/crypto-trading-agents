"""Deterministic regime classification from indicator snapshots.

R55 Vocabulary Mapping:
The outputs from classify_regime() use labels from schemas/llm_strategist.py:
  - RegimeAssessment.regime: 'bull', 'bear', 'range', 'volatile', 'uncertain'
  - AssetState.trend_state:  'uptrend', 'downtrend', 'sideways'
  - AssetState.vol_state:    'low', 'normal', 'high', 'extreme'

These are mapped to R55 canonical labels in services/regime_transition_detector.py:
  - trend_state:  'up', 'down', 'sideways'
  - vol_state:    'low', 'mid', 'high', 'extreme'  (note: 'normal' → 'mid')
  - structure_state: derived from R40 flags + regime label
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schemas.llm_strategist import IndicatorSnapshot, RegimeAssessment


# R55 vocabulary mapping re-exports (canonical entry point for callers)
# Full implementations live in services/regime_transition_detector.py.
def get_r55_vocabulary_mapping_version() -> str:
    """Return the current R55 vocabulary mapping version string."""
    from services.regime_transition_detector import _VOCAB_MAPPING_VERSION  # type: ignore[attr-defined]
    return _VOCAB_MAPPING_VERSION


def classify_regime(snapshot: "IndicatorSnapshot") -> "RegimeAssessment":
    """Classify market regime from indicator snapshot.

    Classification logic based on existing _trend_from_snapshot and _vol_from_snapshot
    patterns from agents/analytics/indicator_snapshots.py.

    Regime rules:
    - bull: price > SMA, RSI > 55, MACD positive (>=2 signals agree)
    - bear: price < SMA, RSI < 45, MACD negative (>=2 signals agree)
    - range: low volatility (<1% ATR), sideways trend
    - volatile: extreme volatility (>5% ATR)
    - uncertain: conflicting signals
    """
    from schemas.llm_strategist import RegimeAssessment

    primary_signals: list[str] = []
    conflicting_signals: list[str] = []

    # Collect trend signals
    close = snapshot.close
    sma_medium = snapshot.sma_medium
    rsi = snapshot.rsi_14
    macd = snapshot.macd
    macd_signal = snapshot.macd_signal
    atr = snapshot.atr_14

    # Trend direction signals
    bullish_count = 0
    bearish_count = 0
    total_signals = 0

    # Price vs SMA signal
    if close and sma_medium:
        total_signals += 1
        if close > sma_medium:
            bullish_count += 1
            primary_signals.append("price_above_sma_medium")
        else:
            bearish_count += 1
            primary_signals.append("price_below_sma_medium")

    # RSI signal
    if rsi is not None:
        total_signals += 1
        if rsi > 55:
            bullish_count += 1
            primary_signals.append(f"rsi_bullish_{rsi:.1f}")
        elif rsi < 45:
            bearish_count += 1
            primary_signals.append(f"rsi_bearish_{rsi:.1f}")
        else:
            conflicting_signals.append(f"rsi_neutral_{rsi:.1f}")

    # MACD signal
    if macd is not None:
        total_signals += 1
        if macd > 0:
            bullish_count += 1
            primary_signals.append("macd_positive")
        else:
            bearish_count += 1
            primary_signals.append("macd_negative")

    # MACD histogram trend
    macd_hist = snapshot.macd_hist
    if macd_hist is not None:
        if macd_hist > 0:
            primary_signals.append("macd_hist_positive")
        else:
            primary_signals.append("macd_hist_negative")

    # Volatility assessment
    vol_ratio = (atr / close) if (atr and close and close > 0) else None
    is_high_vol = vol_ratio is not None and vol_ratio > 0.05
    is_low_vol = vol_ratio is not None and vol_ratio < 0.01
    is_extreme_vol = vol_ratio is not None and vol_ratio > 0.08

    if vol_ratio is not None:
        primary_signals.append(f"atr_ratio_{vol_ratio:.3f}")

    # Determine regime
    if is_extreme_vol:
        regime = "volatile"
        confidence = 0.8
        primary_signals.append("extreme_volatility")
    elif bullish_count >= 2 and bullish_count > bearish_count:
        regime = "bull"
        confidence = min(0.9, 0.5 + (bullish_count / total_signals) * 0.4) if total_signals > 0 else 0.5
        if bearish_count > 0:
            conflicting_signals.append(f"bearish_signals_{bearish_count}")
    elif bearish_count >= 2 and bearish_count > bullish_count:
        regime = "bear"
        confidence = min(0.9, 0.5 + (bearish_count / total_signals) * 0.4) if total_signals > 0 else 0.5
        if bullish_count > 0:
            conflicting_signals.append(f"bullish_signals_{bullish_count}")
    elif is_low_vol:
        regime = "range"
        confidence = 0.6
        primary_signals.append("low_volatility_range")
        if bullish_count > 0:
            conflicting_signals.append(f"bullish_signals_{bullish_count}")
        if bearish_count > 0:
            conflicting_signals.append(f"bearish_signals_{bearish_count}")
    elif is_high_vol:
        regime = "volatile"
        confidence = 0.7
        primary_signals.append("high_volatility")
    else:
        regime = "uncertain"
        confidence = 0.4
        if bullish_count > 0:
            conflicting_signals.append(f"bullish_signals_{bullish_count}")
        if bearish_count > 0:
            conflicting_signals.append(f"bearish_signals_{bearish_count}")

    return RegimeAssessment(
        regime=regime,
        confidence=confidence,
        primary_signals=primary_signals,
        conflicting_signals=conflicting_signals,
    )
