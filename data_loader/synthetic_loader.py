"""Synthetic data generation for deterministic testing.

This module provides waveform generators for testing trigger responsiveness
and execution behavior without relying on historical market data.

Key features:
- Deterministic: same parameters → identical output every time
- Controllable: test specific indicator conditions at known times
- Fast: no API calls, instant data generation
- Composable: combine simple patterns into complex scenarios
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Iterator, List, Literal

import numpy as np
import pandas as pd

from .base import MarketDataBackend


class WaveformType(str, Enum):
    """Supported waveform patterns."""

    SIN = "sin"
    COS = "cos"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    MEAN_REVERT = "mean_revert"
    VOLATILITY_BURST = "volatility_burst"
    RANGE_BOUND = "range_bound"
    COMPOSITE = "composite"


@dataclass
class WaveformParams:
    """Parameters for waveform generation."""

    waveform_type: WaveformType
    base_price: float = 50000.0
    amplitude: float = 1000.0  # Price amplitude for oscillations
    frequency: float = 2.0  # Cycles per day (for sin/cos)
    slope: float = 100.0  # Price change per day (for trends)
    noise_level: float = 0.0  # Random noise as fraction of amplitude
    phase_offset: float = 0.0  # Phase offset in radians (for sin/cos)

    # Mean reversion params
    reversion_speed: float = 0.1  # How fast price reverts (0-1)
    deviation_pct: float = 2.0  # Max deviation from mean (%)

    # Volatility burst params
    burst_start_pct: float = 0.7  # When burst starts (fraction of period)
    burst_magnitude: float = 3.0  # Multiplier for burst amplitude

    # Range bound params
    support: float | None = None
    resistance: float | None = None
    bounce_pct: float = 0.8  # How much of range to use

    # Volume params
    base_volume: float = 1000.0
    volume_multiplier: float = 1.0  # For bursts

    # Seed for reproducibility
    seed: int | None = None


@dataclass
class CompositeWaveform:
    """A composite waveform combining multiple patterns."""

    components: List[tuple[WaveformParams, float]] = field(default_factory=list)
    """List of (params, weight) tuples for combining waveforms."""

    def add(self, params: WaveformParams, weight: float = 1.0) -> "CompositeWaveform":
        """Add a component waveform with a weight."""
        self.components.append((params, weight))
        return self


class SyntheticDataBackend(MarketDataBackend):
    """Generates synthetic OHLCV data with configurable waveforms."""

    name: str = "synthetic"

    def __init__(
        self,
        params: WaveformParams | None = None,
        composite: CompositeWaveform | None = None,
    ) -> None:
        super().__init__()
        self.params = params or WaveformParams(waveform_type=WaveformType.SIN)
        self.composite = composite
        self._rng = np.random.default_rng(self.params.seed)

    def fetch_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        granularity: str,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for the given period."""

        # Parse granularity to timedelta
        freq_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
        }
        delta = freq_map.get(granularity, timedelta(hours=1))

        # Generate time index
        periods = int((end - start) / delta)
        if periods <= 0:
            periods = 1

        idx = pd.date_range(start, periods=periods, freq=delta, tz="UTC")

        # Generate prices based on waveform type
        if self.composite and self.composite.components:
            prices = self._generate_composite(periods, delta)
        else:
            prices = self._generate_waveform(periods, delta, self.params)

        # Generate OHLCV from close prices
        df = self._prices_to_ohlcv(prices, idx)

        return df

    def _generate_waveform(
        self,
        periods: int,
        delta: timedelta,
        params: WaveformParams,
    ) -> np.ndarray:
        """Generate price series for a single waveform type."""

        # Time in days for each bar
        bars_per_day = timedelta(days=1) / delta
        t = np.arange(periods) / bars_per_day

        if params.waveform_type == WaveformType.SIN:
            prices = self._sin_wave(t, params)
        elif params.waveform_type == WaveformType.COS:
            prices = self._cos_wave(t, params)
        elif params.waveform_type == WaveformType.TREND_UP:
            prices = self._trend(t, params, direction=1)
        elif params.waveform_type == WaveformType.TREND_DOWN:
            prices = self._trend(t, params, direction=-1)
        elif params.waveform_type == WaveformType.MEAN_REVERT:
            prices = self._mean_revert(t, params)
        elif params.waveform_type == WaveformType.VOLATILITY_BURST:
            prices = self._volatility_burst(t, params)
        elif params.waveform_type == WaveformType.RANGE_BOUND:
            prices = self._range_bound(t, params)
        else:
            prices = self._sin_wave(t, params)  # Default

        # Add noise if specified
        if params.noise_level > 0:
            noise = self._rng.normal(0, params.amplitude * params.noise_level, periods)
            prices = prices + noise

        return prices

    def _sin_wave(self, t: np.ndarray, params: WaveformParams) -> np.ndarray:
        """Generate sinusoidal price movement."""
        angular_freq = 2 * math.pi * params.frequency
        return params.base_price + params.amplitude * np.sin(angular_freq * t + params.phase_offset)

    def _cos_wave(self, t: np.ndarray, params: WaveformParams) -> np.ndarray:
        """Generate cosine price movement (phase-shifted sin)."""
        angular_freq = 2 * math.pi * params.frequency
        return params.base_price + params.amplitude * np.cos(angular_freq * t + params.phase_offset)

    def _trend(self, t: np.ndarray, params: WaveformParams, direction: int) -> np.ndarray:
        """Generate trending price movement."""
        return params.base_price + direction * params.slope * t

    def _mean_revert(self, t: np.ndarray, params: WaveformParams) -> np.ndarray:
        """Generate mean-reverting price movement using Ornstein-Uhlenbeck process."""
        prices = np.zeros(len(t))
        prices[0] = params.base_price

        max_dev = params.base_price * params.deviation_pct / 100

        for i in range(1, len(t)):
            dt = t[i] - t[i-1] if i > 0 else t[1] - t[0]
            # Mean reversion pull
            pull = params.reversion_speed * (params.base_price - prices[i-1])
            # Random walk component
            shock = self._rng.normal(0, max_dev * 0.1)
            prices[i] = prices[i-1] + pull * dt + shock
            # Clamp to bounds
            prices[i] = np.clip(prices[i], params.base_price - max_dev, params.base_price + max_dev)

        return prices

    def _volatility_burst(self, t: np.ndarray, params: WaveformParams) -> np.ndarray:
        """Generate price with quiet period followed by volatility explosion."""
        prices = np.zeros(len(t))
        burst_idx = int(len(t) * params.burst_start_pct)

        # Quiet period: small oscillation
        quiet_amp = params.amplitude * 0.2
        for i in range(burst_idx):
            prices[i] = params.base_price + quiet_amp * np.sin(2 * math.pi * params.frequency * t[i])

        # Burst period: large movement
        burst_amp = params.amplitude * params.burst_magnitude
        for i in range(burst_idx, len(t)):
            burst_t = t[i] - t[burst_idx]
            prices[i] = prices[burst_idx - 1] + burst_amp * np.sin(2 * math.pi * params.frequency * 2 * burst_t)

        return prices

    def _range_bound(self, t: np.ndarray, params: WaveformParams) -> np.ndarray:
        """Generate price bouncing between support and resistance."""
        support = params.support or (params.base_price - params.amplitude)
        resistance = params.resistance or (params.base_price + params.amplitude)

        range_size = resistance - support
        effective_range = range_size * params.bounce_pct
        center = (support + resistance) / 2

        # Oscillate within range
        angular_freq = 2 * math.pi * params.frequency
        raw = np.sin(angular_freq * t + params.phase_offset)

        return center + (effective_range / 2) * raw

    def _generate_composite(self, periods: int, delta: timedelta) -> np.ndarray:
        """Generate composite waveform by combining multiple patterns."""
        total_weight = sum(w for _, w in self.composite.components)
        if total_weight == 0:
            total_weight = 1.0

        prices = np.zeros(periods)

        for params, weight in self.composite.components:
            component = self._generate_waveform(periods, delta, params)
            # Normalize relative to base_price and add weighted contribution
            normalized = (component - params.base_price) * (weight / total_weight)
            prices += normalized

        # Add back base price (use first component's base)
        if self.composite.components:
            prices += self.composite.components[0][0].base_price

        return prices

    def _prices_to_ohlcv(self, prices: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
        """Convert close prices to OHLCV DataFrame with realistic OHLC spread."""

        # Generate some intra-bar variation
        spread_pct = 0.001  # 0.1% typical spread

        close = pd.Series(prices, index=idx)

        # Generate open prices (close of previous bar with small gap)
        open_prices = close.shift(1).fillna(close.iloc[0])

        # High and low with realistic spread
        high = np.maximum(open_prices, close) * (1 + spread_pct)
        low = np.minimum(open_prices, close) * (1 - spread_pct)

        # Volume based on params
        base_vol = self.params.base_volume
        volume = pd.Series(
            base_vol * self.params.volume_multiplier * (1 + 0.1 * self._rng.random(len(idx))),
            index=idx
        )

        return pd.DataFrame({
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }, index=idx)


# Convenience factory functions

def sin_wave(
    base_price: float = 50000.0,
    amplitude: float = 1000.0,
    frequency: float = 2.0,
    phase_offset: float = 0.0,
    noise_level: float = 0.0,
    seed: int | None = None,
) -> SyntheticDataBackend:
    """Create a sin wave generator."""
    return SyntheticDataBackend(WaveformParams(
        waveform_type=WaveformType.SIN,
        base_price=base_price,
        amplitude=amplitude,
        frequency=frequency,
        phase_offset=phase_offset,
        noise_level=noise_level,
        seed=seed,
    ))


def cos_wave(
    base_price: float = 50000.0,
    amplitude: float = 1000.0,
    frequency: float = 2.0,
    phase_offset: float = 0.0,
    noise_level: float = 0.0,
    seed: int | None = None,
) -> SyntheticDataBackend:
    """Create a cos wave generator."""
    return SyntheticDataBackend(WaveformParams(
        waveform_type=WaveformType.COS,
        base_price=base_price,
        amplitude=amplitude,
        frequency=frequency,
        phase_offset=phase_offset,
        noise_level=noise_level,
        seed=seed,
    ))


def trend(
    base_price: float = 50000.0,
    slope: float = 100.0,
    direction: Literal["up", "down"] = "up",
    noise_level: float = 0.0,
    seed: int | None = None,
) -> SyntheticDataBackend:
    """Create a trending price generator."""
    wtype = WaveformType.TREND_UP if direction == "up" else WaveformType.TREND_DOWN
    return SyntheticDataBackend(WaveformParams(
        waveform_type=wtype,
        base_price=base_price,
        slope=slope,
        noise_level=noise_level,
        seed=seed,
    ))


def mean_reversion(
    base_price: float = 50000.0,
    deviation_pct: float = 2.0,
    reversion_speed: float = 0.1,
    seed: int | None = None,
) -> SyntheticDataBackend:
    """Create a mean-reverting price generator."""
    return SyntheticDataBackend(WaveformParams(
        waveform_type=WaveformType.MEAN_REVERT,
        base_price=base_price,
        deviation_pct=deviation_pct,
        reversion_speed=reversion_speed,
        seed=seed,
    ))


def volatility_burst(
    base_price: float = 50000.0,
    amplitude: float = 1000.0,
    burst_start_pct: float = 0.7,
    burst_magnitude: float = 3.0,
    seed: int | None = None,
) -> SyntheticDataBackend:
    """Create a volatility burst generator (quiet then explosive)."""
    return SyntheticDataBackend(WaveformParams(
        waveform_type=WaveformType.VOLATILITY_BURST,
        base_price=base_price,
        amplitude=amplitude,
        burst_start_pct=burst_start_pct,
        burst_magnitude=burst_magnitude,
        seed=seed,
    ))


def range_bound(
    support: float = 49000.0,
    resistance: float = 51000.0,
    frequency: float = 2.0,
    bounce_pct: float = 0.8,
    seed: int | None = None,
) -> SyntheticDataBackend:
    """Create a range-bound price generator."""
    base_price = (support + resistance) / 2
    amplitude = (resistance - support) / 2
    return SyntheticDataBackend(WaveformParams(
        waveform_type=WaveformType.RANGE_BOUND,
        base_price=base_price,
        amplitude=amplitude,
        support=support,
        resistance=resistance,
        frequency=frequency,
        bounce_pct=bounce_pct,
        seed=seed,
    ))


# Waveform analysis utilities

@dataclass
class WaveformAnalysis:
    """Analysis results for a synthetic waveform."""

    waveform_type: str
    periods: int
    base_price: float
    min_price: float
    max_price: float
    amplitude_actual: float
    frequency_estimated: float | None
    zero_crossings: List[int]  # Bar indices where price crosses base
    peaks: List[int]  # Bar indices of local maxima
    troughs: List[int]  # Bar indices of local minima

    def expected_trigger_bars(
        self,
        condition: Literal["above_base", "below_base", "peak", "trough"],
    ) -> List[int]:
        """Return bar indices where a trigger condition should fire."""
        if condition == "above_base":
            # First bar after each upward zero crossing
            return [zc + 1 for i, zc in enumerate(self.zero_crossings) if i % 2 == 0]
        elif condition == "below_base":
            # First bar after each downward zero crossing
            return [zc + 1 for i, zc in enumerate(self.zero_crossings) if i % 2 == 1]
        elif condition == "peak":
            return self.peaks
        elif condition == "trough":
            return self.troughs
        return []


def analyze_waveform(df: pd.DataFrame, base_price: float) -> WaveformAnalysis:
    """Analyze a generated waveform for testing."""

    close = df["close"].values
    periods = len(close)

    # Find zero crossings (relative to base_price)
    centered = close - base_price
    zero_crossings = []
    for i in range(1, periods):
        if centered[i-1] * centered[i] < 0:
            zero_crossings.append(i)

    # Find peaks and troughs
    peaks = []
    troughs = []
    for i in range(1, periods - 1):
        if close[i] > close[i-1] and close[i] > close[i+1]:
            peaks.append(i)
        elif close[i] < close[i-1] and close[i] < close[i+1]:
            troughs.append(i)

    # Estimate frequency from zero crossings (2 crossings per cycle)
    freq_estimated = None
    if len(zero_crossings) >= 2:
        avg_half_period = np.mean(np.diff(zero_crossings))
        if avg_half_period > 0:
            # periods per full cycle, converted to cycles per day (assuming 24 bars/day for 1h)
            bars_per_cycle = avg_half_period * 2
            freq_estimated = 24 / bars_per_cycle if bars_per_cycle > 0 else None

    return WaveformAnalysis(
        waveform_type="analyzed",
        periods=periods,
        base_price=base_price,
        min_price=float(np.min(close)),
        max_price=float(np.max(close)),
        amplitude_actual=(np.max(close) - np.min(close)) / 2,
        frequency_estimated=freq_estimated,
        zero_crossings=zero_crossings,
        peaks=peaks,
        troughs=troughs,
    )


__all__ = [
    "WaveformType",
    "WaveformParams",
    "CompositeWaveform",
    "SyntheticDataBackend",
    "WaveformAnalysis",
    "sin_wave",
    "cos_wave",
    "trend",
    "mean_reversion",
    "volatility_burst",
    "range_bound",
    "analyze_waveform",
]
