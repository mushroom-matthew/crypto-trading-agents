"""Tests for synthetic waveform generation primitives.

These tests verify that waveforms are generated correctly and deterministically,
serving as the foundation for trigger responsiveness testing.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.synthetic_loader import (
    SyntheticDataBackend,
    WaveformParams,
    WaveformType,
    CompositeWaveform,
    sin_wave,
    cos_wave,
    trend,
    mean_reversion,
    volatility_burst,
    range_bound,
    analyze_waveform,
)


# ============================================================================
# Test fixtures and helpers
# ============================================================================

@pytest.fixture
def standard_period():
    """Standard 24-hour period with 1h bars."""
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    return start, end, "1h"


@pytest.fixture
def multi_day_period():
    """Multi-day period for testing longer patterns."""
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 4, 0, 0, tzinfo=timezone.utc)
    return start, end, "1h"


# ============================================================================
# SIMPLE WAVEFORM TESTS
# ============================================================================

class TestSinWave:
    """Test sinusoidal waveform generation."""

    def test_sin_wave_generates_correct_shape(self, standard_period):
        """Sin wave should oscillate around base price."""
        start, end, granularity = standard_period
        backend = sin_wave(base_price=50000, amplitude=1000, frequency=2.0, seed=42)

        df = backend.fetch_history("TEST", start, end, granularity)

        assert len(df) == 24
        assert df["close"].min() < 50000  # Goes below base
        assert df["close"].max() > 50000  # Goes above base
        # Amplitude check (approximate due to noise-free)
        assert df["close"].max() - df["close"].min() > 1500  # Should be ~2000

    def test_sin_wave_frequency_affects_range(self, multi_day_period):
        """Higher frequency should complete more cycles over longer periods."""
        start, end, granularity = multi_day_period
        base = 50000

        # 0.5 cycles per day (1 cycle over 2 days)
        slow = sin_wave(base_price=base, frequency=0.5, seed=42)
        df_slow = slow.fetch_history("TEST", start, end, granularity)
        analysis_slow = analyze_waveform(df_slow, base)

        # 2 cycles per day (6 cycles over 3 days)
        fast = sin_wave(base_price=base, frequency=2.0, seed=42)
        df_fast = fast.fetch_history("TEST", start, end, granularity)
        analysis_fast = analyze_waveform(df_fast, base)

        # Fast should have more peaks and troughs
        slow_extrema = len(analysis_slow.peaks) + len(analysis_slow.troughs)
        fast_extrema = len(analysis_fast.peaks) + len(analysis_fast.troughs)
        assert fast_extrema > slow_extrema, f"Expected fast ({fast_extrema}) > slow ({slow_extrema})"

    def test_sin_wave_deterministic(self, standard_period):
        """Same seed should produce identical results."""
        start, end, granularity = standard_period

        backend1 = sin_wave(seed=123)
        backend2 = sin_wave(seed=123)

        df1 = backend1.fetch_history("TEST", start, end, granularity)
        df2 = backend2.fetch_history("TEST", start, end, granularity)

        np.testing.assert_array_almost_equal(df1["close"].values, df2["close"].values)

    def test_sin_wave_phase_offset(self, standard_period):
        """Phase offset should shift the wave."""
        start, end, granularity = standard_period
        import math

        sin_0 = sin_wave(phase_offset=0, seed=42)
        sin_90 = sin_wave(phase_offset=math.pi/2, seed=42)  # 90 degrees

        df_0 = sin_0.fetch_history("TEST", start, end, granularity)
        df_90 = sin_90.fetch_history("TEST", start, end, granularity)

        # At t=0, sin(0)=0 but sin(pi/2)=1, so cos starts at max
        # The first close values should be different
        assert abs(df_0["close"].iloc[0] - df_90["close"].iloc[0]) > 500


class TestCosWave:
    """Test cosine waveform generation."""

    def test_cos_wave_starts_at_peak(self, standard_period):
        """Cos wave should start at maximum (base + amplitude)."""
        start, end, granularity = standard_period
        base = 50000
        amp = 1000

        backend = cos_wave(base_price=base, amplitude=amp, frequency=1.0)
        df = backend.fetch_history("TEST", start, end, granularity)

        # First close should be near base + amplitude
        first_close = df["close"].iloc[0]
        assert abs(first_close - (base + amp)) < 100  # Small tolerance


class TestTrendWave:
    """Test trending price generation."""

    def test_trend_up_increases_monotonically(self, multi_day_period):
        """Uptrend should have increasing prices."""
        start, end, granularity = multi_day_period

        backend = trend(base_price=50000, slope=1000, direction="up", seed=42)
        df = backend.fetch_history("TEST", start, end, granularity)

        # With no noise, should be monotonically increasing
        diffs = np.diff(df["close"].values)
        assert np.all(diffs >= 0), "Uptrend should not decrease"

    def test_trend_down_decreases_monotonically(self, multi_day_period):
        """Downtrend should have decreasing prices."""
        start, end, granularity = multi_day_period

        backend = trend(base_price=50000, slope=1000, direction="down", seed=42)
        df = backend.fetch_history("TEST", start, end, granularity)

        # With no noise, should be monotonically decreasing
        diffs = np.diff(df["close"].values)
        assert np.all(diffs <= 0), "Downtrend should not increase"

    def test_trend_slope_affects_rate(self, standard_period):
        """Higher slope should produce faster price change."""
        start, end, granularity = standard_period

        slow = trend(base_price=50000, slope=100)
        fast = trend(base_price=50000, slope=500)

        df_slow = slow.fetch_history("TEST", start, end, granularity)
        df_fast = fast.fetch_history("TEST", start, end, granularity)

        range_slow = df_slow["close"].max() - df_slow["close"].min()
        range_fast = df_fast["close"].max() - df_fast["close"].min()

        assert range_fast > range_slow * 2  # Fast should have much larger range


class TestMeanReversion:
    """Test mean-reverting price generation."""

    def test_mean_reversion_stays_bounded(self, multi_day_period):
        """Mean reversion should stay within deviation bounds."""
        start, end, granularity = multi_day_period
        base = 50000
        deviation_pct = 2.0
        max_deviation = base * deviation_pct / 100

        backend = mean_reversion(
            base_price=base,
            deviation_pct=deviation_pct,
            reversion_speed=0.2,
            seed=42
        )
        df = backend.fetch_history("TEST", start, end, granularity)

        assert df["close"].min() >= base - max_deviation - 1  # Small tolerance
        assert df["close"].max() <= base + max_deviation + 1

    def test_mean_reversion_reverts_to_mean(self, multi_day_period):
        """Mean reversion should tend toward base price."""
        start, end, granularity = multi_day_period
        base = 50000

        backend = mean_reversion(
            base_price=base,
            deviation_pct=2.0,
            reversion_speed=0.5,  # Strong reversion
            seed=42
        )
        df = backend.fetch_history("TEST", start, end, granularity)

        # Average should be close to base price
        avg = df["close"].mean()
        assert abs(avg - base) < base * 0.01  # Within 1% of base


class TestVolatilityBurst:
    """Test volatility burst pattern generation."""

    def test_burst_has_quiet_then_explosive(self, standard_period):
        """Volatility should be low early, high late."""
        start, end, granularity = standard_period

        backend = volatility_burst(
            base_price=50000,
            amplitude=1000,
            burst_start_pct=0.5,  # Burst starts halfway
            burst_magnitude=3.0,
            seed=42
        )
        df = backend.fetch_history("TEST", start, end, granularity)

        # Split into early and late halves
        mid = len(df) // 2
        early = df["close"].iloc[:mid]
        late = df["close"].iloc[mid:]

        early_range = early.max() - early.min()
        late_range = late.max() - late.min()

        # Late period should have larger range due to burst
        assert late_range > early_range


class TestRangeBound:
    """Test range-bound price generation."""

    def test_range_bound_stays_in_range(self, standard_period):
        """Price should stay within support/resistance."""
        start, end, granularity = standard_period
        support = 49000
        resistance = 51000

        backend = range_bound(
            support=support,
            resistance=resistance,
            bounce_pct=1.0,  # Use full range
            seed=42
        )
        df = backend.fetch_history("TEST", start, end, granularity)

        # Allow small tolerance for OHLC spread
        assert df["close"].min() >= support - 100
        assert df["close"].max() <= resistance + 100


# ============================================================================
# COMPOSITE WAVEFORM TESTS
# ============================================================================

class TestCompositeWaveform:
    """Test composite (combined) waveform generation."""

    def test_composite_trend_with_oscillation(self, multi_day_period):
        """Composite should combine trend with oscillation."""
        start, end, granularity = multi_day_period

        composite = CompositeWaveform()
        # Uptrend component
        composite.add(WaveformParams(
            waveform_type=WaveformType.TREND_UP,
            base_price=50000,
            slope=500,
        ), weight=1.0)
        # Oscillation component
        composite.add(WaveformParams(
            waveform_type=WaveformType.SIN,
            base_price=50000,
            amplitude=200,
            frequency=2.0,
        ), weight=0.5)

        backend = SyntheticDataBackend(composite=composite)
        df = backend.fetch_history("TEST", start, end, granularity)

        # Should have overall uptrend
        assert df["close"].iloc[-1] > df["close"].iloc[0]

        # But with oscillations (not monotonic)
        diffs = np.diff(df["close"].values)
        assert np.any(diffs < 0), "Should have some decreases from oscillation"


# ============================================================================
# WAVEFORM ANALYSIS TESTS
# ============================================================================

class TestWaveformAnalysis:
    """Test waveform analysis utilities."""

    def test_analyze_finds_extrema(self, standard_period):
        """Analysis should correctly identify peaks and troughs."""
        start, end, granularity = standard_period
        base = 50000

        # 2 cycles = 2 peaks and 2 troughs
        backend = sin_wave(base_price=base, frequency=2.0, seed=42)
        df = backend.fetch_history("TEST", start, end, granularity)
        analysis = analyze_waveform(df, base)

        # Should find approximately 2 peaks and 2 troughs for 2 cycles
        total_extrema = len(analysis.peaks) + len(analysis.troughs)
        assert 3 <= total_extrema <= 6

    def test_analyze_finds_peaks_and_troughs(self, standard_period):
        """Analysis should identify peaks and troughs."""
        start, end, granularity = standard_period
        base = 50000

        # 2 cycles = 2 peaks and 2 troughs
        backend = sin_wave(base_price=base, frequency=2.0, seed=42)
        df = backend.fetch_history("TEST", start, end, granularity)
        analysis = analyze_waveform(df, base)

        assert len(analysis.peaks) >= 1
        assert len(analysis.troughs) >= 1

    def test_analyze_estimates_frequency(self, standard_period):
        """Analysis should estimate frequency from zero crossings."""
        start, end, granularity = standard_period
        base = 50000
        target_freq = 3.0

        backend = sin_wave(base_price=base, frequency=target_freq, seed=42)
        df = backend.fetch_history("TEST", start, end, granularity)
        analysis = analyze_waveform(df, base)

        if analysis.frequency_estimated is not None:
            # Should be within 50% of target
            assert 0.5 * target_freq < analysis.frequency_estimated < 1.5 * target_freq


# ============================================================================
# DETERMINISM AND REPRODUCIBILITY TESTS
# ============================================================================

class TestDeterminism:
    """Test that waveforms are fully deterministic."""

    def test_all_waveform_types_deterministic(self, standard_period):
        """All waveform types should be deterministic with same seed."""
        start, end, granularity = standard_period
        seed = 999

        waveform_factories = [
            lambda: sin_wave(seed=seed),
            lambda: cos_wave(seed=seed),
            lambda: trend(direction="up", seed=seed),
            lambda: mean_reversion(seed=seed),
            lambda: volatility_burst(seed=seed),
            lambda: range_bound(seed=seed),
        ]

        for factory in waveform_factories:
            backend1 = factory()
            backend2 = factory()

            df1 = backend1.fetch_history("TEST", start, end, granularity)
            df2 = backend2.fetch_history("TEST", start, end, granularity)

            np.testing.assert_array_almost_equal(
                df1["close"].values,
                df2["close"].values,
                err_msg=f"Waveform not deterministic"
            )

    def test_different_seeds_produce_different_results(self, standard_period):
        """Different seeds should produce different noise."""
        start, end, granularity = standard_period

        # Use noise to see seed effect
        backend1 = sin_wave(noise_level=0.1, seed=1)
        backend2 = sin_wave(noise_level=0.1, seed=2)

        df1 = backend1.fetch_history("TEST", start, end, granularity)
        df2 = backend2.fetch_history("TEST", start, end, granularity)

        # Should NOT be identical
        assert not np.allclose(df1["close"].values, df2["close"].values)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_bar_generation(self):
        """Should handle single bar periods."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc)

        backend = sin_wave()
        df = backend.fetch_history("TEST", start, end, "1m")

        assert len(df) >= 1

    def test_very_high_frequency(self, standard_period):
        """Should handle high frequency oscillations."""
        start, end, granularity = standard_period

        backend = sin_wave(frequency=12.0, seed=42)  # 12 cycles per day
        df = backend.fetch_history("TEST", start, end, granularity)

        # Should still generate valid data
        assert len(df) == 24
        assert not df["close"].isna().any()

    def test_zero_amplitude(self, standard_period):
        """Zero amplitude should produce flat line at base price."""
        start, end, granularity = standard_period
        base = 50000

        backend = sin_wave(base_price=base, amplitude=0)
        df = backend.fetch_history("TEST", start, end, granularity)

        # All closes should equal base price
        np.testing.assert_array_almost_equal(
            df["close"].values,
            np.full(len(df), base),
            decimal=1
        )
