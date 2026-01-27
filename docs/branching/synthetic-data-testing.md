# Branch: synthetic-data-testing

## Purpose
Add deterministic synthetic data generation (sin/cos waves, predictable patterns) for testing trigger responsiveness and execution behavior without relying on historical market data.

## Problem Statement
Testing with real market data has limitations:
- Non-deterministic: results vary based on market conditions
- Hard to test edge cases: specific indicator values, rapid regime changes
- Slow iteration: need to run full backtests to validate trigger logic
- Difficult to test opposing signals and conflict resolution

Synthetic waveforms provide:
- Deterministic: same output every time for reproducible tests
- Controllable: test specific trigger conditions (RSI oversold at exact time)
- Fast: no API calls, instant data generation
- Edge cases: gaps, spikes, regime transitions

## Existing Foundation
Investigation (2026-01-27) found **existing synthetic data capability**:

```python
# tests/integration/test_high_budget_activity.py lines 22-29
def _wiggle(start: datetime, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    base = 1000 + 10 * np.sin(np.linspace(0, 4 * np.pi, periods))
    noise = np.linspace(0, 5, periods)
    prices = pd.Series(base - noise, index=idx)
    return pd.DataFrame({"open": prices, "high": prices + 2, "low": prices - 2, "close": prices, "volume": 1})
```

## Scope
- Create `data_loader/synthetic_loader.py` implementing `MarketDataBackend`
- Add CLI flag `--synthetic-mode` with pattern parameters
- Implement pattern library: sin/cos, trend, mean-reversion, volatility regimes
- Add trigger responsiveness test harness with known-outcome scenarios

## Pattern Library (Proposed)
| Pattern | Use Case | Parameters |
|---------|----------|------------|
| `sin` | Oscillating mean-reversion | frequency, amplitude, base_price |
| `cos` | Phase-shifted oscillation | frequency, amplitude, base_price |
| `trend_up` | Consistent uptrend | slope, noise_level |
| `trend_down` | Consistent downtrend | slope, noise_level |
| `mean_revert` | Price around moving average | deviation_pct, reversion_speed |
| `volatility_burst` | Quiet → explosive | burst_time, spike_magnitude |
| `range_bound` | Support/resistance levels | support, resistance, bounce_pct |

## Key Files
- data_loader/synthetic_loader.py (NEW)
- data_loader/base.py (MarketDataBackend interface)
- backtesting/dataset.py (routing)
- backtesting/cli.py (CLI flags)
- tests/synthetic_scenarios.py (NEW - test harness)

## Architecture
```
backtesting/cli.py
  --synthetic-mode sin --base-price 50000 --amplitude 5 --frequency 2
         ↓
backtesting/dataset.py: load_ohlcv()
         ↓
data_loader/synthetic_loader.py: SyntheticDataBackend
         ↓
Generate OHLCV DataFrame with sin wave pattern
         ↓
backtesting/simulator.py: _compute_features()
         ↓
Strategy evaluation (same as real data path)
```

## Test Scenarios (Proposed)
1. **Trigger Fire Timing**: Price crosses Donchian band at known bar → trigger fires
2. **Opposing Signals**: Both RSI oversold and overbought conditions → correct priority
3. **Min-Hold Enforcement**: Quick price reversal → position held minimum bars
4. **Volume Burst Detection**: Synthetic volume spike → vol_burst indicator fires
5. **Regime Detection**: Trend → range → trend transition → correct regime classification

## Acceptance Criteria
- `uv run backtest --synthetic-mode sin` generates valid OHLCV and runs
- Sin wave with 2 cycles/day produces predictable indicator values
- Test harness validates trigger fires at expected bars
- Deterministic: same parameters → identical results

## Dependencies / Coordination
- No conflicts expected with other branches
- Extends existing data_loader infrastructure

## Complexity Assessment
- **Low**: Basic sin/cos generator with CLI flag (2-4 hours)
- **Medium**: Full pattern library with test harness (4-8 hours)
- **High**: Known-outcome scenario automation (8+ hours)

## Test Plan (required before commit)
- uv run pytest -k synthetic -vv
- uv run backtest --synthetic-mode sin --help (verify CLI)
- uv run python -c "from data_loader.synthetic_loader import SyntheticDataBackend"

## Human Verification (required)
- Run backtest with synthetic sin wave
- Verify indicators compute correctly (no NaN, reasonable values)
- Verify triggers fire at predictable times
- Paste run output below

## Git Workflow
```bash
git checkout main && git pull
git checkout -b synthetic-data-testing

# After implementation
git add data_loader/synthetic_loader.py \
  backtesting/cli.py \
  backtesting/dataset.py \
  tests/synthetic_scenarios.py

uv run pytest -k synthetic -vv
git commit -m "Testing: add synthetic data generation for trigger testing"
```

## Change Log (update during implementation)

### 2026-01-27: Implementation Complete

1. **Synthetic Data Generator** (`data_loader/synthetic_loader.py`)
   - `SyntheticDataBackend` implementing `MarketDataBackend` interface
   - `WaveformParams` dataclass for configuring waveforms
   - `CompositeWaveform` for combining multiple patterns
   - 7 waveform types: sin, cos, trend_up, trend_down, mean_revert, volatility_burst, range_bound
   - Factory functions: `sin_wave()`, `cos_wave()`, `trend()`, `mean_reversion()`, `volatility_burst()`, `range_bound()`
   - `analyze_waveform()` utility for extracting peaks, troughs, and frequency estimates
   - Full determinism with seed support

2. **Waveform Primitive Tests** (`tests/synthetic/test_waveforms.py`)
   - 21 tests covering all waveform types
   - Tests for: shape, frequency, determinism, phase offset, trends, mean reversion, volatility burst, range bound
   - Composite waveform tests
   - Edge case tests (single bar, high frequency, zero amplitude)

3. **Trigger Responsiveness Tests** (`tests/synthetic/test_trigger_responsiveness.py`)
   - 9 tests for parametric trigger validation
   - `TestPriceThresholdTriggers`: Above/below threshold fires
   - `TestTrendTriggers`: Uptrend breakout timing
   - `TestRangeBoundTriggers`: Support/resistance bounces
   - `TestParametricWaveformTriggers`: Known peak/trough timing, frequency-based trigger count
   - `TestCompoundTriggers`: Dual threshold validation
   - `generate_trigger_report()`: Human-readable test report

### Files Created
- `data_loader/synthetic_loader.py` - Core synthetic data generator
- `tests/synthetic/__init__.py` - Test package
- `tests/synthetic/test_waveforms.py` - 21 waveform tests
- `tests/synthetic/test_trigger_responsiveness.py` - 9 trigger tests + report generator

### Deferred for Later
- CLI flag `--synthetic-mode` (requires dataset.py changes)
- Integration with backtesting/cli.py

## Test Evidence (append results before commit)

```
$ uv run pytest tests/synthetic/ -vv
============================= test session starts ==============================
collected 30 items

tests/synthetic/test_trigger_responsiveness.py::TestPriceThresholdTriggers::test_price_above_threshold_fires PASSED
tests/synthetic/test_trigger_responsiveness.py::TestPriceThresholdTriggers::test_price_below_threshold_fires PASSED
tests/synthetic/test_trigger_responsiveness.py::TestTrendTriggers::test_uptrend_breakout_trigger PASSED
tests/synthetic/test_trigger_responsiveness.py::TestRangeBoundTriggers::test_range_bounce_entry_and_exit PASSED
tests/synthetic/test_trigger_responsiveness.py::TestParametricWaveformTriggers::test_known_peak_timing PASSED
tests/synthetic/test_trigger_responsiveness.py::TestParametricWaveformTriggers::test_known_trough_timing PASSED
tests/synthetic/test_trigger_responsiveness.py::TestParametricWaveformTriggers::test_frequency_determines_trigger_count PASSED
tests/synthetic/test_trigger_responsiveness.py::TestCompoundTriggers::test_dual_threshold_trigger PASSED
tests/synthetic/test_trigger_responsiveness.py::TestTriggerReport::test_generate_trigger_report PASSED
tests/synthetic/test_waveforms.py::TestSinWave::test_sin_wave_generates_correct_shape PASSED
tests/synthetic/test_waveforms.py::TestSinWave::test_sin_wave_frequency_affects_range PASSED
tests/synthetic/test_waveforms.py::TestSinWave::test_sin_wave_deterministic PASSED
tests/synthetic/test_waveforms.py::TestSinWave::test_sin_wave_phase_offset PASSED
tests/synthetic/test_waveforms.py::TestCosWave::test_cos_wave_starts_at_peak PASSED
tests/synthetic/test_waveforms.py::TestTrendWave::test_trend_up_increases_monotonically PASSED
tests/synthetic/test_waveforms.py::TestTrendWave::test_trend_down_decreases_monotonically PASSED
tests/synthetic/test_waveforms.py::TestTrendWave::test_trend_slope_affects_rate PASSED
tests/synthetic/test_waveforms.py::TestMeanReversion::test_mean_reversion_stays_bounded PASSED
tests/synthetic/test_waveforms.py::TestMeanReversion::test_mean_reversion_reverts_to_mean PASSED
tests/synthetic/test_waveforms.py::TestVolatilityBurst::test_burst_has_quiet_then_explosive PASSED
tests/synthetic/test_waveforms.py::TestRangeBound::test_range_bound_stays_in_range PASSED
tests/synthetic/test_waveforms.py::TestCompositeWaveform::test_composite_trend_with_oscillation PASSED
tests/synthetic/test_waveforms.py::TestWaveformAnalysis::test_analyze_finds_extrema PASSED
tests/synthetic/test_waveforms.py::TestWaveformAnalysis::test_analyze_finds_peaks_and_troughs PASSED
tests/synthetic/test_waveforms.py::TestWaveformAnalysis::test_analyze_estimates_frequency PASSED
tests/synthetic/test_waveforms.py::TestDeterminism::test_all_waveform_types_deterministic PASSED
tests/synthetic/test_waveforms.py::TestDeterminism::test_different_seeds_produce_different_results PASSED
tests/synthetic/test_waveforms.py::TestEdgeCases::test_single_bar_generation PASSED
tests/synthetic/test_waveforms.py::TestEdgeCases::test_very_high_frequency PASSED
tests/synthetic/test_waveforms.py::TestEdgeCases::test_zero_amplitude PASSED

============================== 30 passed in 3.52s ==============================
```

## Human Verification Evidence (append results before commit when required)

Human verification deferred - CLI integration not yet implemented.
Core synthetic data generator and tests are complete with 30 passing tests.
