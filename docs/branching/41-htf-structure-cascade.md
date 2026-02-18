# Branch: htf-structure-cascade

## Purpose
Human traders never set stops and targets in a vacuum — they anchor them to visible structure: yesterday's high and low, the weekly range, prior session pivots. The current system computes support/resistance from swing points on the *trading timeframe*, which gives fine-grained levels but misses the structural anchors that matter most to the market. A 1h trader who ignores the daily high/low is flying blind.

This runbook adds a structural "anchor layer" — always loading a daily (and weekly) candle alongside the trading timeframe, regardless of what timeframe the strategy uses. The resulting `daily_*` and `weekly_*` fields in the feature vector let the LLM and trigger rules anchor stops below yesterday's low, target the prior day's high, and gate entries based on position within the daily range.

This is a data pipeline change, not a strategy change. It makes structural anchors available; the Compression Breakout template (Runbook 40) and Level-Anchored Stops (Runbook 42) consume them.

## Scope
1. **`backtesting/dataset.py`** — always load daily bars alongside the base timeframe
2. **`agents/analytics/indicator_snapshots.py`** — compute HTF structural fields and add to snapshot
3. **`schemas/llm_strategist.py`** — add `htf_*` fields to `IndicatorSnapshot`
4. **`backtesting/llm_strategist_runner.py`** — pass daily context to per-bar indicator computation
5. **`prompts/strategy_plan_schema.txt`** — document HTF identifiers and stop/target anchoring guidance
6. **`prompts/llm_strategist_prompt.txt`** — add guidance on using HTF levels for stops and targets
7. **`tests/test_htf_structure_cascade.py`** — unit tests for HTF field computation

## Out of Scope
- Weekly timeframe in backtesting (insufficient history for most pairs — add later)
- Intraday sub-timeframe anchor layers (e.g., 4h anchor when trading 1h)
- S/R level detection on the daily timeframe beyond prior-session high/low/open (deferred)
- Multi-day rolling high/low windows beyond 5-day lookback

## Key Files
- `backtesting/dataset.py`
- `agents/analytics/indicator_snapshots.py`
- `schemas/llm_strategist.py`
- `backtesting/llm_strategist_runner.py`
- `prompts/strategy_plan_schema.txt`
- `prompts/llm_strategist_prompt.txt`
- `tests/test_htf_structure_cascade.py` (new)

## Implementation Steps

### Step 1: Define HTF fields in `schemas/llm_strategist.py`

Add after candlestick fields (Runbook 38) or after Fibonacci fields:
```python
# Higher-timeframe structural anchor layer (always daily bars)
htf_daily_open: float | None = None         # Current session's daily open
htf_daily_high: float | None = None         # Current session's daily high (rolling intraday)
htf_daily_low: float | None = None          # Current session's daily low (rolling intraday)
htf_daily_close: float | None = None        # Prior completed daily close
htf_prev_daily_high: float | None = None    # Prior completed session's high
htf_prev_daily_low: float | None = None     # Prior completed session's low
htf_prev_daily_open: float | None = None    # Prior completed session's open
htf_daily_atr: float | None = None          # ATR(14) on daily bars (volatility normalizer)
htf_daily_range_pct: float | None = None    # (daily_high - daily_low) / daily_close (session range %)
htf_price_vs_daily_mid: float | None = None # (close - daily_mid) / daily_atr (position in day, ATR-normalized)
htf_5d_high: float | None = None            # Rolling 5-session high (weekly proxy)
htf_5d_low: float | None = None             # Rolling 5-session low (weekly proxy)
htf_prev_daily_mid: float | None = None     # (prev_daily_high + prev_daily_low) / 2 — regime filter pivot
```

### Step 2: Modify `backtesting/dataset.py` to load daily bars

When loading OHLCV for a non-daily timeframe, also load daily bars for the same date range:

```python
def load_with_htf(
    pair: str,
    start: datetime,
    end: datetime,
    base_timeframe: str,
    use_cache: bool = True,
    backend: MarketDataBackend | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load base timeframe OHLCV and daily OHLCV.

    Returns:
        (base_df, daily_df) where daily_df has the same date range.
        If base_timeframe is already '1d', daily_df == base_df.
    """
    base_df = load_ohlcv(pair, start, end, base_timeframe, use_cache, backend)
    if base_timeframe == "1d":
        return base_df, base_df
    # Add a 14-day buffer for daily ATR warmup
    daily_start = start - timedelta(days=20)
    daily_df = load_ohlcv(pair, daily_start, end, "1d", use_cache, backend)
    return base_df, daily_df
```

### Step 3: Compute HTF fields in `agents/analytics/indicator_snapshots.py`

Add a new function:
```python
def compute_htf_structural_fields(
    bar_timestamp: datetime,
    daily_df: pd.DataFrame,
) -> dict:
    """Compute daily structural anchor fields for a given bar timestamp.

    Finds the current and prior completed daily sessions relative to bar_timestamp.
    """
    if daily_df is None or daily_df.empty:
        return {}

    # Completed daily bars (strictly before current bar's date)
    bar_date = bar_timestamp.date()
    completed = daily_df[daily_df.index.date < bar_date]

    if len(completed) < 2:
        return {}

    # Current session (most recent completed daily bar = "yesterday")
    prev = completed.iloc[-1]
    # Session before that
    prev2 = completed.iloc[-2] if len(completed) >= 2 else prev

    # Daily ATR (14)
    atr_result = tech.atr(completed, 14)
    daily_atr = float(atr_result.series_list[0].series.iloc[-1])

    # 5-session rolling high/low (weekly proxy)
    last_5 = completed.tail(5)
    five_day_high = float(last_5["high"].max())
    five_day_low = float(last_5["low"].min())

    # Position within today's range (intraday context) requires knowing
    # today's current high/low from the base timeframe — passed in as optional
    daily_high = float(prev["high"])
    daily_low = float(prev["low"])
    daily_mid = (daily_high + daily_low) / 2.0

    # We don't have the current intraday bar here, so htf_price_vs_daily_mid
    # is computed in the caller with the live close price

    return {
        "htf_daily_open": float(prev["open"]),
        "htf_daily_high": daily_high,
        "htf_daily_low": daily_low,
        "htf_daily_close": float(prev["close"]),
        "htf_prev_daily_high": float(prev2["high"]),
        "htf_prev_daily_low": float(prev2["low"]),
        "htf_prev_daily_open": float(prev2["open"]),
        "htf_daily_atr": daily_atr,
        "htf_daily_range_pct": (daily_high - daily_low) / max(float(prev["close"]), 1e-9) * 100,
        "htf_5d_high": five_day_high,
        "htf_5d_low": five_day_low,
        "htf_prev_daily_mid": (float(prev2["high"]) + float(prev2["low"])) / 2.0,
    }
    # NOTE: these are *prior session* anchors, not rolling intraday values.
    # htf_daily_high is yesterday's completed high, not today's current high.
    # This is intentional: discretionary traders primarily reference the prior
    # completed session's range, which is static and universally watched.
```

Then in `compute_indicator_snapshot()`, after computing the base indicators, merge HTF fields:
```python
htf_fields = compute_htf_structural_fields(bar_timestamp, daily_df)
if htf_fields:
    close = snapshot_dict["close"]
    daily_atr = htf_fields.get("htf_daily_atr", 1.0)
    daily_high = htf_fields.get("htf_daily_high", close)
    daily_low = htf_fields.get("htf_daily_low", close)
    daily_mid = (daily_high + daily_low) / 2.0
    htf_fields["htf_price_vs_daily_mid"] = (close - daily_mid) / max(daily_atr, 1e-9)
    snapshot_dict.update(htf_fields)
```

### Step 4: Pass `daily_df` through `llm_strategist_runner.py`

The backtest runner loads candles per bar. Update it to:
1. Load `daily_df` once at the start alongside the base timeframe data
2. Pass `daily_df` into each call to `compute_indicator_snapshot()`

```python
# At initialization:
base_df, daily_df = load_with_htf(pair, start, end, timeframe)

# In the per-bar loop, pass daily_df to indicator computation:
snapshot = compute_indicator_snapshot(
    df=bar_slice,
    config=indicator_config,
    daily_df=daily_df,  # new parameter
)
```

### Step 5: Update prompt guidance in `prompts/strategy_plan_schema.txt`

Add a new section:
```
HIGHER-TIMEFRAME STRUCTURAL ANCHOR IDENTIFIERS (daily bars):
  htf_daily_high          — Prior completed session's high (key resistance)
  htf_daily_low           — Prior completed session's low (key support)
  htf_daily_open          — Prior session's open (intraday pivot reference)
  htf_daily_close         — Prior session's close (momentum reference)
  htf_prev_daily_high     — Session before prior's high (2-day lookback high)
  htf_prev_daily_low      — Session before prior's low (2-day lookback low)
  htf_daily_atr           — Daily ATR(14) (volatility normalizer for HTF)
  htf_daily_range_pct     — Prior session's range as % of close
  htf_price_vs_daily_mid  — Current price relative to prior daily midpoint, ATR-normalized
                            >0 = above midpoint, <0 = below, |value|>1 = far from mid
  htf_5d_high             — Rolling 5-session high (weekly proxy resistance)
  htf_5d_low              — Rolling 5-session low (weekly proxy support)
  htf_prev_daily_mid      — Midpoint of the session-before-prior's range: (prev2_high + prev2_low) / 2
                            Use as a directional regime filter: is price above or below 2-session mid?

HTF STOP AND TARGET ANCHORING (required when HTF data is available):
- Long stop below daily structure: "close < htf_daily_low * 0.995"
  (0.5% buffer below yesterday's low — prevents stop hunts from noise)
- Short stop above daily structure: "close > htf_daily_high * 1.005"
- Target at prior high: "close > htf_daily_high * 0.998"
  (approach target as exit rule, not exceed — capture before resistance)
- Weekly proxy target: "close > htf_5d_high * 0.995" for breakout targets
- Position context: htf_price_vs_daily_mid > 1.0 means far above daily midpoint
  (overbought relative to session structure — prefer exits over entries)
  htf_price_vs_daily_mid < -1.0 means far below midpoint
  (oversold relative to session structure — prefer entries over exits for longs)

STOP ANCHORING PRINCIPLE:
  Never use stop_loss_pct alone when HTF levels are available.
  Prefer: "close < htf_daily_low * 0.995" over a fixed percentage stop.
  The level-based stop respects market structure; percentage stops are arbitrary.
```

### Step 6: Update `prompts/llm_strategist_prompt.txt`

Add to the "Optional market-structure telemetry" section:
```
Higher-timeframe structural anchors (htf_* fields):
- htf_daily_high and htf_daily_low are the prior completed session's range — the most
  watched levels by institutional traders. Use these as primary stop and target references.
- For longs: prefer stops at htf_daily_low * 0.995; targets at htf_daily_high * 0.998.
- For shorts: prefer stops at htf_daily_high * 1.005; targets at htf_daily_low * 1.002.
- htf_price_vs_daily_mid > 1.0 signals the price is far above the daily midpoint —
  reduce long confidence or prefer risk_reduce triggers.
- htf_5d_high and htf_5d_low are the 5-session range — major weekly pivots.
  Use as targets for high-confidence breakout setups.
```

## Test Plan
```bash
# Unit: HTF field computation with known daily data
uv run pytest tests/test_htf_structure_cascade.py -vv

# Integration: ensure IndicatorSnapshot with HTF fields serializes correctly
uv run pytest -k "indicator_snapshot" -vv

# Integration: backtest populates htf_* fields (non-null) in daily report
uv run python -m backtesting.cli \
  --pair BTC-USD \
  --start 2024-01-01 --end 2024-02-01 \
  --timeframes 1h \
  --llm-strategist enabled \
  --llm-calls-per-day 1 \
  2>&1 | grep "htf_daily_high"
```

## Test Evidence
```
uv run pytest tests/test_htf_structure_cascade.py -vv
17 passed in 3.2s

tests/test_htf_structure_cascade.py::test_returns_empty_when_daily_df_is_none PASSED
tests/test_htf_structure_cascade.py::test_returns_empty_when_daily_df_is_empty PASSED
tests/test_htf_structure_cascade.py::test_returns_empty_when_fewer_than_2_completed_sessions PASSED
tests/test_htf_structure_cascade.py::test_returns_fields_when_sufficient_history PASSED
tests/test_htf_structure_cascade.py::test_prior_session_high_is_yesterday PASSED
tests/test_htf_structure_cascade.py::test_prior_session_low_is_yesterday PASSED
tests/test_htf_structure_cascade.py::test_prev2_session_high_is_two_days_ago PASSED
tests/test_htf_structure_cascade.py::test_five_day_high_is_max_over_five_sessions PASSED
tests/test_htf_structure_cascade.py::test_five_day_low_is_min_over_five_sessions PASSED
tests/test_htf_structure_cascade.py::test_daily_range_pct_positive PASSED
tests/test_htf_structure_cascade.py::test_prev_daily_mid_is_average_of_prev2_high_low PASSED
tests/test_htf_structure_cascade.py::test_daily_atr_is_positive PASSED
tests/test_htf_structure_cascade.py::test_all_expected_keys_present PASSED
tests/test_htf_structure_cascade.py::test_indicator_snapshot_accepts_htf_fields PASSED
tests/test_htf_structure_cascade.py::test_indicator_snapshot_htf_fields_default_none PASSED
tests/test_htf_structure_cascade.py::test_indicator_snapshot_accepts_candlestick_fields PASSED
tests/test_htf_structure_cascade.py::test_indicator_snapshot_candlestick_fields_default_none PASSED
```

## Acceptance Criteria
- [x] `compute_htf_structural_fields()` returns correct prior-session high/low/open/close
- [x] `htf_price_vs_daily_mid` computed in both `compute_indicator_snapshot()` and `_indicator_snapshot()` runner path
- [x] `htf_5d_high` / `htf_5d_low` reflect the 5-session rolling extremes
- [x] `_load_data()` in runner populates `self.daily_data` per pair via `load_with_htf()`
- [x] `_indicator_snapshot()` applies HTF fields via `model_copy(update=htf)` on snapshot_from_frame path
- [x] Existing tests not broken (new fields are optional — `None` by default, 649 passing)
- [x] LLM prompt guidance includes concrete stop/target anchoring examples using `htf_*` fields
- [ ] Backtest populates `htf_*` fields in bar snapshots (non-null after warmup) — needs live backtest validation

## Human Verification Evidence
```
TODO: Run a 2-week backtest on BTC-USD 1h timeframe.
Inspect a mid-week bar's indicator snapshot JSON.
Verify htf_daily_high equals the prior day's high on an external source (Coinbase chart).
Confirm htf_5d_high is the max high over the prior 5 calendar days.
```

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-18 | Runbook created from product strategy audit | Claude |

## Worktree Setup
```bash
git fetch
git worktree add -b feat/htf-structure-cascade ../wt-htf-structure-cascade main
cd ../wt-htf-structure-cascade

# When finished (after merge)
git worktree remove ../wt-htf-structure-cascade
```

## Git Workflow
```bash
git checkout main
git pull
git checkout -b feat/htf-structure-cascade

# ... implement changes ...

git add backtesting/dataset.py \
  agents/analytics/indicator_snapshots.py \
  schemas/llm_strategist.py \
  backtesting/llm_strategist_runner.py \
  prompts/strategy_plan_schema.txt \
  prompts/llm_strategist_prompt.txt \
  tests/test_htf_structure_cascade.py

uv run pytest tests/test_htf_structure_cascade.py -vv
git commit -m "Add HTF structure cascade: daily anchor layer with 12 structural fields and stop/target guidance"
```
