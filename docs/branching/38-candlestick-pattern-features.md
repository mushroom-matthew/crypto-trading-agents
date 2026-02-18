# Branch: candlestick-pattern-features

## Purpose
Add candlestick morphology features to the indicator pipeline. Human traders recognize doji, hammer, engulfing, inside bar, and pin bar patterns with their eyes and use them to confirm entries, exits, and reversals. The system currently has no candle-level pattern recognition — it sees OHLC as four independent scalars with no relational meaning. This runbook adds a `metrics/candlestick.py` module, wires it into the feature vector, and exposes the boolean and scalar pattern identifiers to the LLM strategist.

This is a prerequisite for the Compression Breakout template (Runbook 40) and the HTF Structure Cascade (Runbook 41), where specific candle shapes at key levels are the entry confirmation signal.

## Scope
1. **`metrics/candlestick.py`** — new module with 12 pattern detectors and 4 scalar features
2. **`agents/analytics/indicator_snapshots.py`** — wire candlestick features into `IndicatorSnapshot` and `compute_indicator_snapshot()`
3. **`schemas/llm_strategist.py`** — add new fields to `IndicatorSnapshot`
4. **`prompts/strategy_plan_schema.txt`** — add candlestick identifiers to the allowed list with usage guidance
5. **`prompts/llm_strategist_prompt.txt`** — add guidance on using candlestick patterns
6. **`tests/test_candlestick_patterns.py`** — new test file with numerical verification of each pattern

## Out of Scope
- Multi-candle pattern sequences beyond 2 bars (e.g., three white soldiers, morning star) — deferred
- Candlestick scoring or probability models
- VLM or image-based pattern recognition

## Key Files
- `metrics/candlestick.py` (new)
- `metrics/__init__.py` — register new module
- `agents/analytics/indicator_snapshots.py`
- `schemas/llm_strategist.py`
- `prompts/strategy_plan_schema.txt`
- `prompts/llm_strategist_prompt.txt`
- `tests/test_candlestick_patterns.py` (new)

## Implementation Steps

### Step 1: Create `metrics/candlestick.py`

The module computes features from OHLCV DataFrames, returning one value per bar. All features use the current bar (index -1) and optionally the prior bar (index -2).

**Scalar features** (continuous values, useful as thresholds in trigger expressions):
```python
def body_pct(df) -> pd.Series:
    """Body as fraction of total range. 0=doji, 1=marubozu."""
    body = (df["close"] - df["open"]).abs()
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    return (body / range_).fillna(0.0)

def upper_wick_pct(df) -> pd.Series:
    """Upper wick as fraction of total range."""
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    return (upper / range_).fillna(0.0)

def lower_wick_pct(df) -> pd.Series:
    """Lower wick as fraction of total range."""
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    return (lower / range_).fillna(0.0)

def candle_strength(df, atr_series) -> pd.Series:
    """Body size normalized by ATR. >1 = impulse candle, <0.3 = indecision."""
    body = (df["close"] - df["open"]).abs()
    return (body / atr_series.replace(0, np.nan)).fillna(0.0)
```

**Boolean pattern features** (True/False per bar):
```python
# Directionality
def is_bullish(df) -> pd.Series:
    return df["close"] > df["open"]

def is_bearish(df) -> pd.Series:
    return df["close"] < df["open"]

# Single-bar reversal patterns
def is_doji(df) -> pd.Series:
    """Indecision: body < 10% of range."""
    return body_pct(df) < 0.10

def is_hammer(df) -> pd.Series:
    """Bullish reversal at low: long lower wick, small body near top.
    Conditions: lower_wick > 2*body, upper_wick < body, candle appears after decline."""
    body = (df["close"] - df["open"]).abs()
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    return (lower > 2 * body) & (upper < body) & (body > 0)

def is_shooting_star(df) -> pd.Series:
    """Bearish reversal at high: long upper wick, small body near bottom.
    Conditions: upper_wick > 2*body, lower_wick < body."""
    body = (df["close"] - df["open"]).abs()
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    return (upper > 2 * body) & (lower < body) & (body > 0)

def is_pin_bar(df) -> pd.Series:
    """Generic: wick on either side > 60% of total range (hammer OR shooting star shape)."""
    return (upper_wick_pct(df) > 0.60) | (lower_wick_pct(df) > 0.60)

# Two-bar patterns (compare current to prior bar)
def is_bullish_engulfing(df) -> pd.Series:
    """Current bullish bar fully engulfs prior bearish bar body."""
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prior_bearish = prev_close < prev_open
    curr_bullish = df["close"] > df["open"]
    engulfs = (df["open"] <= prev_close) & (df["close"] >= prev_open)
    return prior_bearish & curr_bullish & engulfs

def is_bearish_engulfing(df) -> pd.Series:
    """Current bearish bar fully engulfs prior bullish bar body."""
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prior_bullish = prev_close > prev_open
    curr_bearish = df["close"] < df["open"]
    engulfs = (df["open"] >= prev_close) & (df["close"] <= prev_open)
    return prior_bullish & curr_bearish & engulfs

def is_inside_bar(df) -> pd.Series:
    """Current bar's high AND low contained within prior bar's range.
    Indicates compression / indecision within prior bar's range."""
    return (df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))

def is_outside_bar(df) -> pd.Series:
    """Current bar's high exceeds prior high AND low undercuts prior low.
    Indicates expansion / volatility spike (also called 'key reversal bar')."""
    return (df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))

def is_impulse_candle(df, atr_series, min_strength: float = 1.0) -> pd.Series:
    """Body >= min_strength * ATR. Indicates decisive directional move.

    Default threshold is 1.0 (body equals one full ATR) rather than 0.8,
    to avoid over-triggering in high-volatility crypto regimes. Configurable
    via the `min_strength` parameter or the IMPULSE_CANDLE_ATR_MULT env var.
    """
    return candle_strength(df, atr_series) >= min_strength
```

Wrap all into a single entry point:
```python
def compute_candlestick_features(df: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    """Compute all candlestick features and return as a DataFrame with one column per feature."""
    return pd.DataFrame({
        "candle_body_pct": body_pct(df),
        "candle_upper_wick_pct": upper_wick_pct(df),
        "candle_lower_wick_pct": lower_wick_pct(df),
        "candle_strength": candle_strength(df, atr_series),
        "is_bullish": is_bullish(df).astype(float),
        "is_bearish": is_bearish(df).astype(float),
        "is_doji": is_doji(df).astype(float),
        "is_hammer": is_hammer(df).astype(float),
        "is_shooting_star": is_shooting_star(df).astype(float),
        "is_pin_bar": is_pin_bar(df).astype(float),
        "is_bullish_engulfing": is_bullish_engulfing(df).astype(float),
        "is_bearish_engulfing": is_bearish_engulfing(df).astype(float),
        "is_inside_bar": is_inside_bar(df).astype(float),
        "is_outside_bar": is_outside_bar(df).astype(float),
        "is_impulse_candle": is_impulse_candle(
        df, atr_series,
        min_strength=float(os.environ.get("IMPULSE_CANDLE_ATR_MULT", "1.0"))
    ).astype(float),
    }, index=df.index)
```

Boolean features are cast to float (0.0 / 1.0) so they survive the existing serialization pipeline.
In trigger rules, evaluate as: `is_hammer == 1` or `is_hammer > 0.5`.

### Step 2: Add fields to `IndicatorSnapshot` in `schemas/llm_strategist.py`

Add after the existing `bollinger_*` fields:
```python
# Candlestick morphology features
candle_body_pct: float | None = None          # 0=doji, 1=marubozu
candle_upper_wick_pct: float | None = None    # upper wick / range
candle_lower_wick_pct: float | None = None    # lower wick / range
candle_strength: float | None = None          # body / ATR (>0.8 = impulse)
is_bullish: float | None = None               # 1.0 if close > open
is_bearish: float | None = None               # 1.0 if close < open
is_doji: float | None = None                  # 1.0 if body_pct < 0.10
is_hammer: float | None = None                # 1.0 if hammer pattern
is_shooting_star: float | None = None         # 1.0 if shooting star pattern
is_pin_bar: float | None = None               # 1.0 if long wick either side
is_bullish_engulfing: float | None = None     # 1.0 if bullish engulfing
is_bearish_engulfing: float | None = None     # 1.0 if bearish engulfing
is_inside_bar: float | None = None            # 1.0 if range inside prior bar
is_outside_bar: float | None = None           # 1.0 if range outside prior bar
is_impulse_candle: float | None = None        # 1.0 if body >= 0.8 * ATR
```

### Step 3: Wire into `compute_indicator_snapshot()` in `agents/analytics/indicator_snapshots.py`

After computing ATR, call:
```python
from metrics import candlestick as cs

candle_features = cs.compute_candlestick_features(df, atr_series)
last = candle_features.iloc[-1]
# Add each feature to the snapshot dict before constructing IndicatorSnapshot
snapshot_dict["candle_body_pct"] = float(last["candle_body_pct"])
snapshot_dict["is_hammer"] = float(last["is_hammer"])
# ... etc for all 15 features
```

### Step 4: Update allowed identifiers in `prompts/strategy_plan_schema.txt`

Add a new section:
```
CANDLESTICK PATTERN IDENTIFIERS (scalar/boolean):
  candle_body_pct         — body as fraction of range (0=doji, 1=marubozu)
  candle_upper_wick_pct   — upper wick fraction of range
  candle_lower_wick_pct   — lower wick fraction of range
  candle_strength         — body / ATR (>0.8 = impulse move)
  is_bullish              — 1.0 if close > open
  is_bearish              — 1.0 if close < open
  is_doji                 — 1.0 if body_pct < 0.10 (indecision)
  is_hammer               — 1.0 if bullish reversal shape at low
  is_shooting_star        — 1.0 if bearish reversal shape at high
  is_pin_bar              — 1.0 if wick > 60% of range (either side)
  is_bullish_engulfing    — 1.0 if current bar engulfs prior bearish body
  is_bearish_engulfing    — 1.0 if current bar engulfs prior bullish body
  is_inside_bar           — 1.0 if current range inside prior bar (compression)
  is_outside_bar          — 1.0 if current range outside prior bar (expansion)
  is_impulse_candle       — 1.0 if body >= 0.8 * ATR (decisive move)

CANDLESTICK USAGE GUIDANCE:
- Use boolean patterns as CONFIRMATION for entry/exit triggers, not as sole condition.
  Example: "is_hammer == 1 and close > nearest_support and rsi_14 < 40" (hammer + support + oversold)
- Evaluate boolean features as: `is_hammer > 0.5` or `is_hammer == 1` in rule expressions.
- candle_strength > 1.0 suggests an impulse move; use for breakout confirmation.
- is_inside_bar is a compression signal; combine with low BB bandwidth for setup detection.
- is_outside_bar at a key level often precedes a strong directional move.
- Two-bar patterns (engulfing, inside, outside) require prev_close/prev_open — always available.
- Pattern context matters: a hammer is bullish only at a support level or after a downtrend.
  The LLM must combine candle patterns with structural context, not use them in isolation.
```

### Step 5: Create `tests/test_candlestick_patterns.py`

Verify each pattern with known OHLC inputs:
```python
import pandas as pd
import numpy as np
from metrics import candlestick as cs

def make_df(rows):
    """rows: list of (open, high, low, close, volume)"""
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])

def test_doji_detected():
    # Body = 0.5, range = 10 → body_pct = 0.05 < 0.10
    df = make_df([(100, 105, 95, 100.5, 1000)])
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_doji"].iloc[-1] == 1.0

def test_hammer_detected():
    # Long lower wick, small body near top
    df = make_df([(100, 101, 88, 100.5, 1000)])  # lower_wick=12, body=0.5, upper=0.5
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_hammer"].iloc[-1] == 1.0

def test_shooting_star_detected():
    # Long upper wick, small body near bottom
    df = make_df([(100, 112, 99, 100.5, 1000)])  # upper_wick=11.5, body=0.5, lower=1
    atr = pd.Series([5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_shooting_star"].iloc[-1] == 1.0

def test_bullish_engulfing_detected():
    # Row 0: bearish, Row 1: bullish that engulfs row 0
    df = make_df([(102, 103, 99, 100, 1000), (99, 105, 98, 104, 1000)])
    atr = pd.Series([3.0, 3.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_bullish_engulfing"].iloc[-1] == 1.0

def test_inside_bar_detected():
    df = make_df([(100, 110, 90, 105, 1000), (102, 108, 93, 104, 1000)])
    atr = pd.Series([5.0, 5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_inside_bar"].iloc[-1] == 1.0

def test_outside_bar_detected():
    df = make_df([(100, 108, 94, 105, 1000), (97, 112, 91, 106, 1000)])
    atr = pd.Series([5.0, 5.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_outside_bar"].iloc[-1] == 1.0

def test_impulse_candle_detected():
    # body = 4.0, ATR = 4.0 → candle_strength = 1.0 >= 0.8
    df = make_df([(100, 105, 99, 104, 1000)])
    atr = pd.Series([4.0])
    result = cs.compute_candlestick_features(df, atr)
    assert result["is_impulse_candle"].iloc[-1] == 1.0
```

## Test Plan
```bash
# Unit: all pattern detectors with known inputs
uv run pytest tests/test_candlestick_patterns.py -vv

# Integration: ensure IndicatorSnapshot serializes without error
uv run pytest tests/test_indicator_snapshots.py -vv

# Schema validation: extra="forbid" will catch missing fields
uv run pytest -k "snapshot" -vv
```

## Test Evidence
```
TODO
```

## Acceptance Criteria
- [ ] `metrics/candlestick.py` contains all 15 features (4 scalar, 11 boolean)
- [ ] Each pattern detector passes unit test with known OHLC inputs
- [ ] `IndicatorSnapshot` schema includes all 15 new optional fields
- [ ] `compute_indicator_snapshot()` populates candlestick features from live OHLCV
- [ ] Allowed identifiers in `strategy_plan_schema.txt` list all 15 names with usage guidance
- [ ] No existing tests broken (Pydantic `extra="forbid"` satisfied, no field name collisions)

## Human Verification Evidence
```
TODO: After implementation, run a backtest and inspect a daily report JSON.
Verify candlestick fields appear in the indicator snapshot payload (non-null values).
Check that the LLM uses at least one candlestick identifier in a generated plan.
```

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-18 | Runbook created from strategy audit | Claude |

## Worktree Setup
```bash
git fetch
git worktree add -b feat/candlestick-pattern-features ../wt-candlestick-patterns main
cd ../wt-candlestick-patterns

# When finished (after merge)
git worktree remove ../wt-candlestick-patterns
```

## Git Workflow
```bash
git checkout main
git pull
git checkout -b feat/candlestick-pattern-features

# ... implement changes ...

git add metrics/candlestick.py \
  metrics/__init__.py \
  agents/analytics/indicator_snapshots.py \
  schemas/llm_strategist.py \
  prompts/strategy_plan_schema.txt \
  prompts/llm_strategist_prompt.txt \
  tests/test_candlestick_patterns.py

uv run pytest tests/test_candlestick_patterns.py -vv
git commit -m "Add candlestick pattern features: 15 morphology identifiers wired into feature vector"
```
