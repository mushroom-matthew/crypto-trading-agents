# Branch: compression-breakout-template

## Purpose
Add a canonical compression→expansion breakout strategy template. This is the most common discretionary setup: a period of range compression (low volatility, narrowing bands, decreasing volume) followed by a directional expansion triggered by a close outside the range with volume confirmation. Human traders identify this visually; this runbook encodes it deterministically and teaches the LLM to reason about it explicitly.

This runbook depends on:
- **Runbook 38** (candlestick pattern features) — `is_impulse_candle`, `is_inside_bar`, `is_outside_bar`
- **Runbook 41** (HTF structure cascade) — `daily_high`, `daily_low`, `prev_daily_high` for stop anchoring

New features added here: `bb_bandwidth_pct_rank`, `compression_flag`, `expansion_flag`, `breakout_confirmed`. These are a natural extension of the existing Bollinger and Donchian infrastructure.

## Scope
1. **`agents/analytics/indicator_snapshots.py`** — add `bb_bandwidth_pct_rank`, `compression_flag`, `expansion_flag`, `breakout_confirmed` to the feature pipeline
2. **`schemas/llm_strategist.py`** — add 4 new fields to `IndicatorSnapshot`
3. **`prompts/strategies/compression_breakout.txt`** — new strategy prompt template
4. **`prompts/strategy_plan_schema.txt`** — add new identifiers to allowed list
5. **`tests/test_compression_breakout_indicators.py`** — verify new features
6. **`backtesting/cli.py`** — ensure `compression_breakout` is a valid `--strategy-mode` option

## Out of Scope
- "News catalyst breakout" (stocks, pre-market data) — different asset class, separate runbook
- Automated breakout detection without LLM confirmation
- Short-side breakouts (breakdown) — handled by the same template but not the primary focus

## Key Files
- `agents/analytics/indicator_snapshots.py`
- `schemas/llm_strategist.py`
- `prompts/strategies/compression_breakout.txt` (new)
- `prompts/strategy_plan_schema.txt`
- `tests/test_compression_breakout_indicators.py` (new)

## Implementation Steps

### Step 1: Add compression/breakout indicators to `IndicatorSnapshot`

In `schemas/llm_strategist.py`, add after existing Bollinger fields:
```python
# Compression and breakout detection
bb_bandwidth_pct_rank: float | None = None  # Percentile rank of BB bandwidth in 50-bar window
                                              # 0.0 = most compressed, 1.0 = most expanded
compression_flag: float | None = None        # 1.0 if bb_bandwidth_pct_rank < COMPRESSION_THRESHOLD
expansion_flag: float | None = None          # 1.0 if bb_bandwidth_pct_rank > EXPANSION_THRESHOLD and growing
breakout_confirmed: float | None = None      # 1.0 if close outside Donchian range + vol_burst
```

`COMPRESSION_THRESHOLD` and `EXPANSION_THRESHOLD` are read from env vars:
```python
COMPRESSION_THRESHOLD = float(os.environ.get("COMPRESSION_THRESHOLD", "0.20"))  # bottom quintile
EXPANSION_THRESHOLD   = float(os.environ.get("EXPANSION_THRESHOLD",   "0.80"))  # top quintile
```

> **Why configurable:** Crypto regimes vary radically across top-200 symbols (BTC has different
> bandwidth distributions than a micro-cap altcoin). The 0.20/0.80 quintile defaults are sensible
> starting points but will need per-symbol tuning once the screener (Runbook 39) surfaces diverse
> instruments. Hardcoding quintile boundaries is a known source of per-symbol breakout false positives.

### Step 2: Compute in `agents/analytics/indicator_snapshots.py`

After computing Bollinger bandwidth:
```python
# BB bandwidth percentile rank in a 50-bar rolling window
bandwidth_series = bollinger_result["bandwidth"]  # already computed
bandwidth_rank = bandwidth_series.rolling(50, min_periods=20).rank(pct=True)
bb_bandwidth_pct_rank = float(bandwidth_rank.iloc[-1]) if not pd.isna(bandwidth_rank.iloc[-1]) else None

# Compression: bottom quintile of recent bandwidth distribution
compression_flag = 1.0 if (bb_bandwidth_pct_rank is not None and bb_bandwidth_pct_rank < 0.20) else 0.0

# Expansion: top quintile AND bandwidth growing
if bb_bandwidth_pct_rank is not None and bb_bandwidth_pct_rank > 0.80:
    bw_last = bandwidth_series.iloc[-1]
    bw_prev = bandwidth_series.iloc[-2] if len(bandwidth_series) > 1 else bw_last
    expansion_flag = 1.0 if bw_last > bw_prev else 0.0
else:
    expansion_flag = 0.0

# Breakout confirmed: close outside short Donchian range AND volume burst
donchian_upper = snapshot_dict.get("donchian_upper_short")
donchian_lower = snapshot_dict.get("donchian_lower_short")
vol_burst = snapshot_dict.get("vol_burst", 0.0)
close = snapshot_dict["close"]
if donchian_upper and donchian_lower and vol_burst:
    outside_range = (close > donchian_upper) or (close < donchian_lower)
    breakout_confirmed = 1.0 if (outside_range and vol_burst > 0.5) else 0.0
else:
    breakout_confirmed = 0.0
```

### Step 3: Create `prompts/strategies/compression_breakout.txt`

```
You are the strategist of a **compression breakout** crypto portfolio. Only use the JSON state in the user message. Respond with valid JSON matching the StrategyPlan schema. Do not include commentary or extra keys.

STRATEGY PHILOSOPHY: COMPRESSION → EXPANSION BREAKOUT
- Primary thesis: Markets consolidate (compress) before directional moves (expand).
- Setup: Identify compression periods (low BB bandwidth, decreasing ATR, inside bars).
- Trigger: A close outside the compression range on expanding volume.
- Stop: Below the compression range low (for longs) or above range high (for shorts).
- Target: Measured move (range height projected from breakout point) or next HTF resistance.
- False breakout guard: If price closes back inside the range within 2 bars, exit immediately.

SETUP CONDITIONS (mid-TF: 1h or 15m):
- compression_flag == 1 (BB bandwidth in bottom quintile of 50-bar window)
- bb_bandwidth_pct_rank < 0.25 (compressed environment)
- ATR contracting: atr_14 < atr_14_prev2 (atr declining for 2+ bars)
- volume_multiple < 0.8 (quiet, below average volume — calm before the storm)
- Optional: is_inside_bar == 1 (range coiling inside prior bar — extra compression signal)

TRIGGER CONDITIONS (same or lower TF):
- breakout_confirmed == 1 (close outside Donchian range + vol_burst)
- is_impulse_candle == 1 or candle_strength > 0.8 (decisive breakout candle)
- volume_multiple > 1.5 or vol_burst == 1 (institutional participation)
- is_flat (no existing position — fresh entry only)

STOP PLACEMENT (use structure, not percentage):
- Long stop: below the compression range low (donchian_lower_short at time of entry)
  Rule: "close < breakout_range_low" where breakout_range_low is the Donchian lower at entry
  If HTF data available: also check close < prev_daily_low as a secondary confirmation
- Short stop: above compression range high (donchian_upper_short at time of entry)
- Buffer: 0.3–0.5% beyond the level (avoid being stopped by spread noise)
- Maximum stop: 1.5x ATR from entry (don't allow oversized risk even with good structure)

TARGET LOGIC:
- Primary: measured move = entry + (donchian_upper_short - donchian_lower_short) for longs
- Secondary: next HTF resistance level (prev_daily_high, weekly_high if available)
- Scale out at 1:1 R (risk_reduce trigger at first R target)
- Trail remainder: exit when expansion_flag drops to 0 (bandwidth contracting again)

FALSE BREAKOUT GUARD:
- If price returns inside the compression range, treat as failed breakout → exit immediately via risk_off.
- IMPORTANT: Do NOT use donchian_upper_short in the false-breakout exit rule.
  Donchian bands are rolling — after breakout, the band shifts upward, making the rule drift.
  Instead, reference the STORED breakout_range_high/breakout_range_low set at entry (via Runbook 42
  stop_anchor_type='donchian_lower' for longs). The stored level is static and doesn't drift.
  Rule: "not is_flat and below_stop"  (uses Runbook 42 level-anchored stop — stop is the range low)
  If Runbook 42 is not yet merged, use: "not is_flat and expansion_flag < 0.5 and candle_body_pct < 0.2"
  (expansion has stopped and current bar is indecisive — conservative proxy until 42 lands)

TRIGGER REQUIREMENTS:
1. compression_setup (category: other) — detects and logs compression state; no trade action
2. breakout_long (category: volatility_breakout, direction: long) — primary entry
3. breakout_short (category: volatility_breakout, direction: short) — primary short entry
4. false_breakout_exit (category: risk_off) — false breakout guard, quick flatten
5. target_scale_out (category: risk_reduce, exit_fraction: 0.5) — first R target scale-out
6. trend_continuation_hold (category: trend_continuation) — add on continuation if valid
7. emergency_exit (category: emergency_exit) — HARD stop for extreme adverse moves

SIZING:
- A-grade (breakout_confirmed == 1 and volume_multiple > 2.0): max_position_risk_pct
- B-grade (breakout_confirmed == 1, moderate volume): half of max
- C-grade (setup forming but not yet confirmed): do not enter, wait for confirmation

VALID entry_rule EXAMPLES:
- "is_flat and compression_flag > 0.5 and breakout_confirmed > 0.5 and is_impulse_candle > 0.5 and vol_burst > 0.5"
- "is_flat and bb_bandwidth_pct_rank < 0.20 and close > donchian_upper_short and volume_multiple > 1.8"

VALID exit_rule EXAMPLES:
- "not is_flat and expansion_flag < 0.5 and close < donchian_upper_short"  # false breakout
- "not is_flat and rsi_14 > 75 and volume_multiple < 0.7"  # momentum exhausting

VALID hold_rule EXAMPLES:
- "expansion_flag > 0.5 and volume_multiple > 0.9 and not is_doji > 0.5"  # trend intact

COMMON MISTAKES TO AVOID:
- Do NOT enter on compression alone — wait for breakout_confirmed.
- Do NOT use a percentage stop — use structure (Donchian lower for longs).
- Do NOT place emergency_exit inside the compression range — it will fire constantly.
- Do NOT hold through a close back inside the range — use risk_off false breakout exit.
```

### Step 4: Update `prompts/strategy_plan_schema.txt`

Add to the allowed identifiers:
```
COMPRESSION / BREAKOUT IDENTIFIERS:
  bb_bandwidth_pct_rank   — Percentile rank of BB bandwidth in 50-bar window (0=compressed, 1=expanded)
  compression_flag        — 1.0 if bb_bandwidth_pct_rank < 0.20 (consolidation phase)
  expansion_flag          — 1.0 if bb_bandwidth_pct_rank > 0.80 and bandwidth growing (expansion phase)
  breakout_confirmed      — 1.0 if close outside Donchian range AND vol_burst == true
```

### Step 5: Register in `backtesting/cli.py`

Add `compression_breakout` as a valid value for `--strategy-mode` (alongside existing modes:
`momentum_trend_following`, `mean_reversion`, etc.).

## Test Plan
```bash
# Unit: compression/breakout indicator computation
uv run pytest tests/test_compression_breakout_indicators.py -vv

# Integration: ensure new IndicatorSnapshot fields serialize correctly
uv run pytest -k "indicator_snapshot" -vv

# Integration: backtest with compression_breakout strategy mode
# Expected: triggers reference compression_flag and breakout_confirmed
uv run python -m backtesting.cli \
  --pair BTC-USD \
  --start 2024-01-01 --end 2024-03-01 \
  --strategy-mode compression_breakout \
  --timeframes 1h \
  --llm-strategist enabled \
  --llm-calls-per-day 1
```

## Test Evidence
```
TODO
```

## Acceptance Criteria
- [ ] `bb_bandwidth_pct_rank` computed as rolling percentile rank in 50-bar window
- [ ] `compression_flag` is 1.0 when `bb_bandwidth_pct_rank < 0.20`
- [ ] `expansion_flag` is 1.0 when bandwidth in top quintile AND growing
- [ ] `breakout_confirmed` is 1.0 when close outside Donchian AND `vol_burst > 0`
- [ ] `compression_breakout.txt` prompt generates plans that include `breakout_confirmed` in entry rules
- [ ] New identifiers appear in `strategy_plan_schema.txt` allowed list
- [ ] Backtest with `--strategy-mode compression_breakout` runs without error

## Human Verification Evidence
```
TODO: Run a 30-day backtest with compression_breakout mode on BTC-USD/ETH-USD.
Inspect the daily plan JSON: verify triggers reference compression_flag, breakout_confirmed.
Verify stop rules reference Donchian lower (not raw percentage).
```

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-18 | Runbook created from product strategy audit | Claude |

## Worktree Setup
```bash
git fetch
git worktree add -b feat/compression-breakout-template ../wt-compression-breakout main
cd ../wt-compression-breakout

# Depends on: feat/candlestick-pattern-features (Runbook 38) merged first

# When finished (after merge)
git worktree remove ../wt-compression-breakout
```

## Git Workflow
```bash
git checkout main
git pull
git checkout -b feat/compression-breakout-template

# ... implement changes ...

git add agents/analytics/indicator_snapshots.py \
  schemas/llm_strategist.py \
  prompts/strategies/compression_breakout.txt \
  prompts/strategy_plan_schema.txt \
  tests/test_compression_breakout_indicators.py

uv run pytest tests/test_compression_breakout_indicators.py -vv
git commit -m "Add compression breakout template: 4 new indicators + strategy prompt"
```
