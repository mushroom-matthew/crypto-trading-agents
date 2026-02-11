# Branch: hold-rule-calibration

## Purpose
Address overly permissive hold rules that prevent planned exit rules from ever firing. In backtest ebf53879, the hold rule `tf_4h_close > tf_4h_sma_medium and rsi_14 > 45` blocked 12 exit attempts, effectively making the normal exit rule dead code.

## Source Evidence
- Backtest `ebf53879`: 12 `HOLD_RULE` blocks on `btc_trend_continuation_1h`
- Hold rule: `tf_4h_close > tf_4h_sma_medium and rsi_14 > 45`
- RSI > 45 is true ~80% of the time in crypto markets — the hold rule is near-always active
- Normal exit rule `not is_flat and (tf_4h_close < tf_4h_sma_medium or close < sma_short or macd_hist < -50)` could never fire because the hold rule suppressed it
- All BTC exits came through `emergency_exit` (bypasses hold rules) instead of planned exits

## Root Cause
The LLM generates hold rules that are too broad. `rsi_14 > 45` is almost always true in a trending or range-bound crypto market. Combined with `tf_4h_close > tf_4h_sma_medium` (true in any uptrend), the hold rule creates a near-permanent block on exit evaluation.

The LLM has no guidance on what constitutes an effective vs. degenerate hold rule.

## Scope
1. Add **hold rule guidance** to prompts — what makes a good vs. degenerate hold rule
2. Add a **compile-time validator** that flags hold rules with overly common conditions (RSI > 40-45, close > SMA)
3. Add **hold rule binding metrics** to trigger analytics so the judge can see when hold rules are the binding constraint
4. Consider a **max hold rule suppression count** — if a hold rule blocks exits N times consecutively, auto-expire it

## Out of Scope
- Removing hold rules entirely (they serve a valid purpose: preventing premature exits during favorable trends)
- Emergency exit bypass behavior (already correct — emergency exits bypass hold rules)

## Key Files
- `prompts/strategy_plan_schema.txt` — Add hold rule guidance
- `trading_core/trigger_compiler.py` — Add degenerate hold rule detection
- `agents/strategies/trigger_engine.py` — Add consecutive suppression counter and auto-expire
- `ops_api/routers/backtests.py` — Already tracks `min_hold_binding_pct` per day

## Implementation Steps

### Step 1: Prompt guidance
Add to `strategy_plan_schema.txt`:
```
HOLD RULE GUIDANCE:
- Hold rules suppress exit evaluation to prevent premature exits during favorable conditions.
- Hold rules should be NARROW and SPECIFIC — they should be true only when the position is clearly in your favor.
- BAD hold rules (too permissive, blocks almost all exits):
  - rsi_14 > 45 (true ~80% of the time)
  - close > sma_medium (true in any uptrend)
  - tf_4h_close > tf_4h_sma_short (true most of the time)
- GOOD hold rules (specific, time-limited):
  - rsi_14 > 60 and macd_hist > 0 and unrealized_pnl_pct > 0.5
  - close > entry_price and tf_4h_trend_state == 'uptrend' and realized_vol_short < atr_14
- If your hold rule fires >50% of the time, it is too permissive and will prevent planned exits.
```

### Step 2: Compile-time degenerate detection
In `trigger_compiler.py`, flag hold rules that contain only weak conditions:
- `rsi_14 > X` where X < 50
- `close > sma_*` without additional qualifying conditions
- Single-condition hold rules (should require 2+ conditions)

### Step 3: Consecutive suppression counter
In `trigger_engine.py`, track how many consecutive bars a hold rule has suppressed exits:
- After N consecutive suppressions (configurable, default 12), log a warning
- Consider auto-expiring the hold rule for that trigger after 2*N bars
- Report max consecutive hold blocks in daily diagnostics

## Test Plan
```bash
# Unit: degenerate hold rule detection
uv run pytest tests/test_trigger_compiler.py -k hold_rule -vv

# Unit: consecutive suppression counter
uv run pytest tests/test_trigger_engine.py -k hold_suppress -vv

# Integration: backtest should show hold rule binding <20% of exit attempts
```

## Test Evidence
*(to be filled after implementation)*

## Acceptance Criteria
- [ ] Prompt guides LLM toward narrow, specific hold rules with examples
- [ ] Compile-time validator flags degenerate hold rules (RSI < 50, single-condition)
- [ ] Consecutive hold suppression count tracked in trigger analytics
- [ ] Backtest shows hold rule binding <20% of exit evaluations (down from ~40%)

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-10 | Runbook created from backtest ebf53879 analysis | Claude |

## Git Workflow
```bash
git checkout -b fix/hold-rule-calibration
# ... implement changes ...
git add prompts/strategy_plan_schema.txt trading_core/trigger_compiler.py agents/strategies/trigger_engine.py
git commit -m "Calibrate hold rules: prompt guidance, degenerate detection, suppression tracking"
```
