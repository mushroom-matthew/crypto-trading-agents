# Deep Dive: Backtest 4722e226

## Executive Summary

This backtest ran BTC-USD over ~2 days (2021-05-01 to 2021-05-02) with 4 plan generations and 3 judge evaluations. The judge identified critical issues:

| Issue | Severity | Evidence |
|-------|----------|----------|
| High emergency exit rate | **CRITICAL** | Judge explicitly calls this out |
| Complete trigger replacement on replan | **HIGH** | 0 unchanged triggers across all 4 plans |
| Low trigger effectiveness | **HIGH** | 5-6 triggers defined, mostly blocked |
| Zero win rate | **HIGH** | 2 trades executed, both losers |

---

## Timeline Analysis

### Phase 1: Initial Plan (00:00 - 08:00)

**Plan ID**: `plan_0f7a60cd6f714a29b571b6fa73a6fc3e`
**Regime**: `range`
**Triggers**: 5

```
- emergency_exit_1: exit 1h
- tc_short_1: short 1h (RSI > 60, close < SMA)
- tc_long_1: long 1h (RSI < 40, close > SMA)
- vb_long_1: long 5m (breakout above Donchian, ATR > 0.5)
- mr_long_1: long 15m (close < Bollinger lower, RSI < 45)
```

**Result**: Zero trades in 8 hours despite 5 triggers defined.

**Judge Feedback (Score: 45)**:
> "No trades executed indicate a lack of alignment with the configured triggers"

**Replan Reason**: "No trades in 8.0h despite 5 triggers"

---

### Phase 2: First Replan (08:00 - 16:00)

**Plan ID**: `plan_f765f474bd27475690ba3bef711a6859`
**Regime**: `mixed`
**Triggers**: 6 (ALL NEW - zero carried over)

```
- btc_emergency_exit_1: exit 1h (close < nearest_support)
- btc_trend_continuation_short_1: short 1h
- btc_trend_continuation_long_1: long 1h
- btc_volatility_breakout_long_1: long 5m (ATR > 200 ← unrealistic!)
- btc_mean_reversion_long_1: long 15m
- btc_reversal_long_1: long 30m
```

**Result**: 2 trades executed, both losers (-0.01% return)

**Judge Feedback (Score: 30 → actual 21)**:
> "only one trigger executed (with 21 attempts) while others were frequently blocked"
> "addressing the high emergency exit rate which currently hinders performance"

**Critical Issue Identified**: The judge mentioned "21 attempts" for one trigger but others were "frequently blocked". This suggests:
1. `btc_mean_reversion_long_1` fired successfully but entries immediately hit emergency exit
2. Emergency exit trigger (`btc_emergency_exit_1`) was competing with entry triggers

**Judge Response - OVERKILL**:
```json
"disabled_trigger_ids": [
  "btc_volatility_breakout_long_1",
  "btc_trend_continuation_short_1",
  "btc_reversal_long_1",
  "btc_trend_continuation_long_1"
],
"disabled_categories": [
  "trend_continuation",
  "reversal",
  "volatility_breakout",
  "mean_reversion",  ← Even the ONE working trigger category!
  "emergency_exit",
  "other"
]
```

**Problem**: Judge disabled ALL categories including the only one that executed. This is a logical error - it should have kept `mean_reversion` enabled.

---

### Phase 3: Second Replan (16:00 - 00:00)

**Plan ID**: `plan_afe38f0d09674aaa88febc8456021beb`
**Regime**: `range`
**Triggers**: 6 (ALL NEW AGAIN - names changed back)

```
- vb_long_1, emergency_exit_1, tc_short_1, tc_long_1, rev_long_1, mr_long_1
```

**Result**: Zero trades.

**Judge Feedback (Score: 45)**:
> "No trades were executed...none of the active triggers fired"

**Replan Reason**: "No trades in 24.0h despite 6 triggers"

---

### Phase 4: Third Replan (Day 2)

**Plan ID**: `plan_633a54799d064ebc9b166604a85548c5`
**Regime**: `bull`
**Triggers**: 6 (ALL NEW AGAIN - `btc_` prefix returns)

Backtest ended here.

---

## Critical Issues Identified

### Issue 1: Complete Trigger Replacement on Every Replan

Every single replan shows:
```
"changed": 0,
"unchanged": 0
```

This means:
- **No trigger stability** - the LLM generates completely new triggers each time
- **No learning continuity** - can't track which triggers work across replans
- **Naming inconsistency** - triggers flip between `tc_long_1` and `btc_trend_continuation_long_1`

**Root Cause**: The LLM strategist isn't instructed to preserve working triggers. It treats each plan as independent.

**Fix Required**:
1. Pass previous plan triggers to LLM with performance data
2. Instruct LLM to keep unchanged triggers that performed well
3. Use stable trigger ID conventions

---

### Issue 2: High Emergency Exit Rate

The judge explicitly flagged: "high emergency exit rate which currently hinders performance"

**What This Means**:
- Positions were entered but immediately exited by the emergency exit trigger
- The emergency exit rule was too sensitive: `close < nearest_support`
- This created a pattern:
  1. Entry trigger fires → position opened
  2. Emergency exit fires on same/next bar → position closed at loss
  3. Repeat

**Evidence**:
- 2 trades executed with 0% win rate
- Judge mentions "21 attempts" for one trigger but only 2 executed trades
- This suggests ~19 attempts were blocked or immediately closed

**Fix Required**:
1. Add minimum hold period before emergency exit can fire (already in TriggerEngine)
2. Make emergency exit less sensitive (e.g., `close < nearest_support * 0.98`)
3. Don't allow emergency exit to compete with entry triggers on same bar

---

### Issue 3: Unrealistic Trigger Parameters

Plan 2 had: `btc_volatility_breakout_long_1` with `atr_14 > 200`

For BTC at ~$58,000 in May 2021:
- ATR of 200 would be 0.34% of price
- This is extremely low for BTC volatility
- Result: This trigger would NEVER fire

Plan 4 fixed this with: `atr_14 > 0.5 * tf_1h_atr` (relative ATR)

**Fix Required**: Validate trigger parameters are reasonable for the asset's price level.

---

### Issue 4: Judge Over-Correction

When the judge saw poor performance, it disabled ALL categories:
```json
"disabled_categories": ["trend_continuation", "reversal", "volatility_breakout", "mean_reversion", "emergency_exit", "other"]
```

This is **every possible category**, which means nothing can trade.

**Root Cause**: The heuristic/LLM judge doesn't have guardrails against self-defeating constraints.

**Fix Required**: Add validation that at least one entry category remains enabled.

---

## Recommendations

### Immediate (P0)

1. **Preserve Working Triggers Across Replans**
   - When score > 50, keep triggers that executed successfully
   - Only modify triggers that had high block rates or losses

2. **Add Emergency Exit Safeguards**
   - Minimum hold period (4+ bars) before emergency exit eligible
   - Emergency exit should be less sensitive than entry triggers
   - Log when emergency exit competes with entry on same bar

3. **Validate Judge Constraints**
   - Never disable ALL categories
   - Never disable the category of the only working trigger
   - Add sanity check: "at least one entry trigger must be possible"

### Short-Term (P1)

4. **Relative Trigger Parameters**
   - Use percentage-based levels, not absolute values
   - Validate parameters against recent price/volatility data

5. **Trigger ID Stability**
   - Use consistent naming convention across replans
   - Hash trigger signature to detect "same trigger, different name"

6. **Better Trigger Attempt Visibility**
   - Include trigger_attempts in event payload (currently in summary but not persisted)
   - Show which triggers were blocked and why

### Medium-Term (P2)

7. **Trigger Quality Tracking Across Sessions**
   - Persist trigger performance metrics
   - Feed historical trigger quality into replan decisions

8. **Emergency Exit Architecture Review**
   - Consider separate emergency exit mechanism (not trigger-based)
   - Trailing stops that update with price movement
   - Position-level exits vs trigger-level exits

---

## Data Appendix

### Judge Scores Over Time

| Time | Score | Trades | Return | Replan |
|------|-------|--------|--------|--------|
| 08:00 | 45 | 0 | 0% | Yes (no trades) |
| 16:00 | 30 (21) | 2 | -0.01% | Yes (score < 40) |
| 00:00 | 45 | 0 | 0% | Yes (no trades) |

### Trigger Diff Summary

| Plan | Added | Removed | Changed | Unchanged |
|------|-------|---------|---------|-----------|
| 1→2 | 6 | 5 | 0 | 0 |
| 2→3 | 6 | 6 | 0 | 0 |
| 3→4 | 6 | 6 | 0 | 0 |

**100% trigger churn on every replan.**
