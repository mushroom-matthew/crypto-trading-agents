# Deep Dive: Backtest c83df69e (Post-Fix Validation)

## Executive Summary

This backtest validates the implemented fixes for trigger continuity, emergency exit blocking, and constraint validation. Key observations:

| Metric | Before Fix (4722e226) | After Fix (c83df69e) |
|--------|----------------------|---------------------|
| Trigger continuity | 0% unchanged | **100% unchanged** (plans 2-4) |
| Emergency exit same-bar | Not blocked | **Now blocked** |
| All categories disabled | Yes (judge panic) | **No** (validation working) |
| Trades executed | 2 total | **8 total** (more activity) |
| Win rate | 0% | 0% (still needs work) |

**The fixes are working**, but the system still has 0% win rate. The issue now is **trigger quality**, not system mechanics.

---

## Positive Findings: Fixes Working

### 1. Trigger Continuity ✅

```
Plan 1→2: unchanged=8 (100% preserved!)
Plan 2→3: unchanged=8
Plan 3→4: unchanged=8
Plan 4→5: unchanged=7, changed=1 (only emergency_exit renamed)
```

The LLM is now preserving triggers across replans. This is a major improvement.

### 2. Same-Bar Emergency Exit Blocking ✅

The TriggerEngine now has:
```python
if trigger.category == "emergency_exit":
    last_entry_ts = self._last_entry_timestamp.get(trigger.symbol)
    if last_entry_ts and last_entry_ts == bar.timestamp:
        detail = "Emergency exit blocked on same bar as entry"
        self._record_block(block_entries, trigger, "SAME_BAR_ENTRY", detail, bar)
        continue
```

This prevents the immediate exit pattern.

### 3. Category Validation ✅

Judge evaluation 2 disabled multiple categories but **NOT all entry categories**:
```json
"disabled_categories": ["trend_continuation", "reversal", "mean_reversion"]
```

At least `volatility_breakout` remains enabled. The schema validation is preventing complete shutdown.

---

## Issues Still Present

### Issue 1: 0% Win Rate Across All Evaluations

| Evaluation | Trades | Winning | Losing | Win Rate |
|------------|--------|---------|--------|----------|
| 1 (07:10) | 3 | 0 | 3 | 0% |
| 2 (14:05) | 6 | 0 | 6 | 0% |
| 3 (22:05) | 8 | 0 | 8 | 0% |

**Every trade is a loser.** This is not a blocking issue - trades ARE executing but losing.

### Issue 2: High Emergency Exit Rate

Judge notes repeatedly:
> "High emergency exit rate indicates competing signals"
> "only 1 of the 8 active triggers seeing execution (emergency_exit_1)"

The emergency exit trigger is firing more than entry triggers.

### Issue 3: Significant Trigger Blocking

Judge notes:
> "several triggers being blocked (14 for mr_long_1...)"

Many trigger attempts are being blocked. The blocking reasons are **not visible in the event payload** - this is a gap.

---

## Analysis: Why Trades Are Losing

### Hypothesis 1: Entry Conditions Too Loose

Looking at the trigger definitions:
```python
mr_long_1: long 5m cat=mean_reversion
  entry: close < bollinger_lower and rsi_14 < 45

tc_long_1: long 1h cat=trend_continuation
  entry: close > ema_short and position == 'flat' and rsi_14 < 60
```

These conditions may fire in unfavorable market conditions:
- Mean reversion entry at Bollinger lower doesn't confirm reversal
- No volume confirmation
- No trend state validation for mean reversion

### Hypothesis 2: Exit Conditions Too Sensitive

The emergency exit trigger:
```python
emergency_exit_1: exit 1h cat=emergency_exit
  entry: position != 'flat' and close < nearest_support
```

If `nearest_support` is calculated from recent lows, it may be very close to entry prices, causing quick exits.

### Hypothesis 3: Regime Mismatch

Plans cycle through regimes:
- Plan 1: `range`
- Plan 2: `mixed`
- Plan 3: `range`
- Plan 4: `bear`
- Plan 5: `bull`

But all plans use the **same triggers** (due to continuity). Triggers designed for `range` may not work in `bear` or `bull`.

---

## Blocking Analysis

### What's Being Blocked

Based on the code flow, blocks can come from:

| Source | Reason | Description |
|--------|--------|-------------|
| TriggerEngine | `MISSING_INDICATOR` | Required indicator not available |
| TriggerEngine | `EXPRESSION_ERROR` | Syntax error in rule |
| TriggerEngine | `MIN_HOLD_PERIOD` | Position held < min_hold_bars |
| TriggerEngine | `SAME_BAR_ENTRY` | Emergency exit on same bar as entry |
| Backtester | `min_hold` | Time-based hold check failed |
| Backtester | `min_flat` | Not flat long enough before re-entry |
| Backtester | `risk_budget` | Daily risk budget exhausted |
| ExecutionEngine | `DAILY_CAP` | Max trades/day reached |
| ExecutionEngine | `CATEGORY` | Category disabled by judge |
| ExecutionEngine | `DIRECTION` | Direction not allowed |

### Missing Data in Event Payload

The `plan_judged` event does **NOT** include:
- `trigger_attempts` (attempted/executed/blocked per trigger)
- `blocked_by_reason` breakdown
- `fills_since_last_judge` details

This makes debugging difficult. The judge sees this data (it's in the summary passed to LLM) but it's not persisted.

---

## Recommendations

### Immediate: Persist Trigger Attempt Data

Update `_emit_plan_judged_event` to include trigger attempt stats:

```python
self._emit_event(
    "plan_judged",
    {
        "date": day_key,
        "plan_id": plan_id,
        # ... existing fields ...
        "trigger_attempts": summary.get("trigger_attempts"),  # ADD THIS
        "fills_since_last_judge": summary.get("fills_since_last_judge"),  # ADD THIS
        "risk_state": summary.get("risk_state"),  # ADD THIS
    },
    ...
)
```

### Short-Term: Improve Trigger Quality

1. **Add volume confirmation** to mean reversion entries
2. **Widen emergency exit threshold** (e.g., `close < nearest_support * 0.98`)
3. **Add regime-specific triggers** instead of using same triggers across all regimes
4. **Require confidence grades** - currently all triggers have `conf=None`

### Medium-Term: Regime-Adaptive Trigger Selection

When regime changes:
- Don't just preserve triggers blindly
- Evaluate if preserved triggers match new regime
- Consider disabling counter-regime triggers

---

## Data Appendix

### Judge Scores

| Time | Score | Replan Triggered |
|------|-------|------------------|
| 07:10 | 39 (24.6 internal) | Yes - score < 40 |
| 14:05 | 24 (18.6 internal) | Yes - score < 40 |
| 22:05 | 20 (17.0 internal) | Yes - score < 40 |

### Judge Constraints Applied

**Evaluation 1**:
- Disabled triggers: `tc_long_2`, `tc_short_2`
- Risk mode: `conservative`

**Evaluation 2**:
- Disabled triggers: 6 total
- Disabled categories: `trend_continuation`, `reversal`, `mean_reversion`
- Risk mode: `conservative`

**Evaluation 3**:
- Disabled triggers: 5 total
- Disabled categories: `mean_reversion` only
- Risk mode: `conservative`

Note: The judge is correctly reducing disabled categories after seeing no improvement.

### Trigger Stability

```
Plan 1: 8 triggers (initial)
Plan 2: 8 triggers (8 unchanged from Plan 1)
Plan 3: 8 triggers (8 unchanged from Plan 2)
Plan 4: 8 triggers (8 unchanged from Plan 3)
Plan 5: 8 triggers (7 unchanged, 1 renamed)
```

**100% continuity achieved** - the fix is working.
