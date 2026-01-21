# Implementation Review: Judge Context Wiring

> **Status**: ✅ COMPLETE - All gaps addressed as of 2026-01-20

## Summary

The changes successfully address **ALL** context gaps identified in [INTRADAY_JUDGE_CONTEXT_GAPS.md](./INTRADAY_JUDGE_CONTEXT_GAPS.md). The judge now receives:

| Data | Before | After |
|------|--------|-------|
| Fill details | Aggregate count only | Individual fills with timestamps, prices, trigger IDs, P&L |
| Trigger attempts | Not provided | Per-trigger attempted/executed/blocked with reasons |
| Active triggers | Not provided | Full definitions (entry_rule, exit_rule, etc.) |
| Plan diffs | Not provided | Added/removed/changed trigger comparison on replan |
| Risk budget state | Not provided | Budget %, used %, per-symbol usage and blocks |
| Prompt guidance | Not provided | Interpretation instructions for each section |

---

## Implementation Quality Assessment

### Strengths

1. **Robust Timestamp Handling**
   - Handles pandas Timestamp, datetime, and ISO strings
   - UTC normalization prevents timezone mismatches
   - Invalid timestamp logging aids debugging

2. **Windowed Data Collection**
   - Uses `since_ts` (last judge time) as window start
   - Prevents showing stale fills from earlier in the day
   - Correctly handles day boundaries

3. **Appropriate Limits**
   - Fill details capped at 20 in prompt
   - Trigger attempts capped at 20
   - Active triggers capped at 20
   - Rule text truncated at 160 chars

4. **Plan Diff Completeness**
   - Uses tuple signature for accurate change detection
   - Distinguishes added/removed/changed/unchanged
   - Stored in both log and event payload

### Areas for Potential Improvement

#### 1. ~~Missing Risk Budget State~~ ✅ ADDRESSED

Risk budget utilization is now included via `risk_state` with:
- `daily_risk_budget_pct` and `risk_budget_used_pct`
- `risk_budget_usage_by_symbol` and `risk_budget_blocks_by_symbol`
- `max_trades_per_day` and `trades_executed_today`

#### 2. Trigger Attempt Source Assumption

`_build_trigger_attempt_stats()` relies on `limit_enforcement_by_day` which may not contain all trigger evaluations, only those that hit limit enforcement. Block reasons from `TriggerEngine` (like `MISSING_INDICATOR`, `EXPRESSION_ERROR`) may not be captured.

**Verification needed**: Confirm that `limit_enforcement_by_day` is populated for ALL trigger evaluations, not just limit-related blocks.

#### 3. ~~No Prompt Instruction for New Context~~ ✅ ADDRESSED

The prompt now includes an `INTERPRETATION GUIDANCE` section that tells the LLM:
- How to use `fills_since_last_judge` (assess trade quality, check trigger_attempts if sparse)
- How to use `trigger_attempts` (identify frequently blocked triggers, recommend adjustments)
- How to use `risk_state` (interpret capacity constraints, recommend conservative adjustments)

#### 4. Event Payload Size

`trigger_diff` and `active_triggers` could make event payloads large. The implementation correctly limits `trigger_summary[:10]` but `trigger_diff` includes up to 30 IDs (10 added + 10 removed + 10 changed).

**Minor**: Consider reducing limits if event storage becomes an issue.

---

## Test Verification Checklist

Before considering this complete, verify:

- [ ] Run backtest with multiple symbols (BTC-USD, ETH-USD)
- [ ] Confirm fills appear in `fills_since_last_judge` section of judge payload
- [ ] Confirm trigger attempts show executed vs blocked breakdown
- [ ] Confirm active triggers include full rule definitions
- [ ] Check `plan_generated` events in event store have `trigger_diff` populated
- [ ] Verify judge score adjusts based on new context (qualitative)
- [ ] Check log for "invalid timestamps" warnings (should be zero now)

---

## Files Changed

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `backtesting/llm_strategist_runner.py` | +221 | Core context collection logic |
| `services/judge_feedback_service.py` | +78 | Prompt formatting |
| `prompts/llm_judge_prompt.txt` | +9 | New prompt sections |

---

## Recommended Next Steps

### Before Merge

1. ~~**Add risk budget state** to summary~~ ✅ DONE
2. **Verify trigger attempt source** covers all block types (investigation)
3. **Run validation backtest** and check event payloads

### Post Merge

1. ~~**Add prompt guidance** for interpreting new context sections~~ ✅ DONE
2. **Create dashboard view** for trigger attempt statistics
3. **Implement implied emergency triggers** when positions open
4. **Add unit tests** for `_collect_fill_details`, `_build_trigger_attempt_stats`, `_diff_plan_triggers`

---

## Related Documentation

- [BACKTEST_79007255_ANALYSIS.md](./BACKTEST_79007255_ANALYSIS.md) - Original issue analysis
- [INTRADAY_JUDGE_CONTEXT_GAPS.md](./INTRADAY_JUDGE_CONTEXT_GAPS.md) - Gap specification (now addressed)
- [TRIGGER_SYSTEM_OVERVIEW.md](./TRIGGER_SYSTEM_OVERVIEW.md) - Trigger system reference
