# Branch: exit-binding-enforcement

## Purpose
Runbook 22 added prompt documentation and compile-time warnings for exit binding mismatches. Backtest `58cb897f` proves this is **insufficient**: 7 `exit_binding_mismatch` blocks still occurred, all from `btc_reversal_fib_2h_short` attempting to close positions from a different category. The LLM reads the documentation but still generates cross-category exits. We need compile-time **enforcement** that auto-corrects or rejects mismatched exit triggers.

## Source Evidence
- Backtest `58cb897f`: 7 `exit_binding_mismatch` blocks in DB
- Trigger `btc_reversal_fib_2h_short` (reversal category short) blocked 6 times
- All blocks are exits trying to close positions opened by a different trigger category
- Runbook 22 implementation (warnings + prompt docs) did not prevent the issue

## Root Cause
Prompt-only guidance is necessary but not sufficient. The LLM occasionally ignores exit binding rules despite explicit documentation. The system must enforce correctness at compile time rather than relying on runtime blocking.

## Scope
1. **Auto-relabel exit triggers** in `trigger_compiler.py`: when a mismatch is detected, relabel the exit trigger's category to match the entry trigger's category
2. **Reject irreconcilable mismatches**: if a trigger has both entry and exit rules in conflicting categories for the same symbol, emit an error event and strip the trigger
3. **Emit structured events** for every auto-correction so the judge and telemetry can see what happened

## Out of Scope
- Changing the exit binding enforcement itself (it's a valid safety mechanism)
- Modifying emergency_exit bypass behavior
- LLM prompt changes (already addressed in Runbook 22)

## Key Files
- `trading_core/trigger_compiler.py` — Add auto-relabel logic in `warn_cross_category_exits()` (rename to `enforce_exit_binding()`)
- `agents/strategies/trigger_engine.py` — Reference for how exit binding is checked at runtime
- `agents/event_emitter.py` — Emit correction events

## Implementation Steps

### Step 1: Upgrade warn_cross_category_exits() to enforce_exit_binding()
In `trigger_compiler.py`:
1. After compiling triggers, group by symbol
2. For each symbol, identify the entry trigger's category
3. For each exit trigger on that symbol with a DIFFERENT category:
   - Auto-relabel the exit trigger's category to match the entry
   - Emit an event: `exit_binding_auto_corrected` with original and corrected category
   - Log a warning with the correction details
4. If a trigger has BOTH entry AND exit rules but its category doesn't match any existing entry trigger:
   - Strip the exit portion (keep entry only) and emit `exit_trigger_stripped`

### Step 2: Add correction event to event_emitter
Add event type `exit_binding_auto_corrected` with fields:
- `trigger_id`, `symbol`, `original_category`, `corrected_category`, `plan_id`

### Step 3: Update daily report telemetry
In the daily summary builder, add `exit_binding_corrections` count so the judge can see how often the LLM gets this wrong.

## Test Plan
```bash
# Unit: auto-relabel when mismatch detected
uv run pytest tests/test_trigger_compiler.py -k exit_binding_enforce -vv

# Unit: strip trigger when irreconcilable
uv run pytest tests/test_trigger_compiler.py -k exit_binding_strip -vv

# Integration: full backtest should show 0 exit_binding_mismatch blocks
# (corrections happen at compile time, no runtime blocks)
```

## Test Evidence
```
(to be filled after implementation)
```

## Acceptance Criteria
- [ ] `enforce_exit_binding()` auto-relabels mismatched exit trigger categories
- [ ] Event emitted for every auto-correction
- [ ] Backtest shows 0 `exit_binding_mismatch` blocks (corrections happen pre-runtime)
- [ ] Daily report shows `exit_binding_corrections` count
- [ ] Existing tests still pass (no regressions)

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-12 | Runbook created from backtest 58cb897f validation analysis | Claude |

## Git Workflow
```bash
git checkout -b fix/exit-binding-enforcement
# ... implement changes ...
git add trading_core/trigger_compiler.py agents/event_emitter.py tests/test_trigger_compiler.py
git commit -m "Enforce exit binding at compile time: auto-relabel mismatched categories"
```
