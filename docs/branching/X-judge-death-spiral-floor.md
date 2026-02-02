# Runbook: Judge Death Spiral — Minimum Trigger Floor

**Status: DONE** (Runbook 16 also implemented on this branch)

## Overview
The intraday judge can disable all entry triggers, creating an irreversible trading halt for the remainder of the backtest (or live session). Once all triggers are disabled, no trades execute, so the judge sees the same stale snapshot and keeps triggers disabled indefinitely.

**Source:** Backtest `backtest-b836ce01-d8ef-402e-a74d-4afa241f3344` — judge disabled all entry triggers at 06:45 UTC day 1, zero trades for remaining 2.5 days.

## Scope
1. **Minimum trigger floor:** Enforce that the judge cannot disable more than N-2 entry triggers (at least 2 must remain enabled at all times).
2. **Zero-activity re-enablement:** If zero trades execute for N consecutive bars after a judge intervention, automatically re-enable the least-bad trigger(s) with conservative sizing.
3. **Stale snapshot detection:** If the judge evaluates the same snapshot (same equity, same trade count) as the previous evaluation, skip or force a different intervention strategy.

## Key Files
- `schemas/judge_feedback.py` — `JudgeConstraints` validator (N-2 category rule), `apply_trigger_floor()` helper
- `backtesting/llm_strategist_runner.py` — `_apply_judge_constraints()`, zero-activity re-enablement, stale snapshot detection
- `tests/test_judge_death_spiral.py` — 21 unit tests covering all three features

## Acceptance
- A backtest where the judge wants to disable all triggers retains at least 2 entry triggers.
- After N bars of zero activity post-judge-intervention, at least one trigger is re-enabled.
- Judge does not re-evaluate with an unchanged snapshot.

## Out of Scope
- Judge scoring algorithm changes (separate from the floor mechanism).
- Risk budget or daily loss limit changes.

## Implementation Summary

### Feature 1: Minimum Trigger Floor
- **`schemas/judge_feedback.py`**: Tightened `validate_disabled_categories` from "at least 1 entry category" to "at least 2 entry categories" (N-2 rule). Added `ClassVar` constants: `ENTRY_CATEGORIES`, `MIN_ENABLED_ENTRY_CATEGORIES=2`, `MIN_ENABLED_ENTRY_TRIGGERS=2`.
- **`schemas/judge_feedback.py`**: New `apply_trigger_floor()` function trims `disabled_trigger_ids` so at least `MIN_ENABLED_ENTRY_TRIGGERS` entry triggers remain enabled. Non-entry triggers (emergency_exit, other) are unaffected.
- **`backtesting/llm_strategist_runner.py`**: New `_apply_judge_constraints()` method wraps all 3 constraint-assignment sites, applying `apply_trigger_floor()` before storing constraints.

### Feature 2: Zero-Activity Re-enablement
- **`backtesting/llm_strategist_runner.py`**: New state: `bars_since_last_trade`, `zero_activity_threshold_bars=48` (~12h on 15m bars), `last_judge_intervention_time`.
- In the main bar loop: increments `bars_since_last_trade` every bar, resets on trade execution. When threshold reached after judge intervention, clears `disabled_trigger_ids` (preserves `disabled_categories`), logs `zero_activity_reenable` event.

### Feature 3: Stale Snapshot Detection
- **`backtesting/llm_strategist_runner.py`**: New state: `last_judge_snapshot_key` (`(equity_rounded, trade_count)` tuple).
- In `_run_intraday_judge()`: computes snapshot key, compares to previous. If identical, skips judge evaluation entirely, returns `should_replan=False`, logs `stale_snapshot_skip` event.

## Test Evidence

```
$ uv run pytest tests/test_judge_death_spiral.py -vv
tests/test_judge_death_spiral.py::TestDisabledCategoriesFloor::test_disabling_all_four_trims_to_two PASSED
tests/test_judge_death_spiral.py::TestDisabledCategoriesFloor::test_disabling_three_trims_to_two PASSED
tests/test_judge_death_spiral.py::TestDisabledCategoriesFloor::test_disabling_two_is_allowed PASSED
tests/test_judge_death_spiral.py::TestDisabledCategoriesFloor::test_disabling_one_is_allowed PASSED
tests/test_judge_death_spiral.py::TestDisabledCategoriesFloor::test_non_entry_categories_unaffected PASSED
tests/test_judge_death_spiral.py::TestDisabledCategoriesFloor::test_disabled_categories_floor_keeps_two_entry_categories PASSED
tests/test_judge_death_spiral.py::TestTriggerFloor::test_no_trimming_when_enough_enabled PASSED
tests/test_judge_death_spiral.py::TestTriggerFloor::test_trims_when_too_many_disabled PASSED
tests/test_judge_death_spiral.py::TestTriggerFloor::test_trigger_floor_trims_disabled_trigger_ids PASSED
tests/test_judge_death_spiral.py::TestTriggerFloor::test_exit_triggers_not_counted PASSED
tests/test_judge_death_spiral.py::TestTriggerFloor::test_custom_min_enabled PASSED
tests/test_judge_death_spiral.py::TestTriggerFloor::test_empty_triggers_returns_unchanged PASSED
tests/test_judge_death_spiral.py::TestTriggerFloor::test_no_disabled_returns_same_object PASSED
tests/test_judge_death_spiral.py::TestZeroActivityReenablement::test_zero_activity_reenables_triggers PASSED
tests/test_judge_death_spiral.py::TestZeroActivityReenablement::test_no_reenable_below_threshold PASSED
tests/test_judge_death_spiral.py::TestZeroActivityReenablement::test_no_reenable_without_judge_intervention PASSED
tests/test_judge_death_spiral.py::TestZeroActivityReenablement::test_no_reenable_when_no_disabled_triggers PASSED
tests/test_judge_death_spiral.py::TestStaleSnapshotDetection::test_stale_snapshot_skips_judge_evaluation PASSED
tests/test_judge_death_spiral.py::TestStaleSnapshotDetection::test_changed_snapshot_runs_judge PASSED
tests/test_judge_death_spiral.py::TestStaleSnapshotDetection::test_first_evaluation_always_runs PASSED
tests/test_judge_death_spiral.py::TestStaleSnapshotDetection::test_rounding_behavior PASSED
============================== 21 passed in 0.69s ==============================

$ uv run pytest tests/test_trigger_engine.py -vv
============================== 15 passed in 0.70s ==============================

$ uv run pytest tests/test_execution_engine.py -vv
============================== 9 passed in 4.77s ==============================
```

## Human Verification Evidence
I accept the changes.

## Change Log
| Date | Author | Summary |
|------|--------|---------|
| 2026-01-30 | Claude Opus 4.5 | Initial implementation: N-2 category floor, `apply_trigger_floor()` helper, zero-activity re-enablement (48-bar threshold), stale snapshot detection. Files: `schemas/judge_feedback.py`, `backtesting/llm_strategist_runner.py`, `tests/test_judge_death_spiral.py`. |
| 2026-01-30 | Claude Opus 4.5 | Runbook 16 co-implemented: consecutive stale skip forced re-enablement (`stale_reenable_threshold=2`), `stale_judge_evals` daily metric in `_finalize_day()`, `consecutive_stale_skips` counter with reset on changed snapshot. 6 new tests added. |
| 2026-02-02 | Claude Opus 4.5 | Cross-backtest learnings D1: Zero-activity re-enablement now also clears `disabled_categories` (not just `disabled_trigger_ids`). Category vetoes were causing multi-day zero-trade locks. Condition expanded to `disabled_trigger_ids or disabled_categories`, clearing block updates both fields. 3 new tests in `TestZeroActivityReenablementCategories`. |
