# Runbook: Judge Stale Snapshot Detection

**Status: DONE** (implemented on branch `judge-death-spiral-floor` alongside Runbook 13)

## Overview
When the judge re-evaluates after a period of zero trading activity, it receives the exact same snapshot (same equity, same trade count, same positions) as the previous evaluation. This leads to identical conclusions and redundant LLM calls that reinforce the current (possibly broken) state.

**Source:** Backtest `backtest-b836ce01-d8ef-402e-a74d-4afa241f3344` — judge eval at 18:45 UTC used identical snapshot to 06:45 eval (equity $9985.13, 10 trades, 0 wins, no positions). Produced the same score (17.8) and doubled down on disabling triggers.

## Scope
1. **Snapshot diff check:** Before running a judge evaluation, compare the current snapshot to the previous one. If equity, trade count, and positions are unchanged, mark as stale.
2. **Stale handling options:**
   - Skip evaluation entirely and schedule next eval sooner.
   - Run evaluation with a modified prompt: "No trades have occurred since your last evaluation. Consider re-enabling triggers or adjusting constraints."
   - Force at least one trigger re-enablement before the next eval.
3. **Metric:** Track `stale_judge_evals` count in daily reports.

## Key Files
- `backtesting/llm_strategist_runner.py` — `_run_intraday_judge()` stale snapshot detection, `stale_judge_evals_by_day` counter, `consecutive_stale_skips` tracker
- `tests/test_judge_death_spiral.py` — `TestStaleSnapshotDetection`, `TestStaleSnapshotReenablement`, `TestStaleJudgeEvalsMetric`

## Acceptance
- Judge detects when consecutive evaluations use the same snapshot.
- At least one of the stale handling options is implemented.
- Daily report tracks `stale_judge_evals`.

## Out of Scope
- Judge scoring algorithm changes.
- Judge prompt content changes beyond stale-snapshot context.

## Implementation Summary

### Acceptance 1: Snapshot diff check
Snapshot key `(round(equity, 2), trade_count)` is computed in `_run_intraday_judge()` and compared against `last_judge_snapshot_key`. Identical key = stale.

### Acceptance 2: Stale handling (two options implemented)
1. **Skip evaluation entirely** — stale snapshot returns early with `should_replan=False`, no LLM call.
2. **Force trigger re-enablement** — after `stale_reenable_threshold` (default 2) consecutive stale skips with disabled triggers, `disabled_trigger_ids` is cleared and a `stale_snapshot_reenable` event is logged. The `consecutive_stale_skips` counter resets on any changed snapshot.

Option "modified prompt" was not implemented (out of scope per runbook: "Judge prompt content changes beyond stale-snapshot context").

### Acceptance 3: Daily metric
`stale_judge_evals` count is included in the daily report summary via `stale_judge_evals_by_day` dict, popped per day in `_finalize_day()`.

## Test Evidence

```
$ uv run pytest tests/test_judge_death_spiral.py -vv
tests/test_judge_death_spiral.py::TestStaleSnapshotDetection::test_stale_snapshot_skips_judge_evaluation PASSED
tests/test_judge_death_spiral.py::TestStaleSnapshotDetection::test_changed_snapshot_runs_judge PASSED
tests/test_judge_death_spiral.py::TestStaleSnapshotDetection::test_first_evaluation_always_runs PASSED
tests/test_judge_death_spiral.py::TestStaleSnapshotDetection::test_rounding_behavior PASSED
tests/test_judge_death_spiral.py::TestStaleSnapshotReenablement::test_consecutive_stale_skips_force_reenable PASSED
tests/test_judge_death_spiral.py::TestStaleSnapshotReenablement::test_single_stale_skip_no_reenable PASSED
tests/test_judge_death_spiral.py::TestStaleSnapshotReenablement::test_stale_reenable_no_effect_without_disabled PASSED
tests/test_judge_death_spiral.py::TestStaleSnapshotReenablement::test_stale_counter_resets_on_changed_snapshot PASSED
tests/test_judge_death_spiral.py::TestStaleJudgeEvalsMetric::test_stale_evals_counter_accumulates_per_day PASSED
tests/test_judge_death_spiral.py::TestStaleJudgeEvalsMetric::test_stale_evals_pop_returns_zero_for_missing_day PASSED
============================== 27 passed (total) in 0.50s ==============================

$ uv run pytest tests/test_trigger_engine.py tests/test_execution_engine.py -vv
============================== 24 passed in 2.07s ==============================
```

## Human Verification Evidence
I accept the changes.

## Change Log
| Date | Author | Summary |
|------|--------|---------|
| 2026-01-30 | Claude Opus 4.5 | Initial implementation: stale snapshot skip + forced re-enablement after 2 consecutive stale evals + `stale_judge_evals` daily metric. Files: `backtesting/llm_strategist_runner.py`, `tests/test_judge_death_spiral.py`. |
