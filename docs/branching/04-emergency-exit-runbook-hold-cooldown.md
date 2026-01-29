# Runbook: Emergency Exit Hold and Cooldown

## Overview
Split runbook for emergency-exit test gaps tied to min-hold and cooldown enforcement. Source: backtest `backtest-d94320c0`.

## Working rules
- File issues per group; avoid a single mega-branch.
- Judge/strategist loop gaps are design items, not test gaps. Track them separately.
- After issues are filed, delete this runbook.

## Scope (items 2, 7, 8)
### 2. Min hold enforcement on emergency exits
No test for `emergency_exit_veto_min_hold`. Emergency exits should still respect minimum hold periods to avoid sub-bar flip-flops.

### 7. Cooldown recommendation
No test that `cooldown_recommendation_bars` is attached to veto records when an emergency exit is vetoed. Downstream consumers rely on this field.

### 8. Multi-bar min hold
No test for emergency exit respecting min_hold across multiple bars. The single-bar case may pass while multi-bar enforcement fails silently.

## Acceptance
- Tests cover min-hold enforcement for emergency exits on single-bar and multi-bar scenarios.
- Tests verify cooldown recommendations are attached to veto records for emergency exits.

## Out of scope
Judge/strategist loop design gaps:
- Judge "competing signals" diagnosis not actionable.
- No mechanism for judge to alter trigger conflict detection logic.
- Emergency exit metrics (count, pct) computation not tested.
