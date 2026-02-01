# Runbook: min_hold vs Exit Timeframe Validation

## Overview
When `min_hold` equals the smallest exit trigger timeframe, exits are mechanically forced at exactly `min_hold` duration. The strategy's exit signal quality cannot be assessed because holds are never longer than the floor — `min_hold` becomes the binding constraint, not market-driven exit logic.

**Source:** Backtest `backtest-b836ce01-d8ef-402e-a74d-4afa241f3344` — all 5 round-trips held exactly 0.5h (min_hold = 0.5h, exit triggers on 15m/30m timeframes). 4 additional exit attempts were blocked by min_hold.

## Scope
1. **Validation warning:** At plan load time, warn if `min_hold` >= smallest exit trigger timeframe. Log to daily report.
2. **Daily report metric:** Track `min_hold_binding_pct` — percentage of exits where hold duration == min_hold (indicating the floor was the binding constraint).
3. **Consider auto-adjustment:** If `min_hold` blocks > 50% of exit attempts, suggest or auto-increase exit timeframe granularity in the next replan.

## Key Files
- `backtesting/llm_strategist_runner.py` — plan validation, min_hold enforcement, exit trigger evaluation
- Daily report generation — add `min_hold_binding_pct` metric

## Acceptance
- Plan validation emits a warning when min_hold >= smallest exit timeframe.
- Daily report includes `min_hold_binding_pct`.
- Backtest with min_hold < exit timeframe shows variable hold durations.

## Out of Scope
- Changing how min_hold is enforced at execution time.
- Judge/replan logic for adjusting min_hold dynamically.
