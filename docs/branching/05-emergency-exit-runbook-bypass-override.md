# Runbook: Emergency Exit Bypass and Override

## Overview
Split runbook for emergency-exit test gaps tied to bypass or override behavior. Source: backtest `backtest-d94320c0`.

## Working rules
- File issues per group; avoid a single mega-branch.
- Judge/strategist loop gaps are design items, not test gaps. Track them separately.
- After issues are filed, delete this runbook.

## Scope (items 3, 5, 6)
### 3. Hold rule bypass
No test confirms emergency exits ignore `hold_rule` (trigger_engine.py:419) while regular exits respect it. This is an intentional design decision that needs test coverage.

### 5. Risk bypass confirmation
TradeRiskEvaluator interaction with emergency flatten is unclear. No test verifies whether emergency exits bypass risk budget checks or consume budget.

### 6. Judge category disabling
No test for `disabled_categories=["emergency_exit"]`. The judge should be able to disable emergency exits as a category, and this path needs coverage.

## Acceptance
- Tests lock down what emergency exits bypass (hold_rule, risk checks) vs respect.
- Tests confirm judge category disabling works for `emergency_exit`.

## Out of scope
Judge/strategist loop design gaps:
- Judge "competing signals" diagnosis not actionable.
- No mechanism for judge to alter trigger conflict detection logic.
- Emergency exit metrics (count, pct) computation not tested.
