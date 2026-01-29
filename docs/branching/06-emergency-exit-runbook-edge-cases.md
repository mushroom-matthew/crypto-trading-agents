# Runbook: Emergency Exit Edge Cases

## Overview
Split runbook for emergency-exit edge cases. Source: backtest `backtest-d94320c0`.

## Working rules
- File issues per group; avoid a single mega-branch.
- Judge/strategist loop gaps are design items, not test gaps. Track them separately.
- After issues are filed, delete this runbook.

## Scope (item 9)
### 9. Missing exit_rule validation
No test for emergency exit with `exit_rule=None`. The trigger engine should handle missing exit rules gracefully without crashing.

## Acceptance
- Test covers emergency exit with `exit_rule=None` and verifies safe handling.

## Out of scope
Judge/strategist loop design gaps:
- Judge "competing signals" diagnosis not actionable.
- No mechanism for judge to alter trigger conflict detection logic.
- Emergency exit metrics (count, pct) computation not tested.
