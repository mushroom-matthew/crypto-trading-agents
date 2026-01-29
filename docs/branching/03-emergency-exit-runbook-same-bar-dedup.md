# Runbook: Emergency Exit Same-Bar Dedup

## Overview
Split runbook for emergency-exit test gaps tied to same-bar competition and deduplication priority. Source: backtest `backtest-d94320c0`.

## Working rules
- File issues per group; avoid a single mega-branch.
- Judge/strategist loop gaps are design items, not test gaps. Track them separately.
- After issues are filed, delete this runbook.

## Scope (items 1, 4, 10)
### 1. Same-bar entry+exit competition (`emergency_exit_veto_same_bar`)
No test verifies that the veto fires when an emergency exit and a new entry compete on the same bar. The trigger engine should veto the exit in this scenario to prevent whipsaw.

### 4. Confidence override with emergency exits
Deduplication logic does not special-case emergency exits. When a high-confidence entry and an emergency exit arrive on the same bar, priority resolution is untested.

### 10. High-confidence entry vs emergency exit deduplication
No test for priority resolution when a high-confidence entry signal competes with an emergency exit. The deduplication logic should prefer the emergency exit in risk-off scenarios.

## Acceptance
- Tests cover same-bar competition outcomes for emergency exits vs entries.
- Deduplication priority is explicit for high-confidence entries vs emergency exits.

## Out of scope
Judge/strategist loop design gaps:
- Judge "competing signals" diagnosis not actionable.
- No mechanism for judge to alter trigger conflict detection logic.
- Emergency exit metrics (count, pct) computation not tested.
