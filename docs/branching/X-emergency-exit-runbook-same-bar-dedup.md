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
No test for priority resolution when a high-confidence entry signal competes with an emergency exit. Emergency exits always preempt entries; risk-off qualifiers apply only to non-emergency exit/risk-reduction signals.

## Trigger severity levels
- **`emergency_exit`**: Safety interrupt. Always wins dedup regardless of regime or competing entry confidence. Non-negotiable.
- **`risk_reduce`** (future, optional): Regime-dependent soft exit. Can be vetoed or compete with entries based on risk classification.
- **Normal exits**: Strategy logic exits (`_exit`, `_flat`). Evaluated in the standard exit path with confidence-based override rules.

## Acceptance
- Tests cover same-bar competition outcomes for emergency exits vs entries.
- Deduplication priority is unconditional for emergency exits vs any competing signal.
- Preemption of entries by emergency exits emits an auditable `emergency_exit_preempts_entry` block event.

## Change Log
- 2026-01-30: Added same-bar emergency-exit veto and emergency-exit dedup tests; prioritized emergency exits in dedup. Files: agents/strategies/trigger_engine.py, tests/test_trigger_engine.py.
- 2026-01-30: Clarified runbook: emergency exits are unconditional safety interrupts, not regime-dependent. Added `emergency_exit_preempts_entry` block event for auditability. Added guardrail test proving emergency exits win dedup even under fully permissive risk constraints.

## Test Evidence
- `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_trigger_engine.py -vv`
```
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0
collected 7 items

tests/test_trigger_engine.py::test_trigger_engine_records_block_when_risk_denies_entry PASSED [ 14%]
tests/test_trigger_engine.py::test_emergency_exit_trigger_bypasses_risk_checks PASSED [ 28%]
tests/test_trigger_engine.py::test_emergency_exit_vetoes_same_bar_entry PASSED [ 42%]
tests/test_trigger_engine.py::test_emergency_exit_vetoes_min_hold_on_next_bar PASSED [ 57%]
tests/test_trigger_engine.py::test_emergency_exit_min_hold_allows_on_threshold_bar PASSED [ 71%]
tests/test_trigger_engine.py::test_emergency_exit_dedup_overrides_high_conf_entry PASSED [ 85%]
tests/test_trigger_engine.py::test_emergency_exit_dedup_wins_even_with_permissive_risk PASSED [100%]

============================== 7 passed in 0.25s ===============================
```

## Human Verification Evidence
- Not required for this runbook; validation is covered by automated tests.

## Out of scope
Judge/strategist loop design gaps:
- Judge "competing signals" diagnosis not actionable.
- No mechanism for judge to alter trigger conflict detection logic.
- Emergency exit metrics (count, pct) computation not tested.
