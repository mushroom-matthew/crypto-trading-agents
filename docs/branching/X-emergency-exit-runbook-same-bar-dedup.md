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

## Change Log
- 2026-01-30: Added same-bar emergency-exit veto and emergency-exit dedup tests; prioritized emergency exits in dedup. Files: agents/strategies/trigger_engine.py, tests/test_trigger_engine.py.

## Test Evidence
- `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_trigger_engine.py -vv`
```
warning: The `tool.uv.dev-dependencies` field (used in `pyproject.toml`) is deprecated and will be removed in a future release; use `dependency-groups.dev` instead
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0 -- /home/getzinmw/crypto-trading-agents/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/getzinmw/crypto-trading-agents
configfile: pyproject.toml
plugins: anyio-4.11.0, respx-0.22.0, asyncio-1.2.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 4 items

tests/test_trigger_engine.py::test_trigger_engine_records_block_when_risk_denies_entry PASSED [ 25%]
tests/test_trigger_engine.py::test_emergency_exit_trigger_bypasses_risk_checks PASSED [ 50%]
tests/test_trigger_engine.py::test_emergency_exit_vetoes_same_bar_entry PASSED [ 75%]
tests/test_trigger_engine.py::test_emergency_exit_dedup_overrides_high_conf_entry PASSED [100%]

============================== 4 passed in 1.46s ===============================
```

## Human Verification Evidence
- Not required for this runbook; validation is covered by automated tests.

## Out of scope
Judge/strategist loop design gaps:
- Judge "competing signals" diagnosis not actionable.
- No mechanism for judge to alter trigger conflict detection logic.
- Emergency exit metrics (count, pct) computation not tested.
