# Runbook: Emergency Exit Hold and Cooldown

## Overview
Emergency exits must still respect minimum hold enforcement while emitting cooldown recommendations on vetoes. This runbook closes the test gaps around min-hold enforcement (single and multi-bar) and ensures cooldown metadata is attached to emergency exit veto records.

**Source:** backtest `backtest-d94320c0`.

## Scope
1. **Min-hold enforcement (single-bar):** Add test coverage for `emergency_exit_veto_min_hold` when an emergency exit fires within the minimum hold period on the entry bar.
2. **Min-hold enforcement (multi-bar):** Add test coverage for multi-bar hold enforcement where the emergency exit is blocked for N-1 bars and allowed at N.
3. **Cooldown recommendation metadata:** Add tests that `cooldown_recommendation_bars` is included on emergency-exit veto records (same-bar and min-hold vetoes).

## Key Files
- `agents/strategies/trigger_engine.py` (min-hold enforcement, veto records, cooldown metadata)
- `tests/test_trigger_engine.py` (unit tests for emergency exit hold/cooldown behavior)

## Acceptance Criteria
- Emergency exits blocked by min-hold record reason `emergency_exit_veto_min_hold` and include `cooldown_recommendation_bars`.
- Multi-bar min-hold enforcement blocks until `min_hold_bars` has elapsed, then permits the emergency exit.
- Cooldown recommendation uses `max(1, trade_cooldown_bars, min_hold_bars)`.

## Out of Scope
- Hold-rule bypass behavior (covered in runbook 05).
- Risk budget bypass semantics (covered in runbook 05).
- Emergency exit metrics computation (tracked separately as design work).

## Test Plan (required before commit)
```bash
uv run pytest tests/test_trigger_engine.py -vv
```

If tests cannot run locally, obtain user-run output and paste it into the Test Evidence section before committing.

## Human Verification (required)
- Inspect block entries in the emergency exit tests to confirm `cooldown_recommendation_bars` is present.
- Confirm the multi-bar min-hold test includes both blocked and allowed cases.

## Worktree Setup (recommended for parallel agents)
```bash
git fetch
git worktree add -b emergency-exit-hold-cooldown ../wt-emergency-exit-hold-cooldown main
cd ../wt-emergency-exit-hold-cooldown

# When finished
git worktree remove ../wt-emergency-exit-hold-cooldown
```

## Git Workflow (explicit)
```bash
git checkout main
git pull
git checkout -b emergency-exit-hold-cooldown

git status
git diff

git add agents/strategies/trigger_engine.py tests/test_trigger_engine.py

uv run pytest tests/test_trigger_engine.py -vv

git commit -m "Emergency exit: min-hold and cooldown tests"
```

## Change Log (update during implementation)
- 2026-01-29: Expanded runbook format with scope, acceptance, test plan, and git workflow details.
- 2026-01-30: Added emergency-exit min-hold/cooldown tests in `tests/test_trigger_engine.py`.

## Test Evidence (append results before commit)
```bash
uv run pytest tests/test_trigger_engine.py -vv
```
```
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0 -- /home/getzinmw/crypto-trading-agents/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/getzinmw/crypto-trading-agents
configfile: pyproject.toml
plugins: anyio-4.11.0, respx-0.22.0, asyncio-1.2.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 6 items

tests/test_trigger_engine.py::test_trigger_engine_records_block_when_risk_denies_entry PASSED [ 16%]
tests/test_trigger_engine.py::test_emergency_exit_trigger_bypasses_risk_checks PASSED [ 33%]
tests/test_trigger_engine.py::test_emergency_exit_vetoes_same_bar_entry PASSED [ 50%]
tests/test_trigger_engine.py::test_emergency_exit_vetoes_min_hold_on_next_bar PASSED [ 66%]
tests/test_trigger_engine.py::test_emergency_exit_min_hold_allows_on_threshold_bar PASSED [ 83%]
tests/test_trigger_engine.py::test_emergency_exit_dedup_overrides_high_conf_entry PASSED [100%]

============================== 6 passed in 0.32s ===============================
```

## Human Verification Evidence (append results before commit)
- 2026-01-30: Verified same-bar and min-hold veto tests assert `cooldown_recommendation_bars` and multi-bar test covers blocked (bars 1-2) and allowed (bar 3) emergency exits.
