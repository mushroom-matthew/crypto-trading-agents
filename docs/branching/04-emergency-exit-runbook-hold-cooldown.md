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

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit)
