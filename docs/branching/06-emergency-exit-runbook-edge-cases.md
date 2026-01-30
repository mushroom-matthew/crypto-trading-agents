# Runbook: Emergency Exit Edge Cases

## Overview
Emergency exit edge cases must be safe and deterministic. This runbook covers missing exit rule handling and the required test coverage to ensure emergency exits never crash the trigger engine.

**Source:** backtest `backtest-d94320c0`.

## Scope
1. **Missing exit_rule handling:** Emergency exits with `exit_rule=None` or empty strings should be blocked safely with an explicit veto record.

## Key Files
- `agents/strategies/trigger_engine.py` (exit rule validation and veto records)
- `tests/test_trigger_engine.py` (unit test for missing exit rule)

## Acceptance Criteria
- Emergency exit with missing `exit_rule` is blocked without exceptions.
- Block record uses reason `emergency_exit_missing_exit_rule` and includes `cooldown_recommendation_bars`.

## Out of Scope
- Min-hold and cooldown enforcement (covered in runbook 04).
- Hold-rule bypass and judge category disabling (covered in runbook 05).
- Judge/strategist design gaps (tracked separately).

## Test Plan (required before commit)
```bash
uv run pytest tests/test_trigger_engine.py -vv
```

If tests cannot run locally, obtain user-run output and paste it into the Test Evidence section before committing.

## Human Verification (required)
- Inspect the missing-exit-rule block entry and confirm cooldown metadata is present.

## Worktree Setup (recommended for parallel agents)
```bash
git fetch
git worktree add -b emergency-exit-edge-cases ../wt-emergency-exit-edge-cases main
cd ../wt-emergency-exit-edge-cases

# When finished
git worktree remove ../wt-emergency-exit-edge-cases
```

## Git Workflow (explicit)
```bash
git checkout main
git pull
git checkout -b emergency-exit-edge-cases

git status
git diff

git add agents/strategies/trigger_engine.py tests/test_trigger_engine.py

uv run pytest tests/test_trigger_engine.py -vv

git commit -m "Emergency exit: edge case tests"
```

## Change Log (update during implementation)
- 2026-01-29: Expanded runbook format with scope, acceptance, test plan, and git workflow details.

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit)
