# Runbook: Emergency Exit Bypass and Override

## Overview
Emergency exits have special bypass behavior (hold-rule ignore, risk checks, judge overrides). This runbook locks down the intended bypass semantics with explicit tests and ensures judge category disabling works for `emergency_exit`.

**Source:** backtest `backtest-d94320c0`.

## Scope
1. **Hold-rule bypass:** Emergency exits should ignore `hold_rule` while regular exits respect it.
2. **Risk budget bypass:** Define and test whether emergency exits bypass risk budget checks or consume budget.
3. **Judge category disabling:** `disabled_categories=["emergency_exit"]` must block emergency exits with an explicit veto record.

## Key Files
- `agents/strategies/trigger_engine.py` (hold-rule bypass, judge constraints)
- `agents/strategies/trade_risk.py` (risk evaluation semantics)
- `trading_core/execution_engine.py` (execution gating, emergency exit handling)
- `tests/test_trigger_engine.py` (trigger engine unit coverage)
- `tests/risk/test_exit_bypass.py` (risk bypass invariants)
- `tests/test_execution_engine.py` (execution-level bypass checks)

## Acceptance Criteria
- Emergency exits bypass `hold_rule`, while non-emergency exits still respect it.
- Emergency exits either bypass risk budgets or consume budget, with tests documenting the chosen behavior.
- Judge category disabling blocks emergency exits and records a block with reason `CATEGORY`.

## Out of Scope
- Min-hold and cooldown metadata enforcement (covered in runbook 04).
- Missing `exit_rule` handling (covered in runbook 06).
- Judge/strategist design changes (tracked separately).

## Test Plan (required before commit)
```bash
uv run pytest tests/test_trigger_engine.py -vv
uv run pytest tests/risk/test_exit_bypass.py -vv
uv run pytest tests/test_execution_engine.py -vv
```

If tests cannot run locally, obtain user-run output and paste it into the Test Evidence section before committing.

## Human Verification (required)
- Inspect block entries for judge category disables to confirm category-level veto is surfaced.
- Confirm hold-rule bypass is only applied to emergency exits.

## Worktree Setup (recommended for parallel agents)
```bash
git fetch
git worktree add -b emergency-exit-bypass-override ../wt-emergency-exit-bypass-override main
cd ../wt-emergency-exit-bypass-override

# When finished
git worktree remove ../wt-emergency-exit-bypass-override
```

## Git Workflow (explicit)
```bash
git checkout main
git pull
git checkout -b emergency-exit-bypass-override

git status
git diff

git add agents/strategies/trigger_engine.py \
  agents/strategies/trade_risk.py \
  trading_core/execution_engine.py \
  tests/test_trigger_engine.py \
  tests/risk/test_exit_bypass.py \
  tests/test_execution_engine.py

uv run pytest tests/test_trigger_engine.py -vv
uv run pytest tests/risk/test_exit_bypass.py -vv
uv run pytest tests/test_execution_engine.py -vv

git commit -m "Emergency exit: bypass and override tests"
```

## Change Log (update during implementation)
- 2026-01-29: Expanded runbook format with scope, acceptance, test plan, and git workflow details.

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit)
