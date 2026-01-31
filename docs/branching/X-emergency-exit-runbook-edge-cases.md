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
- 2026-01-30: Added 5 tests covering missing exit_rule edge cases: 3 schema-level rejection tests (None, empty, whitespace) and 2 runtime defense-in-depth tests (empty, whitespace bypassing Pydantic).

### Files changed
- `tests/test_trigger_engine.py`: Added `test_emergency_exit_missing_exit_rule_rejected_by_schema`, `test_emergency_exit_none_exit_rule_rejected_by_schema`, `test_emergency_exit_missing_exit_rule_runtime_defense`.

## Test Evidence (append results before commit)

```
tests/test_trigger_engine.py: 15 passed in 0.95s
  test_trigger_engine_records_block_when_risk_denies_entry PASSED
  test_emergency_exit_trigger_bypasses_risk_checks PASSED
  test_emergency_exit_vetoes_same_bar_entry PASSED
  test_emergency_exit_vetoes_min_hold_on_next_bar PASSED
  test_emergency_exit_min_hold_allows_on_threshold_bar PASSED
  test_emergency_exit_dedup_overrides_high_conf_entry PASSED
  test_emergency_exit_dedup_wins_even_with_permissive_risk PASSED
  test_emergency_exit_bypasses_hold_rule PASSED
  test_regular_exit_respects_hold_rule PASSED
  test_judge_disabled_category_blocks_emergency_exit PASSED
  test_emergency_exit_missing_exit_rule_rejected_by_schema[empty] PASSED
  test_emergency_exit_missing_exit_rule_rejected_by_schema[whitespace] PASSED
  test_emergency_exit_none_exit_rule_rejected_by_schema PASSED
  test_emergency_exit_missing_exit_rule_runtime_defense[empty] PASSED
  test_emergency_exit_missing_exit_rule_runtime_defense[whitespace] PASSED
```

## Human Verification Evidence (append results before commit)

### Missing exit_rule block entry inspection
- Schema-level: Pydantic rejects `TriggerCondition(category="emergency_exit", exit_rule="")` and `exit_rule=None` with validation errors. This prevents invalid triggers from being constructed normally.
- Runtime defense: When bypassing Pydantic via `model_construct`, the trigger engine at `trigger_engine.py:379` catches `not (trigger.exit_rule or "").strip()` and produces a block with:
  - `reason: "emergency_exit_missing_exit_rule"`
  - `cooldown_recommendation_bars` present, equal to `max(1, trade_cooldown_bars, min_hold_bars)`
- No exceptions raised in any case — the engine handles all variants safely.
