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
- 2026-01-30: Implemented bypass/override semantics. Fixed execution engine bug where emergency exits bypassed judge `disabled_categories` — now judge constraints apply to all triggers including emergency exits. Added 5 new tests across 3 test files.

### Files changed
- `trading_core/execution_engine.py`: Removed `not is_emergency_exit` guard on judge constraints check so `disabled_categories=["emergency_exit"]` blocks emergency exits.
- `tests/test_trigger_engine.py`: Added `test_emergency_exit_bypasses_hold_rule`, `test_regular_exit_respects_hold_rule`, `test_judge_disabled_category_blocks_emergency_exit`.
- `tests/risk/test_exit_bypass.py`: Added `test_emergency_exit_bypasses_risk_budget`, `test_regular_entry_blocked_by_zero_risk_budget`.
- `tests/test_execution_engine.py`: Added `test_judge_disabled_category_blocks_emergency_exit_in_execution_engine`, `test_emergency_exit_still_bypasses_daily_cap_when_category_not_disabled`.

## Test Evidence (append results before commit)

```
tests/test_trigger_engine.py: 10 passed in 0.60s
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

tests/risk/test_exit_bypass.py: 4 passed in 0.40s
  test_exit_does_nothing_when_flat PASSED
  test_exit_flattens_when_position_exists PASSED
  test_emergency_exit_bypasses_risk_budget PASSED
  test_regular_entry_blocked_by_zero_risk_budget PASSED

tests/test_execution_engine.py: 9 passed in 4.82s
  test_simulate_day_enforces_judge_trade_cap PASSED
  test_simulate_day_respects_judge_disabled_category PASSED
  test_run_live_step_accumulates_day_state PASSED
  test_session_trade_multipliers_scale_limits PASSED
  test_emergency_exit_bypasses_daily_cap PASSED
  test_symbol_trigger_budget_enforced PASSED
  test_timeframe_trigger_cap_enforced PASSED
  test_judge_disabled_category_blocks_emergency_exit_in_execution_engine PASSED
  test_emergency_exit_still_bypasses_daily_cap_when_category_not_disabled PASSED
```

## Human Verification Evidence (append results before commit)

### Hold-rule bypass (trigger_engine.py:419)
- Confirmed: `if trigger.hold_rule and trigger.category != "emergency_exit":` — emergency exits skip hold_rule evaluation entirely.
- Regular exits with active hold_rule produce a `HOLD_RULE` block entry and are suppressed.
- Emergency exits with active hold_rule produce no `HOLD_RULE` block and fire normally.

### Judge category disabling (trigger_engine.py:357-364, execution_engine.py:232-252)
- **Bug fixed:** `execution_engine.py` previously wrapped judge constraints in `if constraints and not is_emergency_exit:`, causing emergency exits to bypass `disabled_categories`. Changed to `if constraints:` so judge constraints apply to all triggers.
- `trigger_engine.py` already applied judge constraints to all triggers (no emergency exemption) — no change needed.
- Block entries for judge category disables show `reason: "category"` with detail mentioning the disabled category name.
