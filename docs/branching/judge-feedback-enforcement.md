# Branch: judge-feedback-enforcement

## Purpose
Ensure judge feedback recommendations are actually enforced by the execution engine, not just passed as LLM context that can be ignored.

## Problem Statement
Investigation (2026-01-27) found critical gaps in judge→strategist handoff:

### Gap 1: Trigger Vetoes Not Enforced
- Judge generates `disabled_trigger_ids` and `disabled_categories` in `JudgeConstraints`
- These are stored but **never propagated to execution engine**
- Triggers the judge said to disable can still fire

### Gap 2: Strategist Constraints Are Context-Only
- `must_fix`, `vetoes`, `boost`, `regime_correction` are passed to LLM prompt
- LLM has **no obligation** to follow them
- No parsing or enforcement mechanism

### Gap 3: Risk Mode Not Used
- `risk_mode: Literal["normal", "conservative", "emergency"]` is set but **never consumed**
- No code path applies "emergency" mode constraints

### Gap 4: Limited Sizing Adjustment Parsing
- `sizing_adjustments` uses fragile regex ("cut risk by 25%")
- Variations may not parse correctly

## Scope
- Propagate `disabled_trigger_ids` to trigger engine (block disabled triggers from firing)
- Propagate `disabled_categories` to trigger engine (block categories)
- Parse `must_fix`/`vetoes`/`boost` into machine-enforceable constraints where possible
- Implement `risk_mode` emergency behavior (tighten all limits)
- Add audit trail logging when judge recommendations are followed/ignored

## Out of Scope / Deferred
- Judge unification (separate branch)
- LLM prompt changes
- UI changes

## Key Files
- services/risk_adjustment_service.py (apply_judge_risk_feedback)
- services/strategist_plan_service.py (_resolve_final_caps)
- agents/strategies/trigger_engine.py (add disabled check)
- trading_core/execution_engine.py (enforce disabled triggers)
- schemas/judge_feedback.py (may need schema updates)

## Current Flow (What Exists)
```
JudgeConstraints {
    max_trades_per_day      → ✓ Enforced in plan
    max_triggers_per_symbol → ✓ Enforced in plan
    symbol_risk_multipliers → ✓ Applied as adjustments
    disabled_trigger_ids    → ✗ NOT USED
    disabled_categories     → ✗ NOT USED
    risk_mode               → ✗ NOT USED
}
```

## Target Flow (After Fix)
```
JudgeConstraints {
    max_trades_per_day      → ✓ Enforced in plan
    max_triggers_per_symbol → ✓ Enforced in plan
    symbol_risk_multipliers → ✓ Applied as adjustments
    disabled_trigger_ids    → ✓ Block in trigger engine
    disabled_categories     → ✓ Block in trigger engine
    risk_mode               → ✓ Scale all limits when emergency
}
```

## Dependencies / Coordination
- Coordinate with judge-unification branch
- Avoid conflicts with comp-audit-trigger-cadence changes

## Acceptance Criteria
- Triggers in `disabled_trigger_ids` do not fire during backtest
- Categories in `disabled_categories` do not generate triggers
- `risk_mode = "emergency"` reduces max positions by 50% and tightens stops
- Audit log shows "Judge blocked trigger X (reason: disabled_trigger_ids)"

## Test Plan (required before commit)
- uv run pytest -k judge -vv
- uv run pytest -k trigger_engine -vv
- uv run pytest -k execution_engine -vv

## Human Verification (required)
- Run backtest with judge shim that sets `disabled_trigger_ids = ["trigger_1"]`
- Verify trigger_1 never fires
- Paste evidence below

## Git Workflow
```bash
git checkout main && git pull
git checkout -b judge-feedback-enforcement

# After implementation
git add services/risk_adjustment_service.py \
  services/strategist_plan_service.py \
  agents/strategies/trigger_engine.py \
  trading_core/execution_engine.py

uv run pytest -k judge -vv
git commit -m "Judge: enforce disabled triggers and risk mode"
```

## Change Log (update during implementation)

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)
