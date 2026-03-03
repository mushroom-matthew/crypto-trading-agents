# Runbook 66: Judge Validation Gate

## Purpose

Wire `JudgePlanValidationService.validate_plan()` (R53) as a deterministic gate
immediately after plan generation. If the verdict is REJECT, trigger the revision loop
(`JudgePlanRevisionLoopOrchestrator`, max 2 revisions). If still failing, emit a
stand-down for the current cycle.

Currently `JudgePlanValidationService` is implemented and tested but is only called from
`judge_revision_loop.py` — never directly from the execution path. Plans can pass
validation in isolation but still be structurally problematic (empty triggers, wrong
playbook, contradicted by memory evidence).

**Pre-condition:** R53 implemented. R62 (playbook-first) should be done first so that
`playbook_regime_tags` and memory bundle are available for validation.

## Scope

1. `tools/paper_trading.py`
   - `generate_strategy_plan_activity` — after plan dict is validated as `StrategyPlan`,
     call `JudgePlanValidationService.validate_plan()`; handle REJECT by requesting
     revision; if revision exhausted, emit stand-down event and return None
   - `PaperTradingWorkflow._generate_plan()` — handle `None` return from activity
     (stand-down: skip this cycle, do not post trades until next plan generation)

2. `services/judge_revision_loop.py` — confirm `JudgePlanRevisionLoopOrchestrator` can
   be called synchronously from an activity context (it makes no LLM calls itself)

3. `ops_api/schemas.py` — add `"plan_validation_rejected"` and `"plan_stand_down"` to
   `EventType` Literal

4. `tests/test_judge_validation_gate.py` — new test file

## Out of Scope

- Changing `JudgePlanValidationService` logic
- Changing `JudgePlanRevisionLoopOrchestrator` revision logic
- Backtest runner wiring (covered by R67)

## Implementation Steps

### Step 1: Call validation after plan generation

In `generate_strategy_plan_activity`, after constructing `plan` (a `StrategyPlan`):

```python
from services.judge_validation_service import JudgePlanValidationService
from schemas.judge_feedback import JudgeValidationVerdict

validator = JudgePlanValidationService()

# Build validation context from plan and memory bundle
is_hold_lock = policy_state.get("kind") == "HOLD_LOCK" if policy_state else False
is_thesis_armed = policy_state.get("kind") == "THESIS_ARMED" if policy_state else False

verdict: JudgeValidationVerdict = validator.validate_plan(
    plan,
    memory_bundle=memory_bundle,  # from MemoryRetrievalService if available
    playbook_regime_tags=eligible_regime_tags,  # from PlaybookRegistry R62
    is_thesis_armed=is_thesis_armed,
    is_hold_lock=is_hold_lock,
)

if verdict.decision == "reject":
    activity.logger.warning(
        "Plan validation rejected: %s", verdict.reason
    )
    # Attempt revision
    from services.judge_revision_loop import JudgePlanRevisionLoopOrchestrator
    loop = JudgePlanRevisionLoopOrchestrator(max_revisions=2)
    revision_result = await loop.run(
        plan=plan,
        verdict=verdict,
        revision_callback=_plan_revision_callback,  # calls LLM again
    )
    if revision_result.revision_budget_exhausted or not revision_result.final_plan:
        # Stand-down: no plan this cycle
        return None
    plan = revision_result.final_plan

return plan.model_dump()
```

### Step 2: Handle None plan in workflow

In `PaperTradingWorkflow._generate_plan()`:

```python
plan_dict = await workflow.execute_activity(
    generate_strategy_plan_activity,
    args=[...],
    ...
)

if plan_dict is None:
    # Plan validation failed after revision attempts — stand down this cycle
    await workflow.execute_activity(
        emit_paper_trading_event_activity,
        args=[self.session_id, "plan_stand_down", {
            "cycle": self.state.cycle_count,
            "reason": "validation_exhausted",
        }],
        schedule_to_close_timeout=timedelta(seconds=10),
        retry_policy=RetryPolicy(maximum_attempts=1),
    )
    return  # Skip this cycle — keep existing plan, no new trades
```

### Step 3: Add revision_callback

The `revision_callback` is a callable that takes a `JudgePlanRevisionRequest` and
returns a revised `StrategyPlan`. Implement as an async closure that re-calls
`llm_client.generate_plan()` with `repair_instructions` from the revision request.

```python
async def _plan_revision_callback(
    revision_request: JudgePlanRevisionRequest,
) -> StrategyPlan | None:
    """Re-call LLM with revision instructions."""
    revised_dict = await _call_llm_with_repair(
        repair_instructions=revision_request.revision_instructions,
        failing_criteria=revision_request.failing_criteria,
    )
    if revised_dict is None:
        return None
    try:
        return StrategyPlan.model_validate(revised_dict)
    except Exception:
        return None
```

### Step 4: Add new EventTypes

In `ops_api/schemas.py`:

```python
EventType = Literal[
    ...,
    "plan_validation_rejected",
    "plan_stand_down",
]
```

## Acceptance Criteria

- [ ] `JudgePlanValidationService.validate_plan()` called after every fresh plan generation
- [ ] REJECT verdict triggers revision loop (max 2 revisions)
- [ ] If revision loop exhausted, activity returns None
- [ ] Workflow treats None plan as stand-down — no trades, no crash
- [ ] `"plan_validation_rejected"` and `"plan_stand_down"` events emitted to event stream
- [ ] Emergency exits (`category="emergency_exit"`) in existing triggers are never
  affected by plan stand-down (existing positions continue executing stop/target rules)
- [ ] APPROVE verdict proceeds normally with no additional latency
- [ ] All existing paper trading tests still pass

## Test Plan

```bash
uv run pytest tests/test_judge_validation_gate.py -vv
uv run pytest tests/test_judge_revision_loop.py -vv

# Regression
uv run pytest -x -q
```

## Human Verification Evidence

```text
1. Empty-trigger plan → JudgePlanValidationService.validate_plan() returns REJECT with
   "STRUCTURAL: plan has no triggers" reason (confirmed in test_reject_empty_triggers).
2. Revision loop invoked with max_revisions=2; callback is a synchronous closure that
   re-calls llm_client.generate_plan() with failing_criteria from the revision request.
3. If revision loop exhausted (accepted_plan_id=None), activity returns None.
   Workflow handles None by emitting plan_validation_rejected + plan_stand_down events
   and returning early without setting self.current_plan. Existing plan (with emergency
   exits) stays active for open positions.
4. Valid plan (triggers present, IDLE state, no policy blocks) → verdict=approve,
   gate exits without revision, plan returned normally.
5. THESIS_ARMED and HOLD_LOCK states correctly derived from policy_state_machine_record
   dict via current_state field (confirmed in parametrised test_flag_derivation tests).
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-01 | Runbook created — judge validation gate wiring (R66) | Claude |
| 2026-03-02 | Implemented: validation gate in activity + stand-down handling in workflow + EventTypes + 26 tests | Claude |

## Test Evidence

```text
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
collected 26 items

tests/test_judge_validation_gate.py::TestJudgePlanValidationServiceBasic::test_approve_valid_plan PASSED
tests/test_judge_validation_gate.py::TestJudgePlanValidationServiceBasic::test_reject_empty_triggers PASSED
tests/test_judge_validation_gate.py::TestJudgePlanValidationServiceBasic::test_reject_thesis_armed_without_override PASSED
tests/test_judge_validation_gate.py::TestJudgePlanValidationServiceBasic::test_approve_thesis_armed_with_safety_override PASSED
tests/test_judge_validation_gate.py::TestJudgePlanValidationServiceBasic::test_reject_regime_not_in_playbook_tags PASSED
tests/test_judge_validation_gate.py::TestJudgePlanValidationServiceBasic::test_approve_when_regime_in_playbook_tags PASSED
tests/test_judge_validation_gate.py::TestJudgePlanRevisionLoopGate::test_approve_path_no_revision PASSED
tests/test_judge_validation_gate.py::TestJudgePlanRevisionLoopGate::test_reject_path_returns_none_accepted PASSED
tests/test_judge_validation_gate.py::TestJudgePlanRevisionLoopGate::test_revision_succeeds_on_second_attempt PASSED
tests/test_judge_validation_gate.py::TestJudgePlanRevisionLoopGate::test_budget_exhausted_returns_stand_down PASSED
tests/test_judge_validation_gate.py::TestJudgePlanRevisionLoopGate::test_callback_none_triggers_stand_down PASSED
tests/test_judge_validation_gate.py::TestPolicyStateFlagDerivation::test_flag_derivation[psm0-True-False] PASSED
tests/test_judge_validation_gate.py::TestPolicyStateFlagDerivation::test_flag_derivation[psm1-False-True] PASSED
tests/test_judge_validation_gate.py::TestPolicyStateFlagDerivation::test_flag_derivation[psm2-False-False] PASSED
tests/test_judge_validation_gate.py::TestPolicyStateFlagDerivation::test_flag_derivation[psm3-False-False] PASSED
tests/test_judge_validation_gate.py::TestPolicyStateFlagDerivation::test_flag_derivation[psm4-False-False] PASSED
tests/test_judge_validation_gate.py::TestPolicyStateFlagDerivation::test_thesis_armed_state_triggers_reject PASSED
tests/test_judge_validation_gate.py::TestPolicyStateFlagDerivation::test_hold_lock_state_triggers_reject PASSED
tests/test_judge_validation_gate.py::TestEventTypeLiteral::test_plan_validation_rejected_accepted PASSED
tests/test_judge_validation_gate.py::TestEventTypeLiteral::test_plan_stand_down_accepted PASSED
tests/test_judge_validation_gate.py::TestEventTypeLiteral::test_invalid_event_type_rejected PASSED
tests/test_judge_validation_gate.py::TestEventTypeLiteral::test_existing_event_types_still_valid PASSED
tests/test_judge_validation_gate.py::TestWorkflowStandDownPath::test_none_accepted_plan_id_triggers_stand_down PASSED
tests/test_judge_validation_gate.py::TestWorkflowStandDownPath::test_set_accepted_plan_id_proceeds_normally PASSED
tests/test_judge_validation_gate.py::TestWorkflowStandDownPath::test_reject_verdict_also_produces_none_accepted_plan PASSED
tests/test_judge_validation_gate.py::TestWorkflowStandDownPath::test_emergency_exit_triggers_not_affected_by_stand_down_concept PASSED

26 passed in 5.01s

Regression (excluding pre-existing DB_DSN collection errors and venv-level
pandas version mismatch in test_factor_loader):
2151 passed, 2 skipped in 95.77s
```

## Worktree Setup

```bash
git worktree add -b feat/r66-judge-validation ../wt-r66-judge-validation main
cd ../wt-r66-judge-validation
```

## Git Workflow

```bash
git checkout -b feat/r66-judge-validation

git add tools/paper_trading.py \
        ops_api/schemas.py \
        tests/test_judge_validation_gate.py \
        docs/branching/66-judge-validation-gate.md \
        docs/branching/README.md

git commit -m "feat: wire JudgePlanValidationService as execution gate with revision loop (R66)"
```
