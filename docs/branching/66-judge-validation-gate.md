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
[To be filled after implementation]
1. Construct a plan with an empty triggers list. Confirm plan_validation_rejected event
   appears in the event stream.
2. Confirm revision attempt fires (second LLM call visible in Langfuse).
3. After max revisions exhausted, confirm plan_stand_down event emitted and no trades
   are posted that cycle.
4. Confirm a valid plan passes validation without emitting rejection events.
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-01 | Runbook created — judge validation gate wiring (R66) | Claude |

## Test Evidence

```text
[Paste test output here before committing]
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
