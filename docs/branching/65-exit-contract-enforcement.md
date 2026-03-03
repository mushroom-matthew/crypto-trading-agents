# Runbook 65: Exit Contract Enforcement

## Purpose

Fix `exit_binding_mismatch` — the most significant source of backtest trading loss
identified in backtest `dad709b8`. When the judge replans mid-trade, the new plan's exit
triggers fire against the old open position, closing it at wrong levels.

**Root cause:** Open positions do not record which plan generated them. After a replan,
the new plan's exit triggers are applied to all open positions indiscriminately.

**Fix:** At fill time, record `originating_plan_id` in `position_meta`. In trigger
evaluation, the trigger engine skips exit triggers from a plan whose `plan_id` does not
match the position's `originating_plan_id`.

This implements the pinning contract described in R60 (`PositionExitContract`) without
requiring the full contract schema — the `originating_plan_id` field is sufficient to
prevent cross-plan exit binding.

**Pre-condition:** `position_meta` dict exists and is accessible during trigger
evaluation. `StrategyPlan` has `plan_id` field.

## Scope

1. `tools/paper_trading.py`
   - `_execute_order()` / fill handler — store `originating_plan_id` in `position_meta`
   - `evaluate_triggers_activity` — pass `position_originating_plans: Dict[str, str]`
     (symbol → plan_id) to trigger engine

2. `agents/strategies/trigger_engine.py`
   - Accept `position_originating_plans: Dict[str, str] | None = None`
   - In exit trigger evaluation: skip if `current_plan_id != position_originating_plans[symbol]`
   - Emit `exit_binding_mismatch_blocked` telemetry counter when skipped

3. `SessionState` — add `position_originating_plans: Dict[str, str] = {}` to track
   per-symbol originating plan IDs

4. `tests/test_exit_contract_enforcement.py` — new test file

## Out of Scope

- Full `PositionExitContract` schema enforcement (R60's remaining phases)
- Replan-on-position-close logic (position stays open, only exit binding is enforced)
- Emergency exits (`category="emergency_exit"`) — these ALWAYS bypass pinning per
  domain invariant

## Key Design Decisions

**Emergency exits bypass enforcement** (existing invariant): if `trigger.category ==
"emergency_exit"`, always fire regardless of `originating_plan_id`. Emergency exits are
safety mechanisms, not strategy exits.

**Flatten actions bypass enforcement**: if `action == "flatten"`, bypass pinning (these
are protective).

**New plan's entry triggers are unaffected**: only exit triggers are pinned. New entries
from a replanned plan are allowed normally.

**What "same plan" means**: when the judge replans, the new plan gets a new `plan_id`.
Positions opened under the old plan_id should only close via the old plan's exit rules.
Positions opened after the replan use the new plan_id.

## Implementation Steps

### Step 1: Add originating_plan_ids to SessionState

```python
# Exit contract enforcement: tracks which plan opened each position (Runbook 65)
position_originating_plans: Dict[str, str] = {}  # symbol → plan_id
```

### Step 2: Record originating_plan_id at fill time

In `_execute_order()`, after the fill is recorded:

```python
plan_id = self.state.current_plan.get("plan_id") if self.state.current_plan else None
if plan_id and symbol:
    self.state.position_originating_plans[symbol] = plan_id
```

On position close (stop/target sweep):

```python
# Clear originating plan when position closes
self.state.position_originating_plans.pop(symbol, None)
```

### Step 3: Pass to evaluate_triggers_activity

In the call to `evaluate_triggers_activity`, add:

```python
position_originating_plans=self.state.position_originating_plans,
current_plan_id=self.state.current_plan.get("plan_id") if self.state.current_plan else None,
```

Update `evaluate_triggers_activity` signature:

```python
async def evaluate_triggers_activity(
    plan_dict: Dict[str, Any],
    market_data: Dict[str, Dict[str, Any]],
    portfolio_state: Dict[str, Any],
    exit_binding_mode: str = "category",
    conflicting_signal_policy: str = "reverse",
    position_originating_plans: Dict[str, str] | None = None,  # NEW
) -> Dict[str, Any]:
```

### Step 4: Enforce in trigger engine

In `trigger_engine.py`, in the exit trigger evaluation path:

```python
def _should_apply_exit_trigger(
    self,
    trigger: TriggerCondition,
    symbol: str,
    current_plan_id: str | None,
    position_originating_plans: dict[str, str] | None,
) -> bool:
    """Return False if exit trigger belongs to a different plan than opened the position."""
    # Emergency exits always apply
    if trigger.category == "emergency_exit":
        return True
    # Flatten actions always apply
    if trigger.direction not in ("long", "short"):
        return True
    # No position plan tracking available — allow (degrade gracefully)
    if position_originating_plans is None or current_plan_id is None:
        return True

    originating_plan_id = position_originating_plans.get(symbol)
    if originating_plan_id is None:
        return True  # No recorded origin — allow

    if current_plan_id != originating_plan_id:
        # Exit trigger is from a different plan than opened this position — BLOCK
        self._exit_binding_mismatch_count += 1
        return False

    return True
```

Emit telemetry: add `exit_binding_mismatch_blocked: int` counter to the trigger engine
result dict.

### Step 5: Telemetry in response

In the dict returned by `evaluate_triggers_activity`, add:

```python
"exit_binding_mismatch_blocked": trigger_engine.exit_binding_mismatch_count,
```

Emit as a `"trade_blocked"` event with `reason="exit_binding_mismatch"` when count > 0.

## Acceptance Criteria

- [ ] `SessionState.position_originating_plans` tracks symbol → plan_id at fill time
- [ ] On position close, the symbol is removed from `position_originating_plans`
- [ ] `evaluate_triggers_activity` accepts `position_originating_plans` parameter
- [ ] Trigger engine skips exit triggers whose `current_plan_id` ≠ position's
  `originating_plan_id`
- [ ] Emergency exits (`category="emergency_exit"`) always bypass the pinning check
- [ ] Flatten actions (`direction` not long/short) always bypass the pinning check
- [ ] `exit_binding_mismatch_blocked` counter returned in activity result
- [ ] All existing paper trading and trigger engine tests pass

## Test Plan

```bash
uv run pytest tests/test_exit_contract_enforcement.py -vv
uv run pytest tests/test_trigger_engine.py -vv

# Regression
uv run pytest -x -q
```

## Human Verification Evidence

```text
1. test_on_bar_blocks_exit_from_replanned_plan: TriggerEngine.on_bar() returns zero orders
   and one block with reason=exit_binding_plan_mismatch when current plan_id != originating
   plan_id. Counter increments to 1. ✓

2. test_on_bar_emergency_exit_bypasses_plan_pinning: emergency_exit category trigger fires
   orders=[Order(side='sell')] even when plan_id mismatches. Counter stays 0. ✓

3. test_evaluate_triggers_activity_returns_mismatch_blocked_key: evaluate_triggers_activity
   returns exit_binding_mismatch_blocked >= 1 when exit_rule=True fires with plan mismatch.
   trade_blocked event emitted with reason=exit_binding_mismatch. ✓
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-01 | Runbook created — exit contract enforcement via originating_plan_id pinning (R65) | Claude |
| 2026-03-02 | Implemented R65: position_originating_plans in SessionState + workflow __init__/_snapshot_state/_restore_state; originating_plan_id recorded at entry fill, cleared at exit fill in _execute_order; evaluate_triggers_activity accepts position_originating_plans param and returns exit_binding_mismatch_blocked; TriggerEngine.__init__ accepts position_originating_plans, _should_apply_exit_trigger method, exit_binding_mismatch_count property, on_bar enforcement after _exit_binding_allows; 21 new tests all pass | Claude |

## Test Evidence

```text
$ uv run pytest tests/test_exit_contract_enforcement.py -vv
collected 21 items — all 21 passed

$ uv run pytest tests/test_trigger_engine.py -vv
collected 41 items — all 41 passed

$ uv run pytest -q (regression, excluding 4 pre-existing DB_DSN collection errors in worktree)
2125 passed, 2 skipped — no regressions
(2104 from main + 21 new R65 tests)
```

## Worktree Setup

```bash
git worktree add -b feat/r65-exit-contract ../wt-r65-exit-contract main
cd ../wt-r65-exit-contract
```

## Git Workflow

```bash
git checkout -b feat/r65-exit-contract

git add tools/paper_trading.py \
        agents/strategies/trigger_engine.py \
        tests/test_exit_contract_enforcement.py \
        docs/branching/65-exit-contract-enforcement.md \
        docs/branching/README.md

git commit -m "feat: enforce exit contract pinning via originating_plan_id to fix exit_binding_mismatch (R65)"
```
