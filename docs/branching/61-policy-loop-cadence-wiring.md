# Runbook 61: Policy Loop Cadence Wiring

## Purpose

Wire `PolicyLoopGate` (R54) and `RegimeTransitionDetector` (R55) into the paper trading
main loop so the LLM strategist is called only at policy boundaries — regime transitions,
position open/close events, and heartbeat expiry — rather than on every evaluation cycle.

Also wire the `PolicyStateMachine` state transitions (IDLE → THESIS_ARMED →
POSITION_OPEN → HOLD_LOCK → COOLDOWN) at fill time and position close time.

**Pre-condition:** R54 and R55 are implemented (schemas, services, tests). This runbook
wires them into runtime execution. No schema changes to the services themselves.

## Scope

1. `tools/paper_trading.py`
   - `SessionState` — add `policy_state_machine_record`, `regime_detector_state`,
     `last_policy_eval_at` fields (all Optional, default None for backwards compat)
   - `PaperTradingWorkflow._generate_plan()` — gate on `PolicyLoopGate.evaluate()`
   - `PaperTradingWorkflow._eval_loop()` (or per-bar path) — call
     `RegimeTransitionDetector.evaluate()` per bar; collect trigger events
   - `PaperTradingWorkflow._execute_order()` or fill handler — transition state machine
     IDLE → THESIS_ARMED → POSITION_OPEN → HOLD_LOCK on fill
   - Position close handler (stop/target sweep) — call `state_machine.close_position()`
   - `_snapshot_state()` / `_restore_state()` — serialize/deserialize new SessionState fields

2. `tests/test_paper_trading_cadence.py` — new test file covering:
   - Gate blocks plan generation when state is HOLD_LOCK
   - Gate allows plan generation on regime transition trigger
   - Gate allows plan generation on heartbeat expiry
   - State machine transitions on fill and position close
   - Detector state persisted and restored across continue-as-new

## Out of Scope

- Changes to `PolicyLoopGate`, `PolicyStateMachine`, or `RegimeTransitionDetector` logic
- Backtest runner wiring (covered by R67)
- High-level reflection cadence (Layer 3 — separate concern)

## Implementation Steps

### Step 1: Extend SessionState

Add to `SessionState` in `tools/paper_trading.py`:

```python
# Policy loop cadence (Runbook 61)
policy_state_machine_record: Optional[Dict[str, Any]] = None
regime_detector_state: Optional[Dict[str, Any]] = None
last_policy_eval_at: Optional[str] = None  # ISO datetime string
```

These are dicts (not typed models) so they survive continue-as-new JSON round-trips
without importing the service schemas into paper_trading.py directly.

### Step 2: Per-bar regime detection

In the main evaluation loop, after `fetch_indicator_snapshots_activity` returns and
before `generate_strategy_plan_activity` is considered:

```python
from services.regime_transition_detector import (
    RegimeTransitionDetector, RegimeTransitionDetectorState,
    build_regime_fingerprint,
)
from schemas.reasoning_cadence import get_cadence_config

# Restore or create detector
detector = RegimeTransitionDetector(symbol=primary_symbol)
if self.state.regime_detector_state:
    detector.load_state(
        RegimeTransitionDetectorState.model_validate(self.state.regime_detector_state)
    )

fingerprint = build_regime_fingerprint(indicator_snapshot, asset_state)
transition_event = detector.evaluate(fingerprint, current_ts=bar_ts)

# Persist updated detector state
self.state.regime_detector_state = detector.state.model_dump()

# Collect policy trigger events
policy_triggers: list[PolicyLoopTriggerEvent] = []
if transition_event.fired:
    policy_triggers.append(PolicyLoopTriggerEvent(kind="regime_state_changed"))
if position_opened_this_bar:
    policy_triggers.append(PolicyLoopTriggerEvent(kind="position_opened"))
if position_closed_this_bar:
    policy_triggers.append(PolicyLoopTriggerEvent(kind="position_closed"))
```

### Step 3: Gate plan generation

Replace the direct call to `generate_strategy_plan_activity` with a gate check:

```python
from services.policy_loop_gate import PolicyLoopGate
from schemas.policy_state import PolicyStateMachineRecord

gate = PolicyLoopGate()
state_record = PolicyStateMachineRecord.model_validate(
    self.state.policy_state_machine_record or {}
)
last_eval_at = (
    datetime.fromisoformat(self.state.last_policy_eval_at)
    if self.state.last_policy_eval_at else None
)

allowed, skip_event = gate.evaluate(
    scope=self.session_id,
    state_record=state_record,
    trigger_events=policy_triggers,
    last_eval_at=last_eval_at,
    indicator_timeframe=self.state.indicator_timeframe,
)

if not allowed:
    # Log skip reason and continue using existing plan
    if skip_event:
        await workflow.execute_activity(
            emit_paper_trading_event_activity,
            args=[self.session_id, "policy_loop_skipped", skip_event.model_dump()],
            schedule_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
    # Skip plan generation — use current_plan as-is
else:
    try:
        plan_dict = await workflow.execute_activity(
            generate_strategy_plan_activity,
            args=[...],
            ...
        )
        self.state.last_policy_eval_at = datetime.utcnow().isoformat()
    finally:
        gate.release(self.session_id)
```

### Step 4: State machine transitions

**On fill** (inside `_execute_order` or fill handler, after ledger record):

```python
from services.policy_state_machine import PolicyStateMachine
from schemas.policy_state import PolicyStateMachineRecord

sm = PolicyStateMachine()
record = PolicyStateMachineRecord.model_validate(
    self.state.policy_state_machine_record or {}
)
record, _ = sm.activate_position(record, position_id=fill_order_id)
record = sm.lock_hold(record)
self.state.policy_state_machine_record = record.model_dump()
```

**On position close** (stop sweep / target sweep):

```python
sm = PolicyStateMachine()
record = PolicyStateMachineRecord.model_validate(
    self.state.policy_state_machine_record or {}
)
record = sm.close_position(record)
self.state.policy_state_machine_record = record.model_dump()
```

### Step 5: Serialize/restore in continue-as-new

Verify `_snapshot_state()` already copies all `SessionState` fields verbatim (dict-based
fields survive JSON round-trip automatically). No extra work if `SessionState` uses
`model_dump()` for serialization. Confirm in `_restore_state()` path.

### Step 6: Add `"policy_loop_skipped"` to EventType

Add `"policy_loop_skipped"` to the `EventType` Literal in `ops_api/schemas.py`.

## Acceptance Criteria

- [ ] `SessionState` has `policy_state_machine_record`, `regime_detector_state`,
  `last_policy_eval_at` (all Optional, default None)
- [ ] Per bar: `RegimeTransitionDetector.evaluate()` is called; state is persisted to
  `SessionState.regime_detector_state`
- [ ] On regime transition: `policy_triggers` includes `kind="regime_state_changed"`
- [ ] Gate blocks plan generation when policy state is `HOLD_LOCK`
- [ ] Gate allows plan generation when heartbeat expires (even with no trigger events)
- [ ] Gate allows plan generation on regime transition trigger
- [ ] `gate.release()` is always called (even on exception — use try/finally)
- [ ] On fill: state machine transitions to POSITION_OPEN → HOLD_LOCK
- [ ] On position close: state machine transitions to COOLDOWN
- [ ] State machine record serialized to `SessionState` and survives continue-as-new
- [ ] `"policy_loop_skipped"` added to `EventType` Literal
- [ ] All existing paper trading tests still pass

## Test Plan

```bash
# New cadence wiring tests
uv run pytest tests/test_paper_trading_cadence.py -vv

# Regression: existing paper trading tests
uv run pytest tests/test_paper_trading.py -vv

# Regression: full suite
uv run pytest -x -q
```

## Human Verification Evidence

```text
Verified via unit tests (2026-03-02):
1. test_gate_blocks_hold_lock — confirms policy_frozen_hold_lock skip emitted
2. test_gate_allows_on_regime_transition — confirms gate allows on regime_state_changed
3. test_gate_allows_on_heartbeat_expiry — confirms gate allows when heartbeat expires
4. test_state_machine_fill_transitions — IDLE → THESIS_ARMED → POSITION_OPEN → HOLD_LOCK
5. test_state_machine_close_position — HOLD_LOCK → COOLDOWN on position close
6. test_session_state_roundtrip_cadence — policy_state_machine_record survives model_dump round-trip

Runtime verification pending: requires live Temporal session.
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-01 | Runbook created — policy loop cadence wiring (R61) | Claude |
| 2026-03-02 | Implemented: SessionState fields, regime detection, gate in _generate_plan(), state machine transitions in _execute_order() and _sweep_stop_target(), snapshot/restore, EventType, 13 tests | Claude |

## Test Evidence

```text
=== New cadence tests ===
uv run pytest tests/test_paper_trading_cadence.py -vv

tests/test_paper_trading_cadence.py::test_gate_blocks_hold_lock PASSED
tests/test_paper_trading_cadence.py::test_gate_blocks_thesis_armed_no_trigger PASSED
tests/test_paper_trading_cadence.py::test_gate_allows_on_regime_transition PASSED
tests/test_paper_trading_cadence.py::test_gate_allows_on_heartbeat_expiry PASSED
tests/test_paper_trading_cadence.py::test_gate_blocks_when_no_trigger_and_heartbeat_not_expired PASSED
tests/test_paper_trading_cadence.py::test_state_machine_fill_transitions PASSED
tests/test_paper_trading_cadence.py::test_state_machine_fill_from_thesis_armed PASSED
tests/test_paper_trading_cadence.py::test_state_machine_close_position PASSED
tests/test_paper_trading_cadence.py::test_state_machine_close_from_position_open PASSED
tests/test_paper_trading_cadence.py::test_session_state_has_cadence_fields PASSED
tests/test_paper_trading_cadence.py::test_session_state_roundtrip_cadence PASSED
tests/test_paper_trading_cadence.py::test_session_state_none_cadence_fields_survive_roundtrip PASSED
tests/test_paper_trading_cadence.py::test_gate_release_frees_scope PASSED
13 passed in 4.57s

=== Regression suite (excluding pre-existing failures) ===
uv run pytest -q --ignore=tests/integration --ignore=tests/reports --ignore=tests/risk
4 failed, 2040 passed, 2 skipped in 409.65s
(4 failures are pre-existing on main: test_exit_direction_not_blocked,
test_simple_plan_executes_one_trade, test_logs_risk_block_details,
test_daily_risk_budget_blocks_orders — confirmed failing on main before this branch)
```

## Worktree Setup

```bash
git worktree add -b feat/r61-policy-loop-wiring ../wt-r61-policy-loop main
cd ../wt-r61-policy-loop
```

## Git Workflow

```bash
git checkout -b feat/r61-policy-loop-wiring

git add tools/paper_trading.py \
        ops_api/schemas.py \
        tests/test_paper_trading_cadence.py \
        docs/branching/61-policy-loop-cadence-wiring.md \
        docs/branching/README.md

git commit -m "feat: wire PolicyLoopGate and RegimeTransitionDetector into paper trading loop (R61)"
```
