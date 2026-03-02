# Runbook 67: Backtest-Paper Trading Parity

## Purpose

Mirror all Phase 8 features from `tools/paper_trading.py` into
`backtesting/llm_strategist_runner.py` so backtest results are comparable to paper
trading results. Currently the backtest runner is missing several features that paper
trading has, making backtest vs. paper comparisons misleading.

**Prerequisite:** R61–R66 should be implemented and validated in paper trading first.
This runbook then ports the same wiring into the backtest runner.

## Gaps to Close (Ordered by Impact)

| Feature | Paper Trading | Backtest Runner |
|---------|--------------|-----------------|
| PolicyLoopGate (cadence gating) | ✅ (R61) | ❌ runs LLM every N bars |
| RegimeTransitionDetector | ✅ (R61) | ❌ |
| PlaybookRegistry.list_eligible | ✅ (R62) | ❌ |
| MemoryRetrievalService | ✅ (A5/R63) | ❌ |
| SetupEventGenerator | ✅ (R63) | ❌ |
| build_episode_record on close | ✅ (R63) | ❌ |
| build_tick_snapshot | ✅ (R64) | ❌ |
| StructuralTargetSelector | ✅ (R64) | ❌ |
| exit_binding_mismatch enforcement | ✅ (R65) | ❌ |
| JudgePlanValidationService | ✅ (R66) | ❌ |
| Directional bias hint | ✅ | ❌ |
| indicator_timeframe threading | ✅ (R57) | ❌ |

## Scope

1. `backtesting/llm_strategist_runner.py`
   - All wiring listed above
   - Add `episode_records: List[EpisodeMemoryRecord]` to `StrategistBacktestResult`
   - Add `exit_binding_mismatch_blocked: int` to result stats

2. `backtesting/activities.py`
   - Include `episode_records` in the persisted results payload (extends B1 from prior work)

3. `backtesting/persistence.py`
   - Include episode records in `_build_results_payload()`

4. `tests/test_backtest_parity.py` — new test file verifying that identical data through
   paper and backtest produces matching trigger counts, fill counts, episode counts

## Out of Scope

- Changing backtest engine performance characteristics
- Adding real DB writes from backtest (episode records are in-memory only in backtest)
- Backtest-specific features (e.g. look-ahead prevention) — unchanged

## Implementation Steps

### Step 1: PolicyLoopGate + RegimeTransitionDetector

In the backtest runner's per-bar loop, add:

```python
from services.regime_transition_detector import RegimeTransitionDetector, build_regime_fingerprint
from services.policy_loop_gate import PolicyLoopGate
from schemas.reasoning_cadence import get_cadence_config

# Initialize once per run (outside bar loop)
detector = RegimeTransitionDetector(symbol=self.symbol)
gate = PolicyLoopGate(config=get_cadence_config())
last_policy_eval_at: datetime | None = None
policy_state_record = PolicyStateMachineRecord()

# Per bar:
fingerprint = build_regime_fingerprint(indicator, asset_state)
transition_event = detector.evaluate(fingerprint, current_ts=bar_ts)

policy_triggers = []
if transition_event.fired:
    policy_triggers.append(PolicyLoopTriggerEvent(kind="regime_state_changed"))
if position_opened_this_bar:
    policy_triggers.append(PolicyLoopTriggerEvent(kind="position_opened"))
if position_closed_this_bar:
    policy_triggers.append(PolicyLoopTriggerEvent(kind="position_closed"))

allowed, skip_event = gate.evaluate(
    scope=f"backtest-{self.run_id}",
    state_record=policy_state_record,
    trigger_events=policy_triggers,
    last_eval_at=last_policy_eval_at,
    indicator_timeframe=self.indicator_timeframe or "1h",
)

if not allowed:
    # Skip LLM call — reuse current plan
    continue
# ... proceed with LLM call ...
last_policy_eval_at = bar_ts
gate.release(f"backtest-{self.run_id}")
```

### Step 2: PlaybookRegistry + Memory Retrieval

Before each LLM call in the backtest runner:

```python
from services.playbook_registry import PlaybookRegistry
from services.memory_retrieval_service import MemoryRetrievalService

registry = PlaybookRegistry()
eligible_playbooks = registry.list_eligible(
    regime=indicator.regime or "unknown",
    htf_direction=_extract_htf_direction(indicator),
)

# Load episodes accumulated so far in this backtest run
memory_store = EpisodeMemoryStore()
for ep in self._episode_records:
    memory_store.add(ep)

memory_bundle = None
if memory_store.size() > 0:
    memory_bundle = MemoryRetrievalService(memory_store).retrieve(
        MemoryRetrievalRequest(
            symbol=self.symbol,
            regime_fingerprint=fingerprint.normalized_features,
            playbook_id=current_plan.playbook_id if current_plan else None,
            timeframe=self.indicator_timeframe,
        )
    )
```

Pass `eligible_playbooks` and `memory_bundle` to the plan generation call.

### Step 3: SetupEventGenerator at trigger fire

```python
from services.setup_event_generator import SetupEventGenerator

generator = SetupEventGenerator()
for order in fired_orders:
    setup_event = generator.generate(
        trigger=trigger,
        indicator=indicator,
        plan_id=current_plan.plan_id if current_plan else None,
        template_id=current_plan.template_id if current_plan else None,
    )
    order["setup_event_id"] = setup_event.setup_event_id
```

### Step 4: Episode records on position close

```python
from services.episode_memory_service import build_episode_record

if position_closed:
    signal_event = _get_signal_event_for_position(position)
    if signal_event:
        episode = build_episode_record(
            signal_event=signal_event,
            pnl=realized_pnl,
            r_achieved=r_achieved,
            hold_bars=hold_bars,
            exit_ts=close_ts,
        )
        self._episode_records.append(episode)
```

Initialize `self._episode_records: list[EpisodeMemoryRecord] = []` in `__init__`.

### Step 5: build_tick_snapshot + StructuralTargetSelector

```python
from services.market_snapshot_builder import build_tick_snapshot

try:
    tick_snapshot = build_tick_snapshot(indicator)
except Exception:
    tick_snapshot = None

# Pass to trigger engine (same as paper trading R64)
fired_orders = trigger_engine.on_bar(
    indicator=indicator,
    portfolio=portfolio,
    tick_snapshot=tick_snapshot,
)
```

### Step 6: Exit contract enforcement

```python
# Track originating_plan_id per symbol (same as paper trading R65)
if fill occurred:
    position_originating_plans[symbol] = current_plan.plan_id

# Pass to trigger evaluation
fired_orders = trigger_engine.on_bar(
    ...,
    position_originating_plans=position_originating_plans,
    current_plan_id=current_plan.plan_id if current_plan else None,
)
```

### Step 7: JudgePlanValidationService gate

```python
from services.judge_validation_service import JudgePlanValidationService

validator = JudgePlanValidationService()
verdict = validator.validate_plan(plan, memory_bundle=memory_bundle)
if verdict.decision == "reject":
    # In backtest: log and skip (no LLM revision — too slow for backtest loops)
    self._validation_rejected_count += 1
    continue  # reuse prior plan
```

### Step 8: Directional bias and timeframe threading

```python
# Pass to plan generation (same as paper trading)
plan = plan_provider.get_plan(
    ...,
    direction_bias=self.direction_bias,
    indicator_timeframe=self.indicator_timeframe,
)
```

### Step 9: Add fields to StrategistBacktestResult

```python
episode_records: list[EpisodeMemoryRecord] = []
exit_binding_mismatch_blocked: int = 0
validation_rejected_count: int = 0
policy_loop_skip_count: int = 0
```

### Step 10: Include episode_records in persistence

In `backtesting/activities.py` and `backtesting/persistence.py`:

```python
llm_data["episode_records"] = [ep.model_dump() for ep in result.episode_records]
llm_data["exit_binding_mismatch_blocked"] = result.exit_binding_mismatch_blocked
llm_data["validation_rejected_count"] = result.validation_rejected_count
llm_data["policy_loop_skip_count"] = result.policy_loop_skip_count
```

## Acceptance Criteria

- [ ] PolicyLoopGate wired; `policy_loop_skip_count` in result
- [ ] RegimeTransitionDetector wired; policy events drive LLM calls
- [ ] PlaybookRegistry.list_eligible wired; ELIGIBLE_PLAYBOOKS in backtest LLM prompts
- [ ] MemoryRetrievalService wired; MEMORY_CONTEXT in prompts after ≥1 episode
- [ ] SetupEventGenerator wired; setup_event_id on orders
- [ ] build_episode_record called on position close; episodes accumulate in `_episode_records`
- [ ] `episode_records` in `StrategistBacktestResult` and persisted to DB
- [ ] build_tick_snapshot wired; tick_snapshot passed to trigger engine
- [ ] StructuralTargetSelector wired at entry
- [ ] Exit contract enforcement (originating_plan_id) wired; `exit_binding_mismatch_blocked` in result
- [ ] JudgePlanValidationService wired; `validation_rejected_count` in result
- [ ] Directional bias and indicator_timeframe threaded through
- [ ] `scripts/check_wiring.py` exits 0 (all targets ✅ for both paper_trading and backtest_runner)
- [ ] Parity test: identical dataset through paper and backtest produces trigger counts
  within 5% of each other (when gating is consistent)

## Test Plan

```bash
# New parity tests
uv run pytest tests/test_backtest_parity.py -vv

# Backtest runner regression
uv run pytest tests/test_llm_strategist_runner.py -vv

# Full suite
uv run pytest -x -q

# Wiring audit
uv run python scripts/check_wiring.py
```

## Human Verification Evidence

```text
[To be filled after implementation]
1. Run a backtest. Confirm policy_loop_skip_count > 0 in results (gate is active).
2. Confirm episode_records in DB results JSON (non-empty for runs with fills).
3. Run same data through paper trading (replay mode or comparable setup). Compare
   trigger counts and fill counts — should be within 5%.
4. Run check_wiring.py — all targets should show ✅.
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-01 | Runbook created — backtest-paper parity wiring (R67) | Claude |

## Test Evidence

```text
[Paste test output here before committing]
```

## Worktree Setup

```bash
git worktree add -b feat/r67-backtest-parity ../wt-r67-parity main
cd ../wt-r67-parity
```

## Git Workflow

```bash
git checkout -b feat/r67-backtest-parity

git add backtesting/llm_strategist_runner.py \
        backtesting/activities.py \
        backtesting/persistence.py \
        tests/test_backtest_parity.py \
        docs/branching/67-backtest-paper-parity.md \
        docs/branching/README.md

git commit -m "feat: wire all Phase 8 services into backtest runner for paper-parity (R67)"
```
