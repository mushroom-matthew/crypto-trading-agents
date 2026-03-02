# Runbook 63: Position Lifecycle Completion

## Purpose

Wire three position-lifecycle services that are fully implemented but never called from
the execution path:

1. **`SetupEventGenerator`** (R44) — frozen feature snapshot + template provenance at
   trigger fire time, providing reproducible post-trade analysis
2. **`AdaptiveTradeManagementState`** (R45) — R-multiple state machine
   (EARLY → MATURE → EXTENDED → TRAIL) updated each bar while a position is open
3. **`build_episode_record()` + `EpisodeMemoryStore.persist_episode()`** (R51) — episode
   memory record construction when a position closes, powering MEMORY_CONTEXT injection
   on subsequent plan calls

**Pre-condition:** R44, R45, R51 schemas and services are implemented. This runbook wires
them into `tools/paper_trading.py`. The `EpisodeMemoryStore` DB table (`episode_memory`)
must exist (migration applied).

## Scope

1. `tools/paper_trading.py`
   - `evaluate_triggers_activity` — call `SetupEventGenerator` at trigger fire
   - `SessionState` — add `adaptive_management_states: Dict[str, Any] = {}`
     (per-position state keyed by symbol); add `episode_memory_store_state: List[Dict] = []`
   - Per-bar loop — call `AdaptiveTradeManagementState.tick()` for each open position
   - Stop/target close handler — call `build_episode_record()`, `store.add()`,
     `store.persist_episode()`
   - `generate_strategy_plan_activity` — load episode store from `SessionState`, call
     `MemoryRetrievalService.retrieve()` and inject `DiversifiedMemoryBundle` into prompt

2. `app/db/models.py` — add `EpisodeMemory` SQLAlchemy model (if not already present)

3. New Alembic migration — `make migrate name="add_episode_memory_table"`

4. `tests/test_position_lifecycle.py` — new test file

## Out of Scope

- Backtest runner wiring (covered by R67)
- `SetupEventGenerator` changes beyond the call site
- `EpisodeMemoryStore` in-memory logic changes (already correct)

## Implementation Steps

### Step 1: Add EpisodeMemory DB model

In `app/db/models.py`, add:

```python
class EpisodeMemory(Base):
    __tablename__ = "episode_memory"

    episode_id = Column(String, primary_key=True)
    signal_id = Column(String, nullable=True, index=True)
    trade_id = Column(String, nullable=True)
    symbol = Column(String, nullable=False, index=True)
    timeframe = Column(String, nullable=True)
    playbook_id = Column(String, nullable=True)
    template_id = Column(String, nullable=True)
    trigger_category = Column(String, nullable=True)
    direction = Column(String, nullable=True)
    entry_ts = Column(DateTime(timezone=True), nullable=True)
    exit_ts = Column(DateTime(timezone=True), nullable=True)
    pnl = Column(Float, nullable=True)
    r_achieved = Column(Float, nullable=True)
    hold_bars = Column(Integer, nullable=True)
    hold_minutes = Column(Float, nullable=True)
    mae_pct = Column(Float, nullable=True)
    mfe_pct = Column(Float, nullable=True)
    outcome_class = Column(String, nullable=False)  # win/loss/neutral
    failure_modes = Column(JSON, nullable=False, default=list)
    regime_fingerprint = Column(JSON, nullable=True)
    snapshot_hash = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

Run migration: `make migrate name="add_episode_memory_table"` then `uv run alembic upgrade head`.

### Step 2: Wire SetupEventGenerator at trigger fire

In `evaluate_triggers_activity`, after a trigger fires and before the order dict is
returned, call `SetupEventGenerator`:

```python
from services.setup_event_generator import SetupEventGenerator

generator = SetupEventGenerator()
for order in fired_orders:
    setup_event = generator.generate(
        trigger=trigger,
        indicator=indicator_snapshot,
        plan_id=plan_dict.get("plan_id"),
        template_id=plan_dict.get("_retrieved_template_id"),
    )
    order["setup_event_id"] = setup_event.setup_event_id
    order["setup_event"] = setup_event.model_dump()
```

### Step 3: Add adaptive management state to SessionState

```python
# Adaptive trade management per-position state (Runbook 63)
adaptive_management_states: Dict[str, Any] = {}
```

### Step 4: Per-bar adaptive management tick

In the main evaluation loop, after indicator snapshots are fetched and for each open
position:

```python
from services.adaptive_trade_management import AdaptiveTradeManagementState

for symbol, pos in open_positions.items():
    state_dict = self.state.adaptive_management_states.get(symbol, {})
    mgmt_state = AdaptiveTradeManagementState.model_validate(state_dict) if state_dict else AdaptiveTradeManagementState.initial(pos)
    mgmt_state = mgmt_state.tick(current_price=current_price, indicator=indicator)
    self.state.adaptive_management_states[symbol] = mgmt_state.model_dump()
```

### Step 5: Build episode record on position close

In the stop/target close handler (after the closing fill is recorded):

```python
from services.episode_memory_service import build_episode_record, EpisodeMemoryStore
from schemas.signal_event import SignalEvent

# Retrieve the SignalEvent for this position (stored in position_meta["signal_event"])
signal_event_dict = position_meta.get("signal_event")
if signal_event_dict:
    signal_event = SignalEvent.model_validate(signal_event_dict)
    episode = build_episode_record(
        signal_event=signal_event,
        pnl=realized_pnl,
        r_achieved=r_achieved,
        hold_bars=hold_bars,
        mae_pct=mae_pct,
        mfe_pct=mfe_pct,
        exit_ts=close_ts,
    )
    store = EpisodeMemoryStore()
    store.add(episode)
    store.persist_episode(episode)  # non-fatal DB write

    # Append to SessionState in-memory list for retrieval on next plan gen
    self.state.episode_memory_store_state = (
        self.state.episode_memory_store_state or []
    )[-99:]  # cap at 100 episodes in-memory
    self.state.episode_memory_store_state.append(episode.model_dump())
```

### Step 6: Memory retrieval in generate_strategy_plan_activity

The activity already has skeleton memory retrieval (A5 from prior work). Confirm it:
1. Loads `EpisodeMemoryStore` from the `episode_memory_store_state` list passed in
   `market_context`
2. Calls `MemoryRetrievalService(store).retrieve(request)`
3. Formats the bundle as `MEMORY_CONTEXT` block in the LLM prompt

If this path already exists (from commit `0eaaf6b`), verify it is reachable and test
with actual episode records in the store.

## Acceptance Criteria

- [ ] `EpisodeMemory` DB model exists; migration applied
- [ ] `SetupEventGenerator.generate()` called at trigger fire; `setup_event_id` in order dict
- [ ] `SessionState.adaptive_management_states` added; per-position state updated each bar
- [ ] `build_episode_record()` called on stop/target close
- [ ] `store.persist_episode()` called (non-fatal; wrapped in try/except)
- [ ] Episode record appended to `SessionState.episode_memory_store_state`
- [ ] `MemoryRetrievalService.retrieve()` is called in `generate_strategy_plan_activity`
  when episode records are present
- [ ] LLM prompt contains `MEMORY_CONTEXT` block when prior episodes exist
- [ ] All existing paper trading tests still pass
- [ ] `scripts/check_wiring.py` shows `build_episode_record` ✅ for paper_trading.py

## Test Plan

```bash
# New lifecycle tests
uv run pytest tests/test_position_lifecycle.py -vv

# Verify memory retrieval integration
uv run pytest -k "episode_memory or memory_retrieval" -vv

# Regression
uv run pytest -x -q
```

## Human Verification Evidence

```text
[To be filled after implementation]
1. Trigger a fill in paper trading. Confirm order dict has setup_event_id field.
2. Let position run for 3+ bars. Query SessionState adaptive_management_states —
   should show MATURE or EXTENDED state for old positions.
3. Close a position (stop hit). Query episode_memory DB table — should have a row.
4. On next plan generation, confirm MEMORY_CONTEXT block in LLM prompt (visible in
   Langfuse trace or logs).
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-01 | Runbook created — position lifecycle completion (R63) | Claude |

## Test Evidence

```text
[Paste test output here before committing]
```

## Worktree Setup

```bash
git worktree add -b feat/r63-position-lifecycle ../wt-r63-lifecycle main
cd ../wt-r63-lifecycle
```

## Git Workflow

```bash
git checkout -b feat/r63-position-lifecycle

git add tools/paper_trading.py \
        app/db/models.py \
        alembic/versions/ \
        tests/test_position_lifecycle.py \
        docs/branching/63-position-lifecycle-completion.md \
        docs/branching/README.md

git commit -m "feat: wire SetupEventGenerator, AdaptiveTradeManagement, episode records (R63)"
```
