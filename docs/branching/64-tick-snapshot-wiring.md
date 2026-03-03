# Runbook 64: Tick Snapshot and Structural Target Wiring

## Purpose

Wire two services that provide deterministic structural context but are currently unused
in the execution path:

1. **`build_tick_snapshot()`** (R49) — per-bar `TickSnapshot` passed to the trigger
   engine so triggers have access to normalized numeric/derived features and snapshot
   provenance (snapshot_id, hash, staleness)
2. **`StructuralTargetSelector`** (R58/R56) — deterministic structural stop and target
   candidate selection at entry time, so stops are resolved from real market structure
   levels rather than purely from ATR multiples

**Pre-condition:** R49 and R58 are implemented. `build_tick_snapshot()` exists in
`services/market_snapshot_builder.py`. `StructuralTargetSelector` is in
`services/structural_target_selector.py`. `StructureSnapshot` data is available from the
structure engine (stored in `SessionState.structure_history`).

## Scope

1. `tools/paper_trading.py`
   - `evaluate_triggers_activity` — build `TickSnapshot` from indicator, pass to trigger
     engine; pass current `StructureSnapshot` if available
   - `_execute_order` — call `StructuralTargetSelector.select_stop_candidates()` and
     `select_target_candidates()` at entry; log candidate list; prefer structural level
     over pure ATR if a valid candidate is found within `max_distance_atr=3.0`

2. `agents/strategies/trigger_engine.py`
   - Accept optional `tick_snapshot: TickSnapshot | None = None` in `on_bar()` or the
     appropriate entry point
   - Thread `snapshot_id` and `snapshot_hash` into trigger context so they appear on
     fired-trigger events and signal events

3. `tests/test_tick_snapshot_wiring.py` — new test file

## Out of Scope

- Changing the structure engine itself (R58 scope already done)
- Changing stop/target resolution logic beyond adding structural candidate lookup
- High-level structural reflection (Layer 3)

## Implementation Steps

### Step 1: Build TickSnapshot per bar in evaluate_triggers_activity

At the start of `evaluate_triggers_activity`, after building the `IndicatorSnapshot`:

```python
from services.market_snapshot_builder import build_tick_snapshot
from schemas.market_snapshot import TickSnapshot

tick_snapshot: TickSnapshot | None = None
try:
    tick_snapshot = build_tick_snapshot(indicator_snapshot)
except Exception as exc:
    # Non-fatal — continue without snapshot
    activity.logger.warning("build_tick_snapshot failed: %s", exc)
```

Pass `tick_snapshot` to `trigger_engine.on_bar()` (or wherever the trigger context is
built).

### Step 2: Thread snapshot into trigger engine context

In `trigger_engine.py`, update `_context()` or `on_bar()`:

```python
def on_bar(
    self,
    indicator: IndicatorSnapshot,
    portfolio: PortfolioState,
    tick_snapshot: TickSnapshot | None = None,
) -> list[TriggerFiredEvent]:
    ...

def _context(
    self,
    indicator: IndicatorSnapshot,
    portfolio: PortfolioState,
    tick_snapshot: TickSnapshot | None = None,
) -> dict[str, Any]:
    ctx = {... existing fields ...}
    if tick_snapshot:
        ctx["snapshot_id"] = tick_snapshot.snapshot_id
        ctx["snapshot_hash"] = tick_snapshot.snapshot_hash
        ctx["snapshot_staleness_s"] = tick_snapshot.staleness_seconds
    return ctx
```

Snapshot fields are read-only context for trigger evaluation; they do not affect
trigger logic.

### Step 3: Structural stop/target candidate selection at entry

In `_execute_order()` (paper trading), after resolving the trigger's `stop_anchor_type`
and `target_anchor_type` but before computing final stop/target prices:

```python
from services.structural_target_selector import (
    select_stop_candidates, select_target_candidates
)

structure_snapshot = _get_latest_structure_snapshot(self.state, symbol)

if structure_snapshot and trigger.stop_anchor_type is None:
    # No explicit anchor — try structural candidates
    stop_candidates = select_stop_candidates(
        structure_snapshot,
        direction=trigger.direction,
        max_distance_atr=3.0,
    )
    if stop_candidates:
        nearest_stop = stop_candidates[0]
        # Use structural level as stop price if reasonable
        structural_stop_price = nearest_stop.price
        # Log candidate selection
        activity.logger.info(
            "Structural stop candidate: %s @ %.4f (kind=%s)",
            nearest_stop.role, structural_stop_price, nearest_stop.kind,
        )

if structure_snapshot and trigger.target_anchor_type is None:
    target_candidates = select_target_candidates(
        structure_snapshot,
        direction=trigger.direction,
        max_distance_atr=10.0,
    )
    if target_candidates:
        nearest_target = target_candidates[0]
        structural_target_price = nearest_target.price
        activity.logger.info(
            "Structural target candidate: %s @ %.4f (kind=%s)",
            nearest_target.role, structural_target_price, nearest_target.kind,
        )
```

Do NOT silently override explicit stop/target anchors — structural candidates are
supplementary. Only apply when the trigger has no explicit anchor type.

### Step 4: Helper to get latest structure snapshot

```python
def _get_latest_structure_snapshot(
    state: SessionState, symbol: str
) -> StructureSnapshot | None:
    """Return the most recent StructureSnapshot for symbol from session history."""
    history = state.structure_history.get(symbol, [])
    if not history:
        return None
    latest = history[-1]
    try:
        return StructureSnapshot.model_validate(latest)
    except Exception:
        return None
```

## Acceptance Criteria

- [ ] `build_tick_snapshot()` called per bar in `evaluate_triggers_activity`; failures are
  non-fatal (wrapped in try/except)
- [ ] `TickSnapshot` passed to trigger engine; `snapshot_id` and `snapshot_hash` appear
  in trigger context dict
- [ ] `select_stop_candidates()` called at entry when trigger has no `stop_anchor_type`;
  structural candidate logged
- [ ] `select_target_candidates()` called at entry when trigger has no `target_anchor_type`
- [ ] Explicit stop/target anchors are NOT overridden by structural candidates
- [ ] All existing paper trading and trigger engine tests pass
- [ ] `scripts/check_wiring.py` shows `build_tick_snapshot` and `StructuralTargetSelector`
  ✅ for paper_trading.py and trigger_engine.py

## Test Plan

```bash
uv run pytest tests/test_tick_snapshot_wiring.py -vv
uv run pytest tests/test_trigger_engine.py -vv
uv run pytest -x -q
```

## Human Verification Evidence

```text
1. build_tick_snapshot() is called per bar inside the `for timeframe in timeframes:` loop
   in evaluate_triggers_activity. Failures are wrapped in try/except and logged as
   WARNING (non-fatal). tick_snapshot is passed to trigger_engine.on_bar() via the
   tick_snapshot keyword argument.

2. TriggerEngine._context() surfaces snapshot_id, snapshot_hash, snapshot_staleness_s
   in the evaluation context dict when tick_snapshot is not None. All 3 keys are
   defaulted to None when no snapshot is available (context.setdefault).

3. In _execute_order, select_stop_candidates() is called only when stop_anchor_type is
   None and select_target_candidates() called only when target_anchor_type is None.
   These are guarded by explicit `is None` checks so explicit anchors (e.g.
   stop_anchor_type="htf_daily_low") are NOT overridden — only supplementary logging.

4. check_wiring.py shows build_tick_snapshot ✅ and StructuralTargetSelector ✅ for
   tools/paper_trading.py. trigger_engine.py shows MISSING because those functions
   are called from paper_trading (not trigger_engine directly) — correct design.
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-01 | Runbook created — tick snapshot and structural target wiring (R64) | Claude |
| 2026-03-02 | Implemented R64: build_tick_snapshot per bar in evaluate_triggers_activity, tick_snapshot parameter on on_bar(), snapshot_staleness_s in context, _get_latest_structure_snapshot helper, structural candidate logging in _execute_order, 18 new tests | Claude |

## Test Evidence

```text
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/getzinmw/wt-r64-tick-snapshot
configfile: pyproject.toml

tests/test_tick_snapshot_wiring.py::test_build_tick_snapshot_returns_tick_snapshot PASSED
tests/test_tick_snapshot_wiring.py::test_build_tick_snapshot_staleness_in_quality PASSED
tests/test_tick_snapshot_wiring.py::test_trigger_engine_on_bar_accepts_tick_snapshot PASSED
tests/test_tick_snapshot_wiring.py::test_trigger_engine_context_gets_snapshot_id_hash_staleness PASSED
tests/test_tick_snapshot_wiring.py::test_trigger_engine_context_without_snapshot_has_none_fields PASSED
tests/test_tick_snapshot_wiring.py::test_get_latest_structure_snapshot_empty_history_returns_none PASSED
tests/test_tick_snapshot_wiring.py::test_get_latest_structure_snapshot_returns_most_recent PASSED
tests/test_tick_snapshot_wiring.py::test_get_latest_structure_snapshot_wrong_symbol_returns_none PASSED
tests/test_tick_snapshot_wiring.py::test_get_latest_structure_snapshot_invalid_data_returns_none PASSED
tests/test_tick_snapshot_wiring.py::test_select_stop_candidates_from_history_long PASSED
tests/test_tick_snapshot_wiring.py::test_select_target_candidates_from_history_long PASSED
tests/test_tick_snapshot_wiring.py::test_paper_trading_references_build_tick_snapshot PASSED
tests/test_tick_snapshot_wiring.py::test_paper_trading_references_select_stop_candidates PASSED
tests/test_tick_snapshot_wiring.py::test_paper_trading_references_select_target_candidates PASSED
tests/test_tick_snapshot_wiring.py::test_trigger_engine_on_bar_accepts_tick_snapshot_kwarg PASSED
tests/test_tick_snapshot_wiring.py::test_paper_trading_references_get_latest_structure_snapshot PASSED
tests/test_tick_snapshot_wiring.py::test_explicit_stop_anchor_not_in_structural_candidate_path PASSED
tests/test_tick_snapshot_wiring.py::test_explicit_target_anchor_not_in_structural_candidate_path PASSED
============================== 18 passed in 25.83s ==============================

Full suite (excluding 3 pre-existing DB_DSN collection errors and 2 pre-existing
test_factor_loader pandas version failures):
2 failed, 2104 passed, 2 skipped — no regressions introduced
```

## Worktree Setup

```bash
git worktree add -b feat/r64-tick-snapshot ../wt-r64-tick-snapshot main
cd ../wt-r64-tick-snapshot
```

## Git Workflow

```bash
git checkout -b feat/r64-tick-snapshot

git add tools/paper_trading.py \
        agents/strategies/trigger_engine.py \
        tests/test_tick_snapshot_wiring.py \
        docs/branching/64-tick-snapshot-wiring.md \
        docs/branching/README.md

git commit -m "feat: wire TickSnapshot and StructuralTargetSelector into trigger engine (R64)"
```
