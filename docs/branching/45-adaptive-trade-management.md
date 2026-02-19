# Branch: adaptive-trade-management

## Purpose

Deterministic, state-based position adjustment that protects open profit without collapsing
expected R. Goals:

- Never let a +1R winner become a full loss
- Lock a guaranteed blended floor at +2R via partial exit
- Preserve asymmetric upside on the remainder
- Remain fully auditable and replayable (signal ledger integrity)

**Philosophy: balance leaning toward 1-2R wins.**

This runbook depends on:
- **Runbook 42** — provides `stop_price_abs` in `position_meta` and the Runbook 42 R42
  resolution functions. `initial_risk_abs` is computed from the Runbook 42 stop at entry
  and must never be recomputed afterward.

---

## Definitions (exact; do not drift from these)

```
initial_risk_abs   = abs(entry_price - initial_stop_abs)
                     Set ONCE at entry from position_meta["stop_price_abs"].
                     Constant for the entire trade lifecycle. Never updated.
                     Partial exits and stop adjustments do NOT change it.

current_R          = (mark_price - entry_price) / initial_risk_abs   [for longs]
                   = (entry_price - mark_price) / initial_risk_abs   [for shorts]
                     Computed each bar from mark_price (bar close).

mfe_r              = running max(current_R) since entry (peak unrealized R)
mae_r              = running min(current_R) since entry (worst drawdown in R)
                     Both updated every bar before state transitions check.

position_fraction  = current_qty / entry_qty
                     Decremented by partial exits. Used for blended R math.

blended_floor_R    = sum(exit_R_i * fraction_i) + trailing_stop_R * position_fraction
                     Only meaningful after at least one partial exit.
```

**Critical constraint:** R is always relative to `initial_risk_abs`. If you re-anchor R to
the adjusted stop, the ladder becomes self-referential and expectancy stats become garbage.

---

## State Machine

### States

```
EARLY     0 → r1_threshold R      No interference. Hard structural stop only.
MATURE    r1_threshold+ R         Stop advanced to entry - r1_stop_offset * initial_risk
EXTENDED  r2_threshold+ R         Partial exit emitted; stop locked at entry + r2_stop_r_lock
TRAIL     r3_threshold+ R         Stop tightened to entry + r3_stop_r_lock
```

### Transition Table

```
From      To          Trigger (current_R crosses)   Actions
────────────────────────────────────────────────────────────────────────────────
EARLY   → MATURE      r1_threshold (default 1.0R)   move_stop(entry - 0.25R)
MATURE  → EXTENDED    r2_threshold (default 2.0R)   partial_exit(50%),
                                                    move_stop(entry + 0.5R)
EXTENDED→ TRAIL       r3_threshold (default 3.0R)   move_stop(entry + 1.5R)
```

### Multi-Rung Jump Rule (required for gap handling)

If price moves from 0.8R to 2.4R in a single bar (crypto gap/spike), apply all missed
transitions **in order** before returning:

```python
for rung in [r1, r2, r3]:
    if not rung.triggered and current_R >= rung.threshold:
        apply_rung_transition(rung)
        emit_event(rung, current_R, "jump_catch")
```

Each rung fires at most once. Emit a separate event per rung with `rung_catch=True`.
Do not silently skip rungs.

---

## Config

```python
@dataclass(frozen=True)
class TradeManagementConfig:
    """R-multiple-based trade state progression rules.

    All thresholds and offsets are in R-multiples relative to initial_risk_abs.
    """
    enabled: bool = True

    # MATURE: stop to entry - r1_stop_offset * initial_risk (not break-even)
    r1_threshold: float = 1.0
    r1_stop_offset: float = 0.25       # Negative = below entry (for longs)

    # EXTENDED: partial exit + stop lock
    r2_threshold: float = 2.0
    r2_partial_fraction: float = 0.50  # Fraction of current qty to exit
    r2_stop_r_lock: float = 0.50       # Stop → entry + 0.5R (already profitable)

    # TRAIL: tighten on remainder
    r3_threshold: float = 3.0
    r3_stop_r_lock: float = 1.50       # Stop → entry + 1.5R

    # Wick buffer (optional, default off)
    # Adds buffer below/above the computed rung stop to absorb single wicks
    # Unit: fraction of initial_risk_abs. E.g., 0.10 = 0.10R buffer.
    wick_buffer_r: float = 0.0         # Default: no buffer

    # ATR trail interaction: if True, allow existing ATR trail to tighten beyond
    # the rung stop (advance-only). If False, rung stop always wins.
    allow_atr_tighten: bool = True
```

**Blended floor math for default config (for documentation):**

```
Trade reaches +2R and partial exits:
  50% exits at +2R         = contribution: 0.50 * 2R = 1.00R
  Remainder stop at +0.5R  = contribution: 0.50 * 0.5R = 0.25R (minimum)
  Blended floor            = 1.25R regardless of what the remainder does

Trade reaches +3R:
  50% already exited at +2R
  Remainder stop now at +1.5R
  Blended floor            = 1.00R + 0.50 * 1.5R = 1.75R minimum
```

---

## Actions

### Action: `move_stop(new_stop_price)`

```
1. Compute candidate_stop:
     long:  entry + rung_r_offset * initial_risk_abs
     short: entry - rung_r_offset * initial_risk_abs
     (positive rung_r_offset = above entry for longs = profitable stop)

2. Apply wick buffer (if wick_buffer_r > 0):
     long:  candidate_stop = candidate_stop - wick_buffer_r * initial_risk_abs
     short: candidate_stop = candidate_stop + wick_buffer_r * initial_risk_abs

3. Apply ATR trail interaction (if allow_atr_tighten):
     existing_trail = position_risk_state.trail_price (if active)
     long:  effective_stop = max(candidate_stop, existing_trail or -inf)
     short: effective_stop = min(candidate_stop, existing_trail or +inf)

4. Enforce advance-only:
     long:  if effective_stop <= position_meta["stop_price_abs"]: return (no-op)
     short: if effective_stop >= position_meta["stop_price_abs"]: return (no-op)

5. Write to:
     - PositionRiskState.stop_price = effective_stop
     - position_meta["stop_price_abs"] = effective_stop
     - position_meta["stop_anchor_type"] = f"mgmt_{rung_name}"

6. Emit StopAdjustmentEvent (see Audit section).
```

### Action: `partial_exit(fraction)`

```
1. Compute exit_qty = current_qty * fraction.
2. Emit synthetic Order:
     side = "sell" (long) or "buy" (short)
     quantity = exit_qty
     price = bar.close (current bar)
     reason = "trade_mgmt_r2_partial"
     trigger_category = "risk_reduce"
     intent = "exit"
     exit_fraction = fraction

3. Emit PartialExitEvent (see Audit section).
4. Do NOT touch initial_risk_abs.
5. Update position_fraction_remaining (tracked in PositionRiskState).
```

### Precedence When Multiple Actions Fire (same bar)

```
Order of execution within _advance_trade_state():
  1. Update mfe_r, mae_r (always)
  2. Apply R1 transition (move_stop) if triggered
  3. Apply R2 transition (move_stop) if triggered
  4. Collect R2 partial_exit order (do not execute yet — see ordering below)
  5. Apply R3 transition (move_stop) if triggered

Partial exit orders from step 4 are returned from _advance_trade_state()
and prepended to the order list AFTER trigger_engine.on_bar() runs.
This prevents conflict with simultaneous LLM "risk_off" or "emergency_exit"
orders — trigger engine sees an unchanged position size and the right stop.

Stop adjustments are applied BEFORE trigger_engine.on_bar() so that:
  - below_stop identifier reflects the advanced stop
  - risk engine sees the correct stop state
```

### Binding Constraint (exit_allowed)

The runner already tracks `min_hold_bars` and `trade_cooldown_bars`. When stop advance
would have caused an exit that the binding constraint prevents, log but do not suppress
the stop move:

```python
# In _advance_trade_state, when a partial exit order is generated:
is_in_hold = self._is_in_min_hold(symbol, timestamp)
if is_in_hold:
    order.metadata["exit_blocked_by"] = "min_hold"
    # Still emit the order — the executor decides to process or skip
    # Still emit PartialExitEvent with exit_blocked=True for analysis
```

This means stop adjustments always apply (protective), but partial exit execution can
be deferred by existing hold logic. The event log captures whether a scheduled exit
was blocked so MAE/expectancy stats can account for it.

---

## Audit Events

### `StopAdjustmentEvent`

```python
class StopAdjustmentEvent(BaseModel):
    model_config = {"extra": "forbid"}
    symbol: str
    timestamp: datetime
    rung: str                    # "r1_mature" | "r2_extended" | "r3_trail" | "atr_trail"
    old_stop: Optional[float]
    new_stop: float
    current_R: float
    mfe_r: float
    mae_r: float
    position_fraction: float     # Fraction of original qty still open
    rung_catch: bool = False     # True if this was a jump-catch (multi-rung bar)
    engine_version: str = "45.0.0"
```

### `PartialExitEvent`

```python
class PartialExitEvent(BaseModel):
    model_config = {"extra": "forbid"}
    symbol: str
    timestamp: datetime
    rung: str                    # "r2_extended" (currently only R2 exits)
    fraction_exited: float
    exit_price: float
    exit_R: float                # R at time of partial exit
    mfe_r: float
    initial_risk_abs: float
    position_fraction_before: float
    position_fraction_after: float
    exit_blocked: bool = False   # True if hold constraint prevented execution
    exit_blocked_by: Optional[str] = None
    engine_version: str = "45.0.0"
```

Both event types stored in `LLMStrategistBacktester._trade_mgmt_events: list[dict]`.
Surfaced in `StrategistBacktestResult.trade_mgmt_events`.

---

## Implementation Steps

### Step 1: Add `TradeManagementConfig` (after `ExecutionModelSettings`)

See Config section above. Add to `backtesting/llm_strategist_runner.py`.

### Step 2: Extend `PositionRiskState`

Add fields (verify `PositionRiskState` is `@dataclass`, not frozen):

```python
# Runbook 45 — R tracking
initial_risk_abs: float | None = None  # Computed once at entry; never changed
position_fraction: float = 1.0         # Current qty / entry qty
trade_state: str = "EARLY"
mfe_r: float = 0.0
mae_r: float = 0.0
r1_triggered: bool = False
r2_triggered: bool = False
r3_triggered: bool = False
```

### Step 3: Wire `initial_risk_abs` at fill time

In `_update_position_risk_state()`, after Runbook 42 sets `stop_price_abs`:

```python
stop_abs = meta.get("stop_price_abs")
if stop_abs is not None and state.initial_risk_abs is None:
    risk = abs(order.price - stop_abs)
    if risk > 1e-9:
        state.initial_risk_abs = risk
```

Note the `is None` guard: never overwrite once set. This keeps R math stable even
if stop is subsequently adjusted by Runbook 45.

### Step 4: Add `_advance_trade_state()` and `_apply_stop_adjustment()`

See Actions section above for pseudocode. Key rules:
- `_advance_trade_state()` updates PositionRiskState in-place for stop changes
- Returns list of synthetic partial exit Orders (not yet executed)
- `_apply_stop_adjustment()` enforces advance-only, ATR interaction, wick buffer
- Both write to `position_meta` for trigger engine visibility

Also update `position_meta` each bar with current R tracking values:

```python
meta["current_R"] = round(current_R, 4)
meta["mfe_r"] = round(state.mfe_r, 4)
meta["mae_r"] = round(state.mae_r, 4)
meta["trade_state"] = state.trade_state
meta["position_fraction"] = state.position_fraction
```

### Step 5: Wire into main bar loop

```python
# === Before trigger_engine.on_bar() ===
# 1. Apply stop adjustments (protective — must happen first)
mgmt_partial_orders: list[Order] = []
if self.trade_mgmt_config and self.trade_mgmt_config.enabled:
    for sym in list(self.position_risk_state.keys()):
        close = (self._indicator_snapshot(sym, timeframe, ts) or {}).get("close") or \
                self.portfolio.positions.get(sym, {})  # fallback to last known
        if close:
            partials = self._advance_trade_state(sym, close, ts, self.trade_mgmt_config)
            mgmt_partial_orders.extend(partials)

# 2. Trigger engine evaluates (sees updated stops in position_meta)
orders, blocked_entries = trigger_engine.on_bar(
    bar, indicator, portfolio_state, asset_state,
    market_structure=structure_snapshot,
    position_meta=self.portfolio.position_meta,
)

# 3. Prepend partial exits after trigger engine (avoids qty conflict)
orders = mgmt_partial_orders + orders
```

### Step 6: Expose R fields in trigger engine `_context()`

In the Runbook 42 block (end of `_context()`), add:

```python
current_R_val = active_meta.get("current_R", 0.0) if active_meta else 0.0
mfe_r_val = active_meta.get("mfe_r", 0.0) if active_meta else 0.0
mae_r_val = active_meta.get("mae_r", 0.0) if active_meta else 0.0
context["current_R"] = current_R_val
context["mfe_r"] = mfe_r_val
context["mae_r"] = mae_r_val
context["trade_state"] = active_meta.get("trade_state", "EARLY") if active_meta else "EARLY"
context["position_fraction"] = active_meta.get("position_fraction", 1.0) if active_meta else 1.0
context["r1_reached"] = current_R_val >= 1.0
context["r2_reached"] = current_R_val >= 2.0
context["r3_reached"] = current_R_val >= 3.0
```

### Step 7: Add `StopAdjustmentEvent` and `PartialExitEvent` to `schemas/trade_set.py`

See Audit Events section above.

### Step 8: Update `prompts/strategy_plan_schema.txt`

```
R-TRACKING IDENTIFIERS (available while in a position):
  current_R       — (close - entry) / initial_risk_abs (signed; negative = losing)
  mfe_r           — Peak R reached this trade (max favorable excursion)
  mae_r           — Worst drawdown in R (min, typically negative)
  trade_state     — "EARLY" | "MATURE" | "EXTENDED" | "TRAIL"
  position_fraction — Fraction of original qty still open (1.0 = full, 0.5 = half exited)
  r1_reached      — True when current_R >= 1.0
  r2_reached      — True when current_R >= 2.0

USAGE IN RULES (supplemental logic on top of management engine):
  # Protect a runner: hold as long as R is still expanding
  hold_rule: "mfe_r > 2.0 and current_R > mfe_r * 0.6"

  # Exit if R spikes and then gives back more than 50% of MFE
  exit_rule: "not is_flat and (below_stop or (mfe_r >= 3.0 and current_R < mfe_r * 0.5))"

  # Scale out early if trade stalls after MATURE transition
  exit_rule: "not is_flat and (below_stop or (trade_state == 'MATURE' and current_R < 0.3))"

NOTE: The management engine handles R1/R2/R3 transitions automatically.
      These identifiers let the LLM compose supplemental logic on top.
      Never duplicate management rules in the LLM (e.g., don't write an exit_rule
      that fires at +2R — the engine already handles that via partial exit).
```

---

## Test Plan

```bash
uv run pytest tests/test_adaptive_trade_mgmt.py -vv
uv run pytest tests/test_trigger_engine.py -vv
uv run pytest --tb=short -q
```

## Required Tests (`tests/test_adaptive_trade_mgmt.py`)

```
# R-tracking
test_initial_risk_abs_set_once_at_entry
test_initial_risk_abs_not_updated_on_stop_advance
test_current_R_computed_correctly_long
test_current_R_computed_correctly_short
test_mfe_mae_tracked_each_bar

# State transitions — smooth price path
test_r1_transitions_to_mature_long
test_r1_transitions_to_mature_short
test_r1_stop_moves_to_minus_025R
test_r1_does_not_fire_below_threshold
test_r2_emits_partial_exit_50pct
test_r2_stop_moves_to_plus_05R
test_r3_stop_moves_to_plus_15R

# Idempotency
test_r1_fires_at_most_once
test_r2_fires_at_most_once

# Advance-only invariant
test_stop_never_retreats_after_r1
test_stop_never_retreats_after_r2

# Multi-rung jump rule
test_jump_from_08R_to_24R_applies_r1_then_r2_in_order
test_jump_events_have_rung_catch_true
test_jump_from_08R_to_34R_applies_all_three_rungs

# initial_risk_abs immutability
test_partial_exit_does_not_change_initial_risk_abs
test_position_fraction_updated_on_partial_exit

# ATR trail interaction
test_atr_trail_can_tighten_beyond_rung_stop_if_enabled
test_atr_trail_cannot_loosen_rung_stop

# Binding constraint
test_partial_exit_emitted_during_min_hold_has_exit_blocked_flag
test_stop_adjustment_still_applies_during_min_hold

# Config
test_disabled_config_is_noop
test_wick_buffer_applied_to_rung_stop

# Context namespace
test_trigger_engine_context_has_current_R
test_trigger_engine_context_has_trade_state
test_trigger_engine_context_has_mfe_r

# Audit events
test_stop_adjustment_event_emitted_at_r1
test_partial_exit_event_emitted_at_r2
test_event_contains_initial_risk_abs
test_event_engine_version_field
```

---

## Acceptance Criteria

- [ ] `initial_risk_abs` computed once at entry, never changed by stop advances or partial exits
- [ ] `current_R` always relative to `initial_risk_abs` (not adjusted stop)
- [ ] `position_fraction` decremented by partial exits
- [ ] `mfe_r`, `mae_r` updated every bar before transition checks
- [ ] At +1R: stop moves to `entry - 0.25R` (long) / `entry + 0.25R` (short)
- [ ] At +2R: synthetic `risk_reduce` order with `exit_fraction=0.50`; stop moves to `entry + 0.5R`
- [ ] At +3R: stop moves to `entry + 1.5R`; no additional partial exit
- [ ] Multi-rung jumps: all missed rungs applied in order within a single bar
- [ ] Each rung fires at most once per trade (`r1_triggered`, `r2_triggered`, `r3_triggered`)
- [ ] Stop can only advance in favorable direction (advance-only enforced in `_apply_stop_adjustment`)
- [ ] ATR trail can tighten beyond rung stop when `allow_atr_tighten=True`; cannot loosen it
- [ ] Partial exit orders emitted **after** `trigger_engine.on_bar()` in main loop
- [ ] Stop adjustments applied **before** `trigger_engine.on_bar()` in main loop
- [ ] Min-hold binding constraint: partials carry `exit_blocked=True` flag when blocked; stop still moves
- [ ] `StopAdjustmentEvent` and `PartialExitEvent` emitted for every transition with full audit fields
- [ ] Trigger engine namespace: `current_R`, `mfe_r`, `mae_r`, `trade_state`, `position_fraction`, `r1_reached`, `r2_reached`, `r3_reached`
- [ ] `TradeManagementConfig(enabled=False)` is a complete no-op
- [ ] All existing tests unaffected (all new fields optional or additive)

---

## Key Design Decisions

### R anchor is permanent

`initial_risk_abs` is set from `stop_price_abs` at the moment of entry fill and never
touched again. If the stop advances to +0.5R after the R2 partial, R3 is still computed
as `(price - entry) / initial_risk_abs`, not `(price - adjusted_stop) / initial_risk_abs`.
This keeps all expectancy stats comparable across trades and periods.

### -0.25R stop at +1R, not break-even

BE stops in crypto cluster at round numbers and prior highs/lows — they are targets.
At -0.25R: worst-case loss after +1R is cut from 1.0R to 0.25R (75% reduction) while
still absorbing a normal ATR-width wick without triggering.

### 50% partial at +2R, not 100%

100% exit caps R at exactly 2.0. In a balance-leaning system, the trades above +3R
contribute disproportionately to profit factor. Taking only 50% at +2R preserves
asymmetric participation. The blended floor (1.25R) is already superior to any single
exit below 2.5R.

### Partial exits after trigger engine

Placing management partials before `trigger_engine.on_bar()` would mean the engine
evaluates on a reduced position size. This creates subtle interactions when the engine
also tries to exit (e.g., double-exit a position already half-closed). Prepending after
keeps the trigger engine's sizing math correct.

### Binding constraint: stop advances regardless

If min_hold prevents a partial exit, the stop still advances. This is intentional:
the protective function (stop) should not depend on the discretionary function (minimum
hold time). The partial exit order is still emitted with `exit_blocked=True` so future
analysis can account for cases where an exit was delayed by the hold constraint.

---

## Interaction with Existing Systems

### Runbook 42 (level-anchored stops)

Runbook 42 sets `stop_price_abs` at entry. Runbook 45 reads this as `initial_stop_abs`
to compute `initial_risk_abs`, then subsequently overwrites `stop_price_abs` as the
trade matures. The Runbook 42 initial value is used only for initialization.

### Existing ATR trailing (`trail_distance` / `PositionRiskState.trail_price`)

When `allow_atr_tighten=True` (default):
- ATR trail can move the effective stop above the rung stop (tighter) → allowed
- ATR trail can never move the stop below the rung stop (looser) → rejected

Effective stop = `max(rung_stop, atr_trail)` for longs; `min(rung_stop, atr_trail)` for shorts.
This means Runbook 45 provides the floor; ATR trailing provides the ceiling.

### Runbooks 43/44 (signal ledger, model integration)

Runbook 45 runs in parallel with 43/44 — no shared dependencies. `PartialExitEvent` and
`StopAdjustmentEvent` should be included in the signal ledger schema to give the model
visibility into management actions as features.

---

## Human Verification Evidence

```
TODO: Run a backtest with stop_anchor_type="donchian_lower". Inspect:
  1. initial_risk_abs = entry_price - donchian_lower at entry (constant thereafter)
  2. StopAdjustmentEvent at R1: new_stop = entry - 0.25 * initial_risk_abs
  3. PartialExitEvent at R2: fraction_exited=0.50, exit_R≈2.0
  4. StopAdjustmentEvent at R2: new_stop = entry + 0.5 * initial_risk_abs
  5. Verify no event has new_stop < old_stop (for longs)
  6. Run a trade that jumps 0.8→2.4R in one bar: verify both r1 and r2 events fire
     with rung_catch=True, in that order
```

## Test Evidence

```
TODO
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-18 | Runbook created | Claude |
| 2026-02-18 | Revised: added R definition constraints, multi-rung jump rule, binding constraint, ATR interaction, audit event schemas, wick buffer config | Claude |

---

## Worktree Setup

```bash
git fetch && git pull
git checkout -b feat/adaptive-trade-management
# Depends on: Runbook 42 (feat/level-anchored-stops) merged
```

## Git Workflow

```bash
git add backtesting/llm_strategist_runner.py \
  agents/strategies/trigger_engine.py \
  schemas/trade_set.py \
  prompts/strategy_plan_schema.txt \
  tests/test_adaptive_trade_mgmt.py

uv run pytest tests/test_adaptive_trade_mgmt.py \
             tests/test_trigger_engine.py -vv

git commit -m "Add adaptive trade management: R-multiple state machine with stop progression, partial exits, and audit events"
```
