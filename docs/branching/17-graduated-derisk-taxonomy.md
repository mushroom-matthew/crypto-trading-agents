# Runbook: Exit Taxonomy & Partial Exit Ladder

## Overview
The trigger engine currently treats exits as binary: either a normal strategy exit or an `emergency_exit` safety interrupt. The system jumps from "full risk" to "flat" with no intermediate state. This causes overreaction and churn when conditions are merely deteriorating, and loss of edge in volatile-but-not-failing regimes.

This runbook adds a graduated de-risk ladder: new trigger categories with partial exit capability, strict precedence tiering, and full guardrail coverage.

## Core Design Principle
- **`emergency_exit`** is a **circuit breaker** (hard interrupt). Unconditional. Non-negotiable.
- **`risk_reduce`** and **`risk_off`** are **strategy/risk-management** (negotiable, bounded, subject to normal guardrails).

Do not overload `emergency_exit`. If you want "less conservative emergency" behavior, raise the threshold for when emergency triggers fire (LLM prompt + rule logic), not by weakening emergency's priority.

## Motivation
- Emergency exits are unconditional safety interrupts (hardened in commit 8fc5c76). They should remain rare and non-negotiable.
- Many real deterioration scenarios are better served by reducing exposure 25-50% rather than flattening entirely. A full flatten on every risk signal causes unnecessary whipsaw and re-entry costs.
- The LLM strategist currently has no way to express "trim position" — only "exit" or "hold."
- The graduated ladder gives the system an intermediate state where it de-risks without fully giving up the position, and a way to encode "conditions weakening" separately from "something is broken."

## Category Definitions

| Category | Goal | Flatten? | `exit_fraction` |
|---|---|---|---|
| normal exit | Strategy-driven close | Full (1.0) | N/A |
| `risk_reduce` | Trim exposure by X% | Partial | 0 < f < 1.0 |
| `risk_off` | Move to minimal/defensive exposure | Full (1.0) initially | 1.0 (keep simple; "core exposure" concept deferred) |
| `emergency_exit` | Hard safety interrupt — exposure to zero | Full (1.0) | N/A |

## Precedence Tiering (Dedup)

Strict precedence, no exceptions:

### Tier 0: `emergency_exit`
- Always preempts entries and all other exits.
- Always emits `emergency_exit_preempts_entry` block event when it preempts something.
- Still subject to same-bar veto and min-hold guards.
- Bypasses risk caps that would prevent exiting (but NOT sanity checks like stale data).

### Tier 1: `risk_off`
- Preempts entries only if risk regime is "risk_off" OR a configured risk_off latch is active.
- Otherwise competes like normal exits (confidence-based override applies).
- Does NOT bypass risk checks, hold rules, or min-hold.

### Tier 2: `risk_reduce`
- Competes with other exits and entries using existing confidence-based scoring rules.
- Must never bypass hold rules or risk checks.
- No special dedup priority — uses standard exit-vs-entry resolution.

### Tier 3: normal exit
- Standard strategy exits. Current behavior unchanged.

### Dedup resolution rules
1. `emergency_exit` always wins (existing invariant).
2. `risk_off` beats normal exits; beats entries only in risk-off regime.
3. `risk_reduce` and normal exits compete on confidence like today.
4. `risk_reduce` + normal exit on same bar for same symbol: `risk_reduce` wins (partial reduction is the safer action).
5. Entries can override `risk_reduce` and `risk_off` (non-latched) with sufficient confidence.

## Guardrails

`risk_reduce` and `risk_off` must obey ALL of:
- Same-bar veto rules
- Min-hold rules
- Risk checks (spread, liquidity)
- Min-trade thresholds

`emergency_exit` bypasses risk caps that would prevent exiting, but respects:
- Same-bar veto (anti-whipsaw)
- Min-hold enforcement
- Stale data sanity checks

## Partial Exit Capability

### Approach: `exit_fraction` (fixed fraction)
Use `exit_fraction` (0 < f <= 1.0) on exit triggers. This is composable with the current "position or flat" architecture and is the smallest safe step.

Do NOT implement `target_weight` / `target_qty` in this runbook. That's the direction for the policy engine, but it's a bigger change. Fixed fraction first.

### Schema change
```python
class TriggerCondition(SerializableModel):
    # ... existing fields ...
    exit_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    # None = full exit (backward compatible)
    # 0.5 = close 50% of position
```

### Execution
- Parameterize `_flatten_order` with `fraction=1.0` default, OR add `_reduce_order(fraction)`.
- Partial close must produce clean fills and correct ledger accounting.
- Remaining position must be tracked correctly for subsequent bar evaluation.

## Implementation Phases

### Phase A: TradeSet groundwork (hard prerequisite — see "Hard Dependency" section)
1. Introduce `TradeLeg` and `TradeSet` types in API and UI.
2. Backtest runner emits legs with unique `leg_id` and TradeSet grouping.
3. Deprecate `PairedTrade`; backward-compatible collapsed view for 1-entry/1-exit sets.
4. UI: expandable TradeSet rows.
5. **Acceptance**: metrics reconcile with old pair view. No fill_index collisions.

### Phase 1: Schema + taxonomy (no behavior change)
6. Add `risk_reduce` and `risk_off` to `TriggerCategory` literal in `schemas/llm_strategist.py`.
7. Add `exit_fraction: float | None = None` field to `TriggerCondition` with validator (`0 < f <= 1.0` when set).
8. Default `exit_fraction` to `None` (backward compatible — all existing triggers behave identically).
9. Update prompts to recognize categories but don't require usage yet.
10. **Tests**: schema validation for new categories and exit_fraction bounds.

### Phase 2: Partial exit execution (requires Phase A complete)
11. Parameterize `_flatten_order(fraction=1.0)` or add `_reduce_order(fraction)`.
12. Route `risk_reduce` exits through the partial path when `exit_fraction` is set.
13. Ensure fills/reconciliation handle partial closes cleanly (remaining position correct in portfolio state, WAC updated correctly).
14. **Tests**: partial close produces correct order quantity, portfolio state after partial close is accurate, multiple partial closes accumulate correctly. A 1-entry/3-exit scenario renders as one TradeSet with 4 legs and correct total P&L.

### Phase 3: `risk_reduce` behavior
15. Wire `risk_reduce` into the trigger engine exit path (uses normal exit path, no special dedup priority).
16. Enforce all guardrails: same-bar veto, min-hold, risk checks, min-trade thresholds.
17. **Tests**: risk_reduce respects hold rules, risk checks, min-hold. risk_reduce competes with entries on confidence. emergency_exit still wins over risk_reduce in dedup.

### Phase 4: `risk_off` behavior
18. Add risk_off latch mechanism (regime-dependent activation).
19. Add intermediate dedup priority (Tier 1).
20. risk_off goes through exit path, obeys all guardrails.
21. **Tests**: risk_off preempts entries only in risk-off regime. risk_off loses to emergency_exit. risk_off obeys guardrails.

### Phase 5: Strategist integration + validation
22. Update `prompts/llm_strategist_prompt.txt` and strategy templates with category guidance:
    - `risk_reduce`: "conditions weakening but not critical" (trend losing momentum, approaching resistance, unfavorable volume profile).
    - `risk_off`: "defensive posture warranted" (regime shift signal, correlation breakdown starting, sustained adverse moves).
    - `emergency_exit`: "something is broken, exposure must go to zero" (flash crash, liquidity gap, exchange anomaly, circuit breaker conditions).
23. Backtest with risk_reduce triggers to measure whipsaw reduction vs full-flatten-only baseline.

## Telemetry Events
Each new category should emit block events when it preempts or is preempted:
- `risk_reduce_preempts_exit` — risk_reduce won over a normal exit
- `risk_off_preempts_entry` — risk_off preempted an entry (risk-off regime)
- `emergency_exit_preempts_entry` — already implemented
- `entry_overrides_risk_reduce` — high-confidence entry won over risk_reduce
- `entry_overrides_risk_off` — high-confidence entry won over risk_off (non-latched)

## Key Files

**TradeSet / accounting:**
- `ops_api/routers/backtests.py` — `PairedTrade` → `TradeSet` / `TradeLeg` types, fill indexing
- `ui/src/lib/api.ts` — `PairedTrade` interface → `TradeSet` / `TradeLeg`
- `ui/src/components/BacktestControl.tsx` — round-trip table → expandable TradeSet rows
- `backtesting/llm_strategist_runner.py` — `PortfolioTracker`, trade_log, fill ID assignment

**Trigger taxonomy / partial exits:**
- `schemas/llm_strategist.py` — TriggerCategory, TriggerCondition, exit_fraction field
- `agents/strategies/trigger_engine.py` — _flatten_order(fraction), _deduplicate_orders priority tiers, on_bar routing
- `agents/strategies/trade_risk.py` — risk_reduce and risk_off must NOT bypass risk checks
- `prompts/llm_strategist_prompt.txt` — category guidance for LLM
- `prompts/strategies/*.txt` — per-strategy category emphasis
- `tests/test_trigger_engine.py` — partial close, dedup tiers, guardrail tests

## Acceptance Criteria
- TradeSet is source-of-truth for trade reporting (Phase A gate).
- `risk_reduce` triggers produce partial exit orders using `exit_fraction`.
- `risk_off` triggers flatten with intermediate dedup priority (regime-dependent).
- Precedence: `emergency_exit` > `risk_off` (latched) > `risk_reduce` / normal exits.
- `risk_reduce` and `risk_off` do NOT bypass risk checks, hold rules, or min-hold.
- `emergency_exit` semantics unchanged (unconditional, non-negotiable).
- All preemption/override events are auditable via block events.
- WAC accounting produces correct per-leg and per-set P&L.
- No fill_index collisions when multiple fills share a timestamp.
- A 1-entry/3-exit scenario renders as one TradeSet with 4 legs and correct total P&L.
- Backward compatibility: 1-entry/1-exit TradeSet collapsed view matches old PairedTrade display.
- Strategist prompts updated with clear guidance on when to use each category.
- Backtest evidence showing reduced whipsaw compared to emergency-exit-only baseline.

## Hard Dependency: TradeSet (Position Lifecycle Accounting)

**Partial exits must not be enabled in UI/backtest reports until TradeSet is the source-of-truth representation. Pair-based reporting will be incorrect.**

### Mental model shift

Stop thinking in "pairs." A trade is not a pair (1:1 entry + exit). A trade is a **position lifecycle**: the sequence of legs from flat → non-flat → ... → flat.

Example — one TradeSet with 4 legs:
```
buy 1.0 BTC @ 40000        (leg 1: entry)
sell 0.3 BTC @ 41000        (leg 2: risk_reduce)
buy 0.2 BTC @ 40500         (leg 3: scale back in)
sell 0.9 BTC @ 42000        (leg 4: final close → position returns to 0)
```

No many-to-many "which sell belongs to which buy" linking. That leads to tax-lot accounting (FIFO/LIFO/specific identification) which is complex, brittle, and unnecessary for current system goals.

### Terminology

| Context | Term |
|---|---|
| UI / API | `TradeSet` |
| Internal accounting | `PositionLifecycle` |
| Individual fill | `TradeLeg` |

### Grouping rule

One TradeSet per symbol per continuous non-flat interval:
- **Start**: position goes from 0 → non-zero.
- **End**: position returns to 0.
- **Same-bar flatten + reopen**: two separate TradeSets (cleanest; no merging).

### Accounting: Weighted Average Cost (WAC)

We use WAC accounting for position lifecycle P&L:
- Maintain `position_qty` and `average_entry_price` (WAC basis).
- On buys: WAC recomputed as `total_notional / abs(new_position)`.
- On sells: realized P&L = `sell_qty * (sell_price - WAC_at_time_of_sell)`.
- Per-leg P&L uses WAC basis at the time of the exit leg.
- R-multiple computed per set; leg-level R is optional.

This handles buy-after-risk-reduce cleanly (the scale-in updates the WAC, subsequent exits use the updated basis).

### Data model

```python
@dataclass
class TradeLeg:
    leg_id: str               # Unique ID (uuid assigned at ingestion, NOT timestamp-based)
    side: Literal["buy", "sell"]
    qty: float
    price: float
    fees: float
    timestamp: datetime
    trigger_id: str | None
    category: str | None      # trigger category (risk_reduce, emergency_exit, etc.)
    reason: str | None
    is_entry: bool             # True if this leg opened or added to the position
    exit_fraction: float | None  # set if this came from a risk_reduce trigger

@dataclass
class TradeSet:
    set_id: str
    symbol: str
    timeframe: str
    opened_at: datetime
    closed_at: datetime | None  # None if position still open
    legs: list[TradeLeg]        # ordered by timestamp
    pnl_realized_total: float
    fees_total: float
    # Derived (computed, not stored):
    # num_entries, num_exits, avg_entry_price, avg_exit_price,
    # max_exposure, hold_duration, r_multiple
```

### Fill indexing

The current `fill_index` in `ops_api/routers/backtests.py` uses `(symbol, timestamp)` as key, which collides when multiple fills share a timestamp (common in crypto). Replace with `leg_id` keying. Every fill gets a unique ID at ingestion time — uuid or monotonic sequence number.

### UI rendering

- **Collapsed view**: TradeSet summary row (symbol, side, total P&L, fees, duration, num legs, max exposure, R-multiple).
- **Expanded view**: individual legs with qty, price, reason, category, and cumulative position after each leg.
- **Backward compatibility**: when a TradeSet has exactly 1 entry + 1 exit, the collapsed view looks identical to the current PairedTrade row.

### Phasing (hard dependency)

This is **Phase A** — it must be complete before partial exits (Phase B) can ship:

**Phase A: TradeSet groundwork**
1. Introduce `TradeLeg` and `TradeSet` types in `ops_api/routers/backtests.py` and `ui/src/lib/api.ts`.
2. Backtest runner (`backtesting/llm_strategist_runner.py`) emits legs with unique `leg_id` and TradeSet grouping.
3. Deprecate `PairedTrade`; derive the old paired view from TradeSet when there is exactly 1 entry + 1 exit.
4. UI shows TradeSet summary + expandable legs.
5. **Acceptance**: current pair table still works via TradeSet collapsed view. Metrics reconcile with old view for 1-entry/1-exit sets.

**Phase B: Partial exit execution** (this runbook's Phase 2)
- Only starts after Phase A is merged.
- A 1-entry/3-exit scenario renders as one TradeSet with 4 legs and correct total P&L.
- No key collisions in fill_index when multiple fills share a timestamp.

### Key files
- `ops_api/routers/backtests.py` — `PairedTrade` → `TradeSet` / `TradeLeg` types, fill indexing
- `ui/src/lib/api.ts` — `PairedTrade` interface → `TradeSet` / `TradeLeg`
- `ui/src/components/BacktestControl.tsx` — round-trip table rendering → expandable TradeSet rows
- `backtesting/llm_strategist_runner.py` — `PortfolioTracker._update_position()`, trade_log emission, fill ID assignment

## Out of Scope
- `target_weight` / `target_qty` exit sizing (policy engine direction — bigger change).
- Tax-lot accounting (FIFO/LIFO/specific identification) — WAC is sufficient. Do not try to link individual sells to individual buys.
- Multi-asset correlation-based de-risking (portfolio-level, not trigger-level).
- Dynamic `exit_fraction` based on regime (future optimization after fixed fraction is validated).
- "Core exposure" concept for risk_off (initially risk_off just flattens; selective retention is deferred).

## Change Log
- 2026-01-30: Initial design runbook created.
- 2026-01-30: Expanded with full precedence tiering, guardrail matrix, phased implementation, telemetry events, and explicit `exit_fraction` approach per design review.
- 2026-01-30: Added TradeSet/TradeLeg data model, WAC accounting rule, position lifecycle grouping, fill_id requirement, and Phase A as hard prerequisite for partial exits. Deprecated many-to-many leg linking in favor of sequential lifecycle model.
