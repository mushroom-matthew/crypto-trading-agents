# Paper Trading 4-Session Audit (2026-03-13)

## Scope
Sessions audited:
- `paper-trading-491140c3`
- `paper-trading-edebc022`
- `paper-trading-201cddc0`
- `paper-trading-3f17442d`

Data sources:
- Temporal workflow execution history (including start args and failure events)
- `data/events.sqlite` (`events` table)
- Live workflow queries (`get_plan_history`, `get_structure_snapshots`, `get_session_status`)

Audit timestamp reference:
- Current state check performed on 2026-03-13 around 10:47 ET (14:47 UTC)

## Executive Summary
- Executed trades across these four sessions: `0`.
- Core blockers were not a single bug; they were a combination of cadence, target resolution, and repeated workflow timeout failures.
- `paper-trading-491140c3` is currently `FAILED` (as of 2026-03-13 14:42:56 UTC) due to `fetch_current_prices_activity` ScheduleToClose timeout.
- `paper-trading-3f17442d` is currently `FAILED` (as of 2026-03-13 14:31:59 UTC) due to the same timeout class.
- `paper-trading-edebc022` recovered into a new running run after earlier failed/terminated attempts.
- `paper-trading-201cddc0` continued-as-new and remains running.
- Two sessions produced entry candidates that were blocked by target/R:R gates (`no_target_rr_undefined` or `target_price_unresolvable`).
- One session (`paper-trading-201cddc0`) generated an empty actionable plan after validation exhaustion (`plan_validation_rejected` + `plan_stand_down`).
- For `201cddc0`, worker logs in the same cycle window show validation attempts failing on missing stop definitions for entry triggers, which explains the exhausted revision loop.

## Current Parallel Session State
From Temporal `describe()`:

| session_id | status | run_id | start_utc | close_utc |
|---|---|---|---|---|
| paper-trading-491140c3 | FAILED | 6f42a54b-455a-4c9e-ad2f-3ebe2b29fb9c | 2026-03-13 13:41:01 | 2026-03-13 14:42:56 |
| paper-trading-edebc022 | RUNNING | 54f4e761-3eb3-477f-8a4d-b72d19a4de79 | 2026-03-13 14:45:04 | - |
| paper-trading-201cddc0 | RUNNING | 53349395-773c-443c-895b-bce31cda232c | 2026-03-13 14:45:43 | - |
| paper-trading-3f17442d | FAILED | 5c877b9f-327b-4479-b958-5f2102221b29 | 2026-03-13 13:41:49 | 2026-03-13 14:31:59 |

## Workflow Start Config (UI-selected strategy inputs)
Decoded from `WorkflowExecutionStarted` input payloads:

| session_id | symbols | strategy profile | indicator_tf | plan_interval_h | direction_bias | screener_regime |
|---|---|---|---|---:|---|---|
| 491140c3 | XRP-USD | momentum/trend-following | 6h | 12.0 | long | volatile_breakout |
| edebc022 | XRP-USD | volatility breakout | 5m | 0.5 | neutral | volatile_breakout |
| 201cddc0 | DOGE/LINK/DOT/POL | conservative/defensive | 1h | 4.0 | neutral | volatile_breakout |
| 3f17442d | BTC/POL | momentum/trend-following | 6h | 12.0 | neutral | volatile_breakout |

## Event-Level Diagnostics (current runs)

| session_id | plan_generated | trigger_count | eval_summary_events | triggers_evaluated | trigger_fired | trade_blocked | order_executed |
|---|---:|---:|---:|---:|---:|---:|---:|
| 491140c3 | 1 | 7 | 1 | 7 | 1 | 3 | 0 |
| edebc022 | 3 | 8, 8, 7 | 14 | 111 | 0 | 1 | 0 |
| 201cddc0 | 1 | 0 | 8 | 0 | 0 | 0 | 0 |
| 3f17442d | 1 | 9 | 2 | 9 | 0 | 1 | 0 |

### Block reasons observed
- `paper-trading-491140c3`
  - `priority_skip` x2 (`Max triggers (1) already fired for XRP-USD`)
  - `target_price_unresolvable` x1 (`Target anchor 'htf_daily_high' resolved to None; entry rejected.`)
- `paper-trading-edebc022`
  - `no_target_rr_undefined` x1 (`Risk constraint no_target_rr_undefined prevented sizing`)
- `paper-trading-201cddc0`
  - no `trade_blocked`; plan had no triggers to evaluate
- `paper-trading-3f17442d`
  - `no_target_rr_undefined` x1 (`Risk constraint no_target_rr_undefined prevented sizing`)

## Why No Trades Happened

### 1) `paper-trading-491140c3`
Primary causes:
- Only one evaluation window occurred so far (6h trigger timeframe + 12h replan cadence, plus run age).
- A trend entry fired, but failed downstream target resolution (`target_price_unresolvable`).
- Additional candidate entries were skipped by per-bar priority cap (`max_triggers_per_symbol_per_bar=1` default behavior).
- Workflow later failed on activity timeout and is now stopped:
  - `Activity task timed out`
  - cause: `activity ScheduleToClose timeout`
  - activity: `fetch_current_prices_activity`
  - failure time: 2026-03-13 14:42:56 UTC

Important nuance:
- The fired trigger (`XRP-TRND-2`) used `target_anchor_type=htf_daily_high` at a market price above that HTF level, so target became invalid for long direction (target not above entry), and entry was rejected.

### 2) `paper-trading-edebc022`
Primary causes:
- Strategy produced active triggers and frequent evaluations (5m), but entry that met conditions was blocked by risk gate `no_target_rr_undefined`.
- Trigger set includes `target_hit` in exit behavior but no target anchor declaration, so R:R is undefined and sizing is blocked by design.
- Remaining evaluations mostly did not satisfy entry conditions.

### 3) `paper-trading-201cddc0`
Primary causes:
- Plan generation failed validation loop (`plan_validation_rejected`, reason `validation_exhausted`), then stand-down.
- Accepted/generated plan had `trigger_count=0`; therefore `triggers_evaluated=0` and no possible executions.

Deep root cause (distinct from the other sessions):
- This failure mode was not a runtime risk gate; it was a plan quality failure before execution.
- Timeline from events + worker logs:
  - 2026-03-13 13:43:51 UTC: judge validation reject reported as `STRUCTURAL: plan has no triggers — cannot approve an empty plan`, then stand-down.
  - 2026-03-13 13:45:14 UTC and 13:46:00 UTC: strategy-plan validation attempts failed on entry triggers missing stop definitions (examples in logs: `TREND_CONT_DOT_001`, `TREND_CONT_POL_001`, `POL-meanrev-entry-1`).
  - 2026-03-13 13:46:30 UTC: generated plan persisted with `trigger_count=0`.
- Likely mechanism:
  - Multi-symbol conservative prompt + strict stop invariants caused repair attempts to fail schema/risk requirements before producing an executable trigger set.
  - Because no valid triggers survived validation, the session entered a dead state (`triggers_evaluated=0`).

Implication:
- `201cddc0` should be treated as a plan-construction failure class, separate from trade-time blocking (`no_target_rr_undefined` / `target_price_unresolvable`).

### 4) `paper-trading-3f17442d`
Primary causes:
- Entry candidate blocked by `no_target_rr_undefined` (same class as edebc022).
- Low evaluation opportunities due 6h cadence (only 2 eval summaries across the observed run window).
- Workflow later failed on activity timeout and is now stopped:
  - `Activity task timed out`
  - cause: `activity ScheduleToClose timeout`
  - activity: `fetch_current_prices_activity`
  - failure time: 2026-03-13 14:31:59 UTC

## Parallel-Run Timeout Analysis
Observed multi-run instability points:
- `paper-trading-491140c3` failed due `fetch_current_prices_activity` ScheduleToClose timeout.
- `paper-trading-edebc022` had two earlier attempts before current running run:
  - failed run: timeout in `fetch_current_prices_activity`
  - terminated run: manual/user termination (matching observed restart behavior)
  - subsequent run continued-as-new into current running run
- `paper-trading-3f17442d` failed with the same timeout class and did not auto-recover into a new run.

Interpretation:
- Timeout mode is consistent and activity-specific (`fetch_current_prices_activity`, ScheduleToClose).
- This is operational/concurrency pressure, not strategy logic.

## Trigger-vs-Strategy Alignment

### Aligned cases
- `491140c3` (momentum/trend): trigger mix included trend continuation + breakout + defensive exits, consistent with selected strategy profile.
- `edebc022` (vol breakout): trigger mix strongly favored volatility-breakout categories, consistent with selected profile.

### Misaligned/degraded cases
- `201cddc0` (conservative/defensive): ended with zero actionable triggers after validation exhaustion.
- `3f17442d` (momentum): generated a `mean_reversion` template and mixed long/short trigger set under neutral direction bias; partially inconsistent with momentum-first intent.

## Why Good Triggers Can Get Blocked (and what to change)

### A. `no_target_rr_undefined` blocks viable entries
Current behavior:
- Risk evaluator blocks entries when exit logic implies `target_hit` but no target anchor exists.

Recommendation:
- Enforce target declaration at plan validation time for all entry triggers that can hit `target_hit`.
- If strict enforcement is undesirable, inject deterministic fallback anchors before execution (with telemetry tag `target_anchor_auto_fallback`).

### B. Priority skip can suppress alternatives before downstream validity checks complete
Current behavior:
- Once one trigger fires in-bar, additional same-symbol triggers are skipped (`priority_skip`) even if the fired order is later rejected by stop/target resolution gates.

Recommendation:
- Two-pass in-bar flow:
  1. Generate candidate entries.
  2. Pre-validate stop/target resolvability and RR.
  3. Apply priority selection among only valid candidates.
- This prevents losing valid secondary entries due to an invalid first candidate.

### C. HTF sessions have very sparse evaluation opportunities early in runtime
Current behavior:
- Tier-2 evaluations occur only on new candle boundaries for trigger timeframes.

Recommendation:
- Add explicit intrabar activation refinement mode for HTF templates (evaluate entry activation at lower execution TF while preserving HTF thesis).
- Or add session-level minimum evaluation cadence fallback when run age < N hours.

### D. Parallel timeout resilience
Recommendation:
- Harden `fetch_current_prices_activity` with:
  - tighter retry policy and jittered backoff,
  - fallback data source or cached last-good price for short grace windows,
  - run-level circuit breaker that pauses trading instead of failing workflow when price feed is transiently unavailable.

### E. Plan-construction failure class (`validation_exhausted` / zero-trigger stand-down)
Current behavior:
- Multi-symbol plans can fail validation repeatedly (e.g., missing required stop definitions), exhaust repair attempts, and degrade to zero-actionable-trigger state.

Recommendation:
- Add staged draft validation before full plan acceptance:
  - validate each entry trigger independently (stop + target + direction/category contracts),
  - reject/fix invalid triggers before global plan validation.
- Add bounded fallback synthesis when repair loop is exhausted:
  - generate minimal valid per-symbol trigger set (1 defensive + 1 entry max) instead of persisting an empty executable plan.
- Add complexity guardrails for multi-symbol conservative prompts:
  - cap first-pass symbol count or generate per-symbol subplans, then merge only validated triggers.
- Emit explicit telemetry class `plan_construction_failed` with structured failure reasons (missing_stop, missing_target, schema_mismatch, etc.).

## Immediate Next Steps (pre-implementation)
1. Add a compile/validation invariant: entry triggers requiring `target_hit` must have resolvable target semantics.
2. Refactor trigger selection to apply `priority_skip` after candidate validity checks.
3. Add timeout hardening for `fetch_current_prices_activity` and auto-resume policy for failed runs.
4. Add plan-construction safeguards for multi-symbol sessions (staged trigger validation + non-empty fallback trigger synthesis).
5. Re-run a controlled 4-session smoke with fixed configs and capture block-reason deltas.
