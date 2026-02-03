# Branch: policy-pivot-phase0

**Status: DONE** (implemented incrementally across judge-death-spiral and cross-backtest-learnings commits)

## Purpose
Complete Phase 0 prerequisites for the Continuous Policy Pivot by finalizing no-change replan guard and related telemetry.

## Source Plans
- docs/analysis/CONTINUOUS_POLICY_PIVOT_PLAN.md (Phase 0 item C)

## Scope
- Define and enforce "no-change replan" detection (trigger set, risk limits, policy config).
- Record metrics: replan_rate_per_day and no_change_replan_suppressed_count.
- Ensure decision records capture reason and suppression metadata.

## Out of Scope / Deferred
- Deterministic policy engine (Phase 1) and ML p_hat (Phase 2).
- Emergency-exit semantics and persistence (already marked complete).

## Key Files
- backtesting/llm_strategist_runner.py

## Acceptance Criteria
- No-change replans are suppressed based on explicit equivalence checks. **DONE**
- Suppression and replan metrics are recorded and queryable. **DONE**

## Implementation Summary

All three scope items are implemented in `backtesting/llm_strategist_runner.py`:

### 1. No-change replan detection
- `_is_no_change_replan()` (line 1521): Checks trigger diff (added/removed/changed/unchanged counts match previous plan length) AND constraint signature equality via `_plan_constraints_signature()`.
- Covers both judge-triggered and day-boundary replans (D3 extension).

### 2. Metrics recording
- `replan_rate_per_day` (line 3613): Count of replans per day in daily summary.
- `no_change_replan_suppressed_count` (line 3614): Count of suppressed no-change replans per day.
- `no_change_replan_suppressed_by_day` (line 568): Per-day defaultdict accumulator.

### 3. Decision record metadata
- `replan_suppressed` (line 2765): Boolean in plan_log entries.
- `judge_triggered_replan` (line 2766): Whether the replan was judge-initiated.
- `replan_reasons` (line 2764): Comma-separated reason string (initial_plan, new_day, judge_triggered, plan_expired).
- `stripped_by_judge` (line 2775): Shadow plan of triggers removed by judge constraints (D5).
- `stance` (line 2776): "wait" or "active" indicating empty-plan acceptance (D6).

## Change Log
| Date | Author | Summary |
|------|--------|---------|
| 2026-01-30 | Claude Opus 4.5 | Initial implementation: `_is_no_change_replan()`, `no_change_replan_suppressed_by_day` counter, `replan_rate_per_day` and `no_change_replan_suppressed_count` in daily summary, `replan_suppressed` in plan_log. Implemented on judge-death-spiral branch. |
| 2026-02-02 | Claude Opus 4.5 | D3: Extended no-change suppression from judge-triggered replans only to also cover day-boundary replans. Added `stripped_by_judge` shadow plan and `stance` metadata to plan_log. Runbook closed out — all acceptance criteria met. |
