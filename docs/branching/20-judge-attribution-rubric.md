# Branch: judge-attribution-rubric

## Purpose
Define the Judge Attribution Rubric as a production evaluation contract for a plan-driven, trigger-gated, policy-governed system.

Core mandate:
> **“The Judge’s job is not to fix trades, but to correctly attribute outcomes so the next plan is better than the last.”**

## Source Plans
- docs/analysis/CONTINUOUS_POLICY_PIVOT_PLAN.md
- docs/branching/later/_judge-unification.md (service-unification concepts reused where applicable)

## Scope
- Define strict attribution buckets across `plan -> trigger -> policy -> execution -> safety`.
- Enforce single-primary attribution and evidence-based decisions.
- Standardize Judge output schema and allowed actions.
- Map telemetry expectations to attribution decisions.
- Keep Judge behavior consistent across live and backtest paths.

## Out of Scope / Deferred
- Intra-plan trade micromanagement.
- Trigger or policy changes applied directly by the Judge mid-plan.
- Execution engine mechanics (order routing/slicing).
- Strategy prompt redesign unrelated to attribution.

## First Principles (Non-Negotiable)
- Judge does not intervene intra-plan.
- Judge does not micromanage trades.
- Judge attributes outcomes over an evaluation window, then decides if replan is warranted.
- Every evaluation must land in exactly one primary attribution bucket.

## Attribution Layers (Ordered, Mutually Exclusive)

### Layer A — Plan / Strategist Attribution
Question: Was the plan wrong for the regime?

Primary signals:
- Multiple triggers fire correctly, but expectancy and directional alignment are persistently poor.
- Losses continue even when policy exposure is conservative.
- Underperformance is broad across symbols/timeframes.

Judge action:
- `replan`
- Suggest trigger-set/regime-filter/symbol-scope changes.
- Do not lead with policy parameter changes.

Canonical verdict:
> “The active triggers were structurally misaligned with market regime; losses are attributable to plan selection.”

### Layer B — Trigger Attribution
Question: Did triggers fire at the wrong times or with poor signal quality?

Primary signals:
- Frequent firing in chop/noise.
- High false-positive behavior and rapid directional flips.
- Policy constrained exposure correctly, but entries remain net-negative.

Judge action:
- `replan`
- Tighten trigger definitions/thresholds/cooldowns and scope.

Canonical verdict:
> “Triggers fired permissively in low-quality conditions; attribution is to trigger signal quality, not policy behavior.”

### Layer C — Policy Attribution
Question: Given correct triggers, did sizing/de-risking behave poorly?

Primary signals:
- Direction broadly correct, but sizing too large or de-risk too late.
- Drawdowns are spiky and concentrated.
- Risk caps/stand_down frequently rescue oversized exposure.

Judge action:
- `policy_adjust`
- Typical adjustments: `tau` up, `vol_target` down, `w_max` down.
- Triggers may be secondary notes, not primary fix.

Canonical verdict:
> “Directional permission was sound, but exposure scaling amplified adverse moves; attribution is to policy risk expression.”

### Layer D — Execution Attribution
Question: Was intent correct but execution poor?

Primary signals:
- Slippage/fees dominate losses.
- Fill quality or latency materially degrades realized outcomes.
- Large divergence between theoretical and realized P&L.

Judge action:
- `investigate_execution` (no replan)
- Optional `stand_down` recommendation for affected symbols.

Canonical verdict:
> “Intent and sizing were appropriate; losses are attributable to execution inefficiencies.”

### Layer E — Safety / Emergency Attribution
Question: Did safety controls fire appropriately?

Primary signals:
- Emergency exits, risk caps, or stand_down overrides were activated.

Judge responsibility:
- Verify correctness and timing.
- Never penalize emergency controls for opportunity cost.

Canonical verdict:
> “Emergency controls engaged as designed; outcome acceptable given safety priority.”

## Required Output Contract

For every evaluation window, Judge must output:

```json
{
  "primary_attribution": "plan | trigger | policy | execution | safety",
  "secondary_factors": ["optional list"],
  "confidence": "low | medium | high",
  "recommended_action": "hold | policy_adjust | replan | investigate_execution | stand_down",
  "evidence": {
    "metrics": ["..."],
    "trade_sets": ["..."],
    "events": ["..."]
  }
}
```

Rules:
- Exactly one `primary_attribution`.
- Evidence must reference persisted telemetry/trade-set/event records.
- `replan` is allowed only when `primary_attribution` is `plan` or `trigger`.
- `policy_adjust` is primary only when `primary_attribution` is `policy`.
- `safety` attribution is a control audit, not a performance blame bucket.

## Anti-Patterns (Forbidden)
- “A bit of everything” as primary conclusion.
- Equal blame between trigger and policy.
- Trigger edits proposed when policy sizing is primary fault.
- Penalizing emergency exits for missed upside.
- Mid-plan parameter thrashing.

## Telemetry Minimums for Attribution
Judge decisions must be backed by:
- Window metrics: expectancy, win rate, hold time, MAE/MFE, drawdown profile.
- Policy metrics: target weights (`raw/policy/capped/final`), overrides, stand_down events.
- Trigger metrics: fire counts, direction flips, post-fire outcome windows.
- Execution metrics: slippage/fees/fill quality/latency proxies.
- Safety events: emergency exits and override reason codes.

## Implementation Pattern (from Judge Unification Concepts)
- Use a unified Judge evaluation service across live and backtest paths (e.g., `JudgeFeedbackService`).
- Inject deterministic heuristic pre-analysis into Judge context before LLM verdict generation.
- Support deterministic shim mode in backtests for reproducible attribution tests.
- Keep attribution schema and action rules identical across transports.

## Acceptance Criteria
- Every evaluation emits exactly one primary attribution with evidence references.
- Replan recommendations occur only for Plan or Trigger attributions.
- Policy recommendations are isolated to policy parameters (`tau`, `vol_target`, `w_max`, etc.).
- Safety events are audited, not treated as P&L mistakes.
- Live/backtest judge paths produce schema-compatible outputs under the same rubric.

## Key Files
- `services/judge_feedback_service.py` (unified attribution/evaluation service)
- `schemas/judge_feedback.py` (attribution output schema)
- `agents/judge_agent_client.py` (live judge path)
- `trading_core/judge_agent.py` (core judge path)
- `backtesting/llm_strategist_runner.py` (backtest judge path + shim support)

## Test Plan (required before commit)
- `uv run pytest tests/test_judge_attribution_schema.py -vv`
- `uv run pytest tests/test_judge_attribution_rules.py -vv`
- `uv run pytest tests/test_judge_replan_gating.py -vv`
- `uv run pytest tests/test_judge_unified_service.py -vv`
- `uv run pytest -k judge -vv`

If tests cannot run locally, obtain user-run output and paste it in Test Evidence before committing.

## Human Verification (required)
- Run one evaluation window per attribution class (or representative fixtures) and verify exactly one primary attribution each time.
- Confirm policy-fault windows yield `policy_adjust`, not `replan`.
- Confirm trigger/plan-fault windows are the only ones yielding `replan`.
- Confirm emergency-control windows are classified as `safety` without opportunity-cost criticism.

## Worktree Setup (recommended for parallel agents)
Use a linked worktree so multiple branches can be worked on in parallel from one clone.

```bash
git fetch
git worktree add -b judge-attribution-rubric ../wt-judge-attribution-rubric main
cd ../wt-judge-attribution-rubric

# When finished (after merge)
git worktree remove ../wt-judge-attribution-rubric
```

## Git Workflow (explicit)
```bash
git checkout main
git pull
git checkout -b judge-attribution-rubric

git status
git diff

git add services/judge_feedback_service.py schemas/judge_feedback.py agents/judge_agent_client.py trading_core/judge_agent.py backtesting/llm_strategist_runner.py tests docs/branching/20-judge-attribution-rubric.md docs/branching/README.md

uv run pytest tests/test_judge_attribution_schema.py -vv
uv run pytest tests/test_judge_attribution_rules.py -vv
uv run pytest tests/test_judge_replan_gating.py -vv
uv run pytest tests/test_judge_unified_service.py -vv

# Commit only after Test Evidence + Human Verification Evidence are recorded
git commit -m "Judge: add attribution rubric contract and action gating"
```

## Change Log (update during implementation)
- 2026-02-05: Judge Attribution Rubric implementation complete.
  - Added attribution type definitions to `schemas/judge_feedback.py`: `AttributionLayer`, `AttributionConfidence`, `RecommendedAction`
  - Added `AttributionEvidence` and `JudgeAttribution` schemas with model validators for action gating
  - Added `attribution` field to `JudgeFeedback` schema
  - Implemented `compute_attribution()` method in `services/judge_feedback_service.py`
  - Added helper methods: `_is_policy_attribution()`, `_is_trigger_attribution()`, `_is_plan_attribution()`
  - Modified `_feedback_from_heuristics()` to include attribution in fallback path
  - Created 3 test files with 62 tests covering all acceptance criteria

## Test Evidence (append results before commit)

```
uv run pytest tests/test_judge_attribution_schema.py tests/test_judge_attribution_rules.py \
  tests/test_judge_replan_gating.py -v

============================== 62 passed in 9.16s ==============================

Tests cover:
- Attribution schema validation (21 tests)
- Attribution layer selection based on metrics (16 tests)
- Replan/policy_adjust action gating (25 tests)
- Evidence requirement validation
- Serialization/deserialization
```

## Human Verification Evidence (append results before commit when required)

**Verified via unit tests:**
- `test_high_emergency_exit_rate_triggers_safety` - safety attribution with hold action
- `test_decent_win_rate_poor_profit_factor` - policy attribution with policy_adjust action
- `test_low_win_rate_triggers_trigger_attribution` - trigger attribution with replan action
- `test_very_low_score_triggers_plan_attribution` - plan attribution with replan action
- `test_replan_blocked_for_policy/execution/safety` - action gating enforcement
