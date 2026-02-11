# Branch: judge-stance-enforcement

## Purpose
Enforce `recommended_stance` deterministically so the strategist cannot ignore defensive or wait guidance.

## Source Evidence
- Backtest `7c860ae1`: `recommended_stance` was advisory only and ignored in 22/22 plans.

## Root Cause
`recommended_stance` lives in `strategist_constraints` but is only passed as prompt context, not enforced.

## Scope
- Add `recommended_stance` to judge JSON contract and prompt instructions.
- Enforce stance at plan generation time with deterministic caps or overrides.
- Add audit telemetry when a plan conflicts with recommended stance.

## Out of Scope / Deferred
- Prompt redesign or stylistic changes.
- Policy engine changes beyond stance gating.

## Key Files
- `prompts/llm_judge_prompt.txt`
- `schemas/judge_feedback.py`
- `services/strategist_plan_service.py`
- `backtesting/llm_strategist_runner.py`
- `agents/strategies/plan_provider.py`

## Frontend Impact
- Event Timeline should surface stance overrides (e.g., `stance_override=defensive`).
- Agent Inspector should display the enforced stance and trigger caps.

## Implementation Steps

### Step 1: Add recommended_stance to judge JSON
Update judge prompt JSON schema to include `recommended_stance` explicitly.

### Step 2: Enforce stance in plan service
If `recommended_stance=defensive`:
- Cap `max_triggers_per_symbol_per_day` to a low ceiling (e.g., 2–4).
- Reject “aggressive” plans and force `plan.stance="defensive"` if present.
If `recommended_stance=wait`:
- Allow empty-trigger plans and enforce `max_trades_per_day=0`.

### Step 3: Emit stance enforcement events
Emit `judge_action_applied` with `stance_override` when the plan is forced into defensive or wait mode.

## Test Plan
```bash
uv run pytest tests/test_judge_recommended_stance.py -vv
uv run pytest tests/test_strategist_plan_service.py -k stance -vv
```

## Test Evidence
```
TODO
```

## Acceptance Criteria
- Defensive stance caps trigger count deterministically.
- Wait stance produces empty-trigger plans and blocks trades.
- Plan logs show stance overrides with explicit reasons.

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-11 | Runbook created from backtest 7c860ae1 analysis | Claude |

## Worktree Setup (recommended for parallel agents)
```bash
git fetch
git worktree add -b judge-stance-enforcement ../wt-judge-stance-enforcement main
cd ../wt-judge-stance-enforcement

# When finished (after merge)
git worktree remove ../wt-judge-stance-enforcement
```

## Git Workflow
```bash
git checkout main
git pull
git checkout -b judge-stance-enforcement

# ... implement changes ...

git add prompts/llm_judge_prompt.txt \
  schemas/judge_feedback.py \
  services/strategist_plan_service.py \
  backtesting/llm_strategist_runner.py \
  agents/strategies/plan_provider.py

uv run pytest -k stance -vv
git commit -m \"Judge: enforce recommended stance in plan generation\"
```
