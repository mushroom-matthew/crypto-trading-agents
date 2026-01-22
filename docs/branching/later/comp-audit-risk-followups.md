# Branch: comp-audit-risk-followups (later)

## Purpose
Track follow-ups discovered during comp-audit-risk-core that were explicitly out of scope (risk-budget test failure and LLM scale-in/short-stop alignment).

## Source Plans
- docs/analysis/COMPUTATION_AUDIT_PLAN.md (Phase 0 follow-ups)
- docs/branching/comp-audit-risk-core.md (Out-of-scope notes and test failures)

## Scope
- Fix LLMStrategistBacktester risk-budget allowance fallback when initial_cash is missing (tests/risk/test_daily_budget_reset).
- Align LLM strategist prompt/schema so shorts include stop_loss_pct (to avoid short_stop_required blocks).
- Decide/implement scale-in behavior for LLM trigger path (allow scale-ins or explicitly block + document).

## Out of Scope / Deferred
- Phase 1+ cadence/indicator/metrics/UI items.
- Risk budget base-equity semantics beyond the test failure.

## Key Files
- backtesting/llm_strategist_runner.py
- prompts/strategy_plan_schema.txt
- prompts/llm_strategist_prompt.txt
- agents/strategies/trigger_engine.py

## Dependencies / Coordination
- Coordinate with comp-audit-risk-core to avoid conflicting RiskEngine signature changes.
- If prompt/schema edits overlap prompt-focused branches, coordinate with comp-audit-indicators-prompts.

## Acceptance Criteria
- tests/risk/test_daily_budget_reset.py passes.
- LLM plan generation includes stop_loss_pct for short triggers in sample runs.
- Scale-in behavior is explicitly supported or explicitly blocked/documented for LLM triggers.

## Test Plan (required before commit)
- uv run pytest tests/risk/test_daily_budget_reset.py -vv
- uv run pytest tests/risk/test_risk_at_stop.py -vv
- uv run pytest tests/test_trigger_engine.py -vv

If tests cannot be run, obtain user-run output and paste it below before committing.

## Human Verification (required)
- Run a targeted LLM backtest that includes at least one short trigger and confirm stop_loss_pct is present in the generated plan.
- If scale-ins are enabled, confirm scale-in entries are either executed or explicitly blocked with a documented reason.
- Paste run id and observations in the Human Verification Evidence section.

## Worktree Setup (recommended for parallel agents)
Use a linked worktree so multiple branches can be worked on in parallel from one clone.

```bash
# From the main repo directory
git fetch
git worktree add -b comp-audit-risk-followups ../wt-comp-audit-risk-followups comp-audit-risk-followups
cd ../wt-comp-audit-risk-followups

# When finished (after merge)
git worktree remove ../wt-comp-audit-risk-followups
```

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b comp-audit-risk-followups

# Work, then review changes
git status
git diff

# Stage changes
git add backtesting/llm_strategist_runner.py \
  prompts/strategy_plan_schema.txt \
  prompts/llm_strategist_prompt.txt \
  agents/strategies/trigger_engine.py

# Run tests (must succeed or be explicitly approved by user with pasted output)
uv run pytest tests/risk/test_daily_budget_reset.py -vv
uv run pytest tests/risk/test_risk_at_stop.py -vv
uv run pytest tests/test_trigger_engine.py -vv

# Commit ONLY after test evidence is captured below
git commit -m "Risk follow-ups: budget reset + short stop + scale-in"
```

## Change Log (update during implementation)
- YYYY-MM-DD: Summary of changes, files touched, and decisions.

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)
