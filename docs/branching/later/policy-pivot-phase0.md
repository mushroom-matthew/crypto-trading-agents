# Branch: policy-pivot-phase0 (later)

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
- backtesting/activities.py
- ops_api/event_store.py or metrics emission files (if required for telemetry)

## Dependencies / Coordination
- Coordinate with comp-audit-trigger-cadence due to shared runner and trigger logic.
- Avoid overlapping prompt/template changes with strategist-tool-loop branch.

## Acceptance Criteria
- No-change replans are suppressed based on explicit equivalence checks.
- Suppression and replan metrics are recorded and queryable.

## Test Plan (required before commit)
- uv run pytest tests/test_trigger_engine.py -vv
- uv run pytest -k llm_strategist_runner -vv

If tests cannot be run, obtain user-run output and paste it below before committing.

## Human Verification (required)
- Run a backtest with frequent replans and confirm no-change replans are suppressed.
- Verify suppression metrics (replan_rate_per_day, no_change_replan_suppressed_count) are recorded.
- Paste run id and observations in the Human Verification Evidence section.

## Worktree Setup (recommended for parallel agents)
Use a linked worktree so multiple branches can be worked on in parallel from one clone.

```bash
# From the main repo directory
git fetch
git worktree add -b policy-pivot-phase0 (later) ../wt-policy-pivot-phase0 (later) policy-pivot-phase0 (later)
cd ../wt-policy-pivot-phase0 (later)

# When finished (after merge)
git worktree remove ../wt-policy-pivot-phase0 (later)
```

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b policy-pivot-phase0

# Work, then review changes
git status
git diff

# Stage changes
git add backtesting/llm_strategist_runner.py \
  backtesting/activities.py

# Run tests (must succeed or be explicitly approved by user with pasted output)
uv run pytest tests/test_trigger_engine.py -vv
uv run pytest -k llm_strategist_runner -vv

# Commit ONLY after test evidence is captured below
git commit -m "Policy pivot: no-change replan guard"
```

## Change Log (update during implementation)
- YYYY-MM-DD: Summary of changes, files touched, and decisions.

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)

