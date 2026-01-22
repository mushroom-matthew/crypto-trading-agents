# Branch: comp-audit-trigger-cadence

## Purpose
Increase scalper cadence and unblock throughput by adjusting min-hold and priority skip behavior, and serialize concurrent signals to prevent racey risk checks.

## Source Plans
- docs/analysis/COMPUTATION_AUDIT_PLAN.md (Phase 1 item 5, Phase 5 item 18)

## Scope
- Reduce min-hold to 1-3 bars for scalp profiles; relax priority skip rules for high-confidence triggers.
- Ensure concurrent signal processing serializes risk checks per symbol per tick.
- Update backtest runner and activity defaults to align with new cadence controls.

## Out of Scope / Deferred
- Indicator/prompt changes (Phase 1 items 6/8) are in comp-audit-indicators-prompts.
- Risk budget and sizing (Phase 0) are in comp-audit-risk-core.
- UI changes for scalper settings.

## Key Files
- agents/strategies/trigger_engine.py
- backtesting/llm_strategist_runner.py
- backtesting/activities.py

## Dependencies / Coordination
- Coordinate with comp-audit-risk-core if any new config wiring is added to plan_provider.
- Avoid touching prompt templates in this branch to keep conflicts minimal.

## Acceptance Criteria
- Execution rate exceeds 50% of valid triggers for scalping configs (measured in backtests).
- Risk checks see updated exposure state before subsequent signals on the same tick.
- No regression in emergency exit or guard logic.

## Test Plan (required before commit)
- uv run pytest tests/test_trigger_engine.py -vv
- uv run pytest tests/test_execution_engine.py -vv
- uv run pytest -k llm_strategist_runner -vv

If any test cannot be run, record the failure reason and obtain user-run output before committing.

## Human Verification (required)
- Run a scalper backtest (5m/15m) and record execution rate, trade count, and min-hold behavior.
- Confirm execution rate exceeds 50% of valid triggers and no duplicate risk checks occur per symbol per tick (from logs or counters).
- Paste run id and observed metrics in the Human Verification Evidence section.

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b comp-audit-trigger-cadence

# Work, then review changes
git status
git diff

# Stage changes
git add agents/strategies/trigger_engine.py \
  backtesting/llm_strategist_runner.py \
  backtesting/activities.py

# Run tests (must succeed or be explicitly approved by user with pasted output)
uv run pytest tests/test_trigger_engine.py -vv
uv run pytest tests/test_execution_engine.py -vv
uv run pytest -k llm_strategist_runner -vv

# Commit ONLY after test evidence is captured below
git commit -m "Trigger cadence: scalper min-hold and signal serialization"
```

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)

