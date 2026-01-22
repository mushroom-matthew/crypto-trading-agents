# Branch: comp-audit-risk-core

## Purpose
Implement Computation Audit Plan Phase 0 risk correctness and budget integrity changes (items 1-4). This is the highest priority branch.

## Source Plans
- docs/analysis/COMPUTATION_AUDIT_PLAN.md (Phase 0 items 1-4)

## Scope
- Remove or conditionalize the min trades/day floor so daily risk budget cannot be exceeded.
- Enforce aggregate risk on scale-in entries using blended entry and shared stop.
- Require a stop for shorts or apply a stricter notional cap when no stop is provided.
- Align simulator risk usage with stop-based sizing (risk_used_abs and actual_risk_at_stop).

## Out of Scope / Deferred
- Phase 1+ cadence/indicator/metrics/UI items.
- Prompt or strategy template changes.
- Phase 5 gap/slippage risk (item 17) unless explicitly pulled in later.

## Key Files
- agents/strategies/plan_provider.py
- services/strategist_plan_service.py
- agents/strategies/risk_engine.py
- agents/strategies/trade_risk.py
- backtesting/simulator.py

## Dependencies / Coordination
- Potential overlap on agents/strategies/plan_provider.py with comp-audit-indicators-prompts; coordinate to avoid conflicts.
- If any RiskEngine changes intersect with metrics-parity branch, align on function signatures before merging.

## Acceptance Criteria
- Derived trade cap never implies total risk above daily budget.
- Scale-in entry blocks when combined risk exceeds cap; risk snapshots include combined/incremental deltas.
- Shorts without stops are blocked or capped safely.
- Simulator risk_used_abs and actual_risk_at_stop reflect stop distance when present.

## Test Plan (required before commit)
Run the most specific tests available; if exact files do not exist, use the -k patterns.

- uv run pytest -k risk_engine -vv
- uv run pytest -k trade_risk -vv
- uv run pytest -k simulator -vv
- uv run python -c "from agents.strategies import risk_engine; from backtesting import simulator"

Do not commit until test output is recorded in the Test Evidence section below. If tests cannot be run locally, obtain user-run output and paste it here before committing.

## Human Verification (required)
- Run a targeted backtest that forces: (a) daily budget near per-trade risk, (b) scale-in entries with a shared stop, (c) a short without an explicit stop.
- Confirm in logs or results: trade cap never implies total risk above budget, scale-in blocked when combined risk exceeds cap, short without stop is blocked or capped.
- Paste run id and observations in the Human Verification Evidence section.

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b comp-audit-risk-core

# Work, then review changes
git status
git diff

# Stage changes
git add agents/strategies/plan_provider.py \
  services/strategist_plan_service.py \
  agents/strategies/risk_engine.py \
  agents/strategies/trade_risk.py \
  backtesting/simulator.py

# Run tests (must succeed or be explicitly approved by user with pasted output)
uv run pytest -k risk_engine -vv
uv run pytest -k trade_risk -vv
uv run pytest -k simulator -vv
uv run python -c "from agents.strategies import risk_engine; from backtesting import simulator"

# Commit ONLY after test evidence is captured below
git commit -m "Risk: enforce daily budget and aggregate stop risk"
```

## Change Log (update during implementation)
- YYYY-MM-DD: Summary of changes, files touched, and decisions.

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)

