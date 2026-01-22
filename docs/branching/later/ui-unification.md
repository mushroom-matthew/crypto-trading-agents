# Branch: ui-unification (later)

## Purpose
Address optional enhancements from the UI Unification Plan (maintenance work only).

## Source Plans
- docs/UI_UNIFICATION_PLAN.md (Optional Enhancements section)

## Scope
- Agent Inspector tab polish (if already partially implemented).
- Market Monitor tab with dedicated candle view.
- Performance optimizations per OPS_API_PERFORMANCE_PROFILE.
- A/B backtest comparison UI (if not already covered elsewhere).
- DB connection pooling configuration (ops_api).

## Out of Scope / Deferred
- Core backtest/live functionality (already complete).
- Trade-level stats (comp-audit-ui-trade-stats).

## Key Files
- ui/src/components/AgentInspector.tsx (or existing files)
- ui/src/components/MarketMonitor.tsx
- ops_api/** (performance optimizations)
- ui/src/lib/api.ts
- docs/OPS_API_PERFORMANCE_PROFILE.md (reference)

## Dependencies / Coordination
- Coordinate with comp-audit-ui-trade-stats and scalper-mode to avoid UI conflicts.

## Acceptance Criteria
- Optional UI enhancements compile and render without regressions.
- Performance bottlenecks addressed per profile doc.

## Test Plan (required before commit)
- uv run pytest -k ops_api -vv
- (Optional if UI linting exists) cd ui && npm run lint

If tests cannot be run, obtain user-run output and paste it below before committing.

## Human Verification (required)
- Launch the UI and verify new/updated tabs render without regressions.
- If performance optimizations were made, capture basic response-time or query-count observations.
- Paste observations in the Human Verification Evidence section.

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b ui-unification

# Work, then review changes
git status
git diff

# Stage changes (adjust list as needed based on actual edits)
git add ui/src/components \
  ui/src/lib/api.ts \
  ops_api

# Run tests (must succeed or be explicitly approved by user with pasted output)
uv run pytest -k ops_api -vv

# Commit ONLY after test evidence is captured below
git commit -m "UI: optional unification enhancements"
```

## Change Log (update during implementation)
- YYYY-MM-DD: Summary of changes, files touched, and decisions.

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)

