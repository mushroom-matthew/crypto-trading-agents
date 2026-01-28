# Branch: comp-audit-ui-trade-stats

## Purpose
Surface per-trade risk and performance stats in UI for backtest and live views.

## Source Plans
- docs/analysis/COMPUTATION_AUDIT_PLAN.md (Phase 6 item 19)

## Scope
- Add trade-level fields to API responses (risk_used_abs, actual_risk_abs, R-multiple, fees, timestamps, trigger context).
- Render per-trade detail panel or expandable rows in backtest and live monitors.
- Ensure UI shows stop distance, risk multipliers, and budget utilization if tracked.

## Out of Scope / Deferred
- Backend risk math changes (comp-audit-risk-core).
- Indicator, trigger cadence, or metrics parity changes.

## Key Files
- ui/src/components/BacktestControl.tsx
- ui/src/components/LiveTradingMonitor.tsx
- ui/src/lib/api.ts
- ops_api/routers/backtests.py (if trade detail fields need exposure)
- ops_api/routers/live.py (if live trade endpoints need expansion)

## Dependencies / Coordination
- Coordinate with comp-audit-metrics-parity if new fields are introduced in portfolio/trade metrics.
- Ensure API changes are merged before UI expects new fields.

## Acceptance Criteria
- Each trade row exposes risk-weighted stats and sizing context.
- Backtest and live views show consistent trade-level metrics.
- UI handles missing fields gracefully if a legacy run lacks stats.

## Test Plan (required before commit)
- uv run pytest -k backtests -vv
- uv run pytest -k live -vv
- (Optional if UI linting exists) cd ui && npm run lint

If tests cannot be run, obtain user-run output and paste it below before committing.

## Human Verification (required)
- Launch the UI and verify trade-level fields render in both Backtest and Live views (risk_used_abs, actual_risk_abs, R-multiple, fees).
- Confirm legacy runs without new fields do not break the UI.
- Paste a brief UI observation or screenshot description in the Human Verification Evidence section.

## Worktree Setup (recommended for parallel agents)
Use a linked worktree so multiple branches can be worked on in parallel from one clone.

```bash
# From the main repo directory
git fetch
git worktree add -b comp-audit-ui-trade-stats ../wt-comp-audit-ui-trade-stats comp-audit-ui-trade-stats
cd ../wt-comp-audit-ui-trade-stats

# When finished (after merge)
git worktree remove ../wt-comp-audit-ui-trade-stats
```

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b comp-audit-ui-trade-stats

# Work, then review changes
git status
git diff

# Stage changes
git add ui/src/components/BacktestControl.tsx \
  ui/src/components/LiveTradingMonitor.tsx \
  ui/src/lib/api.ts \
  ops_api/routers/backtests.py \
  ops_api/routers/live.py

# Run tests (must succeed or be explicitly approved by user with pasted output)
uv run pytest -k backtests -vv
uv run pytest -k live -vv

# Commit ONLY after test evidence is captured below
git commit -m "UI: trade-level risk and performance stats"
```

## Change Log (update during implementation)
- 2026-01-28: Implemented trade-level risk stats exposure
  - ops_api/routers/backtests.py: Extended BacktestTrade schema with risk_used_abs, actual_risk_at_stop, stop_distance, allocated_risk_abs, profile_multiplier, r_multiple; updated both trade list builders to populate these fields
  - ops_api/routers/live.py: Extended Fill schema with fee, pnl, trigger_id, risk_used_abs, actual_risk_at_stop, stop_distance, r_multiple; updated endpoint to map new FillRecord fields
  - ops_api/schemas.py: Extended FillRecord with fee, pnl, trigger_id, risk_used_abs, actual_risk_at_stop, stop_distance, r_multiple
  - ops_api/materializer.py: Updated list_fills() to extract risk fields from event payload and compute R-multiple
  - ui/src/lib/api.ts: Extended BacktestTrade interface with risk_used_abs, actual_risk_at_stop, stop_distance, allocated_risk_abs, profile_multiplier, r_multiple
  - ui/src/components/BacktestControl.tsx: Added Fee, Risk Used, Actual Risk, R columns to trades table with color-coded R-multiple
  - ui/src/components/LiveTradingMonitor.tsx: Extended Fill interface; added risk stats row to fill cards showing P&L, Risk, and R when available

## Test Evidence (append results before commit)
```
uv run pytest -k live -vv
=========== 15 passed, 1 skipped, 249 deselected, 1 warning in 8.81s ===========

uv run python -c "from ops_api.routers.backtests import BacktestTrade; from ops_api.routers.live import Fill; from ops_api.schemas import FillRecord; print('Imports OK')"
Imports OK
```

Note: -k backtests returned 0 selected tests (no backtest-specific test files), but import verification passed.

## Human Verification Evidence (append results before commit when required)
PENDING: Launch UI and verify trade-level fields render in Backtest and Live views.

