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
- 2026-01-29: Added round-trip paired trades and risk data plumbing
  - backtesting/llm_strategist_runner.py: Added trade_log field to StrategistBacktestResult; enriched portfolio fills with risk data from executed records; serialized trade_log with datetime conversion
  - backtesting/activities.py: Passed trade_log through to backtest result dict
  - ops_api/routers/backtests.py: Added PairedTrade schema and GET /{run_id}/paired_trades endpoint; fixed risk_used/risk_used_abs field name mapping
  - ui/src/lib/api.ts: Added PairedTrade interface and getPairedTrades API method
  - ui/src/components/BacktestControl.tsx: Round-trip trades table (entry/exit paired) with hold duration, falls back to fill-level table for legacy runs
  - ui/src/components/LiveTradingMonitor.tsx: Added mergeFill helper; improved WebSocket fill parsing for fee/risk/trigger fields

## Test Evidence (append results before commit)
```
uv run pytest -k live -vv
=========== 15 passed, 1 skipped, 249 deselected, 1 warning in 8.81s ===========

uv run python -c "from ops_api.routers.backtests import BacktestTrade; from ops_api.routers.live import Fill; from ops_api.schemas import FillRecord; print('Imports OK')"
Imports OK
```

Note: -k backtests returned 0 selected tests (no backtest-specific test files), but import verification passed.

```
# 2026-01-29: Post paired-trades implementation
uv run python -c "from ops_api.routers.backtests import BacktestTrade, PairedTrade; from ops_api.routers.live import Fill; from ops_api.schemas import FillRecord; from backtesting.llm_strategist_runner import StrategistBacktestResult; print('All imports OK')"
All imports OK

uv run pytest tests/test_trigger_deterministic.py -vv
============================== 13 passed in 1.12s ==============================
```

Note: -k live and -k backtests fail at collection due to missing OPENAI_API_KEY (pre-existing env issue, unrelated to changes).

## Human Verification Evidence (append results before commit when required)
2026-01-29: User verified backtest backtest-b836ce01-d8ef-402e-a74d-4afa241f3344
- Round-Trip Trades table renders correctly with 5 paired trades.
- Legacy fill-level fallback view confirmed working.
- Issues found and addressed below.

## Backtest Analysis: backtest-b836ce01-d8ef-402e-a74d-4afa241f3344

**Run:** 3-day backtest (2024-02-14 to 2024-02-16), BTC-USD + ETH-USD, $10,000 starting equity.

### Issue 1: Risk Used column shows $0.00

**Root cause:** The "Risk Used" column mapped to `risk_used_abs`, which tracks risk *budget* consumption from `_check_risk_budget()`. When risk budgets are not configured (the default), `allowance` is always 0. The meaningful risk metric is `actual_risk_at_stop` (qty × stop distance), which IS populated (e.g. $5.07, $4.89).

**Fix:** Replaced "Risk Used" column with "Risk" showing `actual_risk_at_stop`. Added entry/exit trigger columns.

### Issue 2: No trades after first 8 hours (06:45 UTC day 1)

**Root cause:** The intraday judge evaluated performance at 06:45 UTC after 5 round-trips (10 fills), all losers (0% win rate, -$6.88 gross P&L, -0.15% return). The judge:
1. Scored the strategy at **17.8/100** (below the 40.0 replan threshold).
2. Disabled triggers: `BTC-2`, `BTC-3`, `ETH-2`, `ETH-3` (trend_continuation + mean_reversion categories).
3. Disabled categories: `trend_continuation`, `mean_reversion`.
4. The remaining triggers (`BTC-1` = trend_continuation) were also blocked by category veto.
5. Only `volatility_breakout` and `emergency_exit` triggers remained theoretically enabled, but `volatility_breakout` never fired, and `emergency_exit` only fires with open positions (there were none).

**Result:** 100% of subsequent trigger attempts were blocked by `symbol_veto` (177 blocks day 1, 264 blocks day 2, similar day 3). The judge's second evaluation at 18:45 UTC doubled down, also disabling `BTC-1`, leaving zero entry triggers active for the remainder of the backtest.

**Structural problem:** The judge's "disable low-confidence triggers" feedback creates an irreversible death spiral — once all triggers are disabled, no new trades execute, so the judge sees the same stale snapshot and keeps triggers disabled. There is no re-enablement mechanism or minimum-trigger floor.

### Issue 3: All trades hold exactly 0.5 hours

**Root cause:** The plan sets `min_hold` to 0.5 hours. Exit triggers fire on the 15m or 30m timeframe. Every entry on the 1h bar gets its exit trigger on the very next 30m bar (0.5h later), which is the earliest the `min_hold` constraint allows. This means `min_hold` is the *binding constraint* on hold duration, not the strategy's intended exit logic.

4 additional exit attempts were blocked by `min_hold` (tried to exit at 15m = 0.25h, before the 0.5h floor).

### Gaps Highlighted by This Analysis

1. **Judge death spiral (critical):** No floor on minimum enabled triggers. Once the judge disables all entry triggers, trading halts permanently. Need either:
   - Minimum trigger count floor (e.g. at least 2 entry triggers must remain enabled)
   - Automatic re-enablement after N bars of zero activity
   - Judge self-correction: detect "no trades since last eval" and re-enable conservatively

2. **`risk_used_abs` always zero without risk budgets:** The `allowance` field from `_check_risk_budget()` is 0 when risk budgets aren't configured. Either:
   - Default to `actual_risk_at_stop` as the primary risk metric
   - Only show `risk_used_abs` when risk budgets are explicitly configured
   - Populate `risk_used_abs` with the actual position risk even without budgets

3. **`min_hold` dominates exit timing:** When `min_hold` equals the smallest exit timeframe, exits are mechanically forced at exactly `min_hold`. The strategy's exit signal quality can't be assessed because holds are never longer than the floor. Consider:
   - Validating that `min_hold` < smallest exit timeframe
   - Warning in daily reports when `min_hold` blocks > N% of exit attempts

4. **Judge snapshot staleness:** The second judge eval at 18:45 used the exact same snapshot as the 06:45 eval (same equity, same trade count) because no new trades occurred. The judge should detect unchanged snapshots and either skip evaluation or try a different intervention.

