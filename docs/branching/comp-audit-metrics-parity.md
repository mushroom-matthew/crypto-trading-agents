# Branch: comp-audit-metrics-parity

## Purpose
Bring live metrics and backtest metrics into parity, and resolve win rate/Sharpe inconsistencies.

## Source Plans
- docs/analysis/COMPUTATION_AUDIT_PLAN.md (Phase 2 items 9-11, Phase 3 item 14)

## Scope
- Live daily risk budget tracker based on equity and budget_pct.
- Ledger leverage and fee clarity (ensure leverage reflects exposure / equity, fees included or explicitly documented).
- Centralized win/loss calculation and stable rolling Sharpe/Sortino with crypto 365 annualization.
- Consistent annualization factor for Sharpe in tools/performance analysis.

## Out of Scope / Deferred
- Indicator tuning and scalper cadence.
- UI trade-level visibility (comp-audit-ui-trade-stats).
- Judge unification (later branch).

## Key Files
- services/live_daily_reporter.py
- agents/workflows/execution_ledger_workflow.py
- trading_core/trade_quality.py
- agents/analytics/portfolio_state.py
- tools/performance_analysis.py
- app/strategy/trade_executor.py (only if fee handling changes)

## Dependencies / Coordination
- Coordinate with judge-unification later branch to avoid conflicting changes in trade_quality or portfolio_state.
- Avoid touching trigger_engine or prompt templates here.

## Acceptance Criteria
- Live risk budget scales with equity and matches backtest math.
- Leverage reflects current exposure / equity.
- Fees are included in ledger PnL or explicitly documented as excluded.
- Win rate and Sharpe are consistent across live/backtest; annualization is configurable and uses 365 for crypto.

## Test Plan (required before commit)
- uv run pytest -k trade_quality -vv
- uv run pytest -k portfolio_state -vv
- uv run pytest -k performance_analysis -vv
- uv run python -c "from trading_core import trade_quality; from agents.analytics import portfolio_state"

If tests cannot be run, obtain user-run output and paste it below before committing.

## Human Verification (required)
- If live services are running, fetch the live daily report and confirm risk budget scales with equity and leverage/win rate/Sharpe fields are populated.
- Run a backtest and confirm win rate and Sharpe align with realized PnL and annualization uses 365.
- Paste endpoint output or run id and observations in the Human Verification Evidence section.

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b comp-audit-metrics-parity

# Work, then review changes
git status
git diff

# Stage changes
git add services/live_daily_reporter.py \
  agents/workflows/execution_ledger_workflow.py \
  trading_core/trade_quality.py \
  agents/analytics/portfolio_state.py \
  tools/performance_analysis.py

# Run tests (must succeed or be explicitly approved by user with pasted output)
uv run pytest -k trade_quality -vv
uv run pytest -k portfolio_state -vv
uv run pytest -k performance_analysis -vv
uv run python -c "from trading_core import trade_quality; from agents.analytics import portfolio_state"

# Commit ONLY after test evidence is captured below
git commit -m "Metrics: live parity, leverage, win rate, sharpe"
```

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)

