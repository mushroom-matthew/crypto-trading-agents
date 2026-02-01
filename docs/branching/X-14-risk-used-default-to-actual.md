# Runbook: Default risk_used_abs to Actual Risk at Stop

## Overview
`risk_used_abs` in fill records is always 0.0 when risk budgets are not configured, because it tracks budget consumption (`allowance` from `_check_risk_budget()`). The meaningful per-trade risk metric is `actual_risk_at_stop` (qty x stop distance), which is always populated.

**Source:** Backtest `backtest-b836ce01-d8ef-402e-a74d-4afa241f3344` — all 5 paired trades showed Risk Used = $0.00 while actual_risk_at_stop had real values ($4.68–$5.76).

## Scope
1. **Populate `risk_used` with actual risk when budgets are off:** In the fill enrichment path, if `allowance` is 0 or None and `actual_risk_at_stop` is available, set `risk_used` = `actual_risk_at_stop`.
2. **Daily report `allocated_risk_abs`:** Similarly shows 0.0 when budgets are off. Consider defaulting to sum of `actual_risk_at_stop` across trades.

## Key Files
- `backtesting/llm_strategist_runner.py` — fill enrichment around line 2715, `risk_used` assignment
- `agents/strategies/risk_engine.py` — `_check_risk_budget()`, `allowance` return value

## Acceptance
- Fills from a backtest without risk budgets show non-zero `risk_used_abs` reflecting actual position risk.
- Daily report `allocated_risk_abs` is non-zero when trades execute.

## Out of Scope
- Changes to how risk budgets themselves work when enabled.
- UI changes (already addressed in `comp-audit-ui-trade-stats`).
