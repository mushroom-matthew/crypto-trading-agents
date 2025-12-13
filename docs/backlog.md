# Backlog (triage candidates)

These items were in archived plans; triage and prioritize before execution.

## Risk/RPR follow-ups
- Rebuild RPR surfaces using higher-density runs; add archetype-only multipliers first, defer hour slices until trigger coverage is reliable.
- Define regime-aware risk tiers (trend/range/chop × vol) with per-trade risk, daily budget, caps, session multipliers, and regime confidence smoothing.

## Trigger/telemetry enhancements
- Add trigger-density requirements in plans; log/prune reasons for trigger rejection to identify “1 trade, 0 blocks” days.
- Expand telemetry for trigger quality and load (including per-symbol/timeframe budgets that fire).

## Factor/hedging experiments
- Factor exposure analysis and hedging experiments (see archived HEDGING_AND_EDGE_PLAN.md for ideas if needed).

## PnL/backtesting hygiene
- Validate PnL accounting and backtest/live alignment items from backtesting_remediation_roadmap if still relevant.

## Risk guardrails (if not already covered)
- Confirm daily budgeting and guardrail checks from RISK_REMEDIATION_PLAN/RISK_RPR_ACTION_PLAN are satisfied under the new architecture; re-run any missing safety assertions.
