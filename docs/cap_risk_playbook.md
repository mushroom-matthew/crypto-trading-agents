# Cap & Risk Playbook

## Current behavior
- Caps: fixed mode uses env floors (trades=30, triggers=40 recommended); daily_cap blocks are 0; session caps use the larger of policy/derived.
- Triggers: resolved caps honor env floors; judge caps are ignored in fixed mode unless explicitly present for the current plan; trimming uses resolved caps.
- Risk budget: main brake; throughput scales with `max_daily_risk_budget_pct / max_position_risk_pct`.

## Running the matrices
### Cap baseline matrix
- Script: `scripts/run_cap_matrix.sh` (fixed vs legacy, multipliers on/off; default caps 30/40; default 1 LLM call/day).
- Outputs: `.runs/cap_matrix/cap-*/run_summary.json` plus daily reports.

### Risk geometry matrix
- Script: `scripts/run_cap_matrix_risk.sh` (fixed caps 30/40; sweeps budget% and per-trade risk with/without multipliers).
- Defaults (edit inside the script):
  - R1: 10% / 0.30%
  - R2: 10% / 0.40%
  - R3: 12% / 0.40%
  - R4: 15% / 0.50% (stretch)
- Outputs: `.runs/cap_matrix_risk/risk-*/run_summary.json`.

## Metrics to compare (per run summary)
- `trade_count_mean`
- `blocked_by_risk_budget_mean` (or per-day `limit_stats.risk_budget`)
- `blocked_by_daily_cap_mean`, `blocked_by_plan_limits_mean`, `blocked_by_session_cap_mean` (should be ~0 in fixed runs)
- `risk_budget_used_pct_mean` and bucket distribution
- `cap_state.resolved.max_trades_per_day` / `max_triggers_per_symbol_per_day` (min/mean/max)
- `cap_state.session_caps` when multipliers are on
- `rpr_actual` vs baseline when available

## Interpreting outcomes
- If risk_budget blocks are high and trades low: lower per-trade risk or raise the daily rail (e.g., 10–12% rail with 0.3–0.4% per-trade).
- If daily_cap/plan_limit > 0 in fixed mode: investigate symbol/timeframe budgets or judge caps; caps themselves should not throttle.
- If trades are 0–1 with no blocks: trigger scarcity—inspect the plan and trimming for that day.

## Defaults for ongoing testing
- Fixed mode: `STRICT_FIXED_CAPS=true`, caps 30/40, judge caps off for triggers/trades.
- Risk geometry: start with `max_daily_risk_budget_pct=10–12`, `max_position_risk_pct=0.3–0.4`.
- Multipliers: start without; add archetype/session multipliers once trigger density is reliable.
