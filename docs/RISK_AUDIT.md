# Risk Computation & Reporting Audit

This document maps how risk is computed, applied, and reported in the backtesting/LLM strategist pipeline, and flags bugs/defects as TODOs for follow-up fixes.

## Surfaces & Inputs
- **Plan-level constraints (LLM + overrides):** `schemas.llm_strategist.RiskConstraint` (`max_position_risk_pct`, `max_symbol_exposure_pct`, `max_portfolio_exposure_pct`, `max_daily_loss_pct`, optional `max_daily_risk_budget_pct`), with CLI/config overrides resolved in `backtesting/risk_config.py`.
- **Sizing rules:** `PositionSizingRule` per symbol (`fixed_fraction`, `vol_target`, `notional`) passed into `RiskEngine`.
- **Judge risk profile:** `RiskProfile` multipliers (global + per-symbol) from `services/risk_adjustment_service.py` fed into `RiskEngine` for per-order scaling.
- **Session/load/archetype caps:** Applied in `backtesting/llm_strategist_runner.py` before execution (session windows, timeframe caps, archetype load, trigger load).

## Per-Trade Risk Computation Path
- `RiskEngine.size_position(...)` (agents/strategies/risk_engine.py)
  - Uses plan constraints and `PositionSizingRule` to derive desired notional.
  - Caps by:
    - `max_position_risk_pct` → equity * pct (interpreted as risk-per-trade fraction, not volatility-based).
    - `max_symbol_exposure_pct` → remaining headroom vs abs(position) * price.
    - `max_portfolio_exposure_pct` → remaining headroom vs (equity - cash).
  - Daily loss gate: `_within_daily_loss` compares current equity vs `daily_anchor_equity`.
  - Multiplies sizing by `RiskProfile` (judge feedback) before caps.
  - Returns quantity; sets `last_block_reason` when blocked.
- `TradeRiskEvaluator.evaluate(...)`
  - Lets exits/emergency_exit bypass risk caps (quantity=0), otherwise calls `RiskEngine`.
  - Blocks if indicator snapshot missing.

## Daily Risk Budget (max_daily_risk_budget_pct)
- Tracked in `llm_strategist_runner.py` via `daily_risk_budget_state` keyed by `day_key`.
- `_risk_budget_allowance(...)`:
  - Pct = plan/CLI `max_daily_risk_budget_pct`.
  - If prior daily summary exists, uses `used_pct` to short-circuit (blocks when ≥100%).
  - Computes remaining notional cap = `equity * pct` minus accumulated `used_abs` for the day; returns allowance per order.
- `_commit_risk_budget(...)` records per-symbol usage and used_pct/utilization_pct.
- Block reasons surfaced as `risk_budget` and reported in `limit_stats`/`risk_budget` daily section.

## Reporting / Telemetry
- Per-order execution detail logs (`executed_details`) include `risk_used`, `latency_seconds`.
- Aggregations in `llm_strategist_runner.py`:
  - `risk_usage_by_day` + `risk_usage_events_by_day` (trigger/timeframe/hour buckets).
  - `risk_budget` summary per day (used_pct, utilization_pct, per-symbol usage/blocks).
  - Quality attribution (`trigger_quality`, `timeframe_quality`, `hour_quality`) accumulate `risk_used_abs` and compute `rpr = pnl / risk_used_abs`.
- Run-level summaries in `backtesting/reports.py`:
  - `risk_budget_used_pct_mean/median`, utilization distribution, correlation of risk utilization vs returns.
  - Risk-per-trade and RPR per trigger/timeframe/hour.

## Known Issues / Bugs (TODO)
- **TODO (daily loss anchoring):** `RiskEngine.daily_anchor_equity` is set to the equity at plan generation time (`llm_strategist_runner` around plan compile). Because plans refresh intra-day, the anchor drifts, effectively disabling `max_daily_loss_pct` after each refresh. Fix: anchor at start-of-day and only reset on day boundary.
- **TODO (report duplication):** `backtesting/reports.py` repeats `load_sum`/`load_count` aggregation multiple times inside `trigger_quality` accumulation (copy/paste). This overstates load counts and biases derived load metrics. Deduplicate the increments.
- **TODO (risk budget carry-over semantics):** `_risk_budget_allowance` short-circuits using `latest_daily_summary` `used_pct` when present, regardless of `day_key`. If a prior day hit 100%, the next day can be blocked before any trade. Scope the carry-over to same `day_key` or remove cross-day enforcement.
- **TODO (risk_used vs actual price move):** `risk_used` is tracked as allocated budget (equity * pct) at order time, not realized risk based on stop distance/volatility. RPR can be misleading when stops are far/close. Add actual per-trade risk calc (position size * stop distance) alongside allocated budget.
- **TODO (position risk vs exposure):** `max_position_risk_pct` is treated as notional fraction of equity, not true risk-at-stop. For strategies with tight stops, this misstates risk and may under- or over-size. Consider adding a stop-distance-aware mode.
- **TODO (flattening bypass):** Exits/emergency_exit bypass risk caps entirely (quantity=0). If a plan accidentally labels an entry as `emergency_exit`, it skips caps. Add validation that emergency exits must be directionally reducing an existing position.
- **TODO (vol-target sizing clamp):** `vol_target` sizing clamps scale to `<=1.0` of equity regardless of target; if vol is very low this prevents larger notional even when allowed by exposure caps. Confirm intended behavior or allow scale >1 with exposure caps as backstop.
- **TODO (session/archetype load telemetry):** Risk blocks for session_cap/archetype_load are counted, but not reported in daily risk summaries (`limit_stats` focuses on daily cap/plan/risk_budget). Add them to risk_limit_hints/risk_block_breakdown for visibility.
- **TODO (daily risk budget units):** `_risk_budget_allowance` interprets pct against *current equity* per order, not start-of-day equity, so intra-day PnL drift changes remaining budget. Specify intended base and align computation.
- **TODO (risk budget vs actual fill price):** Budget is committed before `portfolio.execute` and not reconciled to slippage/fill differences. Consider logging actual filled notional and adjusting usage accordingly.

## Observations / Gaps
- Risk telemetry is mostly budget/exposure-based; no VaR/vol/drawdown-aware sizing in execution path (drawdown is only a report metric elsewhere).
- Judge/strategist prompts can scale risk via `RiskProfile` multipliers, but there’s no guardrail to prevent multipliers pushing exposure above configured caps if caps are unset/zeroed.
- Per-trade MAE/MFE and latency are tracked; combining with `risk_used_abs` enables RPR, but MAE/MFE are percentage-of-price not percentage-of-risk, so “R-multiples” are approximate.

## Suggested Next Steps
- Fix the TODO items above, starting with daily loss anchoring and risk budget cross-day leakage.
- Add true risk-at-stop computation and log both allocated and realized risk.
- Extend daily reports to include counts for session_cap/archetype_load/trigger_load blocks and their impact on utilization.
- Clarify and document the base for daily risk budget (start-of-day equity) and enforce consistently. !*** End Patch**"
