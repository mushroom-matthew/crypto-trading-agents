# Backtesting Remediation Roadmap

Checklist-driven plan to address advisor feedback across backtesting and live alignment. Each section highlights intent, concrete tasks, and validation steps.

## PnL Accounting Fixes
- [x] In `backtesting/simulator.py`, compute `equity_return_pct = end_equity / start_equity - 1` per day. *(daily reporting now anchors on start/end equity; component net deltas logged)*
- [x] Define `gross_trade_pct` as (realized trade PnL excluding forced flatten) / `start_equity` and `flattening_pct` as forced EOD flatten PnL / `start_equity` to avoid double-counting. Fees remain per-day over start equity.
- [x] Compute `net_equity_pct` from components (gross + flatten + fees + carryover) and reconcile to `equity_return_pct`; carryover captures unrealized/mark-to-market when not flattening.
- [x] Add a synthetic unit test: 1–2 trades, asserts component sums and `abs(net_equity_pct - equity_return_pct) < 1e-9` (`tests/test_pnl_components.py`).
- [ ] Re-run backtest to confirm no gross/fees/flatten vs net/equity_return_pct mismatches. *(Pending)*

## Daily Reporting Clarity
- [x] Add currency fields: `realized_pnl_abs`, `fees_abs`, `flattening_pnl_abs`, and `daily_cash_flows` (default 0.0).
- [x] Add `carryover_pnl` when `flatten_positions_daily = false` (captures unrealized / mark-to-market residual).
- [x] Ensure JSON exports include new fields; verify on a sample day. *(Validated on `short-telemetry-smoke` daily reports.)*

## Direction Semantics & Exit Handling
- [x] Normalize exit triggers to a consistent `direction` (supporting `exit`/`flat_exit`, normalizing `flat` ➜ `exit`) instead of plain `flat`.
- [x] In limit enforcement, always allow exits that reduce absolute exposure regardless of `allowed_directions` (direction check bypassed for exits).
- [x] At plan compile-time, reject or normalize any trigger direction not in `allowed_directions`; avoid execution-time blocks.
- [x] Tests: (1) emergency exit with an open position is not blocked and flattens; (2) invalid direction is rejected at compile-time (`tests/test_direction_semantics.py`).

## Risk Usage Visibility
- [x] Add a run-level backtest summary (`backtesting/reports.py`): mean/median `risk_budget_used_pct`, % of days with <10% usage, mean trade_count, blocked_by_daily_cap/direction/plan_limit, heuristic correlation of `risk_budget_used_pct` vs `equity_return_pct`.
- [x] Emit summary JSON alongside daily reports; verified `run_summary.json` generated for `short-telemetry-smoke-3` (risk usage still 0% → highlights underuse).

## Trade Budget Regimes (daily_cap vs plan_limit)
- [x] Instrument per-run stats showing which brake was active each day (daily_cap vs plan_limit) and execution rate (in run summary, plus `active_brake`/`execution_rate` per daily report going forward).
- [x] Introduce risk-budget-derived daily cap: if `max_daily_risk_budget_pct` is set, derive `max_trades_per_day` from per-trade risk (min sizing rule target or `max_position_risk_pct`) and log `derived_max_trades_per_day` in plan limits.
- [ ] Add validation/backtest to confirm no overlapping brakes and expected execution rate under the unified cap; tune derived cap thresholds as needed. *(Risk usage still <10%; daily_cap active on some days; day-1 plans sometimes lack `derived_max_trades_per_day` in limits.)*

## Trigger Catalog Hygiene
- [x] Prune or refactor dead variants (`*_exit_exit`, `*_exit_flat`) that never execute; remove from the catalog or merge semantics.
- [x] Ensure entry/exit triggers share consistent direction semantics and align with `allowed_directions` (normalized).
- [x] Add a catalog validation step to catch unused/always-blocked triggers before backtests run. *(Pruning implemented; needs further validation for always-blocked triggers.)*

## Flattening Policy & Fee Drag
- [ ] Make flatten PnL explicit in reports; ensure fees for open + flatten are reflected in `fees_pct`.
- [ ] Parameterize flatten policy: daily vs “flatten before 00:00 UTC” (session-based) to match 1h intraday behavior.
- [ ] Add a test/backtest slice demonstrating fee impact of flattening policy choices.

## Timeframe & Time-of-Day Alignment
- [x] Add timeframe/hour-of-day execution rate telemetry in `run_summary` (timeframe_execution_rate, hour_execution_rate).
- [ ] Revisit `max_trades_per_day` and risk budgets for 1h-dominant, early-UTC activity; consider per-session or per-1h-bar budgeting. *(Initial bias: cap derived per symbol per day and thin non-1h duplicates.)*
- [ ] Evaluate whether 4h triggers should be deprioritized or given separate limits; update plan generation accordingly.
- [ ] Add reporting slices by timeframe and hour-of-day for executed trades.

## Risk-On Expression
- [ ] Adjust risk sizing so high-conviction setups can consume a larger fraction of the 3.75% budget when appropriate.
- [ ] Wire judge feedback into sizing/risk knobs to reduce chronic underuse; surface under-10%-usage days in summaries.

## Suggested Implementation Order
- [x] PnL accounting + tests (foundation for trustworthy metrics).
- [x] Direction/exit semantics + catalog hygiene (reduce pointless blocks).
- [ ] Summary reporting + risk usage visibility (diagnose regime).
- [ ] Re-run backtests, then tune budget/flatten policies based on new visibility.
