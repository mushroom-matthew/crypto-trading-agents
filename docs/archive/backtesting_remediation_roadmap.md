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
- [x] Replace the “<10% usage” equity-based stat with budget utilization: `risk_budget_utilization_pct` now recorded per-day and aggregated (mean/median plus % days under_25/between_25_75/over_75); downstream prompts/adjusters still need to consume it.

## Trade Budget Regimes (daily_cap vs plan_limit)
- [x] Instrument per-run stats showing which brake was active each day (daily_cap vs plan_limit) and execution rate (in run summary, plus `active_brake`/`execution_rate` per daily report going forward).
- [x] Introduce risk-budget-derived daily cap: if `max_daily_risk_budget_pct` is set, derive `max_trades_per_day` from per-trade risk (min sizing rule target or `max_position_risk_pct`) and log `derived_max_trades_per_day` in plan limits.
- [x] Add strict cap flag (`STRATEGIST_STRICT_FIXED_CAPS`) so derived caps become telemetry-only when desired; honors configured caps instead of shrinking them.
- [ ] Validate with backtests that caps are no longer the dominant brake when strict mode is on (recent runs still show plan_limit/daily_cap blocking heavily; needs re-run with env set and higher fixed caps).

## Trigger Catalog Hygiene
- [x] Prune or refactor dead variants (`*_exit_exit`, `*_exit_flat`) that never execute; remove from the catalog or merge semantics.
- [x] Ensure entry/exit triggers share consistent direction semantics and align with `allowed_directions` (normalized).
- [x] Add a catalog validation step to catch unused/always-blocked triggers before backtests run. *(Pruning implemented; needs further validation for always-blocked triggers.)*

## Flattening Policy & Fee Drag
- [ ] Parameterize flatten policy: `none`, `daily_close`, and `session_close_utc` (e.g., 23:00/00:00); surface choice in plan/run metadata.
- [ ] Attribute PnL cleanly: `gross_trade_pct` (strategy exits), `flattening_pct` (forced close), and `fees_pct` including flatten orders.
- [ ] Add A/B backtests (never vs daily vs session flatten) and report `flattening_pct_mean`, `fees_pct_mean`, and % days with flatten trades; document fee drag guidance.

## Timeframe & Time-of-Day Alignment
- [x] Add timeframe/hour-of-day execution rate telemetry in `run_summary` (timeframe_execution_rate, hour_execution_rate).
- [ ] Add per-session risk/trade budgets (e.g., per 8h UTC blocks) derived from per-trade risk and session budget; wire caps into execution.
- [ ] Gate or down-weight time-of-day: optional allowed UTC hour window for backtests; use telemetry to bias active hours (early UTC).
- [ ] Evaluate 4h triggers: either prune or cap separately (tiny budget) when 1h dominates; make this configurable.
- [ ] Add reporting slices by session for executed trades and utilization.

## Risk-On Expression
- [ ] Adjust risk sizing so high-conviction setups can consume a larger fraction of the 3.75% budget when appropriate.
- [ ] Wire judge feedback into sizing/risk knobs to reduce chronic underuse; drive adjustments from budget utilization bands instead of raw equity pct.
- [ ] Add conviction-aware sizing: allow triggers to express conviction (low/medium/high) and map to per-trade risk multipliers (clipped by daily budget). Log risk/PnL by conviction.

## Suggested Implementation Order
- [x] PnL accounting + tests (foundation for trustworthy metrics).
- [x] Direction/exit semantics + catalog hygiene (reduce pointless blocks).
- [ ] Budget utilization metrics + judge wiring (make risk usage meaningful).
- [ ] Flatten policy A/B + fee drag measurement.
- [ ] Session/timeframe budgeting aligned to telemetry; validate derived caps.
- [ ] Conviction-aware risk-on expression.
- [ ] Re-run longer backtests to validate PnL/fee/risk behavior post-tuning.

## Phase 2 Focus (post-telemetry learnings)
- **Risk expression tuning:** Raise per-trade risk and/or adaptive multipliers on low-utilization winning days; allow session multipliers to lift caps when utilization <25% without negative returns.
- **Cap alignment:** Ensure daily caps scale with budget intent (avoid tiny caps with unused budget); bias caps toward active hours (0–4 UTC) and 1h timeframe; allow configurable 4h throttling/pruning.
- **Flatten A/B:** Run `none` vs `daily_close` vs `session_close_utc` to measure `flattening_pct_mean`, `fees_pct_mean`, and risk utilization impact; pick a default based on fee drag.
- **Budget sweeps:** Use A/B helper to sweep `max_daily_risk_budget_pct` and `max_position_risk_pct` to verify utilization moves and identify remaining bottlenecks (plan limits vs budget).
- **Judge feedback refinement:** Drive judge sizing/cap hints from budget utilization bands (under 25% with gains → boost; over 75% with losses → tighten) and from timeframe/hour effectiveness.
