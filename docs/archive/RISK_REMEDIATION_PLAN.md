# Risk Remediation Plan — Sharpened (Based on RISK_AUDIT & Expert Feedback)

Clean, surgical sequence to restore correct risk accounting, daily budgeting, and RPR metrics. Includes file paths, expected side-effects, and required assertions.

## P0 — Safety-Stopping Defects (Immediate)
Structural bugs that break risk entirely; telemetry/strategy behavior is untrustworthy until fixed.

- **P0.1 Fix Daily Loss Anchoring (max_daily_loss_pct)**  
  - Problem: Anchor resets on every plan refresh → limit disabled.  
  - Fix: Set `daily_anchor_equity` once per day (day boundary only). Do not change on intra-day plan refresh.  
  - Files: `backtesting/llm_strategist_runner.py` (day boundary handling), `agents/strategies/risk_engine.py` (`_within_daily_loss` reference).  
  - Tests (new: `tests/risk/test_daily_loss_anchor.py`):  
    - T1: Anchor remains constant across multiple plan refreshes same day.  
    - T2: Anchor resets exactly at 00:00 / next candle day-index change.

- **P0.2 Fix Daily Risk Budget Carry-Over (day_key scoping)**  
  - Problem: `_risk_budget_allowance` uses prior-day summaries → Day 2 can be blocked immediately.  
  - Fix: Restrict exhaustion checks to current `day_key`; clear `daily_risk_budget_state[day_key]` at day boundary; explicit midnight/day-index reset.  
  - Files: `backtesting/llm_strategist_runner.py`.  
  - Tests (new: `tests/risk/test_daily_budget_reset.py`):  
    - Day 1 uses full budget → Day 2 starts with full budget.  
    - Prior-day summaries do not affect new-day allowance.

- **P0.3 Replace “Notional Risk” With “Risk-At-Stop” (True Risk)**  
  - Problem: Risk == notional, not risk-at-stop; RPR meaningless.  
  - Fix: In `RiskEngine.size_position()` compute `risk_at_stop = position_size * stop_distance` (prefer trigger stop; else ATR/realized-vol proxy).  
    - Evaluate caps against: `max_position_risk_pct`, `max_symbol_exposure_pct`, `max_portfolio_exposure_pct`, `max_daily_risk_budget_pct`.  
    - Persist both `allocated_risk = equity * pct` and `actual_risk = size * stop_distance`.  
  - Files: `agents/strategies/risk_engine.py` (sizing), supporting stop extraction (plan/trigger plumbing), budget consumers in `backtesting/llm_strategist_runner.py`.  
  - Tests (new: `tests/risk/test_risk_at_stop.py`):  
    - Tight stop → larger size than wide stop.  
    - Risk budget consumption uses `actual_risk`.

## P1 — High-Impact Corrections
Unblock accurate risk expression; not immediately unsafe but distorts telemetry.

- **P1.1 Remove Vol-Target “<=1× Equity” Clamp**  
  - Fix: In `PositionSizingRule._vol_target_size()` remove `scale = min(scale, 1.0)`; rely on exposure caps; log when scale > 1.  
  - Files: `agents/strategies/risk_engine.py` (vol_target branch).  
  - Tests: `tests/sizing/test_vol_target_clamp.py` (target implying 1.5× equity allowed, capped by exposure caps).

- **P1.2 Daily Budget Base = Start-Of-Day Equity**  
  - Fix: Store `start_of_day_equity`; `allowed_budget = start_of_day_equity * max_daily_risk_budget_pct` (no intra-day drift).  
  - Files: `backtesting/llm_strategist_runner.py` (budget base, daily reset).  
  - Tests: `tests/risk/test_budget_base_equity.py` (intraday win must not expand budget; loss must not shrink).

- **P1.3 Reconcile Committed Budget With Fills**  
  - Fix: In `portfolio.execute()`, record `allocated_risk_pre_fill` and `actual_risk_post_fill`; adjust `daily_risk_budget_state.used_abs` with post-fill values; log slippage delta.  
  - Files: `trading_core/portfolio.py` (or equivalent execute path), `backtesting/llm_strategist_runner.py` (budget accounting).

## P2 — Guardrails & Surfaces
Edge-case safety and visibility.

- **P2.1 Exit/Emergency Exit Validation**  
  - Fix: In `TradeRiskEvaluator.evaluate()`, exits must reduce exposure; otherwise treat as entry and apply risk gates; emit `invalid_exit_flag` telemetry.  
  - Files: `agents/strategies/trade_risk.py`.

- **P2.2 Report All Block Types in Daily Telemetry**  
  - Fix: Add `session_cap_blocks`, `archetype_load_blocks`, `trigger_load_blocks` to daily `risk_block_breakdown` and summaries.  
  - Files: `backtesting/llm_strategist_runner.py`, `backtesting/reports.py`.

- **P2.3 Risk-Profile Multiplier Guard**  
  - Fix: If caps are zero/unset, clamp multipliers so risk cannot exceed a sane ceiling (e.g., 5% equity per trade); log warning.  
  - Files: `agents/strategies/risk_engine.py`, `services/risk_adjustment_service.py`.

## P3 — Telemetry & Reporting Quality

- **P3.1 Fix Load-Count Duplication in Trigger Quality**  
  - Fix duplicated `load_sum`/`load_count` increments in `backtesting/reports.py`; regression test ensures `load_count == executed_trigger_calls`.  
  - Files: `backtesting/reports.py`. Tests: `tests/risk/test_load_counts.py` (new).

- **P3.2 Dual RPR Metrics (Allocated + At-Stop)**  
  - Telemetry: `RPR_allocated = pnl / allocated_risk`, `RPR_actual = pnl / actual_risk_at_stop`; include stop/vol proxy used and fallback flag.  
  - Files: `backtesting/llm_strategist_runner.py` (quality attribution), `backtesting/reports.py`.

- **P3.3 Budget Transparency Fields**  
  - Add per-day: `budget_base_equity`, `budget_allocated_abs`, `budget_allocated_pct`, `budget_actual_abs`, `budget_slippage_adjustment`.  
  - Files: `backtesting/llm_strategist_runner.py`, `backtesting/reports.py`.

## P4 — Validation & Tooling

- **Unit Tests (codified)**  
  - `tests/risk/test_daily_loss_anchor.py`  
  - `tests/risk/test_daily_budget_reset.py`  
  - `tests/risk/test_risk_at_stop.py`  
  - `tests/sizing/test_vol_target_clamp.py`  
  - `tests/risk/test_exit_bypass.py`  
  - `tests/risk/test_load_counts.py`

- **Simulation Validation**  
  - After P0–P1 fixes, re-run P1a–P1c: expect risk usage to rise (>5–25% depending on plan), RPR to stabilize, and block breakdown to show session/plan/budget distinctly.

## Ownership & Ordering
1) Land P0 fixes with tests.  
2) Apply P1 budget/vol corrections and fill reconciliation.  
3) Add guardrails/reporting (P2) for visibility.  
4) Telemetry/report correctness (P3) + full test suite (P4).  

Outcome: Functional daily loss protection, correct daily budgeting, and true stop-based risk underpinning meaningful RPR and utilization metrics.***
