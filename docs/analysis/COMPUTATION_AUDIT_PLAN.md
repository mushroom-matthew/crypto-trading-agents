# Computation Audit Follow-Up Plan

Goal: confirm the assumptions/presumptions from the audit review and lay out a
clear implementation plan for the enhancements requested. Primary focus remains
math + risk logic, but this plan also captures backtest-driven scalper
calibration needs and the UI visibility upgrades requested for per-trade stats.

## Confirmed Assumptions And Presumptions

Confirmed as correct based on current code paths:

- Risk multipliers are applied multiplicatively across global/symbol/archetype
  and correctly scale all risk percentages (`agents/strategies/risk_engine.py`).
- Daily loss gate compares drawdown vs a daily anchor and blocks new trades when
  `(anchor - equity) / anchor > max_daily_loss_pct` (scaled). This is correct
  assuming the anchor resets per day (`backtesting/llm_strategist_runner.py`).
- Per-symbol and portfolio exposure caps compute available notional as
  `max_cap - current_exposure` and clamp to zero; this is standard and correct.
- Per-trade risk sizing with a stop uses
  `qty_cap = risk_cap_abs / stop_distance`, `notional = qty_cap * price`,
  which matches the R-risk model (loss at stop ~= risk_cap_abs).
- If no stop is provided, the fallback `position_cap_notional = risk_cap_abs`
  is conservative and mathematically correct (limits size to max risk amount).
- Volatility targeting uses `target_daily = target_annual / sqrt(365)` and
  `scale = target_daily / realized_vol`, which is standard for crypto.
- Simulator P&L uses average-cost basis and realizes
  `pnl = proceeds - fee - basis_portion`, which is correct for partial exits.
- Daily P&L breakdown in the LLM backtester correctly separates realized,
  flattening, fees, and carryover vs start equity.
- RPR (return per risk) uses `pnl / risk_used_abs` and `pnl / actual_risk_abs`,
  which is a valid risk-weighted performance measure.

Notes/constraints acknowledged (not errors):

- Donchian bands currently use close-only windows (not high/low).
- Ledger performance metrics include placeholders for win rate and Sharpe.
- Live daily risk budget is currently a fixed placeholder in the live reporter.

## Backtest-Driven Findings To Address

- Trade frequency is too low for 15m scalping (~5 trades/day).
- Execution rate is constrained (~23%) due to min-hold and priority skips.
- Capital utilization is low (flat most of the time, minimal P&L per trade).
- Metrics appear inconsistent (win rate = 0 with positive P&L, Sharpe extreme).
- Trigger mix skews to slow mean reversion; momentum/breakout coverage is weak.

## Enhancement Implementation Plan

### Phase 0: Risk Correctness And Budget Integrity (highest priority)

1) Daily trade cap vs budget overshoot
- Issue: Minimum 8 trades/day can exceed daily budget when per-trade risk is
  near the daily budget.
- Plan: Remove or conditionalize the `max(8, ...)` floor for scalping runs.
- Target files:
  - `agents/strategies/plan_provider.py`
  - `services/strategist_plan_service.py`
- Acceptance:
  - Derived trade cap never implies total risk > daily budget.
  - Add a config flag to preserve current behavior if desired.

2) Risk aggregation on scale-in entries
- Issue: Multiple entries can pass independently yet exceed total stop risk.
- Plan: Compute aggregate position risk using blended entry and shared stop.
- Target files:
  - `agents/strategies/risk_engine.py`
  - `agents/strategies/trade_risk.py`
- Acceptance:
  - New entries are blocked if combined position risk exceeds risk cap.
  - Risk snapshot reports combined and incremental risk deltas.

3) Short position safety without explicit stops
- Issue: Short risk is unbounded if stop is missing.
- Plan: Require stop for shorts or apply stricter notional caps if missing.
- Target files:
  - `agents/strategies/risk_engine.py`
  - `agents/strategies/trade_risk.py`
- Acceptance:
  - Shorts without stop are blocked or capped by a safe override.

4) Align baseline simulator with stop-based risk
- Issue: `risk_used_abs` uses notional, ignoring stop distance.
- Plan: Add stop-aware risk usage to the simulator (match RiskEngine logic).
- Target files:
  - `backtesting/simulator.py`
- Acceptance:
  - `risk_used_abs` and `actual_risk_at_stop` reflect stop distance when known.

### Phase 1: Scalper Cadence And Utilization

5) Min-hold and priority skip throughput
- Issue: min-hold and priority skips are blocking a large share of triggers.
- Plan: Reduce min-hold to 1-3 bars for scalp profiles and relax priority
  skips for high-confidence signals.
- Target files:
  - `agents/strategies/trigger_engine.py`
  - `backtesting/llm_strategist_runner.py`
  - `backtesting/activities.py` (min-hold defaults)
- Acceptance:
  - Execution rate > 50% of valid triggers for scalping configs.

6) Trade frequency and indicator speed
- Issue: 15m signals + slow indicators (SMA/EMA20+) produce swing-like cadence.
- Plan: Add fast indicator presets (EMA5/EMA8, VWAP touches, vol bursts) and
  support 5m/1m timeframes for scalp runs.
- Target files:
  - `agents/analytics/indicator_snapshots.py`
  - `prompts/llm_strategist_prompt.txt`
  - `agents/strategies/plan_provider.py`
- Acceptance:
  - Backtests hit 20-100+ trades/day on scalp configs.

7) Risk utilization and sizing aggressiveness
- Issue: Under-deployed risk budget; low P&L per trade.
- Plan: Review `max_position_risk_pct` and stop-distance sizing to ensure
  stops are not overstating risk; allow plan/provider to raise risk when usage
  is persistently low.
- Target files:
  - `agents/strategies/risk_engine.py`
  - `agents/strategies/plan_provider.py`
- Acceptance:
  - Risk budget utilization ≥ 70% on active scalp days.

8) Trigger mix: add momentum/breakout scalps
- Issue: Mean-reversion dominates; little reactive momentum coverage.
- Plan: Add momentum/breakout trigger templates with volatility gating.
- Target files:
  - `prompts/llm_strategist_prompt.txt`
  - `prompts/strategies/*.txt`
- Acceptance:
  - Trigger distribution includes momentum/breakout scalps per symbol.

### Phase 2: Live Parity And Metrics Completeness

9) Live daily risk budget tracker
- Issue: Live uses a fixed $1000 budget placeholder.
- Plan: Compute budget from equity * budget_pct, reset daily, track usage.
- Target files:
  - `services/live_daily_reporter.py`
  - (if needed) shared risk-budget helper in `backtesting/llm_strategist_runner.py`
- Acceptance:
  - Live risk budget scales with equity and mirrors backtest math.

10) Ledger leverage + fees clarity
- Issue: Leverage is hard-coded to 1.0; ledger P&L may not reflect fees.
- Plan: Compute leverage = total exposure / equity; verify fee flow into P&L.
- Target files:
  - `agents/workflows/execution_ledger_workflow.py`
  - `app/strategy/trade_executor.py` (if fee handling changes)
- Acceptance:
  - Risk metrics report actual leverage.
  - Ledger P&L explicitly accounts for fees (or documents why not).

11) Win/loss consistency + live Sharpe
- Issue: Win rate and Sharpe show inconsistencies in recent backtests.
- Plan: Centralize win/loss and fix aggregation paths; implement rolling
  Sharpe/Sortino for live and guard against degenerate distributions.
- Target files:
  - `trading_core/trade_quality.py`
  - `agents/analytics/portfolio_state.py`
  - `agents/workflows/execution_ledger_workflow.py`
- Acceptance:
  - One shared win/loss calculator used across live/backtest.
  - Live performance metrics are non-placeholder.
  - Backtest win rate matches realized P&L sign counts.

### Phase 3: Indicator Fidelity And Regime Responsiveness

12) Donchian bands use high/low
- Issue: Close-only Donchian may miss wick breakouts.
- Plan: Switch to high/low for Donchian channel calculations.
- Target files:
  - `agents/analytics/indicator_snapshots.py`
- Acceptance:
  - Donchian bands match standard definition.

13) Volatility window tuning for scalping
- Issue: Vol windows may lag rapid regime shifts.
- Plan: Add shorter windows or EWMA volatility option.
- Target files:
  - `agents/analytics/indicator_snapshots.py`
  - `agents/strategies/risk_engine.py` (vol_target sizing)
- Acceptance:
  - Configurable short-window realized vol and ATR for scalp runs.

14) Sharpe annualization basis
- Issue: Sharpe uses 252 in some contexts; crypto is 365.
- Plan: Use 365 for daily crypto returns or make scaling configurable.
- Target files:
  - `tools/performance_analysis.py`
  - `agents/analytics/portfolio_state.py`
- Acceptance:
  - Consistent and configurable annualization factor.

### Phase 4: Extended Metrics And Performance

15) Implement Sortino, beta, sentiment scaffolds
- Issue: Tier II/III metrics are placeholders.
- Plan: Implement Sortino (downside dev), beta to benchmark, and optional
  sentiment score hooks.
- Target files:
  - `metrics/market_context.py`
  - `metrics/sentiment.py`
  - `agents/analytics/factors.py`
- Acceptance:
  - Metrics return numeric series without NotImplementedError.

16) Indicator compute optimization for low-latency
- Issue: Rolling computations may be too heavy for HFT-style scalp loops.
- Plan: Add incremental updates or caching for EMA/ATR/realized vol, and ensure
  single-source computation per tick.
- Target files:
  - `agents/analytics/indicator_snapshots.py`
  - `metrics/technical.py`
- Acceptance:
  - Indicator update cost reduced to O(1) per tick for hot paths.

### Phase 5: Edge Case Safeguards

17) Gap/slippage-aware risk
- Issue: Stops can gap; risk can exceed intended cap.
- Plan: Add slippage buffer or gap multiplier in stop-based risk sizing.
- Target files:
  - `agents/strategies/risk_engine.py`
- Acceptance:
  - Optional worst-case risk multiplier applied for gap risk.

18) Concurrent signal serialization
- Issue: Multiple signals can pass before shared state updates.
- Plan: Ensure risk checks see the updated state before subsequent orders.
- Target files:
  - `agents/strategies/trigger_engine.py`
  - `backtesting/llm_strategist_runner.py`
- Acceptance:
  - Only one risk check per symbol per tick updates shared exposure state.

### Phase 6: UI Trade-Level Visibility

19) Per-trade stats surfaced in UI (backtest + live)
- Issue: UI shows aggregate win rate/total trades, but not trade-level risk and
  performance stats for each fill/round-trip.
- Plan: Add a trade detail panel (or expandable row) that shows:
  - Entry/exit timestamps, side, qty, price, fees, slippage.
  - Risk-used and actual risk at stop (`risk_used_abs`, `actual_risk_abs`).
  - R-multiple (`pnl / actual_risk_abs`) and P&L.
  - Stop distance used for sizing and any risk multiplier applied.
  - Per-trade MAE/MFE if available, and trigger/category context.
  - Daily risk budget utilization at time of trade (if tracked).
- Target files (likely):
  - `ui/src/components/BacktestControl.tsx`
  - `ui/src/components/LiveTradingMonitor.tsx`
  - `ui/src/lib/api.ts` (ensure trade-level fields are exposed)
- Acceptance:
  - Each trade row exposes risk-weighted stats and sizing context.
  - Consistent stats between backtest and live views.

## Open Questions / Decisions Needed

- Should the trade cap floor (`min trades/day`) be removed globally or only for
  scalping profiles? If conditional, what signal identifies a scalping profile?
- What is the target trade frequency per timeframe (15m vs 5m vs 1m)?
- What stop-loss source is authoritative (explicit stop vs ATR proxy) for risk
  aggregation and short trades?
- Should volatility targeting use 365 or 252 for annualization in all contexts?
- Do we want to support margin/leverage in live metrics immediately, or just
  document that the system assumes spot trading?
- Which per-trade stats are mandatory for the UI (minimum set vs extended
  diagnostics like MAE/MFE and indicator snapshot)?

## Proposed Validation (Math-Only)

- Unit tests for combined position risk with scale-in entries.
- Risk budget tests covering low-budget edge cases (no overshoot).
- Simulator risk usage tests verifying stop-distance scaling.
- Sharpe/Sortino and beta calculation tests with synthetic return series.
- Backtests meet execution-rate and trade-frequency thresholds for scalper configs.

## Proposed Validation (UI)

- Confirm per-trade stats render for both backtest and live runs.
- Validate R-multiple and risk-used fields match backend calculations.

## Sequencing

- Phase 0 and Phase 1 first (correctness + scalper cadence).
- Phase 2 for live parity and metrics completeness.
- Phase 3 for indicator fidelity/tuning.
- Phase 4 and Phase 5 after correctness work is merged.
- Phase 6 after backend trade stats are stable and consistent.
