# Computation Audit: Risk, Sizing, Indicators, P&L

This document summarizes the math used for risk and sizing, indicator/metric
computations, and P&L/summary statistics. It focuses on formulas and gating
logic rather than inputs/outputs.

Sources referenced (non-exhaustive):
- `agents/strategies/risk_engine.py`
- `agents/strategies/trade_risk.py`
- `agents/strategies/plan_provider.py`
- `services/strategist_plan_service.py`
- `backtesting/llm_strategist_runner.py`
- `backtesting/simulator.py`
- `backtesting/reports.py`
- `trading_core/trade_quality.py`
- `agents/analytics/portfolio_state.py`
- `agents/workflows/execution_ledger_workflow.py`
- `services/live_daily_reporter.py`
- `metrics/base.py`
- `metrics/technical.py`
- `metrics/registry.py`
- `agents/analytics/indicator_snapshots.py`
- `agents/strategies/tool_loop.py`
- `agents/analytics/factors.py`
- `tools/performance_analysis.py`

## Risk And Position Sizing

### RiskEngine (core sizing and caps)
File: `agents/strategies/risk_engine.py`

- Risk multipliers are applied as a product:
  - `total_multiplier = max(0, global) * max(0, symbol) * max(0, archetype) * max(0, archetype_hour)`
  - `_scaled_fraction(pct) = (pct / 100) * total_multiplier`
- Daily loss gate:
  - `loss = (anchor_equity - current_equity) / anchor_equity`
  - Trade blocked if `loss > _scaled_fraction(max_daily_loss_pct)`
- Symbol exposure cap:
  - `position_notional = abs(position_qty) * price`
  - `max_symbol = equity * _scaled_fraction(max_symbol_exposure_pct)`
  - `available = max(0, max_symbol - position_notional)`
- Portfolio exposure cap:
  - `portfolio_exposure = max(0, equity - cash)`
  - `max_portfolio = equity * _scaled_fraction(max_portfolio_exposure_pct)`
  - `available = max(0, max_portfolio - portfolio_exposure)`
- Position risk cap:
  - `risk_cap_abs = equity * _scaled_fraction(max_position_risk_pct)`
  - If stop distance exists (explicit stop or ATR proxy):
    - `qty_cap = risk_cap_abs / stop_distance`
    - `position_cap_notional = qty_cap * price`
  - Else: `position_cap_notional = risk_cap_abs`
- Desired notional from sizing rule:
  - `fixed_fraction`: `equity * (target_risk_pct / 100)`
  - `notional`: `max(0, rule.notional)`
  - `vol_target`:
    - `daily_target = target_annual / sqrt(365)`
    - `scale = daily_target / realized_vol`
    - `desired_notional = equity * scale`
- Final notional:
  - `final_notional = min(desired_notional, position_cap_notional, symbol_cap, portfolio_cap)`
  - `quantity = final_notional / price`
  - `actual_risk_abs = quantity * stop_distance` (if stop distance exists)

### TradeRiskEvaluator (entry vs exit gating)
File: `agents/strategies/trade_risk.py`

- Flatten or emergency exits bypass caps and return `allowed=True`.
- Missing indicator blocks the trade.
- Otherwise defers to `RiskEngine.size_position`; a zero quantity blocks the trade.

### Plan-derived caps and risk budget boosting
Files:
- `agents/strategies/plan_provider.py`
- `services/strategist_plan_service.py`

When `max_daily_risk_budget_pct` is set:
- `max_position_risk_pct` is boosted to consume budget:
  - `boosted = min(budget_pct, max_position_risk_pct * 6)`
- Per-trade risk proxy is the smallest non-zero sizing rule, else `max_position_risk_pct`.
- Derived trade cap:
  - `derived_cap = max(8, ceil(budget_pct / per_trade_risk_pct))`
- Sizing rules are bumped to at least `max_position_risk_pct` so allocations match the boosted cap.

### Daily risk budget enforcement (LLM backtester)
File: `backtesting/llm_strategist_runner.py`

- Per-day budget:
  - `budget_abs = start_equity * (budget_pct / 100)`
  - `used_abs` starts at 0, tracked per symbol.
- Per-trade risk allowance:
  - `target_risk_pct = sizing_targets[symbol]` or `max_position_risk_pct`
  - Adaptive multiplier: `3.0` if `used_abs / budget_abs < 10%`, else `1.0`
  - `contribution = start_equity * (target_risk_pct * adaptive_multiplier / 100)`
  - `contribution = min(contribution, remaining_budget)`
- Budget summary:
  - `used_pct = used_abs / budget_abs * 100`
  - `utilization_pct` is computed on the original `start_equity` budget base
  - `symbol_usage_pct = symbol_used_abs / budget_abs * 100`

### Baseline backtesting simulator risk usage
File: `backtesting/simulator.py`

- Per-trade risk usage:
  - `cap = equity * (max_position_risk_pct / 100)`
  - `risk_used_abs = min(notional, cap)` (no explicit stop-distance modeling)
- `actual_risk_at_stop` is set equal to `risk_used_abs`

### Live daily report risk budget (placeholder)
File: `services/live_daily_reporter.py`

- `risk_budget_abs` is hardcoded to `1000.0`
- `used_pct = total_claimed / risk_budget_abs * 100`
- `available_abs = risk_budget_abs - total_claimed`

### Ledger risk metrics
File: `agents/workflows/execution_ledger_workflow.py`

- Portfolio value:
  - `total = cash + sum(qty * last_price)`
- Position concentration:
  - `concentration = position_value / total`
  - `max_position_concentration = max(concentrations)`
- Cash ratio: `cash / total`
- Leverage is fixed to `1.0`

## Indicator And Metric Computations

### Shared utilities
File: `metrics/base.py`

- `prepare_ohlcv_df`: enforces `timestamp, open, high, low, close, volume`, sorts ascending, casts float64.
- Wilder smoothing: `EMA(alpha = 1 / period)`
- Rolling std: population `std(ddof=0)`
- `ensure_min_period`: sets values to NaN until count >= period.
- `safe_divide`: replaces zero denominators with NaN, then fills with 0.

### Technical indicators
File: `metrics/technical.py`

- SMA: rolling mean of close.
- EMA: exponential mean of close.
- WMA: linear weights `1..period` over close.
- MACD:
  - `macd = EMA_fast - EMA_slow`
  - `signal = EMA(macd, signal_period)`
  - `hist = macd - signal`
- RSI:
  - `delta = close.diff()`
  - `gains = max(delta, 0)`, `losses = max(-delta, 0)`
  - `avg_gain = Wilder(gains)`, `avg_loss = Wilder(losses)`
  - `RS = avg_gain / avg_loss`
  - `RSI = 100 - 100 / (1 + RS)` with 0/100 guards
- Bollinger Bands:
  - `basis = SMA(close)`
  - `std = rolling_std(close)`
  - `upper = basis + mult * std`, `lower = basis - mult * std`
  - `bandwidth = (upper - lower) / abs(basis)`
  - `pctB = (close - lower) / (upper - lower)`
- ATR:
  - `TR = max(|high-low|, |high-prev_close|, |low-prev_close|)`
  - `ATR = Wilder(TR)`
- ADX:
  - `+DM`/`-DM` from directional moves
  - `+DI = 100 * Wilder(+DM) / ATR`
  - `-DI = 100 * Wilder(-DM) / ATR`
  - `DX = 100 * |+DI - -DI| / (+DI + -DI)`
  - `ADX = Wilder(DX)`
- ROC: `close / close.shift(period) - 1`
- OBV: cumulative sum of `sign(close.diff()) * volume`
- VWAP:
  - session-based cumulative `(tp * volume) / volume`
  - or windowed version if `window` provided
  - `tp = (high + low + close) / 3`

### Indicator snapshots and derived metrics
File: `agents/analytics/indicator_snapshots.py`

- Realized volatility:
  - `log_returns = log(close / close.shift(1))`
  - `realized_vol = std(log_returns, ddof=0)` over window
- Donchian:
  - `upper = max(close, window)`, `lower = min(close, window)`
- Volume multiple:
  - `volume_multiple = volume / mean(volume, short_window)`
- Cycle metrics (rolling `window`):
  - `cycle_high = max(high)`, `cycle_low = min(low)`
  - `cycle_range = (cycle_high - cycle_low) / close`
  - `cycle_position = (close - cycle_low) / (cycle_high - cycle_low)`
- Fibonacci retracements (from cycle range):
  - `fib_236/382/500/618/786 = cycle_high - range * ratio`
- Expansion/contraction swing model:
  - Find recent max/min within lookback, compute % moves
  - `ratio = expansion_pct / contraction_pct` when both defined
- Trend/volatility states:
  - Trend: SMA stack or EMA band (`ema_short > ema_medium * 1.005`)
  - Volatility: `vol_metric = max(atr / price, realized_vol_short)`

### Volatility regime classification
File: `agents/strategies/tool_loop.py`

- Uses `realized_vol_short / realized_vol_medium`:
  - `> 1.5` => `expanding`
  - `< 0.7` => `contracting`
  - else `stable`

### Factor exposures
File: `agents/analytics/factors.py`

- Returns: `pct_change(close)`
- Align with factor dataframe on timestamps.
- OLS regression with intercept:
  - `betas = argmin ||y - Xb||`
  - `r2 = 1 - SS_res / SS_tot`
- Idiosyncratic volatility: `std(residuals, ddof=1)`

### Unimplemented metric scaffolds
Files:
- `metrics/market_context.py`
- `metrics/sentiment.py`

These raise `NotImplementedError` for Sharpe/Sortino/beta/sentiment placeholders.

## P&L And Summary Statistics

### Backtesting simulator P&L
File: `backtesting/simulator.py`

- Buy fills: `pnl = 0`, cost basis increases by `total_cost = allocation + fee`
- Sell fills:
  - `pnl = proceeds - fee - basis_portion`
  - `basis_portion = cost_basis * fraction_sold`
- Flattening:
  - For long positions: `pnl = notional - fee - basis_total`
- Equity:
  - `equity = cash + sum(qty * price)`
- Summary metrics (uses `tools/performance_analysis.py`):
  - `equity_return_pct = (final_equity / initial_cash - 1) * 100`
  - `Sharpe`: mean/variance of period returns, annualized with 252
  - `max_drawdown`: peak-to-trough drawdown over equity series
  - Win metrics from realized trade PnL: win rate, avg win/loss, profit factor

### Daily P&L breakdown (LLM backtester)
File: `backtesting/llm_strategist_runner.py`

- Daily realized P&L: sum of trade log entries in day
- Breakdown (per day, using start/end equity):
  - `gross_trade_pct = non_flatten_pnl / start_equity * 100`
  - `flattening_pct = flatten_pnl / start_equity * 100`
  - `fees_pct = -fees / start_equity * 100`
  - `carryover_pnl = (end_equity - start_equity) - (non_flatten + flatten - fees)`
  - `carryover_pct = carryover_pnl / start_equity * 100`
  - `component_net_pct = (non_flatten + flatten - fees) / start_equity * 100`
  - `net_equity_pct = (end_equity / start_equity - 1) * 100`
  - `net_equity_pct_delta = net_equity_pct - (component_net_pct + carryover_pct)`

### Run-level aggregation and risk-adjusted P&L
File: `backtesting/reports.py`

- Aggregates `pnl`, `risk_used_abs`, `actual_risk_abs`, wins/losses across:
  - trigger/timeframe/hour/archetype buckets
- Risk-adjusted metrics:
  - `rpr = pnl / risk_used_abs`
  - `rpr_actual = pnl / actual_risk_abs`
  - `mean_r = pnl / trades / (risk_used_abs / trades)`
- Risk usage vs return correlation: Pearson on `(risk_utilization_pct, equity_return_pct)`
- RPR comparison labels:
  - `good` if `rpr >= 0.2` and delta vs baseline within 0.1
  - `bad` if `rpr <= -0.2` or underperforms baseline by >= 0.1

### Deterministic trade quality metrics
File: `trading_core/trade_quality.py`

- Win/loss classification threshold: `pnl > 0.01` win, `< -0.01` loss.
- Win rate, profit factor, risk/reward ratio:
  - `profit_factor = gross_profit / gross_loss` (or inf if no losses)
  - `risk_reward_ratio = avg_win / avg_loss`
- Quality score (0-100) combines win rate, profit factor, risk/reward,
  and penalties for consecutive losses and emergency exits.

### PortfolioState summary metrics
File: `agents/analytics/portfolio_state.py`

- Win rate and profit factor over 30d from trade log P&L.
- Sharpe (30d) from equity returns (PerformanceAnalyzer).
- Max drawdown (90d) from equity series.
- Realized P&L is `equity_window_delta` over 7d/30d.

### Execution ledger P&L
File: `agents/workflows/execution_ledger_workflow.py`

- Weighted average entry price on buys.
- Realized P&L on sells:
  - `(sell_price - entry_price) * qty_sold`
- Unrealized P&L:
  - `(current_price - entry_price) * qty_open`
- Total P&L: `realized + unrealized`
- Profit scraping: `scraped = realized_pnl * profit_scraping_pct` (deducted from cash)
- Performance metrics currently include placeholders:
  - `win_rate = 0.5`, `sharpe_ratio = 0.0` (no returns series)

### Live daily report P&L
File: `services/live_daily_reporter.py`

- `total_pnl` is sum of `realized_pnl` on filled orders.
- `wins/losses` counted from sign of `realized_pnl`.
- `win_rate = wins / trade_count * 100`.

## Risk-Adjusted P&L (Practical Interpretation)

The repo already includes risk-weighted metrics suitable for interpreting P&L:
- `rpr` and `rpr_actual` (return per risk) in `backtesting/reports.py`.
- `mean_r` (average R multiple per trade) in `backtesting/reports.py`.
- `profit_factor` and `risk_reward_ratio` in `trading_core/trade_quality.py`.
- Sharpe and drawdown in `tools/performance_analysis.py` and `agents/analytics/portfolio_state.py`.

If you want a single weighted metric, `rpr_actual` is the closest to "P&L per unit of
risk taken" because it divides realized P&L by the actual stop-distance-based risk.

## UI Gaps (Wins/Losses)

- Wins/losses are computed in multiple places (`trading_core/trade_quality.py`,
  `services/live_daily_reporter.py`, `backtesting/cli.py`) but the UI mostly shows
  win rate and total trades (e.g., `ui/src/components/BacktestControl.tsx`).
- There is no dedicated wins/losses display in the current UI components.
