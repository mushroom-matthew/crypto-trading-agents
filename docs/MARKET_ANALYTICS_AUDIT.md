# Market Analytics Audit

This note inventories every surface that measures the market, summarizes available metrics and time scales, and captures current constraints plus recommendations for improving the LLM’s understanding of temporal context.

## 1. Analysis Surfaces

| Component | Path | Role |
| --- | --- | --- |
| MCP Metrics Tools | `mcp_server/app.py` (`list_technical_metrics`, `compute_technical_metrics`) | FastMCP endpoints wrapping the metrics package for ad-hoc analytics requests. |
| Metrics Library | `metrics/` + `tools/metrics_service.py` | Calculates indicator families (ADX, ATR, BollingerBands, EMA, MACD, OBV, ROC, RSI, SMA, VWAP, WMA) on cached OHLCV data. Default cadence: 1h candles, 500-bar window; callers can override timeframe/limit. |
| Market Data Workflows | `tools/market_data.py`, `tools/feature_engineering.py` | Stream 1s–level Coinbase ticks, preload 60 minutes of 1m candles, and maintain rolling feature vectors available via Temporal queries. |
| Execution Ledger Analytics | `agents/workflows/execution_ledger_workflow.py` (`get_performance_metrics`, `get_risk_metrics`, `get_risk_metrics_with_live_prices`) | Computes rolling trade stats and exposure ratios directly from ledger signals. Default window: 30 days. |
| Performance Analyzer | `tools/performance_analysis.py` | Creates composite reports (Sharpe, drawdown, win metrics, VAR, qualitative scores) used by judge workflows, backtests, and dashboards. |
| Judge Agent | `agents/workflows/judge_agent_workflow.py`, `agents/judge_agent_client.py` | Consumes performance/risk metrics, tracks evaluation windows (typically 7–30 days), evolves prompts, and requests new analyses via MCP tools. |
| Backtesting Stack | `backtesting/` (`simulator.py`, `llm_strategist_runner.py`, `dataset.py`) | Steps through cached OHLCV for 1h/4h/1d candles, computes execution-agent style features, and produces equity/metric curves for historical runs. |

## 2. Metric Catalog & Time Scales

| Metric Group | Source | Default Time Horizon | Notes |
| --- | --- | --- | --- |
| Real-time ticks | `tools/market_data.SubscribeCEXStream` | 1 second interval (configurable) | Streams Coinbase spot data; `HistoricalDataLoaderWorkflow` backfills 60×1m candles per pair on startup. |
| Feature vectors | `tools/feature_engineering.ComputeFeatureVector` | Sliding window sized by `VECTOR_WINDOW_SEC` (default 900s) | Maintains `historical_ticks` query for intraday reasoning. |
| Technical indicators | `metrics/` via MCP | Adjustable per request (default 1h candles, 500 bars) | `MetricsRequest` accepts timeframe (`1m`–`1d`) and sample size; results cached in `data/market_cache`. |
| Ledger performance | `ExecutionLedgerWorkflow.get_performance_metrics(window_days=30)` | Rolling 30-day window | Returns total trades, win rate, avg trade PnL, total PnL, max drawdown, Sharpe placeholder. |
| Ledger risk | `ExecutionLedgerWorkflow.get_risk_metrics()` | Instantaneous | Reports total portfolio value, cash ratio, position concentration, open positions, leverage proxy. |
| Judge evaluation metrics | `agents/judge_agent_client.py` | User-specified `window_days` (default 30) | Aggregates ledger metrics plus computed analytics (annualized return, Sharpe, win rate, drawdown) for prompt updates. |
| Backtest metrics | `backtesting/simulator.py`, `tools/performance_analysis.py` | Depends on dataset request (commonly multi-week/month) | Produces equity curves, trade logs, win metrics, profit factor, VAR, qualitative scores. |
| Strategy horizon metadata | `StrategySpec` (`tools/strategy_spec.py`) | Encodes `timeframe` (e.g., 15m), optional `expiry_ts`, and exit `max_bars_in_trade` | Used both live and in backtests; ensures deterministic time boxing per strategy. |

## 3. Time Horizon & Constraint Inventory

- **Execution gating** (`agents/execution_config.py`): minimum price move 0.5%, max price staleness 1800s, max 60 LLM calls/hour per symbol. Guards against over-trading and stale context.
- **Ledger price freshness** (`agents/workflows/execution_ledger_workflow.py:34`): price staleness threshold of 300s (5 min) for valuation; `get_price_staleness_info` exposes stale symbols so agents can refresh.
- **Strategy run cadence** (`agents/execution_agent_client.py:323-337`): `StrategyRunConfig` defaults to 15m timeframe, 30-day history window, and 24h plan cadence; `_plan_payload_for_spec` sets `valid_until` to now + 24h.
- **LLM strategist pacing** (`backtesting/llm_strategist_runner.py`): `llm_calls_per_day` (default 1) enforces a 24h plan interval; multi-timeframe (1h/4h/1d) candles inform plan adjustments.
- **Judge evaluation windows**: `JudgeAgentWorkflow` stores evaluations with timestamps; the judge client typically calls `get_performance_metrics(window_days=7|30)` and can trigger immediate evaluations when thresholds/time horizons are hit.
- **Risk/time-to-maturity interactions**:
  - `StrategySpec.exit_conditions` support `max_bars_in_trade` and `timed_exit`, forcing positions to close after a set number of candles even if profit targets aren’t met.
  - `StrategySpec.expiry_ts` and plan `valid_until` require revalidation when market regimes change; ignoring expiry increases risk because the execution agent may follow stale logic.
  - Profit-scraping and cash-ratio rules in `ExecutionLedgerWorkflow` may conflict with slow-moving strategies: a long time-to-maturity trade can trigger cash ratio penalties, causing the judge to flag it despite being within plan tolerances.

**Answering the “risk vs. time-to-maturity” question:** yes—whenever an entry signal arrives close to a strategy’s `max_bars_in_trade` or plan `valid_until`, the system balances expected payoff against the remaining time window. If holding the trade would extend beyond these limits, the execution agent is expected to stand down or reduce sizing, otherwise the judge will mark the decision as risky because it violates the time-boxed mandate.

## 4. Opportunities to Improve LLM Time Awareness

1. **Explicit temporal annotations in prompts**  
   - Enrich broker/execution prompts with fields like “current plan expires in X hours,” “last evaluation window start,” and “price data age.”  
   - Wire `ExecutionLedgerWorkflow.get_price_staleness_info` and plan `valid_until` directly into `agents/prompt_manager` templates so the LLM narrates time remaining before decisions expire.

2. **Multi-horizon metric bundles**  
   - Extend MCP tools to expose composite metrics (e.g., `get_performance_metrics` for 7, 30, and 90 days simultaneously) so the LLM can reason about intraday vs. swing performance without separate calls.  
   - Mirror this in `metrics_service` by allowing multiple timeframes (1m/15m/1h/4h/1d) per request and returning alignment hints (“signal strength decays after 4h”).

3. **Time-aware risk scoring**  
   - Enhance `PerformanceAnalyzer` to calculate trade duration stats (average holding time, longest open position) and include them in judge evaluations, reinforcing the importance of respecting time boxes.  
   - Feed these stats back into execution prompts to discourage entering trades when the residual time-to-expiry is shorter than the strategy’s expected holding period.

4. **LLM plan lifecycle tracking**  
   - When `StrategySpecWorkflow` emits a new spec, capture `expiry_ts`/`valid_until` and set reminders via Temporal signals that notify the execution and judge agents (and their prompts) as deadlines approach.  
   - Encourage the broker agent to automatically request a refreshed plan when 80% of the plan’s life has elapsed, minimizing scenarios where a trade runs up against its maturity window.

## 5. Time Constraints for Trades

- **Staleness rules**: No trade should use price data older than 5 minutes; execution-agent gating and ledger checks enforce this.  
- **Plan validity**: Execution plans expire 24h after generation unless explicitly extended; trades initiated after expiry violate governance.  
- **Strategy exit timers**: `max_bars_in_trade` and `timed_exit` exit conditions ensure trades do not exceed their intended maturation period.  
- **Call throttles**: `max_calls_per_hour_per_symbol` protects against over-frequent LLM invocations; combined with `llm_calls_per_day`, this creates upper bounds on decision refresh rates.  
- **Backtest parity**: Simulators respect the same time boxes, so any regression that would cause a “risk/time-to-maturity battle” is visible in historical runs before deployment.

By shining these time-related signals through the prompts, metrics tooling, and judge evaluations, we can keep LLM guidance aligned with real-world temporal constraints and surface actionable improvements whenever the system drifts from its designed horizons.

## 6. Backtest vs. Live Time Scale Mismatch

Backtests primarily consume 1h/4h/1d candles (plus occasional 15m datasets), whereas live trading reasons over 1s ticks, 1m historical bootstraps, and rolling 15m feature windows. This mismatch has both advantages and drawbacks:

| Aspect | Backtest Coarser Data (Pros) | Backtest Coarser Data (Cons) |
| --- | --- | --- |
| Speed & determinism | Fewer candles mean faster simulations, lower storage requirements, and reproducible runs. | Misses microstructure events (spikes, gaps) that trigger execution gating or judge alerts in live trading. |
| Signal clarity | Smoother candles help evaluate longer regimes and reduce noise. | Conceals slippage, liquidity droughts, and rapid reversals that occur within an hour. |
| Resource usage | Fits comfortably inside `data/backtesting` cache; easy to ship/share. | Limits ability to test tactics such as rapid stop-loss enforcement, profit-scraping cadence, or price staleness guards (300s) defined in `ExecutionLedgerWorkflow`. |

### Suggestions for Reconciliation

1. **Finer-grained cache tiers**  
   - Extend `backtesting/dataset.py` to cache 1m and 5m bars alongside existing 1h files. Use symbol/timeframe suffixes (e.g., `BTC-USD_1m.csv`) and optionally bucket data by week to keep file sizes manageable.  
   - Periodically sync these caches using the live `tools/market_data` stream (persist ticks from `record_tick`) so simulations can replay the exact microstructure seen in production.

2. **Hybrid replay mode**  
   - Allow `LLMStrategistBacktester` to mix timeframes: feed 1m data for execution decisions while still computing higher timeframe context (1h/4h) for strategic signals. This mirrors live prompt inputs, where the execution agent gets both recent ticks and aggregated indicators.

3. **Temporal downsampling checks**  
   - Add regression tests that run the same strategy on 1m and 1h data to quantify drift (PnL delta, drawdown differences, trade counts). Highlight discrepancies in judge reports so reviewers know when coarse backtests diverge from expected live behavior.

4. **LLM-aware caching hints**  
   - Include metadata in strategy plans (e.g., “tested on 1h candles, sensitivity to sub-5m volatility unknown”). When finer caches are present, re-run the plan automatically and attach delta metrics to the prompt so the LLM can reason about short-term risk.

5. **Replay queue for price-staleness scenarios**  
   - Store snapshots of scenarios that caused price staleness or gating triggers in production (since live ticks are 1s cadence). Feed these sequences back into the simulator to ensure strategies respect the 5-minute freshness rule even under coarse historical datasets.

These steps keep backtesting fidelity closer to the live 24×7 environment while preserving the reproducibility benefits of coarser datasets. When finer caches are unavailable, documenting the mismatch inside prompts and judge evaluations reduces the risk of over-trusting long-horizon simulations.
