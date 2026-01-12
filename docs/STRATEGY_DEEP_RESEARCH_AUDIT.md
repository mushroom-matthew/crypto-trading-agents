# Strategy Deep Research Audit (Retail Focus)

This audit compares `docs/strategy_deep_research.md` with the repository as architected, with an emphasis on retail-friendly strategies and operational concerns. It highlights what the system already covers well, where coverage is partial, and what is missing relative to the deep research reference.

## Scope and sources
- Primary reference: `docs/strategy_deep_research.md`
- Architecture and risk guidance: `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/SAFETY_GUARDS.md`, `docs/RISK_AUDIT.md`
- Strategy prompts and schemas: `prompts/strategies/*`, `schemas/llm_strategist.py`
- Execution and risk systems: `agents/strategies/trigger_engine.py`, `agents/strategies/risk_engine.py`, `trading_core/execution_engine.py`
- Backtesting and fees: `backtesting/*`, `app/costing/*`
- Exchange and wallet plumbing: `app/coinbase/*`, `app/strategy/trade_executor.py`, `agents/wallet_provider.py`, `docs/MULTI_WALLET_INTEGRATION_PLAN.md`

## Coverage map (retail strategy topics)

### Strategy catalog
- Trend following: Implemented via strategist prompts and trigger categories. Evidence: `prompts/strategies/momentum_trend_following.txt`, `schemas/llm_strategist.py`.
- Mean reversion: Implemented via strategist prompt and trigger categories. Evidence: `prompts/strategies/mean_reversion.txt`, `schemas/llm_strategist.py`.
- Volatility breakout: Implemented via strategist prompt and trigger categories. Evidence: `prompts/strategies/volatility_breakout.txt`, `schemas/llm_strategist.py`.
- Reversal and emergency exits: Implemented as trigger categories with exit handling. Evidence: `schemas/llm_strategist.py`, `agents/strategies/trigger_engine.py`.
- Grid trading: Missing. No grid order management or grid strategy templates present.
- Arbitrage (cross-exchange, triangular): Missing. No multi-venue routing or arbitrage logic.
- Market making/liquidity provision: Missing. No order book maker loop or spread capture logic.
- ML predictive or RL strategies: Missing. LLM is used for plan generation, but no predictive models or RL agents are implemented.

### Trigger-based rules and indicators
- Rule-based triggers: Implemented with a deterministic rule DSL and compiler. Evidence: `agents/strategies/rule_dsl.py`, `trading_core/trigger_compiler.py`.
- Indicator coverage (RSI, MACD, Bollinger, ATR, Donchian): Implemented. Evidence: `schemas/llm_strategist.py`, `metrics/technical.py`, `tools/feature_engineering.py`.
- Market structure overlays (support/resistance): Implemented and wired into trigger context. Evidence: `agents/analytics/market_structure.py`, `agents/strategies/trigger_engine.py`.
- Sentiment/news/on-chain signals: Stubbed or missing. Evidence: `metrics/sentiment.py` (placeholder only).

### Risk management and retail guardrails
- Stop-loss/exit logic: Implemented as trigger exit rules and optional `stop_loss_pct` for sizing. Evidence: `schemas/llm_strategist.py`, `agents/strategies/trigger_engine.py`.
- Position, symbol, portfolio, and daily loss caps: Implemented. Evidence: `agents/strategies/risk_engine.py`, `docs/RISK_AUDIT.md`.
- Daily risk budget: Implemented for throughput control. Evidence: `backtesting/llm_strategist_runner.py`, `docs/RISK_AUDIT.md`.
- Trade cooldown and minimum hold bars: Implemented to avoid churn. Evidence: `agents/strategies/trigger_engine.py`.
- Fee/slippage awareness: Implemented for backtests and live cost gating. Evidence: `backtesting/llm_strategist_runner.py`, `app/costing/gate.py`, `app/costing/fees.py`.
- Live trading safety latches: Implemented with explicit opt-in and multi-layer guards. Evidence: `docs/SAFETY_GUARDS.md`, `agents/runtime_mode.py`.

### Execution and exchange integration (retail constraints)
- Coinbase Advanced Trade integration (spot): Implemented. Evidence: `app/coinbase/advanced_trade.py`.
- Paper trading default and mock ledger: Implemented. Evidence: `tools/execution.py`, `agents/workflows/execution_ledger_workflow.py`, `CLAUDE.md`.
- Real ledger exists but is not the default path for agents: Partial. Evidence: `CLAUDE.md`, `app/ledger/*`.
- Order types: Market/limit supported; stop/OCO/trailing orders not surfaced. Evidence: `app/coinbase/advanced_trade.py`, `app/strategy/trade_executor.py`.
- Shorting/leverage/margin: Partial at the plan level (direction includes "short"), but no margin or derivatives execution and no negative position support in the mock ledger. Evidence: `schemas/llm_strategist.py`, `agents/workflows/execution_ledger_workflow.py`, `app/coinbase/advanced_trade.py`.
- Multi-exchange or DEX execution: Missing. No Solana/Phantom/Jupiter integration in live code; only a plan. Evidence: `docs/MULTI_WALLET_INTEGRATION_PLAN.md`.

### AI/LLM oversight and control loops
- LLM strategist and plan caching: Implemented with call budgeting and deterministic plan validation. Evidence: `agents/strategies/plan_provider.py`, `trading_core/trigger_compiler.py`.
- Judge agent (LLM as judge) for evaluation and prompt updates: Implemented. Evidence: `agents/judge_agent_client.py`, `agents/workflows/judge_agent_workflow.py`.
- Multi-agent coordination and durable workflow orchestration: Implemented. Evidence: `CLAUDE.md`, `docs/ARCHITECTURE.md`.

### On-ramp/off-ramp and custody
- Coinbase internal transfers and ledger tracking: Implemented. Evidence: `app/coinbase/transfers.py`, `app/strategy/trade_executor.py`, `app/db/models.py`.
- Fiat on-ramp/off-ramp flows (bank, ACH, PayPal, etc.): Missing in code. No direct banking or fiat withdrawal endpoints.
- Phantom/Solana wallet support: Missing in code; only a design plan. Evidence: `docs/MULTI_WALLET_INTEGRATION_PLAN.md`.
- Wallet abstraction for live balances: Partial. Live wallet provider is not implemented. Evidence: `agents/wallet_provider.py`.

## What is good for retail strategies
- Clear retail-oriented strategy templates (conservative, balanced, aggressive, mean-reversion, trend, breakout) enable simple, understandable tactics. Evidence: `prompts/strategies/*`.
- Deterministic trigger evaluation keeps LLM output bounded and auditable, which reduces retail risk. Evidence: `agents/strategies/rule_dsl.py`, `trading_core/trigger_compiler.py`.
- Robust risk controls (per-trade, per-symbol, portfolio, daily loss, daily risk budget) align with retail capital preservation goals. Evidence: `agents/strategies/risk_engine.py`, `docs/RISK_AUDIT.md`.
- Backtesting and paper trading reuse the same primitives, which fits retail needs for testing before risking capital. Evidence: `backtesting/*`, `agents/workflows/backtest_workflow.py`.
- Fee and slippage modeling plus a cost gate address the high-fee reality of retail trading. Evidence: `backtesting/llm_strategist_runner.py`, `app/costing/gate.py`.
- Live trading guardrails require explicit acknowledgement and default to paper mode. Evidence: `docs/SAFETY_GUARDS.md`, `agents/runtime_mode.py`.
- LLM oversight (judge agent) provides an adaptive feedback loop without relying on black-box predictive models. Evidence: `agents/judge_agent_client.py`, `agents/workflows/judge_agent_workflow.py`.
- Observability and auditability are strong due to Temporal workflows, Langfuse instrumentation, and event store logging. Evidence: `CLAUDE.md`, `agents/langfuse_utils.py`, `ops_api/event_store.py`.

## What is missing or partial relative to the deep research

### Strategy coverage gaps
- Grid trading is not implemented (no grid order management, no grid strategy prompt).
- Arbitrage and market-making strategies are not implemented (no cross-venue routing or order book making loop).
- AI/ML predictive strategies are not present; only LLM-based plan generation is implemented.

### Execution and market access gaps
- Margin/leverage trading is not supported; Coinbase integration is spot-only and the mock ledger does not support short positions.
- Stop-loss/take-profit orders are implemented logically but not as exchange-native order types (no stop or OCO order placement).
- Multi-venue and DEX integration (Phantom/Solana/Jupiter) is not implemented; only a roadmap exists.

### Retail operations gaps
- Fiat on/off ramps (bank transfer, ACH, PayPal) are not implemented in code. Coinbase transfers are internal only.
- Live wallet balance provider is not implemented; the system defaults to a paper wallet abstraction.
- No explicit stablecoin treasury or base/quote management beyond symbol-level positions.
- Sentiment/news/on-chain data pipelines are stubbed; no real-time qualitative feeds for retail oversight.

## Summary
The repository is strong for retail-friendly, indicator-driven strategies with careful risk and cost controls, robust backtesting, and defensive safety latches. The gaps mostly sit around strategy breadth (grid/arbitrage), leverage/shorting support, richer data inputs (sentiment/on-chain), and full on/off-ramp or DEX integrations. Addressing those missing pieces would materially close the distance to the full scope described in `docs/strategy_deep_research.md`.
