# Runbook: Learning Book

## Overview
- Objective: define the Learning Book as a first-class construct, separate from the Profit Book, and wire it into the stack with explicit risk budgets and tags.
- Purpose: "loss is boring, observation is sharp".
- Scope: Information-Gathering Risk (exploration) only; exploitative risk stays in the Profit Book.

## Why it exists
Exploration trades should be sized, tagged, and accounted for differently than profit-seeking trades. Without a dedicated Learning Book, experiment risk leaks into profit metrics, backtest parity breaks, and guardrails are impossible to enforce.

## Definitions
- Learning position: any order/position where `learning_book=true` and `experiment_id` is present. It exists to answer a question, not to maximize PnL.
- Profit position: any order/position where `learning_book=false` (default). It exists to harvest edge and is evaluated primarily on PnL.
- Tagging model: `learning_book` (bool) + `experiment_id` (string, required for learning) + optional `experiment_variant` (string).
- Boundaries: Learning Book MUST use its own risk budget, MUST respect no-learn zones, MUST be capped by max_hold_time, and MUST NOT increase Profit Book exposure.

## Interfaces & Contracts
### Config surface (feature flags, budgets, sizing knobs)
Implementation MUST be explicit in config and schema, not implicit in prompt text.

- Add a `LearningBookSettings` model in `schemas/strategy_run.py` and wire into `StrategyRunConfig`.
- Surface the same fields in `ops_api/routers/paper_trading.py` (PaperTradingSessionConfig) and `ops_api/routers/backtests.py` (BacktestConfig).
- Provide environment defaults in `app/core/config.py` or `trading_core/config.py` (explicitly documented).

Required config keys (names and semantics):
- `learning_book_enabled` (bool): master feature flag.
- `learning_daily_risk_budget_pct` (float): daily risk budget for learning trades, separate from `RiskLimitSettings.max_daily_risk_budget_pct`.
- `learning_max_position_risk_pct` (float): per-trade risk cap for learning trades.
- `learning_max_portfolio_exposure_pct` (float): hard cap on total learning exposure.
- `learning_max_trades_per_day` (int): throttle learning trade count.
- `learning_sizing_mode` ("fixed_fraction" | "notional" | "vol_target").
- `learning_notional_usd` (float, optional): used when sizing_mode=notional.
- `learning_max_hold_minutes` (int): max hold time for learning positions.
- `learning_allow_short` (bool): allow short learning positions (default false).

### Data model (orders, fills, positions, trades)
Learning tags MUST be propagated end-to-end. Update the following structures:

- `schemas/llm_strategist.py` (TriggerCondition): add optional fields `learning_book: bool = False` and `experiment_id: str | None`.
- `agents/strategies/trigger_engine.py` (Order dataclass): add `learning_book: bool`, `experiment_id: str | None`, `experiment_variant: str | None`.
- `ops_api/schemas.py` (FillRecord): add `learning_book: bool`, `experiment_id: str | None`.
- `ops_api/event_store.py` payloads for `order_submitted`, `fill`, `position_update`, `trade_blocked`: include `learning_book` and `experiment_id`.
- `app/db/models.py`: add `learning_book` + `experiment_id` columns to `orders`, `position_snapshots`, `risk_allocations`, and `block_events`.

### Accounting (PnL attribution and isolation)
- Learning Book PnL MUST be tracked separately from Profit Book PnL in reports and telemetry.
- Learning Book losses MUST NOT reduce Profit Book risk budgets.
- Profit Book performance metrics MUST exclude learning trades unless explicitly requested.
- Learning PnL MAY be rolled up into total equity for safety checks, but budget enforcement MUST remain separate.

## Implementation Notes
Architecture touchpoints (exact integration points to modify):
- Strategy run config: `schemas/strategy_run.py`, `services/strategy_run_registry.py`.
- Plan generation context: `services/strategist_plan_service.py`, `agents/strategies/plan_provider.py` (inject Learning Book context into `LLMInput.global_context`).
- Trigger evaluation and order creation: `agents/strategies/trigger_engine.py`, `agents/strategies/trade_risk.py`.
- Risk sizing: `agents/strategies/risk_engine.py` (create a learning-specific RiskEngine or RiskConstraint).
- Backtest parity: `backtesting/llm_strategist_runner.py`, `backtesting/reports.py`.
- Persistence + events: `ops_api/event_store.py`, `ops_api/materializer.py`, `ops_api/schemas.py`, `app/db/models.py`.

Implementation guidance:
- Use a dedicated LearningBookSettings object (not `RiskLimitSettings`) to avoid conflating profit and learning budgets.
- Always tag learning orders at creation time (in `agents/strategies/trigger_engine.py`) so downstream systems do not guess.
- Derive learning sizing from LearningBookSettings and enforce caps in `TradeRiskEvaluator` when `learning_book=true`.

## Telemetry
Required events and logs:
- `event_type=trade_blocked` with `reason=no_learn_zone` and `learning_book=true` when learning is blocked.
- `event_type=order_submitted` and `event_type=fill` payloads MUST include `learning_book` and `experiment_id`.
- Live dashboards MUST show separate counts and PnL for learning vs profit trades (extend `ops_api/materializer.py`).

## Checklists
### Backtest + live parity checklist
- Backtests use the same `learning_book` tags and `experiment_id` fields as live.
- Backtest risk budgets for learning and profit books are enforced independently.
- Block reasons include `no_learn_zone` in both backtest and live.
- Learning PnL is separated in `backtesting/reports.py` and live reports.

### Minimal viable implementation checklist
- LearningBookSettings wired into `StrategyRunConfig` and API configs.
- `learning_book` + `experiment_id` tagged on orders, fills, positions, and block events.
- Learning risk budgets enforced in `TradeRiskEvaluator` (or equivalent gate).
- Learning PnL and counts surfaced in Ops API.

### Acceptance criteria
- Learning trades are rejected when `learning_book_enabled=false`.
- Learning trades never consume Profit Book risk budgets.
- Learning trades are fully traceable by `experiment_id` in events and DB.
- Profit trades execute normally when Learning Book is disabled or blocked.

## Examples
Example Learning Book config snippet (YAML-style):
```yaml
learning_book_enabled: true
learning_daily_risk_budget_pct: 0.50
learning_max_position_risk_pct: 0.25
learning_max_portfolio_exposure_pct: 2.0
learning_max_trades_per_day: 12
learning_sizing_mode: fixed_fraction
learning_max_hold_minutes: 60
learning_allow_short: false
```
