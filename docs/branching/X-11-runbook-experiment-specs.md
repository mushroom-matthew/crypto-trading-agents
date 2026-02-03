# Runbook: Experiment Specs

## Overview
- Objective: formalize Questions -> Exposure -> Metrics as explicit, reviewable ExperimentSpec objects.
- Scope: learning-only experiments (Learning Book); Profit Book is not a target for these specs.

## Why it exists
Without explicit ExperimentSpec objects, questions are vague, exposure is inconsistent, and results cannot be attributed. This runbook makes experiments reproducible and enforceable in both live and backtest flows.

## Definitions
- ExperimentSpec: a typed schema that binds a question to exposure constraints and metric requirements.
- Question: a falsifiable statement about market behavior or execution quality.
- Exposure: the allowed risk footprint (size, hold time, stop rules, and trade cadence).
- Metrics: the required measurements and thresholds that decide if the experiment graduates to exploitation.

## Interfaces & Contracts
### ExperimentSpec schema (Pydantic)
Create `schemas/experiment_spec.py` and use `SerializableModel` from `schemas/llm_strategist.py` for consistency.

```python
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import Field
from schemas.llm_strategist import SerializableModel, TriggerCategory


class ExposureSpec(SerializableModel):
    scope: Literal["micro", "meso", "macro"]
    max_hold_minutes: int = Field(..., ge=1)
    max_trades_per_day: int = Field(..., ge=1)
    stop_distance_pct: Optional[float] = Field(default=None, ge=0.0)
    stop_distance_atr: Optional[float] = Field(default=None, ge=0.0)
    max_notional_usd: float = Field(..., gt=0.0)
    max_slippage_bps: float = Field(default=25.0, ge=0.0)
    allow_short: bool = Field(default=False)
    entry_style: Literal["taker", "maker", "post_only"] = "taker"


class MetricSpec(SerializableModel):
    name: str
    unit: Literal["bps", "pct", "usd", "count", "ratio", "seconds"]
    aggregation: Literal["mean", "median", "p95", "sum", "count"] = "mean"
    min_samples: int = Field(default=30, ge=1)
    pass_threshold: Optional[float] = None
    direction: Literal["lower_better", "higher_better", "band"] = "band"


class ExperimentSpec(SerializableModel):
    experiment_id: str
    status: Literal["draft", "running", "paused", "invalidated", "ready_for_exploitation"]
    question: str
    hypothesis: str
    owner: str
    book: Literal["learning"] = "learning"
    target_symbols: List[str]
    trigger_categories: List[TriggerCategory]
    risk_budget_pct: float = Field(..., gt=0.0)
    exposure: ExposureSpec
    metrics: List[MetricSpec]
    regime_filters: List[Literal["bull", "bear", "range", "high_vol", "mixed"]] = []
    tags: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
```

### Config surface (how ExperimentSpec enters the system)
- `StrategyRunConfig.metadata` in `schemas/strategy_run.py`: add `experiment_id`, `experiment_spec_path`, `experiment_status_override`.
- `ops_api/routers/backtests.py`: add `experiment_id` and `experiment_spec` to BacktestConfig.
- `ops_api/routers/paper_trading.py`: add `experiment_id` and `experiment_spec` to PaperTradingSessionConfig.

### Data model and storage
Pick one storage path and keep it consistent in live + backtest.

- DB-first (preferred):
  - New tables in `app/db/models.py`:
    - `experiment_specs` (experiment_id PK, spec_json, status, created_at, updated_at).
    - `experiment_runs` (experiment_id FK, run_id, plan_id, started_at, ended_at).
  - All `orders`, `fills`, `position_snapshots`, `risk_allocations`, and `block_events` MUST carry `experiment_id`.
- Event-first (acceptable):
  - Add event types in `ops_api/schemas.py` and `agents/event_emitter.py`:
    - `experiment_created`, `experiment_status_changed`, `experiment_metric`, `experiment_blocked`.
  - Payloads MUST include `experiment_id`, `learning_book`, and a reference to `run_id`.

## Implementation Notes
### Architecture touchpoints
- Plan context injection: `services/strategist_plan_service.py` and `agents/strategies/plan_provider.py` (add ExperimentSpec into `LLMInput.global_context`).
- Trigger tagging: `schemas/llm_strategist.py` (TriggerCondition fields) and `agents/strategies/trigger_engine.py` (Order fields).
- Backtest wiring: `backtesting/llm_strategist_runner.py` (load ExperimentSpec and enforce exposure rules).
- Telemetry: `agents/event_emitter.py`, `ops_api/event_store.py`, `ops_api/schemas.py`.

### Question design patterns
Use one clear question per experiment. Example patterns:
- Microstructure: "Does spread < 5 bps predict >= 60% fill rate within 30s?"
- Execution: "Does post-only entry reduce slippage by 30% vs taker?"
- Regime: "Does breakout mean-reversion under high_vol regimes produce lower MAE?"
- Risk: "Do tighter stops reduce drawdown without degrading win rate?"

### Exposure taxonomy
- micro: seconds to minutes, tight stops, max_hold_minutes <= 60, small notional.
- meso: hours to a few days, balanced stops, max_hold_minutes 60-720.
- macro: days to weeks, wider stops, max_hold_minutes >= 720.

Rules:
- `max_hold_minutes` MUST be set and enforced.
- Exactly one of `stop_distance_pct` or `stop_distance_atr` MUST be set.
- `max_notional_usd` MUST be less than or equal to Learning Book budget for the day.

### Metrics catalog (required fields)
Execution:
- `slippage_bps`, `fill_rate`, `time_to_fill_seconds`, `post_only_reject_rate`.

Microstructure:
- `spread_bps_mean`, `spread_bps_p95`, `realized_vol_short`, `volume_multiple`.

Risk:
- `mae_pct`, `mfe_pct`, `max_drawdown_pct`, `r_multiple`.

Regime slices:
- `metric_by_regime` using `StrategyPlan.regime` and `backtesting/regimes.py` labels.

Rule-violation counters:
- `stop_missing_count`, `hold_time_exceeded_count`, `no_learn_zone_block_count`.

### Experiment lifecycle
Transitions are explicit and MUST be audited:
- draft -> running: approved by operator, learning_book_enabled true.
- running -> paused: manual pause or kill switch.
- running -> invalidated: data quality failure or violated exposure rules.
- running -> ready_for_exploitation: metrics pass thresholds and review complete.

## Telemetry
- Emit `experiment_metric` events with `experiment_id`, `metric_name`, `value`, `unit`, `sample_count`.
- Emit `experiment_status_changed` events on every lifecycle transition.
- Store experiment metadata in `ops_api/event_store.py` for Ops UI to display.

## Checklists
### Acceptance checklist
- ExperimentSpec validates and round-trips through storage.
- Exposure limits are enforced for learning trades.
- Metrics are computed and tagged with `experiment_id`.
- Lifecycle transitions are logged and reversible (pause/resume).

## Examples
ExperimentSpec YAML:
```yaml
experiment_id: exp_spread_001
status: draft
question: "Does spread < 5 bps improve fill rate within 30s?"
hypothesis: "If spread < 5 bps, fill_rate >= 0.60 and slippage_bps <= 8."
owner: "research"
book: learning
target_symbols: ["BTC-USD", "ETH-USD"]
trigger_categories: ["mean_reversion"]
risk_budget_pct: 0.5
exposure:
  scope: micro
  max_hold_minutes: 30
  max_trades_per_day: 12
  stop_distance_pct: 0.4
  max_notional_usd: 500
  max_slippage_bps: 15
  allow_short: false
  entry_style: post_only
metrics:
  - name: fill_rate
    unit: ratio
    aggregation: mean
    min_samples: 50
    pass_threshold: 0.60
    direction: higher_better
  - name: slippage_bps
    unit: bps
    aggregation: p95
    min_samples: 50
    pass_threshold: 12
    direction: lower_better
regime_filters: ["range", "high_vol"]
tags:
  desk: "learning"
  priority: "medium"
```

ExperimentSpec JSON:
```json
{
  "experiment_id": "exp_exec_002",
  "status": "running",
  "question": "Does taker entry improve fill speed without excess slippage?",
  "hypothesis": "time_to_fill_seconds <= 0.25 and slippage_bps <= 10",
  "owner": "execution",
  "book": "learning",
  "target_symbols": ["BTC-USD"],
  "trigger_categories": ["volatility_breakout"],
  "risk_budget_pct": 0.35,
  "exposure": {
    "scope": "micro",
    "max_hold_minutes": 20,
    "max_trades_per_day": 8,
    "stop_distance_pct": 0.3,
    "max_notional_usd": 300,
    "max_slippage_bps": 10,
    "allow_short": false,
    "entry_style": "taker"
  },
  "metrics": [
    {
      "name": "time_to_fill_seconds",
      "unit": "seconds",
      "aggregation": "p95",
      "min_samples": 30,
      "pass_threshold": 0.25,
      "direction": "lower_better"
    }
  ],
  "regime_filters": ["mixed"],
  "tags": {
    "desk": "execution"
  },
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-02T00:00:00Z"
}
```
