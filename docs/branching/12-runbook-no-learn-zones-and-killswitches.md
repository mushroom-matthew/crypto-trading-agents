# Runbook: No-Learn Zones and Kill Switches

## Overview
- Objective: define when learning is forbidden and enforce kill switches for exploration risk.
- Scope: learning trades only; profit trades remain governed by existing risk constraints.

## Why it exists
Information-gathering trades should not run during unstable or low-quality market conditions. Hard rules prevent learning budget waste and protect the Profit Book from contagion risk.

## Definitions
- No-Learn Zone: a market or system condition where learning is temporarily blocked.
- Kill switch: a rule that halts learning trades after a loss or quality threshold is breached.
- Canonical gate: a single boolean `learning_allowed_now` with explicit reason codes.

## Interfaces & Contracts
### Config surface (thresholds and overrides)
- `StrategyRunConfig.metadata` in `schemas/strategy_run.py`: add `learning_gate_thresholds`, `learning_kill_switches`, `learning_override_until`, `learning_override_actor`, `learning_override_reason`.
- `ops_api/routers/backtests.py` and `ops_api/routers/paper_trading.py`: expose the same fields for API-driven runs.

### Canonical gate status
Add `LearningGateStatus` to a new `schemas/learning_gate.py` and emit it via events.

```python
from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import Field
from schemas.llm_strategist import SerializableModel


class LearningGateStatus(SerializableModel):
    learning_allowed_now: bool
    reason_codes: List[str] = Field(default_factory=list)
    observed_at: datetime
    source: str = "learning_gate"
```

Reason codes (MUST use these strings):
- `volatility_spike`
- `liquidity_thin`
- `spread_wide`
- `scheduled_event_window`
- `feed_latency`
- `exchange_error`
- `data_gap`
- `daily_loss_limit`
- `experiment_loss_limit`
- `consecutive_loss_limit`
- `slippage_anomaly`
- `data_quality_anomaly`
- `manual_override_off`

### No-Learn Zones (conditions)
Learning MUST be blocked when any of these are true:
- Volatility spike: `IndicatorSnapshot.realized_vol_short` or `atr_14` exceeds configured threshold.
- Liquidity thin: `IndicatorSnapshot.volume_multiple` below configured minimum.
- Spread wide: bid-ask spread bps exceeds configured max (from market data feed).
- Scheduled event window: in a configured blackout window (macro events or exchange maintenance).
- System instability: feed latency, exchange errors, or data gaps exceed threshold.

### Kill switches (learning only)
Learning MUST halt when any of these triggers fire:
- Daily loss: learning PnL for the day <= `learning_daily_loss_limit_pct`.
- Per-experiment loss: experiment PnL <= `experiment_loss_limit_pct`.
- Consecutive losses: >= N learning losses in a row for the experiment.
- Slippage anomaly: rolling slippage bps > configured max.
- Data-quality anomaly: missing bars, out-of-order ticks, or stale prices.

### Override policy
- Only an operator action or explicit API call can override learning blocks.
- Overrides MUST be time-bound (TTL) and MUST log actor + reason in the event store.
- Suggested config keys in `StrategyRunConfig.metadata`: `learning_override_until`, `learning_override_actor`, `learning_override_reason`.

## Implementation Notes
Architecture touchpoints (exact integration points to modify):
- Gate calculation: `services/market_data_worker.py` and `agents/analytics/market_structure.py` (compute volatility, volume, spread, and data health).
- Enforcement: `agents/strategies/trade_risk.py` and `trading_core/execution_engine.py` (block learning trades only).
- Event emission: `agents/event_emitter.py` and `ops_api/event_store.py` (emit `learning_gate_update` and kill switch events).
- Persistence: `app/db/models.py` (attach reason codes to `BlockEvent`).

Implementation guidance:
- Compute `learning_allowed_now` once per bar and cache it for the trading loop.
- When blocked, emit `trade_blocked` with `reason=no_learn_zone` and include `reason_codes`.
- Profit trades MUST ignore learning gate checks.

## Telemetry
Required logs/events:
- `learning_gate_update` event with `learning_allowed_now` + `reason_codes`.
- `learning_kill_switch_triggered` event with `reason_code` and `experiment_id`.
- `trade_blocked` events include `learning_book=true` and all gate reasons.

Dashboards/alerts:
- Alert when `learning_allowed_now=false` for > N minutes.
- Alert when kill switch triggers (by reason and experiment_id).

## Checklists
### Acceptance checklist
- `learning_allowed_now` is computed once and reused across the loop.
- Learning blocks do not affect Profit Book trades.
- Override TTL expires and resets to normal gating.
- All block events carry reason codes and `experiment_id` when applicable.

### Failure-mode table
| Failure mode | Expected behavior |
| --- | --- |
| Market feed latency spike | learning_allowed_now=false, reason_codes include feed_latency |
| Wide spread on symbol | learning_allowed_now=false, reason_codes include spread_wide |
| Override TTL expires | learning_allowed_now recomputed, manual_override_off removed |
| Data gap in candles | learning_allowed_now=false, reason_codes include data_gap |
| Slippage anomaly | learning_kill_switch_triggered fired, experiment paused |

## Examples
Example gate thresholds (YAML-style):
```yaml
learning_gate:
  max_realized_vol_short: 0.08
  min_volume_multiple: 0.9
  max_spread_bps: 8
  max_feed_latency_ms: 1500
  max_data_gap_seconds: 90
```
