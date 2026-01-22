# Event & Read Model Contracts (Draft)

This draft defines the append-only event schema and derived read models that the Ops API will serve. Breaking changes require a version bump.

## Event Schema
- `event_id: str`
- `ts: datetime`
- `source: str`
- `type: one of [tick, intent, plan_generated, plan_judged, trigger_fired, trade_blocked, order_submitted, fill, position_update, llm_call]`
- `payload: object`
- `dedupe_key: str?`
- `run_id: str?`
- `correlation_id: str?`

## Read Models
- **RunSummary**: run_id, latest_plan_id?, latest_judge_id?, status {running|paused|stopped}, last_updated, mode {paper|live}
- **BlockReasonsAggregate**: run_id?, reasons[{reason, count}]
- **FillRecord**: order_id, symbol, side, qty, price, ts, run_id?, correlation_id?
- **PositionSnapshot**: symbol, qty, mark_price, pnl, ts
- **LLMTelemetry**: run_id?, plan_id?, prompt_hash?, model, tokens_in, tokens_out, cost_estimate, duration_ms, ts

These models are mirrored in `ops_api/schemas.py`. Add new event types sparingly and version them when changing shape.
