# Backtest 6d140887 Fixes: Telemetry and Persistence

Context: `backtest-6d140887-acea-4a0c-93ad-3769573022ae` ran with the live LLM strategist (no shim). Results were persisted inside the worker container (`/app/.cache/backtests/...pkl`). The host `.cache/backtests`, `backtest_runs`, and `block_events` tables are empty, so audits require container access and pickle parsing.

## What Needs Fixing

- Backtest results are not persisted to Postgres or a host volume.
- `block_events` is empty, so block churn cannot be queried.
- Plan logs do not capture actual generation time or replan reasons.

## Fix Options

### 1) Persist Backtest Results to Postgres

Store structured JSON in `backtest_runs.results` and make this the source for the Ops API:
- Save `summary`, `trades`, `plan_log`, and `limit_enforcement`.
- Include `run_id`, `started_at`, `completed_at`, and status.

Suggested hooks:
- `backtesting/llm_strategist_runner.py` (persist after completion)
- `ops_api/routers/backtests.py` (read from DB first, fallback to disk)

Success criteria:
- `backtest_runs` has a row for each run with populated `results`.
- Ops API no longer relies on container-local cache.

### 2) Persist Block Events for Backtests

Emit block events into `block_events` and/or a backtest-scoped event table:
- Store `run_id`, `trigger_id`, `reason`, `detail`, `timestamp`, `symbol`, `timeframe`.
- Populate from `blocked_entries` in `_process_orders_with_limits`.

Suggested hooks:
- `backtesting/llm_strategist_runner.py` (where `blocked_entries` are recorded)
- `ops_api/routers/agents.py` (queryable backtest blocks)

Success criteria:
- `block_events` has rows for backtests.
- Queries can break down block causes per run.

### 3) Mount Backtest Cache to Host

If DB persistence is deferred, mount `/app/.cache/backtests` as a Docker volume:
- Ensures host visibility without container access.

Suggested hooks:
- `docker-compose.yml` (volume mapping)

Success criteria:
- Host `.cache/backtests` contains new run artifacts.

### 4) Record Real Replan Metadata

Augment plan logs with:
- `generated_at_actual` (the timestamp when LLM response was generated)
- `replan_reason` (judge_triggered, new_day, etc.)
- `replan_sequence` (monotonic index)

Suggested hooks:
- `backtesting/llm_strategist_runner.py` (plan_log payload)
- `services/strategist_plan_service.py` (event payload)

Success criteria:
- Churn audits can compute replan cadence without log scraping.

## Recommended Order

1) Persist results to Postgres.
2) Emit block events for backtests.
3) Add plan replan metadata.
4) Add host-mounted cache as a safety net.

