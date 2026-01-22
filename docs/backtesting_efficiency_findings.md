# Backtesting Efficiency Findings and Fix Plan

## Context
Observed increasing backtest slowness and a failure for
`backtest-91797054-9882-40ac-8b39-a564385e9646`. The workflow failed with:

```
Activity task failed
cause: Complete result exceeds size limit (non-retryable)
activity: run_llm_backtest_activity
```

## Key Findings
- The failure is not a memory leak. The LLM backtest activity returned a payload
  larger than Temporal's activity result size limit, so the activity failed
  non-retryably.
- The `backtest_runs` table has no row for the failed run, indicating metadata
  persistence happens after a successful activity completion.
- Progress accounting is inconsistent: a completed run shows
  `candles_processed` > `candles_total`, suggesting miscounted progress and
  misleading UI status.
- The LLM backtest path repeatedly computes indicators and market structure
  from full history per bar, and stores per-bar reports in memory. This makes
  runtime and memory scale poorly with candle count.

## Evidence (High Level)
- Worker logs show `Total candles to process: 1440`, then activity failure with
  "Complete result exceeds size limit".
- Database query shows only a prior backtest persisted; the failed run did not
  create a `backtest_runs` row.
- The activity result includes large payloads: `equity_curve`, `trades`, and
  `llm_data` with `bar_decisions` and `daily_reports`.

## Root Cause
`run_llm_backtest_activity` returns a large payload that grows with candle
count (per-bar reports and histories). Temporal enforces a strict payload
size limit, so longer backtests fail with "Complete result exceeds size limit".

## Secondary Efficiency Risks
- Per-bar slicing (`df[df.index <= ts]`) and full recalculation of indicator
  snapshots on every bar adds significant overhead.
- `slot_reports_by_day`, `bar_decisions`, and `blocked_details` grow unbounded
  within a run, increasing memory and payload size.
- Excessive per-event logging in tight loops adds I/O overhead.

## Suggested Fixes (Prioritized)

### 1) Fix the Activity Payload Size (Immediate Blocker)
- Persist full results to disk/DB inside the activity and return only a small
  summary and a handle/path.
- Keep `equity_curve`, `trades`, and `llm_data` (especially `bar_decisions`) out
  of the activity result.
- Update APIs to load full artifacts from disk/DB when needed.

**Target files**
- `backtesting/activities.py`
- `backtesting/persistence.py`
- `ops_api/routers/backtests.py`

### 2) Precompute Indicators and Avoid Full-History Slicing (Large Runtime Win)
- Precompute indicator series per timeframe and index into them by bar.
- Avoid `df[df.index <= ts]` on every bar; use integer offsets or cached
  rolling windows.

**Target files**
- `backtesting/llm_strategist_runner.py`
- `agents/analytics/indicator_snapshots.py`

### 3) Bound In-Memory Per-Bar Artifacts (Memory + Payload Win)
- Cap or sample `slot_reports_by_day` and `bar_decisions` (e.g., keep last N).
- Stream detailed bar decisions to disk, keep only aggregates in memory.
- Store only summary stats in DB.

**Target files**
- `backtesting/llm_strategist_runner.py`
- `backtesting/persistence.py`

### 4) Chunk Long Backtests (Robustness + Scalability)
- Split into daily/weekly chunks and stitch summaries.
- Persist between chunks and use Temporal continue-as-new for long runs.

**Target files**
- `backtesting/activities.py`
- `tools/backtest_execution.py` (workflow orchestration)

### 5) Logging + Progress Hygiene
- Aggregate blocked-trade logs instead of logging every event.
- Fix `candles_processed` accounting to align with actual bar count.

**Target files**
- `backtesting/llm_strategist_runner.py`
- `backtesting/activities.py`

## Expected Outcomes
- No more Temporal payload-limit failures for large backtests.
- Lower memory use and improved performance as candle count grows.
- Clearer progress reporting and more stable UI behavior.

## Next Steps (Recommended Order)
1) Reduce activity result payload and persist heavy data out-of-band.
2) Precompute indicators and remove per-bar full-history slicing.
3) Cap/stream per-bar artifacts and reduce per-event logging.
4) Implement chunked backtests for long date ranges.
