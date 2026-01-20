# Backtest Analysis: backtest-79007255-b9c8-4a8a-b2a6-cf225a04ab55

## Run Configuration

```json
{
  "run_id": "backtest-79007255-b9c8-4a8a-b2a6-cf225a04ab55",
  "symbols": ["BTC-USD", "ETH-USD"],
  "timeframes": ["5m", "15m", "30m", "1h", "2h", "4h", "8h", "1d"],
  "history_window_days": 1,
  "plan_cadence_hours": 6,
  "flatten_policy": "none",
  "risk_limits": {
    "max_position_risk_pct": 4.0,
    "max_symbol_exposure_pct": 50.0,
    "max_portfolio_exposure_pct": 100.0,
    "max_daily_loss_pct": 6.0
  }
}
```

## Issue 1: Repeated Plans (Multiple Restarts)

### Observed Behavior
The backtest was restarted multiple times, causing the first plan to be regenerated repeatedly. This raises questions about **trigger consistency** across plan generations.

### Root Cause Analysis
When a backtest restarts:
1. The `StrategyPlanProvider` checks for cached plans in `.cache/strategy_plans/{run_id}/{date}/`
2. Plans are cached by **SHA-256 hash of the `LLMInput` JSON**
3. If the same `LLMInput` is provided, the **same cached plan is returned**
4. If `LLMInput` differs (e.g., different market state snapshot), a **new LLM call generates different triggers**

**Key File**: `agents/strategies/plan_provider.py:149-157`
```python
def _cache_path(self, run_id: str, plan_date: datetime, llm_input: LLMInput) -> Path:
    digest = hashlib.sha256(llm_input.to_json().encode("utf-8")).hexdigest()
    date_bucket = plan_date.strftime("%Y-%m-%d")
    return self.cache_dir / run_id / date_bucket / f"{digest}.json"
```

### Trigger Consistency Implications

| Scenario | Result |
|----------|--------|
| Same LLMInput on restart | Same cached plan, identical triggers |
| Different market state | New plan with potentially different triggers |
| New day boundary | Always generates new plan |
| Judge-triggered replan | New plan with feedback-adjusted triggers |

### What This Means for Analysis
- To compare trigger consistency, examine plans in `.cache/strategy_plans/backtest-79007255.../`
- Each `.json` file contains a `StrategyPlan` with its `triggers` array
- The hash in the filename indicates whether the input context was identical

---

## Issue 2: Intraday Judge Claims "No Trades" Despite 4 Positions

### Observed Behavior
The intraday judge repeatedly suggests adjusting triggers and reports no trades were made, even though 4 positions were opened/closed during the backtest.

### Root Cause Analysis

The intraday judge (`_run_intraday_judge`) receives trades via `_get_intraday_performance_snapshot()`:

**Key File**: `backtesting/llm_strategist_runner.py:3618-3649`
```python
def _get_intraday_performance_snapshot(self, ts: datetime, day_key: str, latest_prices):
    # Get today's fills so far
    today_fills = [
        f for f in self.portfolio.fills
        if f.get("timestamp") and f["timestamp"].strftime("%Y-%m-%d") == day_key
    ]
    # ...
    return {
        "trade_count": len(today_fills),
        # ...
    }
```

**Potential Causes for "No Trades" Report**:

1. **Timestamp Mismatch**: Fills may have timestamps that don't match `day_key` format
2. **Fill Recording Timing**: Judge runs before fills are recorded to `portfolio.fills`
3. **Wrong Day Key**: If backtest spans day boundaries, fills may be attributed to different day
4. **Fill Structure**: Missing or malformed `timestamp` field in fill records

### Debugging Checklist

```python
# Check fill timestamps in portfolio.fills
for fill in self.portfolio.fills:
    ts = fill.get("timestamp")
    print(f"Fill: {ts} -> day_key: {ts.strftime('%Y-%m-%d') if ts else 'NONE'}")
```

### Data Flow Trace

```
Order Generated (TriggerEngine.on_bar)
    ↓
Fill Executed (simulator applies order)
    ↓
portfolio.fills.append(fill_record)  ← Must include timestamp
    ↓
Judge Evaluation Triggered
    ↓
_get_intraday_performance_snapshot filters by day_key
    ↓
"trade_count" reported to judge
```

---

## Issue 3: Plan Replans Not Taking Effect

### Observed Behavior
Despite the judge continually suggesting adjustments to triggers, nothing appears to change in execution behavior.

### Root Cause Analysis

When judge triggers replan (`should_replan=True`), the flow is:

**Key File**: `backtesting/llm_strategist_runner.py:1839-1848`
```python
if judge_result.get("should_replan"):
    judge_triggered_replan = True
    # Apply judge feedback immediately
    judge_constraints = judge_result.get("feedback", {}).get("strategist_constraints")
    if judge_constraints:
        self.judge_constraints = judge_constraints  # ← Stored but may not affect current plan
```

**The Problem**:
- `judge_constraints` are stored on the runner instance
- But the **active `current_plan`** is only replaced when `new_plan_needed=True` is processed
- If `adaptive_replanning=True`, plan validity is day-scoped, so **intraday replans don't automatically replace the plan**

### What Actually Happens on Replan

1. Judge sets `should_replan=True`
2. `judge_triggered_replan=True` flag is set
3. `judge_constraints` are updated (for next plan generation)
4. `new_plan_needed` becomes True
5. **If** budget allows, new plan is generated incorporating constraints
6. **But** the new plan must go through the same LLM → triggers → execution pipeline

### Verification Steps

Check in backtest events:
- `plan_generated` events with `judge_feedback` in payload
- Compare trigger definitions before/after replan
- Verify `triggered_by: "judge"` in replan reason

---

## Recommended Investigations

### 1. Extract All Plans from This Run
```bash
find .cache/strategy_plans/backtest-79007255* -name "*.json" | \
  xargs -I{} sh -c 'echo "=== {} ===" && jq ".triggers | length" {}'
```

### 2. Compare Trigger Definitions Across Replans
For each plan, extract:
- `trigger.id`
- `trigger.entry_rule`
- `trigger.exit_rule`
- `trigger.direction`
- `trigger.timeframe`

### 3. Trace Fill → Judge Data Flow
Enable debug logging:
```python
# In _run_intraday_judge
logger.info("Judge evaluation: day=%s trade_count=%d fills=%s",
    day_key, snapshot["trade_count"], [f.get("timestamp") for f in self.portfolio.fills])
```

### 4. Verify Constraint Application
Check if `judge_constraints` are actually being passed to `get_plan()`:
```python
# In plan generation
logger.info("Generating plan with constraints: %s", self.judge_constraints)
```

---

## Key Questions to Answer

1. **Were the 4 positions opened/closed within a single day or across day boundaries?**
   - This affects whether judge sees them in `today_fills`

2. **What was the timing of judge evaluations relative to trade executions?**
   - Judge may have run before trades executed for that bar

3. **Did replans actually generate new plans or hit budget limits?**
   - Check `calls_used` vs `calls_per_day` in logs

4. **Were new triggers materially different after replan?**
   - Compare trigger `entry_rule` expressions before/after

---

## Files to Examine

| File | Purpose |
|------|---------|
| `.cache/strategy_plans/backtest-79007255.../` | Cached plans with triggers |
| `data/events.sqlite` | Event log (plan_generated, plan_judged, fill) |
| `data/strategy_runs/backtest-79007255*.json` | Run configuration |
| `backtesting/llm_strategist_runner.py:3651-3814` | Intraday judge implementation |
