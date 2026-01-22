# Research Summary: Backtest Judge and Trigger System Analysis

## Documents Created

1. **[BACKTEST_79007255_ANALYSIS.md](./BACKTEST_79007255_ANALYSIS.md)** - Specific analysis of the backtest run issues
2. **[INTRADAY_JUDGE_CONTEXT_GAPS.md](./INTRADAY_JUDGE_CONTEXT_GAPS.md)** - Context wiring gaps in the judge system
3. **[TRIGGER_SYSTEM_OVERVIEW.md](./TRIGGER_SYSTEM_OVERVIEW.md)** - Comprehensive trigger documentation

---

## Key Findings

### 1. Why Repeated Plans May Have Different Triggers

Plans are cached by **SHA-256 hash of LLMInput**. If the market state snapshot differs on restart, you get a different plan. However, if the exact same conditions are presented, the **cached plan is returned unchanged**.

**Verification**: Check `.cache/strategy_plans/backtest-79007255.../` for multiple plan files (different hashes = different inputs).

### 2. Why Judge Reports "No Trades" Despite 4 Positions

The intraday judge filters fills by `day_key` string match:
```python
today_fills = [f for f in fills if f["timestamp"].strftime("%Y-%m-%d") == day_key]
```

**Likely causes**:
- Fills have timestamps that don't match the day key format
- Judge evaluation runs before fills are recorded
- Fills span day boundaries

### 3. Why Replans Don't Seem to Change Anything

The replan workflow:
1. Judge sets `should_replan=True`
2. `judge_constraints` stored for next plan
3. New plan requested from LLM
4. **BUT**: Budget limits may prevent new LLM call
5. **AND**: New plan still goes through same trigger evaluation

**Check**: Look for `LLM call budget exhausted` errors in logs.

### 4. Judge Context Gaps

The judge currently receives:
- Aggregate trade counts (not individual fills)
- Performance metrics (win rate, profit factor)
- Position quality (unrealized P&L)

The judge does **NOT** receive:
- Trigger definitions (what should fire)
- Trigger attempt statistics (what tried to fire but was blocked)
- Risk budget utilization
- Individual fill details

---

## Critical Code Locations

| Component | File | Key Lines |
|-----------|------|-----------|
| Intraday Judge | `backtesting/llm_strategist_runner.py` | 3651-3814 |
| Judge Context Build | `backtesting/llm_strategist_runner.py` | 3691-3712 |
| Fill Filter | `backtesting/llm_strategist_runner.py` | 3618-3622 |
| Trigger Evaluation | `agents/strategies/trigger_engine.py` | 274-385 |
| Trigger Model | `schemas/llm_strategist.py` | 112-129 |
| Plan Caching | `agents/strategies/plan_provider.py` | 149-214 |
| Judge Feedback | `services/judge_feedback_service.py` | 127-604 |

---

## Recommended Debugging Steps

### Step 1: Extract Plans and Compare Triggers

```bash
# Find all plans for this run
find .cache/strategy_plans -path "*79007255*" -name "*.json" | head -10

# For each plan, extract trigger count and IDs
for f in $(find .cache/strategy_plans -path "*79007255*" -name "*.json"); do
  echo "=== $f ==="
  cat "$f" | python3 -c "import json,sys; p=json.load(sys.stdin); print(f'Triggers: {len(p.get(\"triggers\",[]))}'); [print(f'  - {t[\"id\"]}: {t[\"direction\"]}') for t in p.get('triggers',[])]"
done
```

### Step 2: Check Event Store for This Run

```python
from ops_api.event_store import EventStore
store = EventStore()

# Get all events for this run
events = store.list_events_filtered(
    run_id="backtest-79007255-b9c8-4a8a-b2a6-cf225a04ab55",
    limit=1000
)

# Filter by type
plan_events = [e for e in events if e.type == "plan_generated"]
judge_events = [e for e in events if e.type == "plan_judged"]
fill_events = [e for e in events if e.type == "fill"]

print(f"Plans: {len(plan_events)}, Judge evals: {len(judge_events)}, Fills: {len(fill_events)}")
```

### Step 3: Verify Fill Timestamps

```python
# In backtest runner after fills recorded
for fill in self.portfolio.fills[-10:]:  # Last 10 fills
    ts = fill.get("timestamp")
    print(f"Fill: {fill.get('symbol')} @ {fill.get('price')} ts={ts} day={ts.strftime('%Y-%m-%d') if ts else 'NONE'}")
```

### Step 4: Add Debug Logging to Judge

```python
# In _run_intraday_judge, before summary construction
logger.info(
    "Judge evaluation: day=%s fills=%d recent=%s",
    day_key,
    len(self.portfolio.fills),
    [f.get("timestamp").isoformat() for f in self.portfolio.fills[-5:] if f.get("timestamp")]
)
```

---

## Proposed Fixes (Priority Order)

### P0: Fix Fill Visibility

**Problem**: Judge can't see fills.
**Fix**: Pass `fills_since_last_judge` list with full details to judge summary.

### P0: Add Trigger Attempt Tracking

**Problem**: Judge doesn't know why triggers aren't firing.
**Fix**: Aggregate `block_entries` from TriggerEngine and include in summary.

### P1: Include Active Trigger Definitions

**Problem**: Judge says "adjust triggers" but can't specify which.
**Fix**: Pass `current_plan.triggers` to judge context.

### P1: Surface Risk Budget State

**Problem**: Judge doesn't know if budget is exhausted.
**Fix**: Add `risk_state` dict with utilization metrics.

### P2: Implement Implicit Emergency Triggers

**Problem**: Positions may not have stop-loss protection.
**Fix**: Auto-generate emergency exit trigger when position opens.

---

## Questions for Follow-Up

1. **Were the 4 positions all on the same day?** Check timestamps against judge day_key.

2. **Did any replan actually generate a new LLM call?** Check `calls_used` vs `calls_per_day` in logs.

3. **What were the actual trigger definitions in each plan?** Extract and diff.

4. **Were triggers firing but blocked by risk limits?** Check for `DAILY_RISK_EXHAUSTED` blocks.

5. **What does the fill data actually look like?** Inspect `portfolio.fills` structure.
