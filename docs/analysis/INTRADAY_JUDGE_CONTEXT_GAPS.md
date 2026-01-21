# Intraday Judge Context Wiring Analysis

> **Status**: Most gaps addressed as of 2026-01-20. See [IMPLEMENTATION_REVIEW.md](./IMPLEMENTATION_REVIEW.md) for details.

## Executive Summary

~~The intraday judge in the backtest system has **incomplete visibility** into executed trades and trigger state.~~

**UPDATE**: All identified gaps have been addressed:
- ✅ Fill details (individual fills with timestamps, prices, trigger IDs, P&L)
- ✅ Active triggers (full definitions including entry/exit rules)
- ✅ Trigger attempt statistics (attempted/executed/blocked with reasons)
- ✅ Plan diff on replan (added/removed/changed triggers)
- ✅ Risk budget utilization (budget %, used %, per-symbol usage and blocks)
- ✅ Prompt interpretation guidance (tells LLM how to use each section)

This document identifies the original gaps and serves as a reference for the implemented solutions.

---

## What the Judge SHOULD See

For effective intraday evaluation, the judge needs:

| Data Category | Description | Purpose |
|--------------|-------------|---------|
| **Positions Opened/Closed** | All fills since last evaluation | Assess execution quality |
| **Market Metrics** | Current prices, volatility, trend | Context for decisions |
| **Active Triggers** | Currently defined triggers with rules | Understand what SHOULD fire |
| **Trigger Attempts** | Which triggers evaluated, why blocked | Diagnose signal issues |
| **Risk State** | Daily budget used, limits hit | Capacity assessment |

---

## What the Judge ACTUALLY Sees

### Data Provided to Judge (Current Implementation)

**Key File**: `backtesting/llm_strategist_runner.py:3691-3712`

```python
summary = {
    "return_pct": snapshot["return_pct"],
    "trade_count": snapshot["trade_count"],
    "positions_end": snapshot["positions"],
    "equity": snapshot["equity"],
    "anchor_equity": snapshot["anchor_equity"],
    "winning_trades": snapshot["winning_trades"],
    "losing_trades": snapshot["losing_trades"],
    # Deterministic quality metrics
    "trade_metrics": trade_metrics.to_dict(),
    "position_quality": [...],
}
```

### Gap Analysis (Updated - All Addressed)

| What's Needed | Status | Implementation |
|--------------|--------|----------------|
| Positions opened since last judge | ✅ FIXED | `fills_since_last_judge` with full details |
| Position close details | ✅ FIXED | Individual P&L per fill included |
| Trigger definitions | ✅ FIXED | `active_triggers` with entry/exit rules |
| Blocked trigger attempts | ✅ FIXED | `trigger_attempts` with block reasons |
| Market conditions at trade time | ✅ FIXED | `market_structure_entry` in fills |
| Risk budget utilization | ✅ FIXED | `risk_state` with budget %, used %, per-symbol |
| Prompt guidance | ✅ FIXED | `INTERPRETATION GUIDANCE` section added |

---

## Critical Wiring Issues

### Issue 1: Fill Details Not Passed to Judge

**Current Flow**:
```
portfolio.fills → filter by day_key → count only → summary["trade_count"]
```

**What's Lost**:
- Individual fill timestamps
- Entry/exit prices
- Trigger IDs that generated fills
- Realized P&L per fill

**Impact**: Judge sees "4 trades" but can't assess individual trade quality.

### Issue 2: Trigger Catalog Not Accessible

**Current Flow**:
```
trigger_catalog = self.plan_limits_by_day.get(day_key, {}).get("trigger_catalog", {})
trade_metrics = compute_trade_metrics(fills, trigger_catalog=trigger_catalog)
```

**What's Lost**:
- The `trigger_catalog` is used for **category mapping only**
- Actual trigger definitions (entry_rule, exit_rule) are NOT passed to judge
- Judge cannot suggest specific trigger modifications

**Impact**: Judge says "adjust triggers" but can't specify WHICH triggers or HOW.

### Issue 3: No Trigger Attempt Tracking

**Current Flow**:
```
# In TriggerEngine.on_bar()
block_entries.append({
    "trigger_id": trigger.id,
    "reason": reason,
    "detail": detail,
    ...
})
# block_entries returned but NOT aggregated for judge
```

**What's Lost**:
- Count of triggers that tried to fire
- Reasons for blocking (risk limits, cooldown, missing data)
- Ratio of attempts vs executions

**Impact**: Judge doesn't know if "no trades" means:
- Triggers didn't fire (signal issue)
- Triggers fired but were blocked (risk/capacity issue)
- Market didn't reach trigger levels (expected behavior)

### Issue 4: Position State Not Correlated with Fills

**Current Flow**:
```python
position_quality = assess_position_quality(
    positions=self.portfolio.positions,        # Current positions
    entry_prices=self.portfolio.avg_entry_price,  # Average entry
    current_prices=latest_prices,              # Current market
    position_opened_times={...},               # Open time
    current_time=ts,
)
```

**What's Lost**:
- Which fills opened each position
- Whether position was sized correctly
- Whether entry was at expected trigger price

**Impact**: Judge sees position is "underwater" but doesn't know if entry was good/bad.

---

## Proposed Context Enhancements

### Enhancement 1: Pass Fill Details

```python
# Add to summary dict
summary["fills_since_last_judge"] = [
    {
        "timestamp": f["timestamp"].isoformat(),
        "symbol": f["symbol"],
        "side": f["side"],
        "quantity": f["quantity"],
        "price": f["price"],
        "trigger_id": f.get("reason", "unknown"),
        "pnl": f.get("realized_pnl", 0.0),
    }
    for f in fills_since_last_evaluation
]
```

### Enhancement 2: Include Active Trigger Definitions

```python
# Add to summary dict
if current_plan:
    summary["active_triggers"] = [
        {
            "id": t.id,
            "symbol": t.symbol,
            "timeframe": t.timeframe,
            "direction": t.direction,
            "category": t.category,
            "confidence": t.confidence_grade,
            "entry_rule": t.entry_rule,
            "exit_rule": t.exit_rule,
        }
        for t in current_plan.triggers
    ]
```

### Enhancement 3: Track Trigger Attempt Statistics

```python
# Aggregate from TriggerEngine blocks
summary["trigger_stats"] = {
    trigger_id: {
        "attempted": count_of_evaluations,
        "fired": count_of_signals,
        "blocked": count_of_blocks,
        "block_reasons": {reason: count for reason, count in blocks.items()},
    }
    for trigger_id in all_triggers
}
```

### Enhancement 4: Include Risk Budget State

```python
# Add to summary dict
summary["risk_state"] = {
    "daily_risk_budget_pct": plan.risk_constraints.max_daily_risk_budget_pct,
    "risk_used_pct": self.daily_risk_used / starting_equity * 100,
    "trades_allowed": plan.max_trades_per_day,
    "trades_used": self.trades_today,
    "per_symbol_usage": {
        sym: usage_pct for sym, usage_pct in self.symbol_risk_usage.items()
    },
}
```

---

## Data Flow Diagram (Current vs Proposed)

### Current Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTRADAY JUDGE (CURRENT)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Inputs:                                                        │
│  ├─ return_pct ────────────────────────────────────► YES       │
│  ├─ trade_count ───────────────────────────────────► YES       │
│  ├─ equity/anchor_equity ──────────────────────────► YES       │
│  ├─ winning_trades/losing_trades ──────────────────► YES       │
│  ├─ position_quality (unrealized P&L) ─────────────► YES       │
│  ├─ trade_metrics (win rate, profit factor) ───────► YES       │
│  │                                                              │
│  ├─ individual fill details ───────────────────────► NO        │
│  ├─ trigger definitions ───────────────────────────► NO        │
│  ├─ trigger attempt/block stats ───────────────────► NO        │
│  ├─ risk budget utilization ───────────────────────► NO        │
│  └─ market conditions at trade time ───────────────► NO        │
│                                                                 │
│  Outputs:                                                       │
│  ├─ score (0-100)                                               │
│  ├─ should_replan (bool)                                        │
│  ├─ constraints (max_trades, disabled_triggers)                 │
│  └─ strategist_constraints (must_fix, vetoes, boost)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Proposed Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                   INTRADAY JUDGE (PROPOSED)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Inputs:                                                        │
│  ├─ return_pct ────────────────────────────────────► YES       │
│  ├─ trade_count ───────────────────────────────────► YES       │
│  ├─ equity/anchor_equity ──────────────────────────► YES       │
│  ├─ winning_trades/losing_trades ──────────────────► YES       │
│  ├─ position_quality ──────────────────────────────► YES       │
│  ├─ trade_metrics ─────────────────────────────────► YES       │
│  │                                                              │
│  ├─ fills_since_last_judge ────────────────────────► NEW       │
│  │   (timestamp, symbol, price, trigger_id, pnl)               │
│  ├─ active_triggers ───────────────────────────────► NEW       │
│  │   (id, symbol, direction, entry_rule, exit_rule)            │
│  ├─ trigger_stats ─────────────────────────────────► NEW       │
│  │   (attempted, fired, blocked, block_reasons)                │
│  ├─ risk_state ────────────────────────────────────► NEW       │
│  │   (budget_used_pct, trades_remaining, per_symbol)           │
│  └─ market_snapshot_at_fills ──────────────────────► NEW       │
│      (price, volatility, trend at each fill time)              │
│                                                                 │
│  Outputs (Enhanced):                                            │
│  ├─ score (0-100)                                               │
│  ├─ should_replan (bool)                                        │
│  ├─ specific_trigger_actions (modify, disable, boost)          │
│  ├─ risk_adjustments (per-symbol sizing changes)               │
│  └─ strategist_constraints (with trigger references)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority

| Enhancement | Effort | Impact | Priority |
|-------------|--------|--------|----------|
| Fill details | Low | High | P0 |
| Trigger stats | Medium | High | P0 |
| Active triggers | Low | Medium | P1 |
| Risk budget state | Medium | Medium | P1 |
| Market snapshot at fills | High | Medium | P2 |

---

## Files Requiring Changes

| File | Change Required |
|------|-----------------|
| `backtesting/llm_strategist_runner.py:3691-3712` | Add new summary fields |
| `trading_core/trade_quality.py` | Enhance fill detail processing |
| `services/judge_feedback_service.py` | Accept and analyze new fields |
| `agents/strategies/trigger_engine.py` | Expose attempt statistics |
