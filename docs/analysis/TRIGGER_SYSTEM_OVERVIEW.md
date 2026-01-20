# Trigger System Overview

## What Is a Trigger?

A **Trigger** is a deterministic rule that converts market conditions into trading decisions. Each trigger defines:
- **When to enter** a position (entry_rule)
- **When to exit** a position (exit_rule)
- **Optional: When to hold** despite exit signals (hold_rule)
- **Risk parameters** (stop_loss_pct)

Triggers are the bridge between LLM-generated strategy plans and actual order execution.

---

## Trigger Lifecycle

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          TRIGGER LIFECYCLE                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. GENERATION (LLM Strategist)                                         │
│     ├─ Market analysis → regime detection                               │
│     ├─ Technical indicators → entry/exit levels                         │
│     └─ Output: StrategyPlan with TriggerCondition[]                    │
│                                                                          │
│  2. STORAGE (Plan Provider)                                             │
│     ├─ Plan cached by hash of LLMInput                                  │
│     ├─ Triggers stored in plan.triggers array                           │
│     └─ Location: .cache/strategy_plans/{run_id}/{date}/{hash}.json     │
│                                                                          │
│  3. EVALUATION (Trigger Engine)                                         │
│     ├─ Each bar: evaluate all triggers for matching symbol/timeframe    │
│     ├─ Entry rule evaluated with current indicators                     │
│     ├─ Exit rule evaluated if in position                               │
│     └─ Output: Order[] or blocked entry record                          │
│                                                                          │
│  4. EXECUTION (Risk Engine → Order Placement)                           │
│     ├─ Position sizing via RiskEngine                                   │
│     ├─ Validate against daily limits                                    │
│     └─ Execute order or record block reason                             │
│                                                                          │
│  5. MONITORING (Judge)                                                  │
│     ├─ Evaluate trigger quality from fills                              │
│     ├─ Assess win rate by category                                      │
│     └─ Recommend adjustments or replan                                  │
│                                                                          │
│  6. ITERATION (Replan if needed)                                        │
│     ├─ Judge constraints passed to next plan                            │
│     ├─ Disabled triggers removed                                        │
│     └─ New triggers generated for regime                                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Trigger Structure (TriggerCondition)

**Definition**: `schemas/llm_strategist.py:112-129`

```python
class TriggerCondition(SerializableModel):
    id: str                           # Unique identifier (e.g., "btc_trend_1h_001")
    symbol: str                       # Trading pair (e.g., "BTC-USD")
    category: TriggerCategory | None  # Classification for analysis
    confidence_grade: "A" | "B" | "C" | None  # Signal quality
    direction: TriggerDirection       # "long", "short", "exit", "flat"
    timeframe: str                    # Bar size (e.g., "1h", "15m")
    entry_rule: str                   # Boolean expression for entry
    exit_rule: str                    # Boolean expression for exit
    hold_rule: str | None             # Optional: suppress exit when True
    stop_loss_pct: float | None       # Emergency exit distance
```

### Trigger Categories

| Category | Description | Example Use Case |
|----------|-------------|------------------|
| `trend_continuation` | Ride established trends | Long when price > EMA in uptrend |
| `reversal` | Catch trend changes | Short when RSI divergence at resistance |
| `volatility_breakout` | Trade range breaks | Long when price breaks Donchian upper |
| `mean_reversion` | Fade overextension | Short when price > 2σ Bollinger |
| `emergency_exit` | Forced risk-off | Exit all when drawdown > 2% |
| `other` | Uncategorized | Custom conditions |

### Confidence Grades

| Grade | Meaning | Execution Priority |
|-------|---------|-------------------|
| A | High conviction, multiple confirmations | Evaluated first, can override exits |
| B | Moderate conviction, single confirmation | Evaluated second |
| C | Low conviction, speculative | Evaluated last |
| None | Ungraded | Lowest priority |

---

## Trigger Rules (DSL)

Triggers use a simple expression language evaluated against indicator context.

### Available Operators

| Operator | Example | Description |
|----------|---------|-------------|
| `>`, `<`, `>=`, `<=` | `close > ema_20` | Numeric comparison |
| `==`, `!=` | `trend_state == "uptrend"` | Equality |
| `and`, `or` | `rsi < 30 and close > sma_50` | Logical combination |
| `not` | `not position == "long"` | Negation |
| `in` | `trend_state in ["uptrend", "sideways"]` | Membership |

### Available Variables

**From IndicatorSnapshot**:
- `close`, `open`, `high`, `low`
- `sma_short`, `sma_medium`, `sma_long`
- `ema_short`, `ema_medium`
- `rsi`, `rsi_14`
- `macd`, `macd_signal`, `macd_hist`
- `atr`, `atr_14`
- `bollinger_upper`, `bollinger_lower`, `bollinger_middle`
- `donchian_upper_short`, `donchian_lower_short`
- `fib_236`, `fib_382`, `fib_500`, `fib_618`, `fib_786`

**From AssetState**:
- `trend_state`: "uptrend" | "downtrend" | "sideways"
- `vol_state`: "low" | "normal" | "high" | "extreme"

**Cross-timeframe (prefixed)**:
- `tf_1h_close`, `tf_4h_rsi`, `tf_1d_ema_short`

**Position state**:
- `position`: "long" | "short" | "flat"

### Example Rules

```python
# Trend continuation entry
entry_rule = "close > ema_20 and rsi > 50 and trend_state == 'uptrend'"

# Mean reversion entry
entry_rule = "close < bollinger_lower and rsi < 30"

# Trailing stop exit
exit_rule = "close < ema_20 or rsi < 40"

# Hold rule (suppress exit)
hold_rule = "trend_state == 'uptrend' and rsi > 45"
```

---

## Trigger Evaluation Flow

**Key File**: `agents/strategies/trigger_engine.py:274-385`

```
For each bar:
    1. Check trade cooldown
       └─ Skip if recently traded

    2. For each trigger (sorted by confidence grade):
       │
       ├─ 3. Check symbol/timeframe match
       │      └─ Skip if not applicable
       │
       ├─ 4. Evaluate EXIT rule (if in position)
       │      ├─ If True: Check hold_rule
       │      │   ├─ Hold active? → Block exit
       │      │   └─ Hold inactive? → Generate flatten order
       │      └─ If False: Continue
       │
       ├─ 5. Evaluate ENTRY rule
       │      ├─ If True: Size position via RiskEngine
       │      │   ├─ Allowed? → Generate entry order
       │      │   └─ Blocked? → Record block reason
       │      └─ If False: Continue
       │
       └─ 6. Track triggers fired for symbol
              └─ Stop if max_triggers_per_symbol reached

    7. Deduplicate orders (exit priority over entry)
       └─ Exception: High-confidence entry can override exit
```

### Block Reasons

| Reason | Description | Resolution |
|--------|-------------|------------|
| `DAILY_RISK_EXHAUSTED` | Max daily loss/risk budget hit | Wait for new day |
| `POSITION_LIMIT` | Max position size reached | Close existing position |
| `COOLDOWN` | Recently traded this symbol | Wait cooldown period |
| `MIN_HOLD_PERIOD` | Exit blocked, position too new | Wait min hold bars |
| `MISSING_INDICATOR` | Required data not available | Check data pipeline |
| `EXPRESSION_ERROR` | Rule syntax invalid | Fix rule expression |
| `SIGNAL_PRIORITY` | Higher priority trigger already fired | Expected behavior |

---

## Implied Actions: Position Opens Trigger Creation

When a position is **opened**, several implicit actions should occur:

### 1. Emergency Exit Trigger (Implicit)

Every entry should implicitly create an emergency exit trigger:

```python
# When BTC long position opened at $50,000 with 2% stop
implicit_emergency = TriggerCondition(
    id=f"{entry_trigger_id}_emergency",
    symbol="BTC-USD",
    category="emergency_exit",
    direction="exit",
    timeframe=entry_trigger.timeframe,
    entry_rule="True",  # Always ready
    exit_rule=f"close < {entry_price * 0.98}",  # 2% stop
    stop_loss_pct=2.0,
)
```

**Why**: Ensures every position has a backstop exit regardless of planned exit rules.

### 2. Trailing Stop Adjustment (Implicit)

As position moves into profit, trailing stop should tighten:

```python
# When position is +1% in profit
update_emergency_trigger(
    trigger_id=f"{entry_trigger_id}_emergency",
    new_exit_rule=f"close < {max_price * 0.99}",  # Trail at 1% from high
)
```

**Why**: Locks in profits and prevents winners from becoming losers.

### 3. Scale-Out Triggers (Optional)

For larger positions, create partial exit triggers:

```python
# When position > 50% of max allocation
scale_out_trigger = TriggerCondition(
    id=f"{entry_trigger_id}_scale_out",
    symbol="BTC-USD",
    direction="exit",
    exit_rule=f"close > {entry_price * 1.02}",  # Take profit at 2%
    # Only exits 50% of position
)
```

**Why**: Reduces risk while letting winners run.

---

## Trigger Metrics and Quality

**Key File**: `trading_core/trade_quality.py`

### Computed Metrics Per Trigger

| Metric | Calculation | Good Value |
|--------|-------------|------------|
| Win Rate | wins / total_trades | > 50% |
| Profit Factor | gross_profit / gross_loss | > 1.5 |
| Avg Win/Loss Ratio | avg_win / avg_loss | > 1.0 |
| Emergency Exit % | emergency_exits / total_exits | < 20% |
| Consecutive Losses | max streak of losses | < 3 |

### Quality Score Formula

```python
score = 50.0  # Baseline

# Win rate contribution (0-30 points)
score += (win_rate * 50) - 15  # Neutral at 50%

# Profit factor contribution (0-25 points)
score += min(25, max(0, (profit_factor - 1.0) * 25))

# Risk/reward contribution (-10 to +20 points)
score += min(20, max(-10, (risk_reward_ratio - 1.0) * 20))

# Penalties
if max_consecutive_losses >= 5: score -= 15
if emergency_exit_pct > 0.3: score -= 10
```

---

## Judge-Trigger Interaction

### What Judge Can Recommend

| Action | Effect | Scope |
|--------|--------|-------|
| `disabled_trigger_ids` | Stop evaluating specific triggers | Per-trigger |
| `disabled_categories` | Stop all triggers of a type | Category-wide |
| `max_triggers_per_symbol_per_day` | Limit trigger fires | Symbol-wide |
| `sizing_adjustments` | Change position sizes | Per-symbol |
| `must_fix` | Guidance for next plan | Strategist |
| `vetoes` | Hard rules to avoid | Strategist |
| `boost` | Prioritize certain setups | Strategist |

### Feedback Loop

```
Judge Evaluation
    │
    ├─ Score < 45? → Trigger Replan
    │   └─ judge_constraints → passed to StrategyPlanProvider
    │       └─ LLM generates new plan with constraints
    │
    └─ Score >= 45? → Continue current plan
        └─ Constraints applied to TriggerEngine
            ├─ disabled_trigger_ids → skip in evaluation
            └─ max_triggers_per_symbol_per_day → cap fires
```

---

## Common Trigger Patterns

### Pattern 1: Trend Following

```python
TriggerCondition(
    id="btc_trend_1h",
    symbol="BTC-USD",
    category="trend_continuation",
    confidence_grade="A",
    direction="long",
    timeframe="1h",
    entry_rule="close > ema_20 and ema_20 > ema_50 and rsi > 50",
    exit_rule="close < ema_20 or rsi < 40",
    hold_rule="trend_state == 'uptrend' and rsi > 45",
    stop_loss_pct=2.0,
)
```

### Pattern 2: Mean Reversion

```python
TriggerCondition(
    id="eth_reversion_15m",
    symbol="ETH-USD",
    category="mean_reversion",
    confidence_grade="B",
    direction="long",
    timeframe="15m",
    entry_rule="close < bollinger_lower and rsi < 30",
    exit_rule="close > bollinger_middle or rsi > 50",
    stop_loss_pct=1.5,
)
```

### Pattern 3: Breakout

```python
TriggerCondition(
    id="btc_breakout_4h",
    symbol="BTC-USD",
    category="volatility_breakout",
    confidence_grade="A",
    direction="long",
    timeframe="4h",
    entry_rule="close > donchian_upper_short and vol_state != 'extreme'",
    exit_rule="close < ema_20",
    stop_loss_pct=3.0,
)
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `schemas/llm_strategist.py` | TriggerCondition, StrategyPlan models |
| `agents/strategies/trigger_engine.py` | Trigger evaluation logic |
| `agents/strategies/rule_dsl.py` | Rule expression parser |
| `agents/strategies/risk_engine.py` | Position sizing |
| `agents/strategies/plan_provider.py` | Plan caching, LLM calls |
| `trading_core/trade_quality.py` | Quality metrics computation |
| `services/judge_feedback_service.py` | Judge feedback generation |
| `prompts/llm_strategist_prompt.txt` | LLM prompt for trigger generation |
