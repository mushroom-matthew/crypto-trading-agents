# Repo Analysis: Emergency Exit Competition with Entries

## Problem Statement

Emergency exit triggers fire immediately after entries, causing high emergency exit rates and poor performance. The judge flagged: "addressing the high emergency exit rate which currently hinders performance."

**Evidence from backtest-4722e226**:
- 21 trigger attempts, only 2 executed trades
- 0% win rate on executed trades
- Judge recommended disabling emergency exits

---

## Root Cause Analysis

### 1. Trigger Evaluation Order: Exits First

**File**: `agents/strategies/trigger_engine.py:303-385`

```python
for trigger in trigger_list:
    # ... symbol/timeframe matching ...

    # EXIT evaluation FIRST (lines 314-356)
    try:
        exit_fired = bool(trigger.exit_rule and
                          self.evaluator.evaluate(trigger.exit_rule, context))
    # ...
    if exit_fired:
        # Check hold_rule - BUT emergency_exit bypasses this!
        if trigger.hold_rule and trigger.category != "emergency_exit":  # Line 332
            # hold_rule can suppress exit
        # Check min hold period
        if self._is_within_hold_period(trigger.symbol):  # Line 348
            # Block exit if held < min_hold_bars
        exit_order = self._flatten_order(...)

    # ENTRY evaluation SECOND (lines 358-378)
    try:
        entry_fired = bool(trigger.entry_rule and
                           self.evaluator.evaluate(trigger.entry_rule, context))
```

**Key Finding**: Emergency exits bypass `hold_rule` checks (line 332) but **still respect** `min_hold_bars` (line 348).

---

### 2. Two Separate Min Hold Systems (Not Synchronized!)

#### TriggerEngine: Bar-Based Min Hold

**File**: `agents/strategies/trigger_engine.py:69, 397-405`

```python
def __init__(self, ..., min_hold_bars: int = 4, ...):
    self.min_hold_bars = max(0, min_hold_bars)

def _is_within_hold_period(self, symbol: str) -> bool:
    if self.min_hold_bars <= 0:
        return False
    entry_bar = self._position_entry_bar.get(symbol)
    if entry_bar is None:
        return False
    bars_held = self._bar_counter - entry_bar
    return bars_held < self.min_hold_bars  # Must hold 4+ bars
```

#### Backtester: Time-Based Min Hold

**File**: `backtesting/llm_strategist_runner.py:351, 1257-1279`

```python
# Default 2 hours
min_hold_hours: float = 2.0

def _min_hold_block(order: Order) -> str | None:
    if self.min_hold_hours <= 0:
        return None
    opened_at = self.last_flat_time_by_symbol.get(order.symbol)
    if opened_at is None:
        return None
    if order.timestamp - opened_at >= self.min_hold_window:
        return None  # Passed min hold
    # Block exit
```

**Problem**: These two systems are **not connected**:
- TriggerEngine counts bars
- Backtester checks wall-clock time
- Neither knows about the other

---

### 3. Emergency Exit Rule Sensitivity

**From backtest plans**:

```python
# Plan 2 emergency exit
entry_rule = "position != 'flat' and close < nearest_support"
```

This fires whenever:
1. Position is open (any position)
2. Price touches support level

**Problem**: Support levels can be very close to entry prices, causing immediate exits.

---

### 4. Deduplication Favors Exits

**File**: `agents/strategies/trigger_engine.py:485-558`

```python
def _deduplicate_orders(self, orders, bar_symbol):
    # Separate exits from entries
    exits = [o for o in symbol_orders
             if o.reason.endswith("_exit") or o.reason.endswith("_flat")]
    entries = [o for o in symbol_orders if o not in exits]

    # High-confidence entries CAN override exits
    high_conf_entries = [
        (o, priority) for o in entries
        if meets_threshold(get_confidence(o))
    ]

    if high_conf_entries:
        result.append(best_entry)  # High-conf entry wins
    elif exits:
        result.append(exits[0])  # EXIT PRIORITY (line 553)
    elif entries:
        result.append(entries[0])
```

**Default behavior**: Exits win unless entry has high confidence grade.

---

### 5. Fill Recording During Same Bar

**File**: `backtesting/llm_strategist_runner.py:2334-2353`

```python
# Process orders for current bar
orders, blocked = trigger_engine.on_bar(bar, indicator, portfolio_state, ...)
executed_records = self._process_orders_with_limits(...)

# After execution, record fills
if fills_after > fills_before:
    for _ in range(fills_after - fills_before):
        self._on_trade_executed(ts)
```

**Timing issue**: Within the same bar evaluation:
1. Entry trigger fires → fill recorded
2. Emergency exit trigger evaluates → sees position exists → fires
3. Both happen before bar advances

---

## Why Min Hold Doesn't Always Work

### Scenario A: Same Bar, Different Triggers

```
Bar N, Trigger A (mean_reversion): entry_rule fires → BUY order generated
Bar N, Trigger B (emergency_exit): exit_rule fires → SELL order generated
Deduplication: Exit wins (unless entry is high-confidence)
Result: No position taken or immediate exit
```

### Scenario B: Multi-Timeframe Same Timestamp

```
Bar N (15m timeframe): Entry trigger fires → position opened
Bar N (1h timeframe): Emergency exit evaluates → close < support → fires
Result: Position held for 0 bars, min_hold_bars check may not help
```

### Scenario C: Min Hold Misconfiguration

```python
# If min_hold_bars = 0 or min_hold_hours = 0
# Both checks pass immediately
# Emergency exit can fire on the same bar as entry
```

---

## Proposed Fixes

### Fix 1: Emergency Exits Should NOT Compete on Same Bar

**File**: `agents/strategies/trigger_engine.py`

```python
def on_bar(self, bar, indicator, portfolio, ...):
    # Track if entry happened this bar
    entry_happened_this_bar = set()

    for trigger in trigger_list:
        # ... exit evaluation ...
        if exit_fired:
            # NEW: Block emergency exit if entry just happened
            if trigger.category == "emergency_exit":
                if trigger.symbol in entry_happened_this_bar:
                    self._record_block(block_entries, trigger,
                        "SAME_BAR_ENTRY", "Cannot emergency exit same bar as entry", bar)
                    continue

        # ... entry evaluation ...
        if entry_fired:
            entry = self._entry_order(...)
            if entry:
                entry_happened_this_bar.add(trigger.symbol)
```

### Fix 2: Synchronize Min Hold Systems

**File**: `backtesting/llm_strategist_runner.py`

```python
def __init__(self, ...):
    # Use consistent min hold
    self.min_hold_bars = 4  # TriggerEngine default
    self.min_hold_hours = self.min_hold_bars * (timeframe_minutes / 60)
```

Or pass the TriggerEngine instance's `min_hold_bars` to the backtester.

### Fix 3: Emergency Exit Should Have Higher Threshold

**File**: `prompts/llm_strategist_prompt.txt`

Add instruction:
```
## EMERGENCY EXIT RULES

Emergency exit triggers should only fire on SIGNIFICANT adverse moves:
- close < nearest_support * 0.98 (2% below support)
- NOT just close < nearest_support

Emergency exits are for protecting against large losses, not normal fluctuations.
```

### Fix 4: Add Emergency Exit Cooldown

**File**: `agents/strategies/trigger_engine.py`

```python
def __init__(self, ..., emergency_exit_cooldown_bars: int = 8):
    self.emergency_exit_cooldown_bars = emergency_exit_cooldown_bars
    self._last_emergency_exit_bar: dict[str, int] = {}

def on_bar(self, ...):
    for trigger in trigger_list:
        if exit_fired and trigger.category == "emergency_exit":
            # Check cooldown
            last_exit = self._last_emergency_exit_bar.get(trigger.symbol, -999)
            if self._bar_counter - last_exit < self.emergency_exit_cooldown_bars:
                self._record_block(..., "EMERGENCY_COOLDOWN", ...)
                continue
```

### Fix 5: Require Higher Confidence for Exit Override

**File**: `agents/strategies/trigger_engine.py:68`

```python
def __init__(self, ...,
             confidence_override_threshold: Literal["A", "B", "C"] | None = "A",
             emergency_exit_min_bars: int = 2,  # NEW
             ...):
    self.emergency_exit_min_bars = emergency_exit_min_bars
```

---

## Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `agents/strategies/trigger_engine.py:303-385` | Block emergency exit on same bar as entry | P0 |
| `agents/strategies/trigger_engine.py:348` | Add separate min_hold for emergency exits | P0 |
| `backtesting/llm_strategist_runner.py:351` | Synchronize with TriggerEngine min_hold | P1 |
| `prompts/llm_strategist_prompt.txt` | Instruct stricter emergency exit rules | P1 |
| `agents/strategies/trigger_engine.py` | Add emergency exit cooldown | P2 |

---

## Validation Criteria

After fix, verify:
- [ ] Emergency exits cannot fire on same bar as entry
- [ ] Emergency exit rate < 20% of total exits
- [ ] Min hold period is consistent between systems
- [ ] Positions have minimum duration before any exit
