# Branch: level-anchored-stops

## Purpose
The current system expresses stop loss as `stop_loss_pct` — a percentage of entry price, re-evaluated each bar. This is how automated systems typically handle stops, but it is not how human traders set them. Human traders set stops at specific price levels: below the manipulation candle's low, below the prior session's low, below a Fibonacci level, below a swing point. The key property: **the stop level is fixed at entry and doesn't move unless intentionally trailed.**

This runbook adds `stop_price_abs` and `target_price_abs` to `TradeLeg`: absolute prices computed at entry using available structural levels and stored for the lifetime of the trade. The trigger engine exposes `below_stop` and `above_target` as built-in identifiers that the LLM can reference in exit rules. This enables strategy templates (especially Runbook 40 and 41) to express "stop below yesterday's low" as a concrete, persistent reference rather than a re-computed percentage.

This runbook depends on:
- **Runbook 41** (HTF structure cascade) — provides `htf_daily_low` as a stop anchor source
- **Runbook 38** (candlestick patterns) — provides candle body/wick information for stop placement

## Scope
1. **`schemas/trade_set.py`** — add `stop_price_abs`, `target_price_abs`, `stop_anchor_type` to `TradeLeg`
2. **`schemas/llm_strategist.py`** — add `stop_anchor_type` to `TriggerCondition`
3. **`agents/strategies/trigger_engine.py`** — expose `below_stop` and `above_target` in identifier namespace
4. **`backtesting/simulator.py`** — resolve stop/target levels at fill time and write to `TradeLeg`
5. **`prompts/strategy_plan_schema.txt`** — document `stop_anchor_type` field and `below_stop`/`above_target` usage
6. **`tests/test_level_anchored_stops.py`** — new test file
7. **UI display** (minor): show `stop_price_abs` and `target_price_abs` in trade detail views

## Out of Scope
- Trailing stop logic (this handles fixed-level stops; trailing is a separate concern)
- Smart order routing (stop is tracked in-software, not as an exchange native stop order)
- Multi-leg scale-in stop adjustment (deferred; first entry's stop is authoritative)

## Key Files
- `schemas/trade_set.py`
- `schemas/llm_strategist.py`
- `agents/strategies/trigger_engine.py`
- `backtesting/simulator.py`
- `prompts/strategy_plan_schema.txt`
- `tests/test_level_anchored_stops.py` (new)

## Implementation Steps

### Step 1: Extend `TradeLeg` in `schemas/trade_set.py`

Add optional fields (all `None` for backward compatibility):
```python
stop_price_abs: Optional[float] = Field(
    default=None,
    description="Absolute stop price set at entry (below = exit for longs, above = exit for shorts)",
)
target_price_abs: Optional[float] = Field(
    default=None,
    description="Absolute profit target price set at entry",
)
stop_anchor_type: Optional[str] = Field(
    default=None,
    description="How the stop was computed: 'pct', 'atr', 'htf_daily_low', 'htf_prev_daily_low', "
                "'donchian_lower', 'fib_level', 'candle_low', 'manual'",
)
target_anchor_type: Optional[str] = Field(
    default=None,
    description="How the target was computed: 'measured_move', 'htf_daily_high', 'htf_5d_high', "
                "'fib_level', 'r_multiple', 'manual'",
)
```

### Step 2: Add `stop_anchor_type` to `TriggerCondition` in `schemas/llm_strategist.py`

```python
stop_anchor_type: Optional[str] = Field(
    default=None,
    description="How to compute the stop price at entry. Options: "
                "'pct' (use stop_loss_pct, default), "
                "'atr' (1.5 * ATR below entry), "
                "'htf_daily_low' (below prior session's low), "
                "'htf_prev_daily_low' (below session before prior), "
                "'donchian_lower' (below Donchian lower at time of entry), "
                "'fib_618' (at the 618 retracement level), "
                "'candle_low' (below the trigger bar's low). "
                "Null/absent defaults to 'pct' behavior.",
)
target_anchor_type: Optional[str] = Field(
    default=None,
    description="How to compute the profit target at entry. Options: "
                "'measured_move' (range height projected from entry), "
                "'htf_daily_high' (prior session high), "
                "'htf_5d_high' (5-day rolling high), "
                "'r_multiple_2' (2R from entry), "
                "'r_multiple_3' (3R from entry), "
                "'fib_618_above' (618 extension above entry). "
                "Null/absent means no stored target; exits via exit_rule only.",
)
```

### Step 3: Resolve stop/target at fill time in `backtesting/simulator.py`

When a fill occurs (entry leg), resolve the stop and target prices from the trigger's anchor specification and the current indicator snapshot:

```python
def _resolve_stop_price(
    trigger: TriggerCondition,
    fill_price: float,
    snapshot: IndicatorSnapshot,
    direction: str,  # "long" or "short"
) -> tuple[float | None, str | None]:
    """Compute absolute stop price from anchor type and current snapshot."""
    anchor = trigger.stop_anchor_type or "pct"

    if anchor == "pct" and trigger.stop_loss_pct:
        pct = trigger.stop_loss_pct / 100.0
        price = fill_price * (1 - pct) if direction == "long" else fill_price * (1 + pct)
        return price, "pct"

    elif anchor == "atr" and snapshot.atr_14:
        mult = 1.5  # configurable
        offset = snapshot.atr_14 * mult
        price = fill_price - offset if direction == "long" else fill_price + offset
        return price, "atr"

    elif anchor == "htf_daily_low" and snapshot.htf_daily_low:
        price = snapshot.htf_daily_low * 0.995  # 0.5% buffer below daily low
        return price, "htf_daily_low"

    elif anchor == "htf_prev_daily_low" and snapshot.htf_prev_daily_low:
        price = snapshot.htf_prev_daily_low * 0.995
        return price, "htf_prev_daily_low"

    elif anchor == "donchian_lower" and snapshot.donchian_lower_short:
        price = snapshot.donchian_lower_short * 0.995
        return price, "donchian_lower"

    elif anchor == "fib_618" and snapshot.fib_618:
        price = snapshot.fib_618  # Already at the level
        return price, "fib_618"

    elif anchor == "candle_low" and snapshot.low:
        price = snapshot.low * 0.998  # Slight buffer below trigger bar's low
        return price, "candle_low"

    # Fallback: no stop resolved
    return None, None


def _resolve_target_price(
    trigger: TriggerCondition,
    fill_price: float,
    stop_price: float | None,
    snapshot: IndicatorSnapshot,
    direction: str,
) -> tuple[float | None, str | None]:
    """Compute absolute target price from anchor type."""
    anchor = trigger.target_anchor_type
    if not anchor:
        return None, None

    if anchor == "htf_daily_high" and snapshot.htf_daily_high:
        price = snapshot.htf_daily_high * 0.998
        return price, "htf_daily_high"

    elif anchor == "htf_5d_high" and snapshot.htf_5d_high:
        price = snapshot.htf_5d_high * 0.998
        return price, "htf_5d_high"

    elif anchor == "measured_move" and snapshot.donchian_upper_short and snapshot.donchian_lower_short:
        range_height = snapshot.donchian_upper_short - snapshot.donchian_lower_short
        price = fill_price + range_height if direction == "long" else fill_price - range_height
        return price, "measured_move"

    elif anchor == "r_multiple_2" and stop_price:
        risk = abs(fill_price - stop_price)
        price = fill_price + 2 * risk if direction == "long" else fill_price - 2 * risk
        return price, "r_multiple_2"

    elif anchor == "r_multiple_3" and stop_price:
        risk = abs(fill_price - stop_price)
        price = fill_price + 3 * risk if direction == "long" else fill_price - 3 * risk
        return price, "r_multiple_3"

    return None, None
```

After fill, write resolved levels to the `TradeLeg`:
```python
leg.stop_price_abs, leg.stop_anchor_type = _resolve_stop_price(trigger, fill_price, snapshot, direction)
leg.target_price_abs, leg.target_anchor_type = _resolve_target_price(trigger, fill_price, leg.stop_price_abs, snapshot, direction)
```

### Step 4: Expose `below_stop` and `above_target` in `agents/strategies/trigger_engine.py`

In the identifier namespace builder, check if there's an active entry leg with stop/target:
```python
# In the namespace dict used to evaluate trigger expressions:
active_stop = self._active_trade_leg.stop_price_abs if self._active_trade_leg else None
active_target = self._active_trade_leg.target_price_abs if self._active_trade_leg else None
current_close = snapshot.close

namespace["below_stop"] = (current_close < active_stop) if (active_stop and not is_flat) else False
namespace["above_target"] = (current_close > active_target) if (active_target and not is_flat) else False
namespace["stop_price"] = active_stop or 0.0
namespace["target_price"] = active_target or 0.0
namespace["stop_distance_pct"] = (
    abs(current_close - active_stop) / current_close * 100
    if active_stop else 0.0
)
namespace["target_distance_pct"] = (
    abs(active_target - current_close) / current_close * 100
    if active_target else 0.0
)
```

### Step 5: Update `prompts/strategy_plan_schema.txt`

Add:
```
LEVEL-ANCHORED STOP/TARGET FIELDS:
  stop_anchor_type (on TriggerCondition):
    Controls how the stop price is computed when the trigger fires.
    Values: 'pct' | 'atr' | 'htf_daily_low' | 'htf_prev_daily_low' |
            'donchian_lower' | 'fib_618' | 'candle_low'
    Default (null): uses stop_loss_pct percentage calculation.
    PREFERRED for structure-based strategies: 'htf_daily_low', 'donchian_lower'.

  target_anchor_type (on TriggerCondition):
    Controls how the profit target is computed at fill time.
    Values: 'measured_move' | 'htf_daily_high' | 'htf_5d_high' |
            'r_multiple_2' | 'r_multiple_3'
    Default (null): no stored target, exits via exit_rule only.

BUILT-IN IDENTIFIERS (available after entry, evaluate to True/False):
  below_stop      — True when current close is below the stored stop_price_abs
  above_target    — True when current close is above the stored target_price_abs
  stop_price      — The resolved stop price (float, 0.0 if no stop set)
  target_price    — The resolved target price (float, 0.0 if no target set)
  stop_distance_pct   — % distance between close and stop
  target_distance_pct — % distance between close and target

USAGE IN EXIT RULES:
  Recommended exit pattern for structure-based entries:
    exit_rule: "not is_flat and (below_stop or above_target)"

  Combined with rule-based exits:
    exit_rule: "not is_flat and (below_stop or above_target or (rsi_14 > 75 and macd_hist < 0))"

  Approach-to-target scale-out (risk_reduce):
    exit_rule: "not is_flat and target_distance_pct < 0.3"  (within 0.3% of target → take profit)
    exit_fraction: 0.5

STOP ANCHORING PRIORITY (use the tightest that still makes structural sense):
  1. candle_low     — tightest (below trigger bar's low); best for high-conviction setups
  2. donchian_lower — below compression range low; canonical for breakout setups
  3. htf_daily_low  — below prior session low; gives more room but structurally important
  4. atr            — 1.5 ATR below entry; volatility-normalized fallback
  5. pct            — last resort; arbitrary, not structure-based
```

## Test Plan
```bash
# Unit: stop/target resolution with known snapshots
uv run pytest tests/test_level_anchored_stops.py -vv

# Unit: below_stop and above_target evaluate correctly
uv run pytest tests/test_trigger_engine.py -k "level_anchored" -vv

# Integration: TradeLeg serializes with new fields
uv run pytest tests/test_trade_set.py -vv

# Pydantic: extra="forbid" satisfied (no unknown fields)
uv run pytest -k "trade_set" -vv
```

## Test Evidence
```
# Initial implementation (Runbook 42):
uv run pytest tests/test_level_anchored_stops.py tests/test_trigger_engine.py tests/test_trade_set.py -vv
90 passed in 9.82s

Full suite: 681 passed, 3 failed (pre-existing isolation failures), 1 skipped
New tests added: 33 (test_level_anchored_stops.py)

# Hotfix: direction-aware anchors + stop_hit/target_hit (fix/r42-direction-aware):
uv run pytest tests/test_level_anchored_stops.py -vv
64 passed in 14.79s

Full suite: 713 passed, 2 failed (pre-existing isolation failures), 1 skipped
New tests added: 31 additional (64 total in test_level_anchored_stops.py)
```

## Acceptance Criteria
- [ ] `TradeLeg` includes `stop_price_abs`, `target_price_abs`, `stop_anchor_type`, `target_anchor_type` (all optional)
- [ ] `TriggerCondition` includes `stop_anchor_type` and `target_anchor_type` (optional)
- [ ] `_resolve_stop_price()` handles all 7 anchor types correctly
- [ ] `_resolve_target_price()` handles all 6 target anchor types correctly
- [ ] Trigger engine exposes `below_stop`, `above_target`, `stop_price`, `target_price`, `stop_distance_pct`, `target_distance_pct` in expression namespace
- [ ] Existing tests not broken (all new fields are optional with `None` defaults)
- [ ] Backtest fills include `stop_price_abs` in trade log output

## Human Verification Evidence
```
All 7 stop anchor types and 5 target anchor types covered by unit tests with hand-crafted
snapshots. Trigger engine context keys (below_stop, above_target, stop_price, target_price,
stop_distance_pct, target_distance_pct) verified via unit tests that simulate the context
dict construction logic. TradeLeg serialization round-trip tested with all new fields.
TriggerCondition schema accepts stop_anchor_type and target_anchor_type without validation errors.
```

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-18 | Runbook created from product strategy audit | Claude |
| 2026-02-18 | Implementation: schemas/trade_set.py — added stop_price_abs, target_price_abs, stop_anchor_type, target_anchor_type to TradeLeg | Claude |
| 2026-02-18 | Implementation: schemas/llm_strategist.py — added stop_anchor_type, target_anchor_type to TriggerCondition | Claude |
| 2026-02-18 | Implementation: backtesting/llm_strategist_runner.py — added _resolve_stop_price_anchored(), _resolve_target_price_anchored(); wired into _update_position_risk_state() | Claude |
| 2026-02-18 | Implementation: agents/strategies/trigger_engine.py — added below_stop, above_target, stop_price, target_price, stop_distance_pct, target_distance_pct to _context() namespace | Claude |
| 2026-02-18 | Implementation: prompts/strategy_plan_schema.txt — added LEVEL-ANCHORED STOP/TARGET FIELDS section | Claude |
| 2026-02-18 | Tests: tests/test_level_anchored_stops.py — 33 new unit tests | Claude |
| 2026-02-18 | Hotfix (fix/r42-direction-aware): _resolve_stop_price_anchored() made direction-aware — long-only anchors return None for shorts; added short-only anchors (htf_daily_high, htf_prev_daily_high, donchian_upper, candle_high); added family anchors (htf_daily_extreme, htf_prev_daily_extreme, donchian_extreme, candle_extreme); ATR mult now configurable via stop_loss_atr_mult | Claude |
| 2026-02-18 | Hotfix: _resolve_target_price_anchored() made direction-aware — htf_daily_high/htf_5d_high long-only; added short targets (htf_daily_low, htf_5d_low); added family targets (htf_daily_extreme, htf_5d_extreme) | Claude |
| 2026-02-18 | Hotfix: trigger_engine._context() — replaced below_stop/above_target with direction-aware stop_hit/target_hit; kept old names as backward-compat aliases | Claude |
| 2026-02-18 | Hotfix: schemas/llm_strategist.py — added stop_loss_atr_mult to TriggerCondition | Claude |
| 2026-02-18 | Hotfix: prompts/strategy_plan_schema.txt — documented all new anchors, stop_hit/target_hit, stop_loss_atr_mult, precedence rules | Claude |
| 2026-02-18 | Hotfix: tests/test_level_anchored_stops.py — 31 new tests (64 total) covering direction-aware behavior | Claude |

## Worktree Setup
```bash
git fetch
git worktree add -b feat/level-anchored-stops ../wt-level-anchored-stops main
cd ../wt-level-anchored-stops

# Depends on:
# - Runbook 41 (HTF structure cascade) merged — provides htf_daily_low etc.
# - Runbook 38 (candlestick patterns) merged — provides candle_low (snapshot.low)

# When finished (after merge)
git worktree remove ../wt-level-anchored-stops
```

## Git Workflow
```bash
git checkout main
git pull
git checkout -b feat/level-anchored-stops

# ... implement changes ...

git add schemas/trade_set.py \
  schemas/llm_strategist.py \
  agents/strategies/trigger_engine.py \
  backtesting/simulator.py \
  prompts/strategy_plan_schema.txt \
  tests/test_level_anchored_stops.py

uv run pytest tests/test_level_anchored_stops.py \
             tests/test_trigger_engine.py \
             tests/test_trade_set.py -vv
git commit -m "Add level-anchored stops: stop/target resolved at entry from structural levels"
```
