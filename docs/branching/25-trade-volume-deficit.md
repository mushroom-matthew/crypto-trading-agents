# Branch: trade-volume-deficit

## Purpose
Address the systemic under-trading problem: 6 entries across 14 days (0.43/day) on a 2-symbol, 1h-bar backtest. The system is 80-90% idle. This runbook tackles the root causes that prevent trade execution rather than symptoms.

## Source Evidence
- Backtest `ebf53879`: 6 entries, 5 exits, 11 total fills in 14 days
- 8 of 14 days had zero trades
- Longest drought: Jan 10-13 (4 consecutive days, zero fills)
- Only 2 of 7 entry triggers ever executed a trade
- `btc_volatility_breakout_8h`: 8 fires, 0 executions (100% blocked)
- `eth_reversal_short_4h`: 0 fires in 54 evaluations
- `btc_risk_reduce_on_weakness`: 0 fires in 204 evaluations
- `eth_risk_off_on_regime_shift`: 0 fires in 210 evaluations

## Root Causes (Multi-Factor)

### 1. Emergency exit whipsaw cycle
Enter → emergency exit 1-3h later → go flat → wait for conditions to re-align → re-enter → repeat.
**Addressed by**: Runbook 21 (emergency exit sensitivity)

### 2. `is_flat` hard gate blocks all entries while positioned
Every entry rule starts with `is_flat`. While holding a BTC position (73h, 29h, etc.), no additional entries are possible on any symbol.
**Fix needed here**: Allow conditioned re-entry or scale-in

### 3. ETH VWAP threshold too restrictive
`vwap_distance_pct < -1.0` fires only 2 of 209 evaluations (0.96%). ETH effectively had 1 trade in 14 days.
**Fix needed here**: Prompt guidance on threshold calibration

### 4. Dead triggers consuming plan slots
3 triggers never fire (0 fires across 600+ evaluations combined). They waste plan complexity without contributing trades.
**Fix needed here**: Dead trigger detection and reporting

### 5. Hold rule → emergency exit cycle
Hold rule blocks normal exits → position held longer → emergency exit eventually fires → goes flat → `is_flat` blocks entries → long drought.
**Addressed by**: Runbook 23 (hold rule calibration)

## Scope
1. **Dead trigger detection** — Identify triggers with 0 fires after N evaluations and report to judge
2. **Entry threshold calibration guidance** — Prompt the LLM about expected fire rates
3. **`is_flat` relaxation guidance** — Teach the LLM about `not is_long` vs `is_flat` for adding to positions
4. **Trade frequency telemetry** — Add trades/day metric to judge snapshot so the judge can react to drought

## Out of Scope
- Emergency exit sensitivity (Runbook 21)
- Hold rule calibration (Runbook 23)
- Exit binding (Runbook 22)

## Key Files
- `prompts/strategy_plan_schema.txt` — Add fire rate guidance, `is_flat` vs `not is_long` docs
- `prompts/llm_strategist_simple.txt` — Add trade frequency expectations
- `backtesting/llm_strategist_runner.py` — Add dead trigger detection, trades/day to judge snapshot
- `agents/strategies/trigger_engine.py` — Track per-trigger fire rate

## Implementation Steps

### Step 1: Add fire rate guidance to prompts
Add to `strategy_plan_schema.txt`:
```
TRIGGER FIRE RATE GUIDANCE:
- Each entry trigger should fire at least once per day on average. If a trigger
  evaluates 100+ times without firing, it is too restrictive and wastes a plan slot.
- Calibrate thresholds to expected market conditions:
  - vwap_distance_pct < -1.0 is very restrictive (fires <1% of bars)
  - vwap_distance_pct < -0.5 is moderate (~5% of bars)
  - rsi_14 < 40 is restrictive in bull/range markets (fires <10%)
  - rsi_14 < 50 is moderate (~40-50% of bars)
- Use `is_flat` for fresh entries. Use `not is_long` or `not is_short` to allow
  adding to existing positions or entering after partial exits.
```

### Step 2: Dead trigger detection
After each plan period, compute fire rate per trigger:
```python
for trigger_id, stats in eval_stats.items():
    fire_rate = stats["fired"] / max(1, stats["evaluated"])
    if stats["evaluated"] >= 48 and fire_rate == 0:  # 48 bars = ~2 days on 1h
        dead_triggers.append(trigger_id)
```
Report `dead_triggers` list in the judge snapshot.

### Step 3: Trades/day metric in judge snapshot
Add to the judge compact summary:
```python
"trades_per_day": total_trades / max(1, days_elapsed),
"days_without_trades": count_of_zero_trade_days,
```
This gives the judge direct visibility into drought periods.

### Step 4: Judge constraint for minimum activity
When `trades_per_day < 0.5` for 3+ consecutive days, the judge should be guided to:
- Suggest loosening entry thresholds
- Recommend removing dead triggers and replacing with more realistic ones
- Flag `is_flat` gates that may be over-constraining

## Test Plan
```bash
# Unit: dead trigger detection
uv run pytest tests/test_trigger_engine.py -k dead_trigger -vv

# Unit: trades_per_day in snapshot
uv run pytest tests/test_llm_strategist_runner.py -k trades_per_day -vv

# Integration: backtest should show >1 trade/day average
# (improved from 0.43/day after combined fixes with runbooks 21-23)
```

## Test Evidence
```
tests/test_trigger_engine.py::test_per_trigger_fire_rate PASSED
```
Per-trigger fire rate tracking works: `_trigger_eval_counts` and `_trigger_fire_counts` correctly track evaluations and fires per trigger ID.

Dead trigger detection implemented in `llm_strategist_runner.py` snapshot builder — triggers with 0 fires after 48+ evaluations are surfaced as `dead_triggers` list. `trades_per_day` metric added to compact judge summary.

Prompt verification: `strategy_plan_schema.txt` contains "TRIGGER FIRE RATE CALIBRATION" section with threshold examples. `llm_strategist_simple.txt` Rule 9 documents fire rate targets.

## Acceptance Criteria
- [x] Prompt includes fire rate guidance with concrete threshold examples
- [x] Dead triggers (0 fires after 48+ evaluations) detected and reported to judge
- [x] `trades_per_day` and `days_without_trades` in judge snapshot
- [ ] Backtest shows improved trade frequency (target: >1 entry/day average) — *requires validation backtest*

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-10 | Runbook created from backtest ebf53879 analysis | Claude |
| 2026-02-11 | Implemented: fire rate guidance in schema + simple prompt, per-trigger eval/fire counting in trigger_engine.py, dead trigger detection + trades_per_day in judge snapshot | Claude |

## Git Workflow
```bash
git checkout -b fix/trade-volume-deficit
# ... implement changes ...
git add prompts/strategy_plan_schema.txt prompts/llm_strategist_simple.txt backtesting/llm_strategist_runner.py
git commit -m "Address trade volume deficit: fire rate guidance, dead trigger detection, drought telemetry"
```
