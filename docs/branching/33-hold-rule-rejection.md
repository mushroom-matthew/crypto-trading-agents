# Branch: hold-rule-rejection

## Purpose
Runbook 23 added detection and warnings for degenerate hold rules. Backtest `58cb897f` proves this is **insufficient**: `HOLD_RULE` blocks still occurred (2 blocks on Jan 7 and Jan 9), specifically from `btc_trend_continuation_1h` with hold rule `tf_4h_close > tf_4h_sma_medium and rsi_14 > 45`. The `rsi_14 > 45` condition is true ~80% of the time, effectively preventing exits. Detection isn't enough -- we need **rejection** of degenerate hold rules at compile time.

## Source Evidence
- Backtest `58cb897f`: 2 `HOLD_RULE` blocks in DB
- Hold rule `rsi_14 > 45` blocks exits on Jan 7 and Jan 9
- All exits for this trigger forced through emergency_exit instead of normal exit rules
- `detect_degenerate_hold_rules()` (Runbook 23) correctly flags these but doesn't prevent them

## Root Cause
The v1 implementation detects degenerate hold rules and emits warnings, but the compiled plan still includes them. The trigger engine suppresses exits when the hold rule is true, so a near-always-true hold rule effectively disables normal exits. We need to either:
1. Strip degenerate hold rules at compile time, OR
2. Replace them with a sensible default

## Scope
1. **Strip degenerate hold rules** in `trigger_compiler.py`: when `detect_degenerate_hold_rules()` flags a rule, remove it from the compiled trigger and emit a structured event
2. **Add hold rule quality scoring** that rejects single-condition rules with known low-discriminating patterns
3. **Add prompt constraint** telling the LLM that hold rules with fire rate >60% will be stripped

## Out of Scope
- Changing how the trigger engine evaluates hold rules when they're valid
- Emergency exit behavior

## Key Files
- `trading_core/trigger_compiler.py` — Upgrade `detect_degenerate_hold_rules()` to strip bad rules
- `agents/strategies/trigger_engine.py` — Reference for hold rule evaluation
- `prompts/strategy_plan_schema.txt` — Add hold rule quality requirement
- `agents/event_emitter.py` — Emit hold_rule_stripped events

## Implementation Steps

### Step 1: Define degenerate patterns
In `trigger_compiler.py`, define known low-discriminating conditions:
- Single-condition rules with `rsi_14 > X` where X < 50 (true ~70-90% of the time)
- Single-condition rules with `close > sma_X` (true ~50% of the time, too broad)
- Any single-condition rule (compound rules are generally safer)

### Step 2: Strip degenerate hold rules at compile time
When `detect_degenerate_hold_rules()` identifies a degenerate rule:
1. Set the trigger's compiled hold rule to `None` (effectively disabling the hold)
2. Emit event: `hold_rule_stripped` with trigger_id, original rule text, reason
3. The trigger then relies only on its normal exit_rule, which is the safer default

### Step 3: Update prompt schema
Add to `strategy_plan_schema.txt`:
```
HOLD RULE QUALITY (Important):
- Hold rules that are true >60% of the time will be automatically stripped.
- Single-condition hold rules (e.g., "rsi_14 > 45") are almost always degenerate.
- Good hold rules combine multiple conditions: "rsi_14 > 60 and adx > 25 and close > ema_20"
- If unsure, omit the hold rule entirely -- the exit_rule alone is sufficient.
```

## Test Plan
```bash
# Unit: degenerate hold rule stripped
uv run pytest tests/test_trigger_compiler.py -k hold_rule_strip -vv

# Unit: compound hold rules preserved
uv run pytest tests/test_trigger_compiler.py -k hold_rule_compound_preserved -vv

# Integration: backtest should show 0 HOLD_RULE blocks
```

## Test Evidence
```
(to be filled after implementation)
```

## Acceptance Criteria
- [ ] Degenerate single-condition hold rules are stripped at compile time
- [ ] Compound hold rules are preserved (no false positives)
- [ ] Event emitted for every stripped hold rule
- [ ] Prompt updated with hold rule quality guidance
- [ ] Backtest shows 0 `HOLD_RULE` blocks
- [ ] Existing tests still pass

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-12 | Runbook created from backtest 58cb897f validation analysis | Claude |

## Git Workflow
```bash
git checkout -b fix/hold-rule-rejection
# ... implement changes ...
git add trading_core/trigger_compiler.py prompts/strategy_plan_schema.txt agents/event_emitter.py tests/test_trigger_compiler.py
git commit -m "Strip degenerate hold rules at compile time to prevent exit suppression"
```
