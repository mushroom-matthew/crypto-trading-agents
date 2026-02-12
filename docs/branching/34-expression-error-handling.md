# Branch: expression-error-handling

## Purpose
LLM-generated trigger expressions sometimes contain identifier typos or invalid variable names (e.g., `realization_vol_short` instead of `realized_vol_short`). These fail silently during compilation and the trigger never fires, contributing to the trade volume deficit. We need pre-compile validation that either auto-corrects common typos or rejects triggers with invalid identifiers, emitting explicit error events.

## Source Evidence
- Backtest `58cb897f`: early judge feedback flagged `expression_error` for `eth_reversal_fib618_guarded`
- The trigger was either replaced later in the plan cache or silently dropped
- No explicit error event was emitted, making debugging difficult
- This is a contributing factor to the 0.29 trades/day rate (triggers that should fire don't)

## Root Cause
The rule DSL compiler (`rule_dsl.py` / `trigger_compiler.py`) compiles expressions into Python but does not validate that identifiers in the expression exist in the available indicator set. Invalid identifiers cause runtime `NameError` which is caught and silently returns False (trigger doesn't fire).

## Scope
1. **Pre-compile identifier validation** in `trigger_compiler.py`: extract all identifiers from the expression and check against the available indicator set
2. **Auto-correct common typos**: maintain a mapping of known typo → correct identifier (e.g., `realization_vol` → `realized_vol`)
3. **Emit structured events** for corrections and rejections

## Out of Scope
- Changing the rule DSL syntax itself
- Modifying how trigger_engine evaluates compiled expressions at runtime

## Key Files
- `trading_core/trigger_compiler.py` — Add identifier validation
- `trading_core/rule_registry.py` — Source of valid identifiers/indicators
- `agents/strategies/rule_dsl.py` — Reference for expression compilation
- `agents/event_emitter.py` — Emit validation events

## Implementation Steps

### Step 1: Extract identifiers from expressions
After compiling a trigger expression, extract all variable names (AST walk or regex on the compiled expression). Compare against the known indicator set from `rule_registry.py`.

### Step 2: Build auto-correct map
Create `KNOWN_TYPOS` dict in `trigger_compiler.py`:
```python
KNOWN_TYPOS = {
    "realization_vol_short": "realized_vol_short",
    "realization_vol_long": "realized_vol_long",
    "bollinger_upper": "bb_upper",
    "bollinger_lower": "bb_lower",
    # Add as discovered
}
```
When a typo is detected, replace it in the expression and emit `expression_auto_corrected` event.

### Step 3: Reject unresolvable identifiers
If an identifier is not in the valid set and not in KNOWN_TYPOS:
- Strip the trigger from the compiled plan
- Emit `expression_error_rejected` event with the invalid identifier and trigger_id

### Step 4: Surface in daily report
Add `expression_corrections` and `expression_rejections` counts to daily summary.

## Test Plan
```bash
# Unit: auto-correct known typo
uv run pytest tests/test_trigger_compiler.py -k expression_auto_correct -vv

# Unit: reject unknown identifier
uv run pytest tests/test_trigger_compiler.py -k expression_reject_unknown -vv

# Unit: valid expression passes through unchanged
uv run pytest tests/test_trigger_compiler.py -k expression_valid_passthrough -vv
```

## Test Evidence
```
(to be filled after implementation)
```

## Acceptance Criteria
- [ ] Known typos auto-corrected at compile time
- [ ] Unknown identifiers cause trigger rejection with event
- [ ] Valid expressions pass through unchanged
- [ ] Daily report surfaces correction/rejection counts
- [ ] Existing tests still pass

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-12 | Runbook created from backtest 58cb897f validation analysis | Claude |

## Git Workflow
```bash
git checkout -b fix/expression-error-handling
# ... implement changes ...
git add trading_core/trigger_compiler.py agents/event_emitter.py tests/test_trigger_compiler.py
git commit -m "Add pre-compile identifier validation with auto-correct for trigger expressions"
```
