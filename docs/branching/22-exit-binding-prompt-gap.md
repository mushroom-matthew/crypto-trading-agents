# Branch: exit-binding-prompt-gap

## Purpose
Teach the LLM strategist that exit rules only close positions opened by the **same trigger category**. Currently the LLM generates cross-category exit triggers (e.g., `volatility_breakout` exit trying to close a `trend_continuation` entry), which are silently blocked by the exit binding mismatch guard, producing dead triggers.

## Source Evidence
- Backtest `ebf53879`: 11 `exit_binding_mismatch` blocks, all from `btc_volatility_breakout_8h` attempting to exit positions opened by `btc_trend_continuation_1h`
- `btc_volatility_breakout_8h` fired 8 times, executed 0 times (100% block rate)
- The LLM generated this trigger in all 21 plans, never understanding it could never fire as an exit

## Root Cause
The trigger engine enforces **exit binding** — an exit rule can only close a position opened by a trigger with the same `category`. This is documented in the code (`trigger_engine.py:716`) but **not communicated to the LLM** in any prompt or schema document.

The LLM sees two BTC long triggers and assumes either can close a BTC position. It has no way to know that `volatility_breakout` category exits cannot close `trend_continuation` category entries.

## Scope
1. Add **exit binding documentation** to `strategy_plan_schema.txt`
2. Add a **compile-time warning** when multiple entry triggers for the same symbol have different categories (cross-category exit risk)
3. Consider adding a **`global_exit`** category or flag that bypasses exit binding (distinct from `emergency_exit`)

## Out of Scope
- Changing the exit binding enforcement itself (it's a valid safety mechanism)
- Emergency exit behavior (already has binding bypass)

## Key Files
- `prompts/strategy_plan_schema.txt` — Add exit binding rules
- `prompts/llm_strategist_simple.txt` — Add exit binding summary
- `trading_core/trigger_compiler.py` — Add cross-category warning
- `agents/strategies/trigger_engine.py` — Reference only (exit binding logic)

## Implementation Steps

### Step 1: Document exit binding in schema
Add to `strategy_plan_schema.txt`:
```
EXIT BINDING RULE (Critical):
- Exit rules ONLY close positions opened by a trigger with the SAME category.
- A "volatility_breakout" exit CANNOT close a position opened by "trend_continuation".
- Only "emergency_exit" and "risk_off" categories bypass this restriction.
- If you have multiple entry categories for the same symbol, each MUST have its own matching exit trigger.
- Do NOT create entry triggers in one category expecting another category's exit to close them.

Example of WRONG plan:
  Entry: btc_trend_long (category=trend_continuation)
  Exit:  btc_breakout_exit (category=volatility_breakout)  ← WILL NEVER CLOSE the trend entry

Example of CORRECT plan:
  Entry: btc_trend_long (category=trend_continuation)
  Exit:  btc_trend_long exit_rule field handles the exit  ← Same category, works correctly
```

### Step 2: Compile-time cross-category warning
In `trigger_compiler.py`, after compiling all triggers for a plan:
- Group entry triggers by symbol
- If a symbol has entries in multiple categories, emit a warning:
  `"Symbol BTC-USD has entries in categories [trend_continuation, volatility_breakout] — each needs category-matched exit rules"`

### Step 3: Consider global_exit category
Evaluate whether a `global_exit` trigger category (distinct from `emergency_exit`) should be added that can close any position regardless of entry category. This would give the LLM a legitimate way to create universal exit rules without abusing `emergency_exit`.

## Test Plan
```bash
# Unit: exit binding documentation present
python3 -c "
text = open('prompts/strategy_plan_schema.txt').read()
assert 'EXIT BINDING' in text or 'exit binding' in text
print('PASS: schema documents exit binding')
"

# Unit: cross-category warning
uv run pytest tests/test_trigger_compiler.py -k cross_category -vv

# Integration: backtest should show 0 exit_binding_mismatch blocks
```

## Test Evidence
*(to be filled after implementation)*

## Acceptance Criteria
- [ ] `strategy_plan_schema.txt` explains exit binding rule with examples
- [ ] Compile-time warning emitted for cross-category entry triggers on same symbol
- [ ] Backtest shows 0 `exit_binding_mismatch` blocks
- [ ] LLM plans correctly pair entry/exit within same category

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-10 | Runbook created from backtest ebf53879 analysis | Claude |

## Git Workflow
```bash
git checkout -b fix/exit-binding-prompt-gap
# ... implement changes ...
git add prompts/strategy_plan_schema.txt prompts/llm_strategist_simple.txt trading_core/trigger_compiler.py
git commit -m "Document exit binding rules in LLM prompts and add cross-category warning"
```
