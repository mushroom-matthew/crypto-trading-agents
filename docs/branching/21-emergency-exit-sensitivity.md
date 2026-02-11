# Branch: emergency-exit-sensitivity

## Purpose
Fix the overly sensitive `global_emergency_exit_vol_spike` trigger that dominates all exits (4 of 5 closed trades in backtest ebf53879) and causes premature position closure within 1–3 hours of entry.

## Source Evidence
- Backtest `ebf53879`: 4/5 exits via `global_emergency_exit_vol_spike_exit`, 0/5 via planned exit rules
- Trade 4: entered Jan 6 23:00, emergency-exited Jan 7 00:00 (1h hold, -1.63R)
- Trade 5: entered Jan 9 15:00, emergency-exited Jan 9 18:00 (3h hold, -0.60R)
- Only Trade 1 (ETH mean reversion) exited via its own exit rule

## Root Cause
The emergency exit rule contains the condition `tf_1d_atr_14 > tf_4h_atr_14`, which is **structurally always true** — daily ATR is inherently larger than 4h ATR due to the longer timeframe. This reduces the effective emergency exit to: `not is_flat and (vol_state == 'extreme') and (close < sma_short)`, which fires on any minor dip below the short SMA during volatile periods.

The LLM generates this condition repeatedly across all 21 plans because it appears logically correct (daily volatility exceeding 4h volatility = bad), but it's a tautology.

## Scope
1. Add a **prompt guardrail** warning the LLM about structurally tautological cross-timeframe ATR comparisons
2. Add a **compile-time validator** in `trigger_compiler.py` that flags always-true ATR comparisons across timeframes
3. Consider adding a **minimum hold time** before emergency exits can fire (e.g., 2 bars minimum)

## Out of Scope
- Changing the emergency exit bypass/override system (already addressed in runbooks 04-06)
- Modifying the min-hold enforcement mechanism (already addressed in runbook 15)

## Key Files
- `prompts/strategy_plan_schema.txt` — Add warning about cross-timeframe ATR tautologies
- `prompts/llm_strategist_simple.txt` — Add emergency exit guidance
- `trading_core/trigger_compiler.py` — Add tautology detection for `tf_Xh_atr > tf_Yh_atr` where X > Y
- `agents/strategies/trigger_engine.py` — Consider per-trigger min-hold before emergency exit fires

## Implementation Steps

### Step 1: Prompt guardrail
Add to `strategy_plan_schema.txt`:
```
WARNING - Cross-timeframe ATR comparisons:
- tf_1d_atr_14 > tf_4h_atr_14 is ALWAYS TRUE (longer timeframes have structurally higher ATR)
- Use RATIO comparisons instead: tf_1d_atr_14 > 2.5 * tf_4h_atr_14 (daily ATR is 2.5x the 4h norm)
- Emergency exits should be RARE events, not the primary exit mechanism
```

### Step 2: Compile-time tautology detection
In `trigger_compiler.py`, add a validator that detects patterns like:
- `tf_1d_X > tf_4h_X` or `tf_4h_X > tf_1h_X` where X is ATR/vol-related
- Emit an `identifier_warning` of type `tautology_suspect`

### Step 3: Emergency exit frequency cap
Add telemetry to track emergency exit frequency per symbol. If emergency exits account for >50% of all exits in a plan period, flag this in the judge snapshot so the judge can correct it.

## Test Plan
```bash
# Unit: tautology detection
uv run pytest tests/test_trigger_compiler.py -k tautology -vv

# Unit: prompt contains warning
python3 -c "
text = open('prompts/strategy_plan_schema.txt').read()
assert 'ALWAYS TRUE' in text or 'tautology' in text.lower()
print('PASS: prompt contains ATR tautology warning')
"

# Integration: run backtest and verify emergency exit ratio
# Emergency exits should be <30% of total exits
```

## Test Evidence
```
tests/test_trigger_compiler.py::test_detects_atr_tautology_1d_vs_4h PASSED
tests/test_trigger_compiler.py::test_detects_atr_tautology_4h_vs_1h PASSED
tests/test_trigger_compiler.py::test_detects_atr_tautology_with_atr_14_suffix PASSED
tests/test_trigger_compiler.py::test_allows_atr_ratio_comparison PASSED
tests/test_trigger_compiler.py::test_allows_same_timeframe_atr PASSED
tests/test_trigger_compiler.py::test_no_tautology_lower_gt_higher PASSED
tests/test_trigger_compiler.py::test_detects_atr_tautology_lt_operator PASSED
```
All 7 ATR tautology tests pass. `detect_atr_tautologies()` correctly identifies `tf_1d_atr > tf_4h_atr` (always true), allows ratio comparisons like `> 2.5 * tf_4h_atr`, and ignores same-timeframe or lower>higher comparisons.

Prompt verification: `strategy_plan_schema.txt` contains "ALWAYS TRUE" warning and ratio comparison guidance. `llm_strategist_simple.txt` Rule 7 warns about ATR tautologies.

## Human Verification Evidence
Verified `prompts/strategy_plan_schema.txt` contains cross-timeframe ATR tautology warning with correct/incorrect examples. `trigger_compiler.py` `detect_atr_tautologies()` uses AST parsing to find Compare nodes matching `_TF_ATR_PATTERN`. Emergency exit pct surfaced in compact judge summary.

## Acceptance Criteria
- [x] Prompt explicitly warns about cross-timeframe ATR tautologies
- [x] Compile-time validator flags `tf_Xh_atr > tf_Yh_atr` where X > Y
- [x] Emergency exit ratio tracked in trigger analytics
- [ ] Backtest shows emergency exits <30% of total exits (down from 80%) — *requires validation backtest*

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-10 | Runbook created from backtest ebf53879 analysis | Claude |
| 2026-02-11 | Implemented: ATR tautology detection in trigger_compiler.py, prompt guardrails in schema + simple prompt, emergency_exit_pct in judge snapshot | Claude |

## Git Workflow
```bash
git checkout -b fix/emergency-exit-sensitivity
# ... implement changes ...
git add prompts/strategy_plan_schema.txt prompts/llm_strategist_simple.txt trading_core/trigger_compiler.py
git commit -m "Fix emergency exit sensitivity: ATR tautology guardrail and prompt warning"
```
