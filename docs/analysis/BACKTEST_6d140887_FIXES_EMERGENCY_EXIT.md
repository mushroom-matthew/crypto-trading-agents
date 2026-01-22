# Backtest 6d140887 Fixes: Emergency Exit Competition

Context: `backtest-6d140887-acea-4a0c-93ad-3769573022ae` ran with the live LLM strategist (no shim). All exits were `emergency_exit_1_flat`, including a same-bar exit at `2021-05-01T07:00:00+00:00`. The emergency exit trigger uses `entry_rule: "position != 'flat'"` and an empty `exit_rule`, so it bypasses same-bar and min-hold guards that only apply to exit rules.

## What Needs Fixing

- Emergency exits fire through the entry/flatten path, bypassing same-bar and min-hold protections.
- Emergency exit rules are tautological (no buffers, no volume confirmation).
- Prompt updates are not enforced at validation time (mean-reversion lacks volume gate).

## Fix Options

### 1) Force Emergency Exits Through Exit Evaluation

Treat `category == emergency_exit` as exit-only:
- Require a non-empty `exit_rule`.
- Ignore `entry_rule` for emergency exits, or migrate `entry_rule` to `exit_rule` if missing.
- Apply same-bar and min-hold checks in the exit path.

Suggested hooks:
- `schemas/llm_strategist.py` (validation: emergency_exit must define exit_rule)
- `agents/strategies/trigger_engine.py` (evaluate emergency exits only in exit path)

Success criteria:
- `SAME_BAR_ENTRY` blocks appear when entry and emergency exit share a timestamp.
- Emergency exits no longer fire on the same bar as entry.

### 2) Add Emergency-Specific Cooldown and Min-Hold

Introduce `emergency_min_hold_bars` (or `emergency_exit_cooldown_bars`) to avoid immediate reversals:
- Applies only to `category == emergency_exit`.
- Defaults to `max(1, min_hold_bars)` for safety.

Suggested hooks:
- `agents/strategies/trigger_engine.py` (exit guard)
- `backtesting/llm_strategist_runner.py` (reporting counters)

Success criteria:
- Average hold duration increases.
- Same-bar exit count drops to zero.

### 3) Enforce Emergency Exit Buffers and Volume Gates

Reject emergency exits without buffer logic, for example:
- Long: `close < nearest_support * 0.98`
- Short: `close > nearest_resistance * 1.02`
Add `volume_multiple` gating when mean-reversion is active.

Suggested hooks:
- `prompts/llm_strategist_prompt.txt` (already updated, make mandatory)
- `agents/strategies/plan_provider.py` (validate rule content, hard fail or downgrade)

Success criteria:
- Emergency exit rules include buffer conditions.
- Mean-reversion rules reference `volume` or `volume_multiple`.

### 4) Make Exit Priority Conditional on Confidence + Hold

If entry just fired, allow entry to stand unless emergency exit passes both:
- buffer rule AND
- emergency-specific hold/cooldown.

Suggested hooks:
- `agents/strategies/trigger_engine.py` (deduplication logic)

Success criteria:
- Emergency exit execution rate drops below entry execution rate.

## Recommended Order

1) Force emergency exits through exit evaluation (most direct fix).
2) Add emergency-specific hold/cooldown.
3) Enforce buffer and volume gates at validation time.
4) Tighten deduplication priority rules.

