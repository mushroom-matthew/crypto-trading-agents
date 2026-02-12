# Branch: block-reason-normalization

## Purpose
Daily summary telemetry lumps `exit_binding_mismatch` and `HOLD_RULE` blocks into the generic `risk` category in `limit_stats.blocked_totals`. The DB has the correct granular reasons, but the summary (which the judge reads) is misleading. This causes the judge to recommend risk adjustments when the actual problem is prompt quality or hold rule degeneracy.

## Source Evidence
- Backtest `58cb897f`: DB shows `exit_binding_mismatch` (7) and `HOLD_RULE` (2) as distinct reasons
- Daily summary `blocked_totals` normalizes both as `risk`
- Judge feedback driven by "high risk blocks" when actual blocks are prompt/calibration issues
- Judge wasted 6+ feedback slots on phantom risk issues in previous backtests (Runbook 26 context)

## Root Cause
The block reason normalization in the daily summary builder maps all non-standard reasons to `risk`. The granular block reasons from the trigger engine are lost before they reach the judge.

## Scope
1. **Add explicit block reason categories** to the daily summary builder
2. **Preserve granular reasons** so the judge sees `exit_binding_mismatch`, `HOLD_RULE`, `daily_budget`, `position_cap`, etc. as separate categories
3. **Update judge snapshot formatter** to include the granular breakdown

## Out of Scope
- Changing how the trigger engine generates block reasons
- Fixing the underlying exit binding or hold rule issues (separate runbooks 32, 33)

## Key Files
- `backtesting/llm_strategist_runner.py` — Daily summary builder (where blocked_totals is computed)
- `tools/execution_tools.py` — Reference for how blocks are generated
- `agents/strategies/trigger_engine.py` — Source of block reasons

## Implementation Steps

### Step 1: Define canonical block reason enum
```python
BLOCK_REASONS = {
    "exit_binding_mismatch",
    "hold_rule",
    "daily_budget_cap",
    "position_cap",
    "category_restricted",
    "stand_down",
    "trigger_disabled",
    "risk_limit",
    "other",
}
```

### Step 2: Update daily summary builder
In the blocked_totals computation, preserve the original reason string and map it to the canonical set instead of collapsing to "risk".

### Step 3: Update judge snapshot
Ensure the judge sees the granular breakdown in its snapshot, so it can distinguish between prompt quality issues (exit_binding, hold_rule) and actual risk issues (daily_budget_cap, position_cap).

## Test Plan
```bash
# Unit: block reasons preserved in summary
uv run pytest tests/test_daily_summary.py -k block_reason_granular -vv

# Unit: unknown reasons map to "other" not "risk"
uv run pytest tests/test_daily_summary.py -k block_reason_unknown -vv
```

## Test Evidence
```
(to be filled after implementation)
```

## Acceptance Criteria
- [ ] Daily summary `blocked_totals` preserves granular block reasons
- [ ] Judge snapshot shows granular breakdown
- [ ] No reason silently mapped to generic "risk"
- [ ] Existing tests still pass

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-12 | Runbook created from backtest 58cb897f validation analysis | Claude |

## Git Workflow
```bash
git checkout -b fix/block-reason-normalization
# ... implement changes ...
git add backtesting/llm_strategist_runner.py tests/test_daily_summary.py
git commit -m "Preserve granular block reasons in daily summary instead of collapsing to risk"
```
