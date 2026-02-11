# Branch: judge-structured-multipliers

## Purpose
Replace fragile text parsing for judge sizing instructions with structured multipliers, and clamp outputs to prevent catastrophic undersizing.

## Source Evidence
- Backtest `7c860ae1`: judge note “capping at 1.5%” was parsed into a multiplier of `0.015`, shrinking sizes by 100x and causing large numbers of risk blocks.

## Root Cause
`services/risk_adjustment_service.py` relies on regex to parse free-text sizing instructions. This can misinterpret numeric values and apply extreme multipliers without bounds.

## Scope
- Add structured `symbol_risk_multipliers` (or `risk_adjustments`) to judge JSON output.
- Update judge prompt template to require structured multipliers when sizing changes are recommended.
- Make structured fields authoritative; text parsing becomes fallback only.
- Clamp multipliers to a safe range (e.g., `0.25 <= m <= 3.0`).

## Out of Scope / Deferred
- Policy engine tuning or p_hat integration.
- Changes to risk engine math outside multiplier handling.

## Key Files
- `prompts/llm_judge_prompt.txt`
- `schemas/judge_feedback.py`
- `services/judge_feedback_service.py`
- `services/risk_adjustment_service.py`
- `backtesting/llm_strategist_runner.py`

## Implementation Steps

### Step 1: Extend judge JSON contract
Add `symbol_risk_multipliers` or `risk_adjustments` to the judge JSON schema and prompt with explicit examples.

### Step 2: Prefer structured multipliers
Update `apply_judge_risk_feedback()` to use structured multipliers first. Free-text parsing remains as a fallback only.

### Step 3: Add clamps
Clamp multipliers to a safe band (configurable via env). Log when clamping occurs.

### Step 4: Add tests
Verify that “1.5%” yields a multiplier of `0.985` or explicit `0.85` only when structured fields are set. Prevent `0.015` scale-in.

## Test Plan
```bash
uv run pytest tests/test_judge_risk_adjustments.py -vv
uv run pytest tests/test_risk_adjustment_clamps.py -vv
```

## Test Evidence
```
TODO
```

## Acceptance Criteria
- Structured multipliers override text parsing.
- Multipliers are clamped to a safe range with explicit logs.
- No multiplier < 0.25 is accepted unless explicitly overridden by config.

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-11 | Runbook created from backtest 7c860ae1 analysis | Claude |

## Worktree Setup (recommended for parallel agents)
```bash
git fetch
git worktree add -b judge-structured-multipliers ../wt-judge-structured-multipliers main
cd ../wt-judge-structured-multipliers

# When finished (after merge)
git worktree remove ../wt-judge-structured-multipliers
```

## Git Workflow
```bash
git checkout main
git pull
git checkout -b judge-structured-multipliers

# ... implement changes ...

git add prompts/llm_judge_prompt.txt \
  schemas/judge_feedback.py \
  services/judge_feedback_service.py \
  services/risk_adjustment_service.py \
  backtesting/llm_strategist_runner.py

uv run pytest tests/test_judge_risk_adjustments.py -vv
git commit -m \"Judge: structured risk multipliers with clamps\"
```
