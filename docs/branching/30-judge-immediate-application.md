# Branch: judge-immediate-application

## Purpose
Apply intraday judge feedback to the active TriggerEngine and RiskEngine without waiting for a replan.

## Source Evidence
- Backtest `7c860ae1`: judge flagged hold-rule issues but enforcement only changed after a replan, allowing 44 additional hold-rule blocks.

## Root Cause
Intraday judge feedback is recorded but does not update the active engine state unless a replan occurs. This delays action on known issues.

## Scope
- When intraday judge feedback arrives, immediately update:
  - `TriggerEngine.judge_constraints`
  - `RiskEngine.risk_profile` or `risk_mode` scaling
  - Any live in-memory caps that are enforced per-bar
- Ensure updates are consistent across backtest and live paths.

## Out of Scope / Deferred
- Full policy engine integration.
- Replan frequency tuning.

## Key Files
- `backtesting/llm_strategist_runner.py`
- `agents/strategies/trigger_engine.py`
- `agents/strategies/risk_engine.py`
- `services/risk_adjustment_service.py`
- `agents/judge_agent_client.py`

## Implementation Steps

### Step 1: Introduce an “apply intraday feedback” helper
Add a method that accepts `JudgeFeedback` and updates the active trigger and risk engines in-place.

### Step 2: Apply constraints immediately
When the judge disables triggers or sets risk_mode, update the current TriggerEngine and RiskEngine on the same evaluation cycle.

### Step 3: Emit action events
Emit `judge_action_applied` events with `scope=intraday` so the timeline reflects immediate enforcement.

## Test Plan
```bash
uv run pytest tests/test_intraday_judge_application.py -vv
uv run pytest -k judge -vv
```

## Test Evidence
```
TODO
```

## Acceptance Criteria
- Intraday judge feedback modifies active trigger constraints within the same bar.
- Risk mode scaling changes position sizing immediately after feedback.
- Timeline shows explicit action events for intraday applications.

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-11 | Runbook created from backtest 7c860ae1 analysis | Claude |

## Worktree Setup (recommended for parallel agents)
```bash
git fetch
git worktree add -b judge-immediate-application ../wt-judge-immediate-application main
cd ../wt-judge-immediate-application

# When finished (after merge)
git worktree remove ../wt-judge-immediate-application
```

## Git Workflow
```bash
git checkout main
git pull
git checkout -b judge-immediate-application

# ... implement changes ...

git add backtesting/llm_strategist_runner.py \
  agents/strategies/trigger_engine.py \
  agents/strategies/risk_engine.py \
  services/risk_adjustment_service.py \
  agents/judge_agent_client.py

uv run pytest -k judge -vv
git commit -m \"Judge: apply intraday feedback immediately\"
```
