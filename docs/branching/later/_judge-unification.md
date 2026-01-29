# Branch: judge-unification (later)

## Purpose
Unify judge behavior across live, trading_core, and backtest by introducing JudgeFeedbackService and using heuristics as LLM context.

## Source Plans
- docs/JUDGE_UNIFICATION_PLAN.md

## Scope
- Create JudgeFeedbackService and HeuristicAnalysis data model.
- Route backtest and live judge flows through the same service.
- Support shim transport for deterministic backtests.
- Update prompts and telemetry to include heuristic pre-analysis.

## Out of Scope / Deferred
- Policy engine changes (policy-pivot branch).
- Any strategy prompt or indicator changes unrelated to judge feedback.

## Key Files
- services/judge_feedback_service.py
- agents/judge_agent_client.py
- trading_core/judge_agent.py
- backtesting/llm_strategist_runner.py
- schemas/judge_feedback.py (if schema updates needed)

## Dependencies / Coordination
- Coordinate with comp-audit-metrics-parity if trade_quality or portfolio_state is updated.
- Ensure transport shim behavior is stable for backtests.

## Acceptance Criteria
- Single judge service used for live and backtests (with shim option).
- Heuristic analysis passed into LLM prompt in non-shim mode.
- Deterministic shim output matches heuristic scoring.

## Test Plan (required before commit)
- uv run pytest -k judge -vv
- uv run pytest -k backtesting -vv
- uv run python -c "from services import judge_feedback_service"

If tests cannot be run, obtain user-run output and paste it below before committing.

## Human Verification (required)
- Run a backtest with judge shim on and off (if supported) and compare outputs for determinism vs LLM.
- Confirm heuristic pre-analysis is included in the prompt/logs for non-shim runs.
- Paste run id and observations in the Human Verification Evidence section.

## Worktree Setup (recommended for parallel agents)
Use a linked worktree so multiple branches can be worked on in parallel from one clone.

```bash
# From the main repo directory
git fetch
git worktree add -b judge-unification (later) ../wt-judge-unification (later) judge-unification (later)
cd ../wt-judge-unification (later)

# When finished (after merge)
git worktree remove ../wt-judge-unification (later)
```

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b judge-unification

# Work, then review changes
git status
git diff

# Stage changes
git add services/judge_feedback_service.py \
  agents/judge_agent_client.py \
  trading_core/judge_agent.py \
  backtesting/llm_strategist_runner.py \
  schemas/judge_feedback.py

# Run tests (must succeed or be explicitly approved by user with pasted output)
uv run pytest -k judge -vv
uv run pytest -k backtesting -vv
uv run python -c "from services import judge_feedback_service"

# Commit ONLY after test evidence is captured below
git commit -m "Judge: unify heuristics and LLM feedback"
```

## Change Log (update during implementation)
- YYYY-MM-DD: Summary of changes, files touched, and decisions.

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)

