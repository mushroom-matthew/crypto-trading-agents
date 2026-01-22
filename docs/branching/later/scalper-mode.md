# Branch: scalper-mode (later)

## Purpose
Implement Scalper Mode features across config, UI, backtesting, and comparison tooling.

## Source Plans
- docs/SCALPER_MODE_IMPLEMENTATION_PLAN.md

## Scope
- Expose aggressive risk/whipsaw/execution gating settings in UI and API.
- Add short-term indicator presets and timeframe support configuration.
- Add leverage comparison backtest workflow and UI view.
- Implement walk-away threshold logic and market-driven replan triggers.

## Out of Scope / Deferred
- Real leverage trading integrations (Phase 7 in plan).

## Key Files
- ops_api/routers/backtests.py
- ops_api/routers/paper_trading.py
- backtesting/llm_strategist_runner.py
- agents/strategies/trigger_engine.py
- ui/src/components/BacktestControl.tsx
- ui/src/components/PaperTradingControl.tsx
- ui/src/components/LeverageComparison.tsx
- ui/src/lib/api.ts
- backtesting/comparison.py
- schemas/llm_strategist.py

## Dependencies / Coordination
- Requires comp-audit-trigger-cadence and comp-audit-indicators-prompts merged first.
- Coordinate with comp-audit-ui-trade-stats to avoid UI conflicts.

## Acceptance Criteria
- Scalper settings visible and functional in backtest and paper trading UI.
- Short-term presets run without errors and produce higher trade frequency.
- Comparison endpoint returns aggregated metrics; UI renders charts and tables.
- Walk-away threshold stops trading at configured profit or loss.

## Test Plan (required before commit)
- uv run pytest -k backtests -vv
- uv run pytest -k paper_trading -vv
- uv run pytest -k trigger_engine -vv
- (Optional if UI linting exists) cd ui && npm run lint

If tests cannot be run, obtain user-run output and paste it below before committing.

## Human Verification (required)
- Use the UI to apply aggressive settings and run a backtest; confirm settings persist and trade frequency increases.
- Run leverage comparison and confirm the comparison table/chart renders expected metrics.
- Paste run id and UI observations in the Human Verification Evidence section.

## Worktree Setup (recommended for parallel agents)
Use a linked worktree so multiple branches can be worked on in parallel from one clone.

```bash
# From the main repo directory
git fetch
git worktree add -b scalper-mode (later) ../wt-scalper-mode (later) scalper-mode (later)
cd ../wt-scalper-mode (later)

# When finished (after merge)
git worktree remove ../wt-scalper-mode (later)
```

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b scalper-mode

# Work, then review changes
git status
git diff

# Stage changes (adjust list as needed based on actual edits)
git add ops_api/routers/backtests.py \
  ops_api/routers/paper_trading.py \
  backtesting/llm_strategist_runner.py \
  agents/strategies/trigger_engine.py \
  ui/src/components/BacktestControl.tsx \
  ui/src/components/PaperTradingControl.tsx \
  ui/src/components/LeverageComparison.tsx \
  ui/src/lib/api.ts \
  backtesting/comparison.py \
  schemas/llm_strategist.py

# Run tests (must succeed or be explicitly approved by user with pasted output)
uv run pytest -k backtests -vv
uv run pytest -k paper_trading -vv
uv run pytest -k trigger_engine -vv

# Commit ONLY after test evidence is captured below
git commit -m "Scalper mode: aggressive configs and comparison tooling"
```

## Change Log (update during implementation)
- YYYY-MM-DD: Summary of changes, files touched, and decisions.

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)

