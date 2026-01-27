# Branch: comp-audit-indicators-prompts

## Purpose
Increase scalper trade frequency and improve indicator fidelity by adding faster indicator presets, momentum/breakout prompts, and indicator compute optimizations.

## Source Plans
- docs/analysis/COMPUTATION_AUDIT_PLAN.md (Phase 1 items 6/8, Phase 3 items 12/13, Phase 4 item 16)

## Scope
- Add fast indicator presets (EMA5/EMA8, VWAP touches, volatility bursts) for 5m/1m scalper runs.
- Add momentum/breakout trigger templates and volatility gating in prompt templates.
- Update Donchian bands to use high/low instead of close-only windows.
- Add shorter vol windows or optional EWMA volatility support for scalp runs (indicator side only).
- Optimize indicator computations with caching or incremental updates to reduce per-tick cost.

## Out of Scope / Deferred
- RiskEngine vol-target sizing changes (coordinate with comp-audit-risk-core if needed).
- Trigger cadence/min-hold changes (comp-audit-trigger-cadence).
- UI exposure of new presets (scalper-mode branch later).

## Key Files
- agents/analytics/indicator_snapshots.py
- metrics/technical.py
- prompts/llm_strategist_prompt.txt
- prompts/strategies/*.txt

## Dependencies / Coordination
- Coordinate with comp-audit-risk-core if vol-target sizing needs RiskEngine changes.
- Avoid editing plan_provider or trigger_engine to prevent conflicts with other Phase 0/1 branches.

## Acceptance Criteria
- Scalper configs support 5m/1m timeframes with fast indicator presets.
- Donchian bands use high/low inputs.
- Indicator compute cost reduced for hot paths (documented or profiled).
- Prompt templates include momentum/breakout coverage with volatility gating.

## Test Plan (required before commit)
- uv run pytest -k indicator -vv
- uv run pytest -k technical -vv
- uv run python -c "from agents.analytics import indicator_snapshots; from metrics import technical"

If tests cannot be run, obtain user-run output and paste it below before committing.

## Human Verification (required)
- Run a short backtest using the new fast indicator preset to ensure indicators compute without errors.
- Confirm Donchian bands use high/low inputs (inspect indicator snapshot or debug output).
- Confirm momentum/breakout prompts appear in plan output or template review.
- Paste run id and observations in the Human Verification Evidence section.

## Worktree Setup (recommended for parallel agents)
Use a linked worktree so multiple branches can be worked on in parallel from one clone.

```bash
# From the main repo directory
git fetch
git worktree add -b comp-audit-indicators-prompts ../wt-comp-audit-indicators-prompts comp-audit-indicators-prompts
cd ../wt-comp-audit-indicators-prompts

# When finished (after merge)
git worktree remove ../wt-comp-audit-indicators-prompts
```

## Git Workflow (explicit)
```bash
# Start from updated main
git checkout main
git pull

# Create branch
git checkout -b comp-audit-indicators-prompts

# Work, then review changes
git status
git diff

# Stage changes
git add agents/analytics/indicator_snapshots.py \
  metrics/technical.py \
  prompts/llm_strategist_prompt.txt \
  prompts/strategies

# Run tests (must succeed or be explicitly approved by user with pasted output)
uv run pytest -k indicator -vv
uv run pytest -k technical -vv
uv run python -c "from agents.analytics import indicator_snapshots; from metrics import technical"

# Commit ONLY after test evidence is captured below
git commit -m "Indicators: fast presets, Donchian highs/lows, optimized compute"
```

## Change Log (update during implementation)
- 2026-01-26: Implemented fast indicator presets and Donchian high/low fix.
  - **agents/analytics/indicator_snapshots.py**: Fixed Donchian bands to use high/low instead of close. Added `scalper_config()` factory, fast indicator fields (ema_fast, ema_very_fast, realized_vol_fast, ewma_vol, vwap, vwap_distance_pct, vol_burst), EWMA volatility helpers.
  - **schemas/llm_strategist.py**: Added new IndicatorSnapshot fields for fast indicators.
  - **prompts/llm_strategist_prompt.txt**: Added scalper-specific guidance for fast indicators.
  - **prompts/strategies/volatility_breakout.txt**: Added vol_burst examples.
  - **prompts/strategies/scalper_fast.txt**: New scalper strategy template.

## Test Evidence (append results before commit)
```
$ uv run pytest -k indicator -vv
tests/test_strategy_executor.py::test_indicator_helpers PASSED
1 passed, 226 deselected

$ uv run pytest -k technical -vv
tests/test_metrics_tools.py::test_list_technical_metrics_tool PASSED
tests/test_metrics_tools.py::test_compute_technical_metrics_from_sample_csv PASSED
2 passed, 225 deselected

$ uv run python -c "from agents.analytics import indicator_snapshots; from metrics import technical"
Imports successful

# Verification of new fast indicators:
=== Standard Config ===
close: 39480.77
ema_fast: 39493.80
ema_very_fast: 39504.39
vwap: 39674.75
vwap_distance_pct: -0.49%
vol_burst: False
ewma_vol: 0.000772
realized_vol_fast: 0.000783
donchian_upper_short: 39630.09 (now uses high)
donchian_lower_short: 39447.90 (now uses low)

=== Scalper Config ===
ema_fast (period=3): 39486.76
ema_very_fast (period=5): 39493.80
vwap (window=20): 39533.57
```

## Human Verification Evidence (append results before commit when required)
Pending: Run short backtest with fast indicator preset to verify indicators compute without errors.

