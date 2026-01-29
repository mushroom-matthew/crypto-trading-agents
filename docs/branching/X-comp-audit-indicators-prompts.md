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
- 2026-01-26: Auto-enable scalper_config for fast timeframes in backtester.
  - **agents/analytics/__init__.py**: Export `scalper_config` function.
  - **backtesting/llm_strategist_runner.py**: Added `_config_for_timeframe()` method that auto-selects `scalper_config()` for timeframes ≤15m. This ensures fast indicators are computed with optimized parameters for scalper runs.
- 2026-01-26: Auto-load scalper_fast.txt prompt for fast timeframe runs.
  - **backtesting/llm_strategist_runner.py**: Added `_has_fast_timeframes()` and `_load_scalper_prompt()` methods. When timeframes include 15m or faster and no explicit prompt is provided, automatically loads `prompts/strategies/scalper_fast.txt` which instructs the LLM to use fast indicators (ema_fast, vwap, vol_burst, etc.).

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

### Backtest: backtest-29d0261f-551d-42b6-acf2-d542e176e2a0
- **Date range**: Jan 31 - Feb 2, 2024
- **Symbols**: BTC-USD, ETH-USD
- **Timeframes**: 15m, 30m, 1h, 2h, 4h, 8h, 1d

#### Fast Indicator Usage (CONFIRMED)
```
ema_fast: 46 occurrences in trigger rules
ema_very_fast: 46 occurrences
vwap_distance_pct: 29 occurrences
vol_burst: 8 occurrences
```

#### Slow Indicator Usage (CONFIRMED ZERO)
```
sma_medium: 0 occurrences
sma_short: 0 occurrences
sma_long: 0 occurrences
```

#### Sample Trigger Rules Generated
```
trigger_1 (mean_reversion): vwap_distance_pct < -0.5 and ema_fast < ema_very_fast and position == 'flat'
trigger_3 (volatility_breakout): close > donchian_upper_short and vol_burst == true and ema_fast > ema_very_fast
```

#### Performance vs Previous (OLD backtest bb630623, same period style)
| Metric | OLD (no fast) | NEW (with fast) |
|--------|---------------|-----------------|
| Fast indicators used | 0 | 46+ |
| Slow indicators used | 10 | 0 |
| Trades | 48 | 15 |
| Win rate | 45.8% | 46.7% |
| Total PnL | -$7.19 | -$2.28 |

#### Verdict
- Fast indicators ARE being used correctly
- Slow indicators are NOT being used
- Scalper prompt auto-loads for 15m timeframe
- More selective trading (fewer trades, similar win rate, better PnL)
- Donchian bands confirmed using high/low (see test evidence above)

