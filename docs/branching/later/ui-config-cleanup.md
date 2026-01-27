# Branch: ui-config-cleanup

## Purpose
Clean up backtest configuration UI to resolve hidden parameters, preset state confusion, and improve clarity.

## Source
Analysis performed 2026-01-27 during comp-audit-indicators-prompts work.

## Issues Identified

### 1. Hidden Parameter: `max_daily_risk_budget_pct`
- **Status**: Defined in API schema (`ops_api/routers/backtests.py` line 237-240) but NOT exposed in frontend UI
- **Backend**: Passed through to activities (line 572-573), used in `llm_strategist_runner.py` (line 478, 1082)
- **Purpose**: Cumulative daily risk allocation (controls position sizing budget)
- **Confusion**: Different from `max_daily_loss_pct` (realized loss stop) but users can't see/set it
- **Fix**: Either add to `AggressiveSettingsPanel` with tooltip explaining difference, or remove from schema

### 2. Auto-Derived: `llm_calls_per_day`
- **Status**: Calculated as `1 + judge_evals_per_day` in `activities.py` line 227
- **Issue**: Frontend doesn't allow direct control; only `judge_cadence_hours` is exposed
- **Fix**: Either expose as separate field or document that it's derived from judge cadence

### 3. Whipsaw Preset + Manual Field Confusion
- **Location**: `ui/src/components/BacktestControl.tsx` - AggressiveSettingsPanel
- **Issue**: UI has presets (Conservative, Moderate, Aggressive, Disabled) PLUS manual input fields
- **Problem**: User can load preset then modify individual fields → state becomes inconsistent
- **Fix**: Track "Custom" state when user modifies preset values; update preset selector to show "Custom"

### 4. Debug Options Verbosity
- **Current**: 3 separate fields (`debug_trigger_sample_rate`, `debug_trigger_max_samples`, `indicator_debug_mode`)
- **Proposal**: Consolidate into single "Debug Mode" selector with off/sample/full options
- **Priority**: Low - current setup works, just verbose

## Key Files
- `ui/src/components/BacktestControl.tsx` (AggressiveSettingsPanel, PlanningSettingsPanel)
- `ui/src/lib/presets.ts` (preset definitions)
- `ops_api/routers/backtests.py` (API schema)
- `backtesting/activities.py` (parameter consumption)

## Validation Performed

### Strategy Templates: All Valid
| Template | Purpose | Status |
|----------|---------|--------|
| `aggressive_active.txt` | Max opportunities, smaller positions | ✓ Used |
| `balanced_hybrid.txt` | Balanced risk/reward | ✓ Used |
| `conservative_defensive.txt` | Risk-averse, fewer trades | ✓ Used |
| `mean_reversion.txt` | Reversion to mean strategy | ✓ Used |
| `momentum_trend_following.txt` | Trend following | ✓ Used |
| `scalper_fast.txt` | High-frequency scalping | ✓ Used (new) |
| `volatility_breakout.txt` | Volatility-based entries | ✓ Used |

### Presets: All Valid
| Preset | Purpose | Status |
|--------|---------|--------|
| `btc-quick` | Baseline testing | ✓ Valid |
| `btc-monthly` | Full month analysis | ✓ Valid |
| `multi-crypto` | Portfolio testing | ✓ Valid |
| `llm-strategist` | LLM test harness | ✓ Valid |
| `scalper-5m` | 5m high-frequency | ✓ Valid |
| `scalper-15m` | 15m high-frequency | ✓ Valid |
| `scalper-leverage-2x` | 2x leverage example | ✓ Valid |
| `conservative-daily` | Full whipsaw protection | ✓ Valid |

### Frontend → Backend Mapping: Complete
All UI-exposed options are consumed by backend. No dead code identified.

## Acceptance Criteria
- [ ] `max_daily_risk_budget_pct` either added to UI or removed from schema
- [ ] Whipsaw preset selector shows "Custom" when user modifies individual values
- [ ] Tooltips added explaining `max_daily_loss_pct` vs `max_daily_risk_budget_pct`
- [ ] (Optional) Debug options consolidated

## Priority
Low - system works correctly, these are UX improvements only.

## Dependencies
None - isolated UI cleanup.
