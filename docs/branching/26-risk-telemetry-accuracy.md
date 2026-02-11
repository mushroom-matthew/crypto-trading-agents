# Branch: risk-telemetry-accuracy

## Purpose
Fix phantom risk metrics in the judge snapshot that cause the judge to waste feedback slots on non-issues. The judge repeatedly flags "BTC position risk=50 exceeds max_position_risk_pct=3.0" and "position risk reporting inconsistency," but these values don't correspond to any real calculation.

## Source Evidence
- Backtest `ebf53879` judge feedback repeatedly contains:
  - "BTC-USD position risk=50, which exceeds max_position_risk_pct 3.0%"
  - "Reconcile reported BTC-USD open position with zero trades/fills"
  - "Reduce BTC position sizing immediately to ensure per-position risk <= 3.0%"
- Actual risk per trade was $0.71-$2.05 on $10K (0.007-0.02%), well within limits
- The value "50" doesn't match any real metric (not a percentage, not absolute dollars, not position size)
- Judge wasted 6+ `must_fix` slots across 14 days on this phantom issue

## Root Cause
The judge snapshot builder computes a `risk` metric for `position_quality` that doesn't align with the risk engine's actual calculation. The snapshot likely uses a simplified formula (e.g., `position_value / equity * 100`) that produces values like 50 for a $300 notional position on a $10K portfolio, while the risk engine uses `qty * stop_distance` for actual risk.

Need to trace the exact snapshot builder code to confirm.

## Scope
1. **Trace the snapshot builder** to find where `risk=50` is computed
2. **Fix the risk metric** to match the risk engine's `actual_risk_at_stop` calculation
3. **Add metric labels** so the judge knows what each number represents (absolute risk, % of equity, exposure %)
4. **Add unit test** verifying snapshot risk matches risk engine output

## Key Files
- `backtesting/llm_strategist_runner.py` — Snapshot builder (`_build_judge_snapshot` or equivalent)
- `services/judge_feedback_service.py` — How the snapshot is consumed by the judge
- `agents/strategies/trigger_engine.py` — How `risk_used_abs` is computed at trade time

## Implementation Steps

### Step 1: Trace the risk=50 source
Search for where `position_quality` or `risk` is computed in the snapshot builder. The value 50 likely comes from:
```python
risk = (position_value / equity) * 100  # This would be exposure %, not risk %
```
Should be:
```python
risk_pct = (actual_risk_at_stop / equity) * 100  # Actual risk as % of equity
```

### Step 2: Fix the calculation
Replace the snapshot risk metric with `actual_risk_at_stop / equity * 100`, which is what `max_position_risk_pct` is compared against.

### Step 3: Label the metric
In the snapshot, include both:
```python
"position_risk_pct": actual_risk_at_stop / equity * 100,  # Risk at stop as % of equity
"symbol_exposure_pct": position_value / equity * 100,      # Gross exposure as % of equity
```
This prevents the judge from confusing exposure (always higher) with risk (much lower when stops are tight).

### Step 4: Add unit test
```python
def test_snapshot_risk_matches_risk_engine():
    # Given a position with known actual_risk_at_stop
    # When snapshot is built
    # Then snapshot.position_risk_pct == actual_risk_at_stop / equity * 100
```

## Test Plan
```bash
# Unit: snapshot risk calculation
uv run pytest tests/test_llm_strategist_runner.py -k snapshot_risk -vv

# Integration: backtest judge should NOT flag "risk=50 exceeds 3.0%"
```

## Test Evidence
```
tests/test_judge_death_spiral.py::TestRiskTelemetry::test_snapshot_risk_renamed PASSED
tests/test_judge_death_spiral.py::TestRiskTelemetry::test_budget_utilization_in_snapshot PASSED
tests/test_judge_death_spiral.py::TestRiskTelemetry::test_must_fix_on_low_budget_utilization PASSED
```
All 3 risk telemetry tests pass. `risk_score` renamed to `risk_quality_score` throughout. `position_risk_pct` and `symbol_exposure_pct` added as separate labeled fields. Judge formatting updated from `risk={risk:.0f}` to `risk_quality={quality:.0f} exposure={exposure:.1f}%`. Budget utilization <10% triggers `must_fix` hint.

## Acceptance Criteria
- [x] Snapshot `position_risk_pct` uses `actual_risk_at_stop / equity * 100`
- [x] Snapshot separately reports `symbol_exposure_pct` for gross exposure
- [x] Judge feedback no longer flags phantom risk violations (metric renamed/labeled)
- [x] Judge `must_fix` slots used for real issues, not telemetry artifacts (budget utilization hint)

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-10 | Runbook created from backtest ebf53879 analysis | Claude |
| 2026-02-11 | Implemented: risk_score→risk_quality_score rename, position_risk_pct + symbol_exposure_pct fields, judge formatting with clear labels, budget utilization telemetry + must_fix hint | Claude |

## Git Workflow
```bash
git checkout -b fix/risk-telemetry-accuracy
# ... implement changes ...
git add backtesting/llm_strategist_runner.py services/judge_feedback_service.py
git commit -m "Fix risk telemetry: use actual_risk_at_stop in judge snapshot, separate risk vs exposure"
```
