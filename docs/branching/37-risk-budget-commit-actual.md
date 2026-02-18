# Branch: risk-budget-commit-actual

## Priority: P0 (Blocking real-money deployment)

## Purpose
Fix the 255x overcharge in `_commit_risk_budget()` that exhausts the daily risk budget after 2–4 trades with microscopic position sizes. The function currently deducts the theoretical `max_position_risk_pct` cap from the budget rather than the actual risk taken at stop. The result: a system configured for 2.5% max risk per trade burns through a 15% daily budget in 6 trades, even when each trade risked <$1 in practice.

## Source Evidence
- Codex analysis of backtest `5386c15c`: median overcharge ratio 255x
  - `actual_risk_at_stop` per trade: $0.79–$6.25
  - `risk_used` charged to budget: $250–$900 per trade
  - Budget exhausts after 2–4 trades regardless of position size
- Session log 2026-02-13: "root: `_risk_budget_allowance()` at `llm_strategist_runner.py:3516-3517`"
- Backtest `5386c15c` validated: `risk_budget_used_pct` went from 0% → 64% mean after exit_binding_exempt fix; but sizing mismatch means actual capital at risk is still wrong

## Root Cause
`_commit_risk_budget()` charges `theoretical_allowance` (= equity × max_position_risk_pct) instead of `actual_risk_at_stop` (= qty × stop_distance). When stop_distance is small (as it often is with tight ATR-based stops on low-priced crypto bars), the actual risk is tiny but the theoretical cap is large.

```python
# CURRENT (WRONG): charges theoretical cap every trade
risk_used = min(position_risk_cap, theoretical_allowance)  # ← always theoretical_allowance

# CORRECT: charges what was actually risked at the stop
risk_used = actual_risk_at_stop  # already computed before this call
```

## Scope
1. **`_commit_risk_budget()`** — change deduction to use `actual_risk_at_stop`
2. **`_risk_budget_allowance()`** — ensure the allowance *gate* still uses theoretical cap (don't allow trades above the cap), but the *deduction* uses actual
3. **Tests** — verify: budget deducts actual risk, not theoretical; budget gate still blocks trades exceeding cap; daily reset behaves correctly
4. **Telemetry** — add `risk_overcharge_ratio` to daily report (theoretical / actual) so regression is visible in future backtests

## Out of Scope
- Changing the sizing math itself (`_size_position()`)
- Changing the risk gate logic (whether a trade is allowed)
- Judge multiplier wiring (Runbook P2 — symbol_risk_multipliers dead code)

## Key Files
- `backtesting/llm_strategist_runner.py` — `_commit_risk_budget()`, `_risk_budget_allowance()` (lines ~3510–3530)
- `tests/risk/test_risk_budget_commit_actual.py` — existing test, likely needs extension
- `tests/integration/test_risk_usage_events_actual_risk.py` — integration test for event emission

## Implementation Steps

### Step 1: Locate the deduction site
In `llm_strategist_runner.py`, find `_commit_risk_budget()`. The current logic:
```python
def _commit_risk_budget(self, ..., actual_risk_at_stop: float, ...) -> None:
    theoretical = self._equity * (self._max_position_risk_pct / 100.0)
    self._daily_risk_used += theoretical  # ← BUG: ignores actual_risk_at_stop
```

### Step 2: Fix the deduction
```python
def _commit_risk_budget(self, ..., actual_risk_at_stop: float, ...) -> None:
    # Gate: still block trades that would exceed the theoretical cap
    theoretical_cap = self._equity * (self._max_position_risk_pct / 100.0)
    # Hard diagnostic: actual exceeding cap means sizing math is wrong upstream
    if actual_risk_at_stop > theoretical_cap * 1.01:
        logger.error(
            "actual_risk_at_stop (%.4f) exceeds theoretical_cap (%.4f) — "
            "stop distance or sizing mismatch upstream. Investigate before scaling capital.",
            actual_risk_at_stop, theoretical_cap,
        )
    # Deduct: charge only what was actually risked (capped at theoretical for safety)
    deduction = min(actual_risk_at_stop, theoretical_cap)
    self._daily_risk_used += deduction
```

> **Note:** The `min()` clamp is correct, but it silently hides upstream sizing bugs. The `logger.error` above surfaces any case where actual risk exceeds the cap. This must NOT be demoted to a warning — it indicates a real discrepancy in stop-distance calculation that needs investigation before deploying real capital.

### Step 3: Add overcharge ratio to daily telemetry
```python
risk_overcharge_ratio = theoretical_cap / max(actual_risk_at_stop, 1e-9)
# Log if ratio > 10 (indicates systemic stop miscalibration)
if risk_overcharge_ratio > 10:
    logger.warning("risk_overcharge_ratio=%.1f (theoretical=%.2f, actual=%.4f)",
                   risk_overcharge_ratio, theoretical_cap, actual_risk_at_stop)
```

Add `risk_overcharge_ratio_median` and `risk_overcharge_ratio_max` to the daily report dict.

### Step 4: Update tests
In `tests/risk/test_risk_budget_commit_actual.py`:
```python
def test_deducts_actual_not_theoretical():
    # Given: equity=10000, max_position_risk_pct=2.5 → cap=250
    # actual_risk_at_stop = 5.00 (tight stop, small position)
    runner._commit_risk_budget(actual_risk_at_stop=5.00)
    assert runner._daily_risk_used == pytest.approx(5.00)  # not 250

def test_gate_blocks_trade_exceeding_cap():
    # actual_risk_at_stop = 300, cap = 250 → should still be blocked at gate level
    allowed = runner._risk_budget_allowance(actual_risk_at_stop=300)
    assert not allowed

def test_capped_deduction_when_actual_exceeds_cap():
    # If actual_risk > cap (shouldn't happen after gate), clamp to cap
    runner._commit_risk_budget(actual_risk_at_stop=300.0)
    assert runner._daily_risk_used <= 250.0
```

## Test Plan
```bash
uv run pytest tests/risk/test_risk_budget_commit_actual.py -vv
uv run pytest tests/integration/test_risk_usage_events_actual_risk.py -vv
uv run pytest -k "risk_budget" -vv
```

## Test Evidence
```
TODO
```

## Acceptance Criteria
- [ ] `_commit_risk_budget()` deducts `actual_risk_at_stop`, not theoretical cap
- [ ] Risk gate (`_risk_budget_allowance()`) still blocks trades where actual risk would exceed cap
- [ ] Daily report includes `risk_overcharge_ratio_median` and `risk_overcharge_ratio_max`
- [ ] Backtest shows `risk_overcharge_ratio_median` < 5.0 (was ~255x)
- [ ] Trade count per day increases (budget no longer exhausts after 2–4 trades)

## Human Verification Evidence
```
TODO: Run backtest after fix. Verify risk_budget_used_pct is proportional to actual stop distances.
Confirm trades_per_day increases from ~0.43 baseline.
```

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-18 | Runbook created from Codex backtest analysis of 5386c15c | Claude |

## Worktree Setup
```bash
git fetch
git worktree add -b fix/risk-budget-commit-actual ../wt-risk-budget-commit-actual main
cd ../wt-risk-budget-commit-actual

# When finished (after merge)
git worktree remove ../wt-risk-budget-commit-actual
```

## Git Workflow
```bash
git checkout main
git pull
git checkout -b fix/risk-budget-commit-actual

# ... implement changes ...

git add backtesting/llm_strategist_runner.py \
  tests/risk/test_risk_budget_commit_actual.py \
  tests/integration/test_risk_usage_events_actual_risk.py

uv run pytest tests/risk/test_risk_budget_commit_actual.py -vv
git commit -m "Fix: _commit_risk_budget() deducts actual_risk_at_stop, not theoretical cap"
```
