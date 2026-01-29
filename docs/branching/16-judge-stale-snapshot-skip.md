# Runbook: Judge Stale Snapshot Detection

## Overview
When the judge re-evaluates after a period of zero trading activity, it receives the exact same snapshot (same equity, same trade count, same positions) as the previous evaluation. This leads to identical conclusions and redundant LLM calls that reinforce the current (possibly broken) state.

**Source:** Backtest `backtest-b836ce01-d8ef-402e-a74d-4afa241f3344` — judge eval at 18:45 UTC used identical snapshot to 06:45 eval (equity $9985.13, 10 trades, 0 wins, no positions). Produced the same score (17.8) and doubled down on disabling triggers.

## Scope
1. **Snapshot diff check:** Before running a judge evaluation, compare the current snapshot to the previous one. If equity, trade count, and positions are unchanged, mark as stale.
2. **Stale handling options:**
   - Skip evaluation entirely and schedule next eval sooner.
   - Run evaluation with a modified prompt: "No trades have occurred since your last evaluation. Consider re-enabling triggers or adjusting constraints."
   - Force at least one trigger re-enablement before the next eval.
3. **Metric:** Track `stale_judge_evals` count in daily reports.

## Key Files
- `backtesting/llm_strategist_runner.py` — judge evaluation scheduling, snapshot construction, `_run_intraday_judge`

## Acceptance
- Judge detects when consecutive evaluations use the same snapshot.
- At least one of the stale handling options is implemented.
- Daily report tracks `stale_judge_evals`.

## Out of Scope
- Judge scoring algorithm changes.
- Judge prompt content changes beyond stale-snapshot context.
