# Runbook: Judge Death Spiral — Minimum Trigger Floor

## Overview
The intraday judge can disable all entry triggers, creating an irreversible trading halt for the remainder of the backtest (or live session). Once all triggers are disabled, no trades execute, so the judge sees the same stale snapshot and keeps triggers disabled indefinitely.

**Source:** Backtest `backtest-b836ce01-d8ef-402e-a74d-4afa241f3344` — judge disabled all entry triggers at 06:45 UTC day 1, zero trades for remaining 2.5 days.

## Scope
1. **Minimum trigger floor:** Enforce that the judge cannot disable more than N-2 entry triggers (at least 2 must remain enabled at all times).
2. **Zero-activity re-enablement:** If zero trades execute for N consecutive bars after a judge intervention, automatically re-enable the least-bad trigger(s) with conservative sizing.
3. **Stale snapshot detection:** If the judge evaluates the same snapshot (same equity, same trade count) as the previous evaluation, skip or force a different intervention strategy.

## Key Files
- `backtesting/llm_strategist_runner.py` — judge evaluation loop, `_apply_judge_feedback`, trigger disable logic
- `agents/strategies/risk_engine.py` — if trigger floor is enforced at the risk engine level

## Acceptance
- A backtest where the judge wants to disable all triggers retains at least 2 entry triggers.
- After N bars of zero activity post-judge-intervention, at least one trigger is re-enabled.
- Judge does not re-evaluate with an unchanged snapshot.

## Out of Scope
- Judge scoring algorithm changes (separate from the floor mechanism).
- Risk budget or daily loss limit changes.
