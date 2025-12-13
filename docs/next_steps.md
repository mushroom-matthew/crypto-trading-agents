# Next Steps (Current State)

## Where we are
- Caps are stable: daily_cap blocks are 0; trigger caps honor env floors in fixed mode; legacy behavior isolated.
- Risk budget is the active brake; per-trade risk vs daily rail sets throughput.
- Planner occasionally undersupplies triggers (some 0–1 trade days with no blocks).
- Recent risk sweeps (fixed caps 30/40): ~9–12 trades/day with 10–12% budget and 0.3–0.4% per-trade risk; budget usage ~35–45%.

## Priorities to accelerate throughput and quality
1) **Boost trigger supply/quality**
   - Require minimum viable triggers per day; log why triggers are pruned (filters/budgets).
   - Add trigger-density telemetry and warnings; review “1 trade, 0 blocks” days to fix prompt/culling.
2) **Settle on a risk regime**
   - Good defaults: `max_daily_risk_budget_pct` 10–12%, `max_position_risk_pct` 0.30–0.40%.
   - Judge caps off in fixed mode; caps fixed at trades=30, triggers=40.
3) **Trim remaining plan_limit friction**
   - Inspect plan_limit-heavy days (r2_nomult/r3_mult); relax per-symbol/timeframe caps if safe.
4) **Rebuild RPR surfaces once throughput is steady**
   - Use the higher-density runs to recompute RPR; add archetype-only multipliers first; defer hour slices until trigger coverage is solid.

## Experiments to run now
- Use `scripts/run_cap_matrix_risk.sh` on 2021-02-10→2021-02-16 and compare:
  - `trade_count_mean`, `blocked_by_risk_budget_mean`, `risk_budget_used_pct_mean`.
  - cap_state resolved caps (trades/triggers/session caps).
  - RPR vs baseline (when available).
- If trades stay low on certain days, inspect those daily reports for trigger scarcity vs risk budget blocks.

## What not to do yet
- Don’t re-enable sticky judge caps in fixed mode.
- Don’t introduce hour-level multipliers until trigger density is reliable.
- Don’t revert to derived-cap min() behavior in fixed mode; policy floors are correct.
