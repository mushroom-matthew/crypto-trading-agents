## Risk Expression & RPR-Guided Strategy Tuning

This note captures the “do it in order” steps we just agreed on. Safety rails stay; we only loosen throughput and make the LLM respond to measured performance.

### Phase 0 – Guardrails (unchanged)
- Keep `max_daily_loss_pct = 3%`, `max_daily_risk_budget_pct = 20%`.
- Keep stop-aware sizing, daily budget reset/anchor, and full risk telemetry (allocated vs actual at stop).

### Phase 1 – Let risk express (exploratory caps)
Goal: move mean daily usage toward ~10–15% while the above guardrails hold.

- Add an “exploratory” profile via env/CLI for test runs:
  - `STRATEGIST_PLAN_DEFAULT_MAX_TRADES=30` (per-plan cap seed).
  - Global daily cap target: ~30–40 trades/day (via judge constraints or plan defaults; keep hard rails intact).
  - Leave `max_position_risk_pct=2`, `max_daily_risk_budget_pct=20`, `max_daily_loss_pct=3`.
- Success on a 30-day slice: risk_budget_used_pct_mean in 7–15%, no daily_loss/risk_budget blowups, blocks driven by risk only in bad regimes (not by daily_cap every day).
- If usage stays <5% even with higher caps, the LLM plan is under-asking; address in Phase 3 prompts.

### Phase 2 – Archetype weights driven by `rpr_actual`
- Define a simple health rule over a 14–30 day lookback:
  - `Good`: trades ≥ N_min (5–10) and `rpr_actual > +0.2`.
  - `Neutral`: trades < N_min or `-0.2 ≤ rpr_actual ≤ +0.2`.
  - `Bad`: trades ≥ N_min and `rpr_actual < -0.2`.
- Map health → caps/multipliers:
  - Good: full cap, optional risk multiplier ~1.2×.
  - Neutral: current caps, no multiplier.
  - Bad: cap cut (≤50%) or risk multiplier ~0.5×; for exits, keep but reduce eagerness.
- Implement via per-archetype caps or a small risk_adjustment helper that consumes `trigger_quality` and emits per-archetype multipliers for the judge.

### Phase 3 – Teach the LLM (strategist/judge) to use RPR + utilization
- Provide a performance snapshot in prompts (recent `rpr_actual`, trades) per archetype and per hour when available.
- Explicit instructions:
  - Favor archetypes with positive `rpr_actual` and enough samples.
  - Limit/justify archetypes with negative `rpr_actual`; cap reversal “probationary” until proven.
  - Target daily risk utilization band 10–30%; if <5% for several days, increase trade frequency or per-trade sizing within existing risk rails.
  - Time-of-day: avoid initiating in hours with consistently negative `rpr_actual`; be more willing in positive hours.

### Phase 4 – Longer, looser P1a and evaluate
- Run a longer window (e.g., full Feb 2021) with exploratory caps and RPR-aware prompts.
- Success metrics:
  - Usage: mean 7–15%; some days >25%, few/no daily_loss hits.
  - Blocks: risk_budget/daily_loss appear only in bad regimes; daily_cap/plan_limit not constant brakes.
  - Archetype quality: sustained positive `rpr_actual` for “good” buckets; reduced contribution from “bad”.
  - Time-of-day: reduced trading in hours with negative `rpr_actual`.

### Phase 5 – Only then consider higher per-trade risk
- If the above looks good over 30–60 days and utilization is still low, experiment with `max_position_risk_pct` 3–4% **without** changing daily loss/budget rails.
