## Risk Expression & RPR-Guided Strategy Tuning

This note captures the “do it in order” steps we just agreed on. Safety rails stay; we only loosen throughput and make the LLM respond to measured performance.

### Phase 0 – Guardrails (unchanged)
- Keep `max_daily_loss_pct = 3%`, `max_daily_risk_budget_pct = 20%`.
- Keep stop-aware sizing, daily budget reset/anchor, and full risk telemetry (allocated vs actual at stop).

### Phase 1 – Let risk express (exploratory caps)
Goal: move mean daily usage toward a healthy band while guardrails hold and caps are not the dominant brake.

- Add an “exploratory” profile via env/CLI for test runs:
  - `STRATEGIST_PLAN_DEFAULT_MAX_TRADES=30` (or higher) and `STRATEGIST_PLAN_DEFAULT_MAX_TRIGGERS_PER_SYMBOL=40`.
  - Turn on `STRATEGIST_STRICT_FIXED_CAPS=true` so derived caps stay telemetry-only and do not shrink the plan caps.
  - Keep `max_daily_risk_budget_pct` in the 10–15% range; set `RISK_PROFILE_GLOBAL_MULTIPLIER=1.0` if you don’t want the rail silently scaled down.
  - Keep `max_position_risk_pct` in the 1–2% range and `max_daily_loss_pct=3`.
- Success on a 30-day slice: risk_budget_used_pct_mean in ~30–60% with fewer daily_cap/plan_limit blocks; blocks driven by risk_budget in bad regimes, not by static caps.
- If usage stays pinned low or caps still dominate, caps/env are not applied or the strategist is under-asking; fix caps first before tuning prompts.

### Phase 2 – Archetype weights driven by `rpr_actual`
- Define a simple health rule over a 14–30 day lookback (adjust thresholds if sample sizes are thin):
  - `Good`: trades ≥ N_min (LLM ≥ 5–10, baseline ≥ 10–15) and `rpr_actual` materially above baseline (e.g., ΔRPR ≥ +0.15).
  - `Neutral`: trades < N_min or `-0.15 ≤ ΔRPR ≤ +0.15`.
  - `Bad`: trades ≥ N_min and ΔRPR ≤ -0.15.
- Map health → caps/multipliers:
  - Good: full cap, risk multiplier ~1.05–1.2×.
  - Neutral: current caps, no multiplier.
  - Bad: cap cut (≤50%) or risk multiplier ~0.5–0.7×; for exits, keep but reduce eagerness.
- Implement via per-archetype multipliers (hour multipliers stay off until coverage is adequate).

### Phase 3 – Teach the LLM (strategist/judge) to use RPR + utilization
- Provide a performance snapshot in prompts (recent `rpr_actual`, trades) per archetype and per hour when available.
- Explicit instructions:
  - Favor archetypes with positive `rpr_actual` and enough samples.
  - Limit/justify archetypes with negative `rpr_actual`; cap reversal “probationary” until proven.
  - Target daily risk utilization band 30–60% once caps are correctly applied; if <25% for several days, increase trade frequency or per-trade sizing within existing risk rails.
  - Time-of-day: avoid initiating in hours with consistently negative `rpr_actual`; be more willing in positive hours (only after hour-level coverage is sufficient).

### Phase 4 – Longer, looser P1a and evaluate
- Run a longer window (e.g., full Feb 2021) with strict fixed caps, archetype-only multipliers, and RPR-aware prompts.
- Success metrics:
  - Usage: mean 30–60% with balanced distribution (not all >75%), few/no daily_loss hits.
  - Blocks: risk_budget/daily_loss appear only in bad regimes; daily_cap/plan_limit not constant brakes.
  - Archetype quality: sustained positive `rpr_actual` vs baseline for “good” buckets; reduced contribution from “bad”.
  - Time-of-day: only revisit hour-level after coverage thresholds are met.

### Phase 5 – Only then consider higher per-trade risk
- If the above looks good over 30–60 days and utilization is still low, experiment with `max_position_risk_pct` 3–4% **without** changing daily loss/budget rails; keep strict caps and RPR weighting in place.
