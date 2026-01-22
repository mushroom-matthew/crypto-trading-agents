## Repo Status Overview
This file summarizes current state, recent efforts, and outstanding roadmap items to help a new agent ramp quickly.

### Recent Efforts (Dec 2025)
- **Risk telemetry & actual risk-at-stop**: Trigger-level `rpr_actual` flows; timeframe/hour aggregation includes actual risk. Baseline and LLM pipelines now produce comparable daily reports and run summaries under `.cache/strategy_plans/<run_id>/`.
- **Baseline logging**: The non-LLM simulator logs fills using a unified schema (`baseline_strategy|1h`, `actual_risk_at_stop`, etc.) and writes daily/run reports like the LLM runs, enabling hour-quality and RPR comparisons.
- **Trigger compiler**: Unary ± and attribute/subscript access (e.g., `recent_tests[0].result`) are permitted to avoid LLM-plan compilation crashes.
- **Prompts**: Strategist prompt updated to consume `rpr_actual`, utilization bands, and performance snapshots; encourages risk expression in a 10–30% band while honoring budget/loss rails.
- **Action plans**: Added `docs/RISK_RPR_ACTION_PLAN.md` (phased plan to raise utilization, apply RPR weighting, integrate performance telemetry).
- **Risk-budget wiring fixes**: LLM path now separates planned `risk_used` vs. `actual_risk_at_stop`; budget usage is scaled correctly (no shrink from actual MAE/MFE). Strict caps flag (`STRATEGIST_STRICT_FIXED_CAPS`) added to keep configured caps from being overwritten by derived caps; derived caps are telemetry only when the flag is on.

### Roadmaps & Status
- **docs/ROADMAP.md**: Broad guidance; not reconciled with recent risk/RPR/baseline insights.
- **docs/backtesting_remediation_roadmap.md**: P0/P1 risk fixes largely done (daily loss anchor, budget reset, actual risk, block telemetry). Remaining: long-window validation, plan-limit tuning, RPR-driven cap multipliers.
- **docs/RISK_AUDIT.md → RISK_REMEDIATION_PLAN.md → RISK_RPR_ACTION_PLAN.md**: P0/P1 implemented; P2/P3 telemetry/reporting refinements and performance-guided strategy shaping still open.
- **docs/HEDGING_AND_EDGE_PLAN.md**, **docs/FACTOR_EXPOSURE.md**: Factor auto-fetch exists; meaningful integration into prompts/judge not yet done.
- **docs/MARKET_STRUCTURE_TELEMETRY.md**: Telemetry/module in place; partial prompt integration; further wiring into strategy exports/LLM context pending.
- **docs/MARKET_ANALYTICS_AUDIT.md**: Follow-ups hinge on enhanced telemetry (vol regime, liquidity, structural edges).
- **docs/dashboard.md**: Idea-level; no recent changes.
- **docs/phase2_risk_expression_plan.md**: Needs update to incorporate baseline-relative learning and time-of-day quality filters.

### Notable Gaps / TODOs
- **Caps still the dominant brake**: Plan/daily caps continue to block most attempts in recent runs (e.g., rpr_weighted4/5/6), muting RPR multipliers. Strict cap flag exists but must be set in the env for it to take effect.
- **Rail shrink via profile multiplier**: Global multiplier defaults to 0.75, silently reducing a 10% rail to ~7.5% unless overridden.
- **Baseline vs. LLM**: Archetype-level RPR improved in some runs (rpr_weighted3) but regressed when caps tightened; current deltas hover neutral/negative because throughput is too low.
- **Hour-level noise**: Hour multipliers are either disabled or based on thin samples; some hour_quality entries show risk_used_abs with zero trades (allocation vs. execution hour semantics need clarity).
- **Exit archetypes**: Some exits show negative `rpr_actual` in both LLM and baseline; need suppressive caps or redesign.
- **Session/time-of-day filters**: Hour-level `rpr_actual` exists; prompts/caps not using it yet (hour multipliers currently off due to low coverage).
- **Factor/hedging integration**: Factors fetched but not used in risk, plan generation, or regime logic.

### Where to Look
- **Telemetry**: `.cache/strategy_plans/<run_id>/daily_reports/` and `run_summary.json` (baseline and LLM share schemas).
- **Baseline backtests**: `backtesting/simulator.py` (synthetic triggers, risk/pnl, RPR).
- **LLM backtests**: `backtesting/llm_strategist_runner.py`; prompt at `prompts/llm_strategist_prompt.txt`.
- **Risk logic**: `agents/strategies/risk_engine.py`, daily/loss guards in `backtesting/llm_strategist_runner.py`, aggregation in `backtesting/reports.py`.
- **Data loading**: `backtesting/dataset.py`, `data_loader/api_loader.py`, `data_loader/caching.py`.

### Suggested Next Moves for a New Agent
1. **Enforce fixed caps via env**: Run with `STRATEGIST_STRICT_FIXED_CAPS=true` and explicit `STRATEGIST_PLAN_DEFAULT_MAX_TRADES`/`...MAX_TRIGGERS_PER_SYMBOL` (e.g., 30/40) so derived caps stay telemetry-only; set `RISK_PROFILE_GLOBAL_MULTIPLIER=1.0` if you want the full budget rail.
2. **Archetype-only pass to rebuild RPR delta**: Use archetype multiplier (btc ~1.08), disable hour multipliers until trade counts per hour are ≥10/15 (LLM/baseline), and re-run Feb 2021 with loosened caps. Target `risk_budget_used_pct_mean` ~30–60% and higher trade_count_mean.
3. **Inspect caps vs. budget in the new run**: Block totals should show reduced `plan_limit`/`daily_cap` dominance; budget blocks present but not primary. Verify plan_limits in daily reports stay at the configured caps.
4. **Reintroduce hour multipliers only after coverage**: Once archetype RPR > baseline with stable throughput, recompute rpr_comparison, apply GOOD/BAD hours with real min-trade gating, and rerun.
5. **Clarify hour telemetry**: Decide whether hour_quality reflects allocation hour vs. execution hour; split metrics or document to avoid “0 trades, non-zero risk” confusion.
6. **Exit/reversal tuning remains**: Suppress exit archetypes with negative `rpr_actual` or redesign them; keep using daily flatten as backstop.
