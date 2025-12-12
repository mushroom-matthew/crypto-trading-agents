## Repo Status Overview
This file summarizes current state, recent efforts, and outstanding roadmap items to help a new agent ramp quickly.

### Recent Efforts (Dec 2025)
- **Risk telemetry & actual risk-at-stop**: Trigger-level `rpr_actual` flows; timeframe/hour aggregation includes actual risk. Baseline and LLM pipelines now produce comparable daily reports and run summaries under `.cache/strategy_plans/<run_id>/`.
- **Baseline logging**: The non-LLM simulator logs fills using a unified schema (`baseline_strategy|1h`, `actual_risk_at_stop`, etc.) and writes daily/run reports like the LLM runs, enabling hour-quality and RPR comparisons.
- **Trigger compiler**: Unary ± and attribute/subscript access (e.g., `recent_tests[0].result`) are permitted to avoid LLM-plan compilation crashes.
- **Prompts**: Strategist prompt updated to consume `rpr_actual`, utilization bands, and performance snapshots; encourages risk expression in a 10–30% band while honoring budget/loss rails.
- **Action plans**: Added `docs/RISK_RPR_ACTION_PLAN.md` (phased plan to raise utilization, apply RPR weighting, integrate performance telemetry).

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
- **Risk usage low**: Exploratory runs still ~2–5% mean daily usage; daily/plan limits are primary brakes; LLM conservatism persists. Baseline shows higher trade frequency and clearer hour-level RPR.
- **LLM vs. baseline**: Baseline outperforms LLM; LLM should weight archetypes relative to their own `rpr_actual` and baseline `rpr_actual` for the same hours/archetypes.
- **Baseline-relative calibration**: Not implemented—no Good/Neutral/Bad classification vs. baseline, no cap adjustments, no use of baseline hour-quality to filter bad hours.
- **Exit archetypes**: Some exits show negative `rpr_actual` in both LLM and baseline; need suppressive caps or redesign.
- **Session/time-of-day filters**: Hour-level `rpr_actual` exists; prompts/caps not using it yet.
- **Factor/hedging integration**: Factors fetched but not used in risk, plan generation, or regime logic.
- **LLM plan pacing**: Strategist under-proposes entries vs. baseline; utilization targets unmet even when caps allow.

### Where to Look
- **Telemetry**: `.cache/strategy_plans/<run_id>/daily_reports/` and `run_summary.json` (baseline and LLM share schemas).
- **Baseline backtests**: `backtesting/simulator.py` (synthetic triggers, risk/pnl, RPR).
- **LLM backtests**: `backtesting/llm_strategist_runner.py`; prompt at `prompts/llm_strategist_prompt.txt`.
- **Risk logic**: `agents/strategies/risk_engine.py`, daily/loss guards in `backtesting/llm_strategist_runner.py`, aggregation in `backtesting/reports.py`.
- **Data loading**: `backtesting/dataset.py`, `data_loader/api_loader.py`, `data_loader/caching.py`.

### Suggested Next Moves for a New Agent
1. **Longer exploratory LLM run**: e.g., full Feb 2021 with higher caps per `RISK_RPR_ACTION_PLAN.md`. Inspect utilization, block_totals, and `rpr_actual` stability across archetypes/hours.
2. **Baseline-relative weighting**: For each archetype, combine LLM `rpr_actual` and baseline `rpr_actual` to classify Good/Neutral/Bad; adjust daily caps or risk multipliers accordingly.
3. **Prompt snapshots**: Feed “performance + baseline snapshots” into strategist/judge prompts to favor archetypes/hours that outperform baseline; target 10–30% utilization while respecting rails.
4. **Exit/reversal tuning**: Clamp low-quality exit/reversal families; let trend entries express more risk where `rpr_actual` is consistently positive.
5. **Data coverage**: Ensure OHLCV spans the full requested window; enable ccxt fetch/factors for multi-month runs.
