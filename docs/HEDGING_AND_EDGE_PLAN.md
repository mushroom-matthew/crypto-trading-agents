# Plan: Hedging, Edge Tests, and Telemetry Upgrades

This roadmap breaks the feedback into PR-sized slices with owners, entry points, and success criteria. Each PR is independent and should ship with tests + docs + prompt notes.

## PR 1 — Factor Exposure Analysis
- Path: `analytics/factors.py`
- Tasks:
  - Load crypto factor proxies (BTC dominance, ETH/BTC ratio, total market cap, alt/mid-cap breadth).
  - Rolling regression: asset returns ~ market factor → betas, residual vol.
  - Expose API: `compute_factor_loadings(frames_by_symbol, factors_df, lookback)` → betas + idiosyncratic volatility.
- Outputs: JSON/telemetry block keyed by symbol/timeframe; unit tests with synthetic data; doc stub.
  - Status: **in progress** (module + tests added; telemetry hookup pending).

## PR 2 — Auto Hedge / Neutralization
- Paths: `judge_agent_client.py` (policy), CLI flag `--auto-hedge market`.
- Tasks:
  - Use factor betas to scale/offset positions (target beta ≈ 0 or vol-balanced).
  - Judge applies size dampening or hedge legs when flag is on.
  - Telemetry: record target beta, achieved beta, hedge size.

## PR 3 — Fast Statistical Edge Test
- Path: `backtesting/quick_test.py`.
- Tasks:
  - Sample historical windows, evaluate trigger logic without full sim.
  - Return mean/median, t-test p-value, bootstrap CI.
  - Cache results per trigger/timeframe; unit tests on synthetic returns.

## PR 4 — Judge Rejection Rule for No Edge
- Paths: judge prompt + enforcement hook.
- Tasks:
  - If quick-test p-value > 0.3 or median return ≤ 0 → veto/size slash trigger.
  - Log edge metrics per trigger in daily reports; add prompt text.

## PR 5 — Strategy Comparison Module
- Path: `analytics/compare_strategies.py`.
- Tasks:
  - Load exported strategy JSONs; compute Sharpe/Sortino/vol/corr.
  - Blend portfolios and rank by risk-adjusted return.
  - CLI helper to print SKFolios-style table; unit tests with fixtures.

## PR 6 — Strategist Feedback Loop
- Paths: `strategist_plan_service`, prompt additions.
- Tasks:
  - Surface trigger RPR, correlations, worst DD, win/loss distribution to LLM context.
  - Summaries stored in `strategy_export.json` for replay.

## PR 7 — Drawdown Telemetry Extension
- Paths: `backtesting/llm_strategist_runner.py`, reports module.
- Tasks:
  - Compute rolling 30d/90d DD, DD duration, recovery time.
  - Emit per-day in reports and run summary.

## PR 8 — Drawdown-Aware Risk Knob
- Paths: judge policy + risk engine.
- Tasks:
  - If rolling DD > threshold, cut position sizes by Y%.
  - Configurable thresholds via risk config/CLI; telemetry of applied cuts.

## PR 9 — Monte Carlo VaR Module
- Path: `analytics/var.py`.
- Tasks:
  - Simulate return paths (use vol + corr), compute 95/99% VaR.
  - Unit tests on deterministic seeds.

## PR 10 — VaR-Guided Risk Adjustment
- Paths: judge policy + CLI flag.
- Tasks:
  - Reduce risk until projected next-day VaR < max_daily_loss_pct.
  - Log VaR inputs/outputs per day.

## PR 11 — QuantStats Tear Sheets
- Paths: `backtesting/reports/tear_sheet.py`.
- Tasks:
  - Generate HTML/PDF after backtests (return curve, DD chart, rolling Sharpe, monthly heatmap, best/worst trades, return distribution).
  - Flag: `--save-tear-sheet local|s3` (PR 12 adds upload).

## PR 12 — Tear Sheet Upload
- Paths: CLI + storage client.
- Tasks:
  - Optional upload to S3/GCS; store URI in run summary.

## PR 13 — Sortino Ratio Everywhere
- Paths: `analysis/performance_metrics.py` (or metrics module), reports.
- Tasks:
  - Downside deviation, rolling Sortino; include in daily/run summaries, strategy exports.

## PR 14 — Prompt Extension for Sortino Preference
- Paths: strategist/judge prompts.
- Tasks:
  - Explicitly prefer triggers with high Sortino over Sharpe when available; hook in context fields.

## Cross-Cutting (Telemetry & UX)
- Telemetry unification: add `daily_metrics` block with sharpe, sortino, var_95/99, rolling DD, factor betas, edge scores, tear_sheet_path.
- Backtesting CLI “risk dashboard”: verbose mode prints regime, risk limits, VaR, expected DD, factor exposure, quick-test edge results, trigger evaluation.
- Planner API: `strategist_request("factor_exposure" | "edge_test" | "var_forecast")` returning structured data; judge brokers responses.
