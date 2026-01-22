# Continuous Policy Pivot Plan (Phased, Reversible)

This plan sequences correctness and observability fixes before introducing a continuous policy engine. It tightens acceptance criteria, defines Phase 1 interfaces up front, and prevents predictable failure modes. The pivot remains reversible by keeping the policy engine deterministic and isolating ML as a plug-in.

## Principles

- Correctness and auditability are prerequisites for any policy/model changes.
- Phase 1 introduces a deterministic policy engine only; ML is deferred to Phase 2.
- Interfaces and telemetry are defined early to prevent scope creep.

## Scope and Sequencing

1) Phase 0: fix emergency exits, persistence, and replan churn.  
2) Phase 1: introduce deterministic policy engine using a proxy p_hat.  
3) Phase 2: optional ML-based p_hat, gated by calibration and trading metrics.

## Phase 0: Correctness + Observability (Prerequisites)

Status:
- A) Emergency-exit semantics: complete (tests: `tests/test_trigger_engine.py`, `tests/test_execution_engine.py`)
- B) Backtest persistence: complete (DB persistence + block_events + DB-first Ops API reads)
- C) No-change replan guard: not started

### A) Emergency-exit semantics (unambiguous)

Requirements:
- Emergency exits must express `target_weight := 0` at the policy boundary, even before Phase 1.
- The trigger engine emits an `ExitIntent(emergency=True)` that the execution layer interprets as a flatten request.
- Emergency exits route through exit guards (same-bar/min-hold).

Acceptance:
- Emergency exits emit distinct block-event reasons:
  - `emergency_exit_veto_same_bar`
  - `emergency_exit_executed`
- Emergency exits include a cooldown recommendation in the decision record (even if applied later).

### B) Backtest persistence: source-of-truth discipline

Requirements:
- Backtest runner writes the following to Postgres for every run:
  - `plan_log`
  - `bar_decisions` (bar-by-bar decisions)
  - `fills/trades`
  - `enforcement/overrides`
- Downstream tooling reads from Postgres by default; container artifacts are debug-only.

Acceptance:
- A backtest run is invalid if any of `results_summary`, `plan_log`, or `bar_decisions` are missing in DB.

### C) No-change replan guard (precise definition)

Define "unchanged" as:
- Trigger set identical (IDs + parameters)
- Risk limits identical
- Policy config identical (once Phase 1 exists)
- Only non-material changes present (timestamps, narration, formatting)

Acceptance:
- Metrics recorded:
  - `replan_rate_per_day`
  - `no_change_replan_suppressed_count`

## Phase 1: Deterministic Policy Engine (No ML)

### A) Policy Engine interface (define now)

Single function signature that survives Phase 2:

Inputs:
- `p_hat` (float in [0, 1])
- `vol_hat` (float; annualized or per-bar, consistent with `vol_target`)
- `policy_config` (tau, vol_target, bounds, etc.)
- `risk_overrides` (stand_down, caps)
- `position_state` (current_weight, equity)

Outputs:
- `target_weight_raw` (before caps/overrides)
- `target_weight_final` (after caps/overrides)
- `reasons[]` (why changed or overridden)

Acceptance:
- Every bar produces a complete decision record, even if no trade.

### B) Proxy p_hat (least-dangerous mapping)

Use a rule-based proxy for Phase 1 to avoid LLM confidence bias.

Example:
- `signal = tanh(k * zscore(momentum))`
- `p_hat = 0.5 * (signal + 1)`

LLM remains a controller of parameters and allowlists only.

### C) Delta-weight execution (idempotent, anti-churn)

Requirements:
- Minimum trade notional threshold.
- Rebalance band: ignore |delta_w| <= epsilon_w.
- Cooldown after execution (1-2 bars) to prevent flip-flop.
- Prefer post-only; allow taker fallback for exit-only paths.

Acceptance:
- Metrics captured:
  - `turnover`
  - `avg_hold_time`
  - `trade_count`
  - `cancel_count`

### D) Stand_down (stateful and auditable)

Stand_down is a state, not a boolean:
- `stand_down_until_ts`
- `stand_down_reason`
- `stand_down_source` (judge, ops, risk engine)

Acceptance:
- Every stand_down decision is recorded in bar_decisions with reason/source.

## Phase 2: ML-based p_hat (Optional, gated)

### A) Baseline before TCN

Train and compare a baseline model before TCN:
- Logistic regression or small MLP
- Gradient boosting on engineered features

If baseline matches TCN, keep the simpler model.

### B) Calibration + trading metrics gate promotion

Required metrics:
- Brier score
- Expected Calibration Error (ECE) or reliability curves
- Trading proxies: churn, slippage, realized PnL volatility

Promotion rule example:
- Do not ship model p_hat if Brier improves but churn or slippage worsens materially.

### C) Model artifact metadata (sticky horizon)

Store with model:
- `timeframe`
- `horizon_bars`
- `label_definition`
- `feature_version`

### Data and compute notes

- Data: multi-regime OHLCV; funding/spread/VWAP where possible.
- Minimum: 6–12 months per timeframe.
- Compute: CPU ok for 15m/1h and few symbols; GPU recommended for 1m or multi-asset.

## Critical Decision (lock early)

- Choose horizon/cadence for v1: 15m recommended unless microstructure edge is proven.

## Stop/Go Gates

Gate 0 -> 1:
- Emergency exits correct and auditable.
- DB persistence complete with `results_summary`, `plan_log`, `bar_decisions`.
- No-change replan guard active with metrics.

Gate 1 -> 2:
- Policy engine stable: turnover, trade count, drawdown within acceptable bands vs baseline.

## Deliverables to Unblock Phase 0/1

1) `PolicyConfig` schema (strategist/judge contract)
2) `DecisionRecord` schema (bar-level telemetry)
3) Override taxonomy (block event reasons and stand_down sources)
