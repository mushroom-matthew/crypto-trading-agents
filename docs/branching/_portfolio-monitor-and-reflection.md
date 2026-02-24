# Backlog: Portfolio Monitor and Reflection Layer

## Status: Backlog - pairs with per-instrument workflows and portfolio control plane

## Problem Statement

Per-instrument strategist workflows improve symbol-specific decision quality, but local
quality does not guarantee portfolio quality. A system can have "good" symbol plans and
still perform poorly due to:

- correlation stacking (same thesis expressed through multiple symbols)
- concentration drift (too much risk in one cohort/template)
- churn/overtrading across symbols despite acceptable per-symbol cadence
- portfolio drawdown acceleration not visible inside any single symbol loop
- regime mismatch between symbol-level wins and portfolio-level allocation behavior

The existing docs describe symbol-local reasoning cadence and cohort/shared regime signals,
but there is no dedicated runbook for a portfolio-level monitor that aggregates outcomes,
diagnoses shared failure modes, and produces monitor-only alerts or shared recommendations.

## Vision

Introduce a portfolio-level monitor/slow-loop workflow that consumes aggregate telemetry
and produces portfolio diagnostics without collapsing back into a single multi-symbol
strategist call.

Principles:

- **Monitor first**: this layer should be observation/diagnostics-first, not direct
  per-tick intervention.
- **Cadence separated**: no per-bar strategist/judge calls; run on policy heartbeat and
  slower scheduled windows.
- **Shared recommendations**: output portfolio-level guidance/constraints that can be
  consumed by a dedicated portfolio control plane or translated into broadcasts.
- **Auditability**: every portfolio alert/recommendation cites metrics and evidence.

## Relationship to Existing Runbooks

- `_per-instrument-workflow.md`: defines symbol-local strategist workflows and notes the
  unresolved question of portfolio-level judge routing.
- `_portfolio-control-plane.md`: enforces deterministic shared constraints (allocator).
- `54-reasoning-agent-cadence-rules.md`: defines event-driven policy loop + slow loop cadence.
- `55-regime-fingerprint-transition-detector.md`: defines symbol-local primary detector and
  cohort/shared detector as advisory cross-symbol signal.
- `50-dual-reflection-templates.md`: provides reflection template structure; this backlog
  runbook applies it at portfolio scope.

## Proposed Workflow Shape

```
ExecutionLedgerWorkflow + Signal/Outcome telemetry + Regime detectors + Playbook stats
    -> PortfolioMonitorWorkflow (aggregate diagnostics)
         -> PortfolioMonitorReport (monitor-only by default)
         -> optional PortfolioRecommendationEvent (shared constraints / mode hints)
              -> PortfolioControlWorkflow (enforce) and/or operator UI
```

Key invariant: portfolio monitoring informs shared policy, but does not directly trigger
symbol strategist reruns from tick noise.

## Scope

1. Portfolio diagnostics aggregation across all active symbols
2. Slow-loop reflection templates for portfolio-level review (daily/weekly)
3. Monitor-only alerts for concentration, drawdown, correlation, churn, and crowding
4. Shared recommendation events (e.g., defensive mode, cohort cap tightening)
5. Telemetry linking recommendations to evidence windows and confidence
6. Evaluation gates for when samples are insufficient (emit monitor-only / no action)

## Out of Scope

- Symbol-level trigger changes or plan generation
- Direct order placement
- Replacement of execution ledger accounting
- Learned optimizer / automated rebalancer (deterministic and evidence-based only)

## Suggested Metrics / Diagnostics

At minimum, monitor:

1. **Portfolio heat / reserved heat / active heat**
2. **Concentration by symbol, template, playbook, and cohort**
3. **Correlation stacking score** (same-direction exposure in correlated buckets)
4. **Drawdown velocity** and drawdown regime state
5. **Turnover / churn rate** across symbols
6. **Realized vs expected profile** (holding time, MAE/MFE, slippage drift)
7. **Idle capital ratio** vs screener opportunity rate
8. **Failure-mode recurrence** from memory/reconciliation clusters

Each alert must include threshold, observed value, lookback window, and affected symbols.

## Suggested Outputs (Monitor-First)

### `PortfolioMonitorReport`

- summary metrics
- top risk concentrations
- anomaly alerts
- sample sufficiency flags
- recommended follow-up actions (monitor-only if insufficient evidence)

### `PortfolioRecommendationEvent` (optional)

- `recommended_portfolio_mode` (`normal` / `defensive` / `risk_off`)
- cap adjustments (heat/cohort/position count)
- confidence / sample size metadata
- expiry / review time
- evidence references (signal ledger windows, drawdown stats, regime context)

## Cadence (Initial)

- **Event-driven (limited):** major portfolio events only (drawdown threshold breach,
  allocation rejection spike, correlation cap breach)
- **Daily:** monitoring summary + calibration diagnostics
- **Weekly+:** structural reflection (playbook/crowding/regime allocation patterns)

No per-tick or per-bar portfolio strategist/judge loop.

## Integration Dependencies

Required before implementation:

1. `_per-instrument-workflow.md` (multi-symbol symbol-local loops)
2. `_portfolio-control-plane.md` (target for shared recommendations/enforcement)
3. Runbook 43 (signal ledger + outcome reconciler) for evidence substrate
4. Runbook 48 (research budget) for playbook evidence separation

Recommended:

- Runbook 50 (dual reflection templates)
- Runbook 51 (memory retrieval/failure-mode clustering)
- Runbook 54 (cadence rules)
- Runbook 55 (cohort/shared regime detector signals)

## Open Questions

1. **Portfolio judge vs monitor**: Is this purely a monitor service, or a portfolio-scope
   judge with structured actions routed to the control plane?
2. **Operator authority**: Should portfolio recommendations auto-apply in paper trading
   only, with manual approval for live?
3. **Confidence gating**: What minimum sample sizes/windows are required before a
   portfolio recommendation can tighten constraints automatically?
4. **Conflict handling**: If symbol-level metrics look healthy but portfolio monitor says
   derisk, how is the override communicated and audited?

## Why Deferred

This layer is only meaningful after the system is actually running multiple concurrent
instrument workflows. Before that, portfolio alerts are synthetic and risk overfitting the
monitor to imagined failure modes.

Do not implement until:

- [ ] `_per-instrument-workflow.md` is validated enough to produce concurrent symbol activity
- [ ] `_portfolio-control-plane.md` authority boundary is defined
- [ ] Signal/outcome telemetry supports portfolio aggregation without schema gaps
- [ ] Cadence and confidence gates are agreed (monitor-only vs auto-apply)

## Acceptance Criteria (for eventual implementation)

- Portfolio monitor emits deterministic, evidence-backed reports on schedule
- Monitor-only mode is default when sample sufficiency gates fail
- Shared recommendations are explicit, typed, and auditable
- No tick-level portfolio loop causes symbol strategist churn
- Operator can inspect why a portfolio recommendation was made (metrics + windows)

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-24 | Backlog runbook created - portfolio-level monitor/reflection layer for per-instrument strategy architecture | Codex |
