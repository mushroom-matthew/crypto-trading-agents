# ADR: Portfolio Judge Routing and Shared Constraint Authority (Per-Instrument Workflows)

## Status

Proposed (draft) - 2026-02-24

## Context

`_per-instrument-workflow.md` proposes one `InstrumentStrategyWorkflow` per active symbol,
which fixes the current multi-symbol `StrategyRun` coupling (template conflicts, state
pollution, screener cadence mismatch). That backlog runbook leaves one core question open:

> With per-symbol workflows, does the judge evaluate per symbol, or does it still evaluate
> the portfolio and broadcast constraints to all active instrument workflows?

If this is left implicit, the implementation will likely recreate hidden coupling via
ad-hoc broadcasts or duplicate portfolio checks inside symbol workflows.

At the same time, the docs already point toward a layered design:

- symbol-local policy cadence should be driven by symbol-local events (Runbook 55)
- shared/cohort signals are advisory, not direct symbol policy triggers (Runbook 55)
- portfolio state/accounting remains shared (execution ledger)

We need a clear authority boundary for portfolio-level reasoning vs enforcement.

## Decision

Adopt a **three-layer routing model**:

1. **Symbol-local judge remains symbol-local**
   - Each `InstrumentStrategyWorkflow{symbol}` receives symbol-specific judge feedback.
   - Symbol judge feedback may change symbol stance, plan constraints, or sizing intent,
     but only within the active portfolio envelope.

2. **Portfolio-level reasoning is monitor/recommendation-oriented**
   - Portfolio-scope diagnostics run in a dedicated portfolio monitor / slow-loop layer.
   - This layer emits typed `PortfolioRecommendationEvent` outputs (for example:
     `defensive_mode`, tighten cohort cap, reduce position count cap).
   - It does not directly mutate symbol plans and does not broadcast raw judge text.

3. **Portfolio control plane is the sole authority for shared constraints**
   - A shared `PortfolioControlWorkflow` receives portfolio recommendations, live portfolio
     state, and symbol allocation intents.
   - It computes deterministic `PortfolioConstraintEnvelope` decisions and broadcasts those
     envelopes to active symbol workflows.
   - All shared risk/concentration/correlation caps are enforced here, not duplicated in
     each symbol workflow.

In short:

`Portfolio monitor/judge -> PortfolioControlWorkflow (deterministic translation/enforcement) -> InstrumentStrategyWorkflow{symbol}`

## Rationale

### Why not direct portfolio judge broadcasts to symbol workflows?

- Reintroduces coupling through side effects (one broadcast can churn all symbols).
- Harder to audit because raw recommendations are mixed with enforcement decisions.
- Encourages duplicate cap logic in symbol workflows ("just in case" checks).
- Makes replayability weaker (LLM outputs become enforcement authority).

### Why centralize enforcement in the control plane?

- Shared constraints are cross-symbol by definition and need one authoritative state.
- Deterministic replay and telemetry are easier when allocation/envelope decisions happen
  in one workflow.
- Symbol workflows stay focused on symbol-specific strategy quality and execution cadence.

## Detailed Rules

### 1. Authority Boundary

- **Symbol workflows own**: symbol thesis, template choice (subject to validated routing),
  trigger plan, symbol-level judge adaptation, local position state.
- **Portfolio control plane owns**: portfolio mode, heat caps, position-count caps,
  cohort/correlation caps, reservation/release accounting, global derisk envelopes.
- **Execution ledger owns**: fills, positions, cash/account truth; it signals state changes
  to the portfolio control plane.

### 2. Conflict Precedence (Required)

When symbol-local intent conflicts with portfolio constraints:

- portfolio envelope wins for execution authority
- symbol workflow logs a constrained/overridden decision reason
- telemetry records both:
  - symbol requested intent (pre-constraint)
  - portfolio-approved envelope/size (post-constraint)

This preserves diagnosability without allowing symbol loops to violate portfolio rules.

### 3. Routing of Portfolio Recommendations

Portfolio recommendations must be structured and typed. No free-text broadcast directly to
symbol workflows.

Examples:

- `set_portfolio_mode(defensive)`
- `tighten_cohort_cap(cohort="alts_l1", cap_risk_abs=...)`
- `reduce_max_new_positions(...)`
- `force_flatten_only=true` (temporary safety state)

The portfolio control plane translates these into per-symbol envelopes and broadcasts them.

### 4. Cadence Rules

- No per-tick portfolio judge/monitor loop.
- Portfolio monitor runs on:
  - major portfolio events (drawdown threshold breach, repeated allocation rejections)
  - daily monitoring cadence
  - weekly+ structural reflection cadence
- Symbol policy cadence remains symbol-local (event-driven + heartbeat).

### 5. Audit and Replay Requirements

Every portfolio-constrained symbol action should be reconstructable from:

- symbol intent event
- portfolio control state at decision time
- allocation/envelope decision
- execution result (if any)

This is required before live-capital use.

## Consequences

### Positive

- Preserves per-instrument strategist/judge separation.
- Prevents portfolio logic from leaking into every symbol workflow.
- Makes shared constraints deterministic and testable.
- Supports a monitor-first portfolio layer without reintroducing monolithic strategist calls.

### Costs / Tradeoffs

- Adds one more workflow/service boundary and messaging contracts.
- Requires careful reservation/release reconciliation.
- Requires explicit telemetry for "symbol wanted X, portfolio allowed Y" events.

## Alternatives Considered

### A. Keep one portfolio judge that broadcasts directly to all symbols

Rejected as primary design. It recreates cross-symbol coupling and makes enforcement too
LLM-dependent.

### B. Put full portfolio checks inside each symbol workflow

Rejected. Shared state becomes inconsistent and race-prone under concurrent symbol events.

### C. No portfolio-level reasoning; symbol-only forever

Rejected. Portfolio performance failure modes (concentration, correlation stacking,
drawdown velocity) cannot be handled adequately at symbol scope alone.

## Implementation Implications (Backlog Mapping)

- `_per-instrument-workflow.md`
  - Open question #4 is resolved by this ADR: use symbol-local judge plus portfolio
    control-plane envelopes, not direct portfolio judge broadcasts.
- `_portfolio-control-plane.md`
  - Implements deterministic shared constraint authority and broadcast envelopes.
- `_portfolio-monitor-and-reflection.md`
  - Implements portfolio diagnostics / recommendations feeding the control plane.

## Deferred Items (Follow-up ADRs or runbooks)

- Auto-apply vs manual approval of portfolio recommendations in live trading
- Correlation bucket methodology (static taxonomy vs rolling covariance)
- Reservation timing (signal vs order vs fill-attempt)
- Emergency authority path for forced flatten (control plane vs execution safety subsystem)

## Decision Checklist

- [x] Per-instrument strategist/judge separation preserved
- [x] Shared portfolio strategy/constraints have a single authority
- [x] No direct raw portfolio-judge broadcasts to symbol workflows
- [x] Conflict precedence defined (portfolio envelope wins)
- [ ] Backed by paper-trading operational evidence (pending)

## References

- `docs/branching/_per-instrument-workflow.md`
- `docs/branching/_portfolio-control-plane.md`
- `docs/branching/_portfolio-monitor-and-reflection.md`
- `docs/branching/55-regime-fingerprint-transition-detector.md`
- `docs/branching/54-reasoning-agent-cadence-rules.md`
