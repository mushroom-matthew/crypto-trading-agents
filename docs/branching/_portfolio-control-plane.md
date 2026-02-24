# Backlog: Portfolio Control Plane (Shared Allocation + Constraints)

## Status: Backlog - depends on per-instrument workflow architecture

## Problem Statement

The backlog design for per-instrument strategist workflows solves symbol-level plan
coupling, but it creates a new coordination problem: independent symbol workflows can
make locally valid decisions that are collectively invalid at the portfolio level.

Examples:

- BTC and ETH both request long risk at the same time and exceed portfolio heat.
- Three symbols select the same breakout template and create hidden correlation
  concentration.
- A symbol workflow replans into a higher-risk template without visibility into pending
  reservations from other symbols.
- The judge (or a portfolio safety rule) wants to derisk globally, but there is no
  single authority that can broadcast and enforce shared constraints consistently.

If each instrument owns its own strategist/judge loop, we need a shared deterministic
control plane for portfolio budgeting, exposure limits, and cross-symbol conflict
resolution.

## Vision

Introduce a portfolio-level coordinator that is authoritative for shared constraints,
while leaving symbol selection and template-specific plan generation inside each
`InstrumentStrategyWorkflow`.

High-level split:

- **Instrument workflow (symbol-local):** thesis, template selection, trigger plan,
  symbol-level judge feedback, signal generation.
- **Portfolio control plane (shared):** risk budget allocation, concentration caps,
  correlation/crowding controls, drawdown mode, global derisk constraints.

This preserves the per-instrument strategist benefits without regressing into a single
multi-symbol strategist call.

## Proposed Workflow Shape

```
UniverseScreenerWorkflow
  -> InstrumentStrategyWorkflow{symbol} (N symbols)
       -> submits PortfolioIntent{symbol, side, requested_risk, template_id, thesis_meta}
            -> PortfolioControlWorkflow (shared authority)
                 -> returns PortfolioConstraintEnvelope / AllocationDecision
       -> TriggerEngine evaluates with portfolio envelope applied
       -> orders/fills -> ExecutionLedgerWorkflow
                          -> signals PortfolioControlWorkflow (actual fills / releases)
```

Key invariant: portfolio-level constraints are computed once, centrally, and applied
uniformly across all active instrument workflows.

## Scope

1. Shared portfolio state model (cash, reserved risk, active risk, exposure by cohort)
2. Deterministic allocation / rejection decisions for symbol requests
3. Broadcast constraint envelopes to active instrument workflows
4. Reservation + release semantics tied to actual fills and position close
5. Global derisk / risk-off mode propagation
6. Telemetry for rejected or clipped symbol intents (why an otherwise-valid plan was
   constrained by portfolio rules)
7. Replayability: given the same intents and fills, portfolio decisions are identical

## Out of Scope

- Replacing symbol-local strategist or judge logic
- Tick-level trigger evaluation (still symbol-local)
- Execution adapter fill logic (stays in execution ledger / broker path)
- Learned portfolio optimizer / ML allocator (this is a deterministic coordinator)

## Proposed Contracts (Sketch)

### `schemas/portfolio_control.py`

```python
class PortfolioIntent(BaseModel):
    symbol: str
    side: Literal["long", "short", "flat"]
    requested_risk_abs: float
    requested_notional_abs: float | None = None
    template_id: str | None = None
    playbook_id: str | None = None
    regime_state: str | None = None
    timestamp: datetime
    correlation_bucket: str | None = None


class AllocationDecision(BaseModel):
    approved: bool
    approved_risk_abs: float
    decision_code: str  # approved_full / approved_clipped / rejected_heat / ...
    reasons: list[str]
    constraint_envelope: "PortfolioConstraintEnvelope"
    reservation_ttl_sec: int | None = None


class PortfolioConstraintEnvelope(BaseModel):
    portfolio_mode: Literal["normal", "defensive", "risk_off"]
    max_new_risk_abs: float
    max_additional_positions: int | None = None
    symbol_max_risk_abs: float | None = None
    size_multiplier_cap: float | None = None
    force_flatten_only: bool = False
```

### `workflows/portfolio_control_workflow.py`

- `request_allocation(intent)` -> `AllocationDecision`
- `on_fill_update(fill_event)` -> reserve/commit/release reconciliation
- `broadcast_constraints()` -> push updated envelopes after major portfolio events
- `set_portfolio_mode(mode, reason)` -> judge/safety override channel
- `get_portfolio_state()` -> query for debugging and audit

## Deterministic Rules (Initial)

Start simple and explicit before adding sophisticated correlation logic:

1. **Portfolio heat cap**: total active + reserved risk cannot exceed configured cap
2. **Per-symbol cap**: symbol risk cannot exceed configured max
3. **Cohort cap**: sum of symbols in the same cohort/template bucket capped
4. **Position count cap**: max concurrent positions
5. **Drawdown mode scaling**: tighten caps after drawdown thresholds
6. **Reservation TTL**: stale unfilled reservations expire deterministically

All decisions must emit a reason code and numeric context in telemetry.

## Integration Dependencies

Required before implementation:

1. `_per-instrument-workflow.md` (symbol workflows must exist first)
2. Runbook 39 (Universe Screener) in paper trading, surfacing multiple symbols
3. Runbook 46 validated (template routing correctness)
4. Runbook 47 validated (hard template binding) so symbol intents carry stable template IDs
5. Signal/Execution ledger telemetry sufficient to reconcile reservation vs actual fill risk

Recommended but parallelizable:

- Runbook 55 (symbol-local + cohort detector scopes) for better cohort/correlation bucketing
- Runbook 49 (snapshot contract) for consistent regime metadata in intents

## Open Questions

1. **Authority boundary**: Should the portfolio control plane only clip/reject new risk,
   or may it force symbol-level de-risk/flatten directly?
2. **Judge routing**: Does portfolio-level judge feedback target this workflow only, then
   get translated into envelopes for symbols?
3. **Reservation model**: Reserve on signal emission, order placement, or fill attempt?
4. **Correlation buckets**: Start with static mappings (BTC/L1/alts) or derive from
   rolling correlations/covariance?
5. **Conflict precedence**: If symbol judge says "increase conviction" and portfolio mode
   says defensive, which telemetry and action path records the override?

## Why Deferred

This should not be built before the per-instrument workflow split is validated in paper
trading. Otherwise we risk designing portfolio coordination around assumptions that will
change once real multi-symbol cadence, fan-out latency, and screener churn are observed.

Do not implement until:

- [ ] `_per-instrument-workflow.md` design questions are resolved
- [ ] Paper trading shows >= 3 concurrent active symbols with distinct templates
- [ ] Reservation/release telemetry requirements are confirmed from live paper runs
- [ ] Portfolio-level constraint policy is agreed (heat, cohort cap, drawdown modes)

## Acceptance Criteria (for eventual implementation)

- Portfolio allocation decisions are deterministic and replayable
- Symbol workflows receive and honor shared constraint envelopes
- Risk reservations reconcile to actual fills/releases without drift accumulation
- Telemetry explains every clipped/rejected allocation request
- A portfolio risk-off command can propagate to all active symbol workflows within one
  policy loop cycle

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-24 | Backlog runbook created - shared portfolio control plane for per-instrument workflows | Codex |
