# AI-Led Portfolio Session Plan V0 (Draft)

## Goal
Replace user-selected fixed strategy templates with an AI-led portfolio manager that:
- Scans market universe continuously.
- Selects 5-10 instruments with meaningful opportunity.
- Manages the portfolio to target 5-10 round-trip trades per 24h session.
- Optimizes for risk-adjusted PnL and explicit hypothesis tracking, not symbol loyalty.

## Motivation (from current live-paper behavior)
- Strategy-template selection can mismatch actual regime.
- Trigger systems can fail closed (zero triggers or blocked triggers) even when market is active.
- Session outcomes depend too heavily on static upfront choices (single strategy profile, fixed symbol basket).

## Product Requirements
### Functional
- Universe scan every 5-15 minutes across tradable symbols.
- Opportunity ranking model chooses 5-10 symbols per session window.
- Portfolio manager allocates risk budget dynamically across selected symbols.
- Trigger generation and execution policy can adapt intraday as regime shifts.
- Session objective enforces target cadence: 5-10 completed round trips / 24h.
- Full hypothesis ledger for each trade thesis, expected invalidation, and post-trade attribution.

### Risk and Controls
- Hard caps: max daily loss, max concurrent exposure, per-symbol concentration limits.
- Soft caps: dynamic de-risking when drawdown/volatility increases.
- Block taxonomy must separate:
  - valid risk blocks (desired),
  - quality-gate false negatives (undesired),
  - data-quality/infra blocks (undesired).

### Observability
- Real-time metrics:
  - opportunities detected, selected, and rejected,
  - trigger fire rate, block rate by reason,
  - round-trip completion count,
  - win rate, expectancy, realized R multiple,
  - hypothesis confidence calibration.

## Proposed Control Plane

## 1. Market Opportunity Scanner
Inputs:
- Volatility expansion/compression, structure quality, liquidity, spread, trend persistence, anomaly markers.

Output:
- `OpportunityCard` per symbol with score, confidence, and expected holding horizon.

Initial scoring model (required before Phase 1 implementation):
- Normalize each component to `[0, 1]`.
- Status: V0 placeholder. Weights are provisional; component definitions and calibration are finalized in Phase 1 design.
- Define:
  - `vol_edge`: expansion/compression opportunity score
  - `structure_edge`: support/resistance + structural clarity score
  - `trend_edge`: persistence/alignment score
  - `liquidity_score`: depth/volume quality
  - `spread_penalty`: normalized transaction-cost penalty
  - `instability_penalty`: anomaly/latency/data-quality penalty
- Weighted score:
  - `opportunity_score = 0.28*vol_edge + 0.24*structure_edge + 0.18*trend_edge + 0.20*liquidity_score - 0.07*spread_penalty - 0.03*instability_penalty`
- Expected raw score range:
  - theoretical min = `-0.10` (all positive components 0, penalties 1)
  - theoretical max = `0.90` (all positive components 1, penalties 0)
- Normalized score (for consumers expecting `[0, 1]`):
  - `opportunity_score_norm = clamp((opportunity_score + 0.10) / 1.00, 0, 1)`
- Ranking output must include full component breakdown for auditability (not just final score).

## 2. Session Portfolio Planner (AI)
Responsibilities:
- Select top 5-10 symbols from opportunity cards.
- Assign per-symbol trade budget and risk budget.
- Define preferred playbook style per symbol (trend/breakout/reversion) as a derived decision, not user-selected template.

Output:
- `SessionIntent`:
  - selected symbols,
  - planned trade cadence range,
  - risk budget map,
  - thesis map.

## 3. Trigger Synthesizer
Responsibilities:
- Generate trigger packs per selected symbol from current context.
- Must pass deterministic compile-time invariants:
  - target semantics present when `target_hit` logic is used,
  - resolvable stop/target anchors,
  - direction and category consistency.

Output:
- `ValidatedTriggerPack` with explicit fallback policy metadata.

## 4. Execution and Recovery Layer
Responsibilities:
- Candidate pre-validation before in-bar priority arbitration.
- Retry/fallback logic for transient market data outages.
- Graceful degrade mode: pause entries but preserve position protection when data feed unstable.

## 5. Hypothesis and Learning Ledger
For each round trip:
- thesis text,
- expected market condition,
- invalidation condition,
- result class,
- attribution tags (model quality vs execution vs market regime).

Feeds next-cycle symbol selection and risk weighting.

## Session Objective Framework (24h)
Hard target:
- 5-10 completed round trips.

Constraint hierarchy:
1. Safety constraints (loss/exposure) never violated.
2. Quality floor for entries.
3. Cadence objective (5-10 round trips) achieved opportunistically.

If cadence is below target:
- widen search breadth and instrument coverage only within risk guardrails.
- prefer smaller risk allocations over lowering hard safety constraints.
- maintain invariant quality gates (no relaxation of target-anchor validity, stop validity, or minimum R:R thresholds).

If cadence is above target:
- throttle new risk and reserve budget for high-conviction setups.

Anti-gaming clause (hard invariant):
- Cadence governor must never lower validation quality to hit trade-count targets.
- Prohibited adaptations:
  - reducing minimum R:R thresholds,
  - permitting unresolved/implicit target anchors,
  - bypassing stop-anchor requirements,
  - downgrading compile-time trigger validity checks.
- Permitted adaptations:
  - increasing symbol breadth,
  - reallocating risk budgets,
  - reducing per-trade size,
  - adjusting opportunity-ranking cutoff while keeping quality invariants fixed.

## Decision Policies to Standardize
- Symbol selection policy: ranked utility = edge score * execution feasibility * risk cost.
- Replacement policy: when a held symbol loses opportunity rank, rotate into stronger candidate.
- Trigger block policy: classify and react differently for
  - risk-valid blocks,
  - configuration-quality blocks,
  - data/infra blocks.

## Implementation Phases (iterative)

### Phase 0: Instrumentation and invariants
- Add explicit block reason taxonomy quality flags.
- Enforce target/anchor invariants pre-execution.
- Add timeout resilience metrics and alerts.

Exit criteria:
- `no_target_rr_undefined` and `target_price_unresolvable` reduced to near-zero in controlled runs.
- No session failures attributable to transient `fetch_current_prices_activity` ScheduleToClose timeouts in the controlled run window.

Phase gate (hard dependency):
- Phase 1 must not begin until Phase 0 exit criteria are met in live controlled re-runs across parallel sessions.

### Phase 1: Opportunity scanner + ranking
- Build `OpportunityCard` generator and ranking API.
- Expose scanner telemetry in UI.

Exit criteria:
- Stable top-10 opportunity list updates without overload.

### Phase 2: AI portfolio planner
- Introduce `SessionIntent` artifact and symbol/risk allocation decisions.
- Decouple from user-selected static strategy prompt.

Exit criteria:
- Session starts with AI-selected 5-10 symbols and explicit risk map.

### Phase 3: Objective-driven trigger synthesis
- Generate triggers from `SessionIntent` with deterministic post-validation.
- Add cadence governor to move toward 5-10 round trips/day.

Exit criteria:
- Median sessions produce 5-10 completed round trips under risk limits.

### Phase 4: Hypothesis feedback loop
- Trade outcome attribution updates scanner/planner priors.
- Add confidence calibration and playbook performance decay rules.

Exit criteria:
- Improved risk-adjusted PnL and reduced block-induced dead sessions.

## Success Metrics
Primary:
- Round trips per 24h: median in [5, 10].
- Risk-adjusted return: improvement vs current template baseline.
- Block quality: >= 90% of blocks are risk-valid, <= 10% are avoidable quality/infra blocks.

Secondary:
- Time in dead session state (no actionable triggers) reduced by >= 80%.
- Failure recovery: transient data timeout does not kill session workflow.

## Open Design Questions
- Universe size and refresh cadence tradeoff vs infrastructure cost.
- Whether to keep optional user constraints (risk profile only) while removing strategy template selection.
- How aggressively to rotate symbols intraday to hit cadence targets without overtrading.
- Minimum explainability payload needed for every AI selection decision.

## Suggested Next Review
- Confirm acceptance criteria for "5-10 round trips" (strict vs soft target).
- Confirm anti-gaming invariants for cadence governor as non-negotiable requirements.
- Validate OpportunityCard scoring weights and component definitions.
- Finalize invariant set for trigger validity before execution.
- Approve Phase 0 controlled re-run plan and enforce Phase 0 -> Phase 1 gate.
