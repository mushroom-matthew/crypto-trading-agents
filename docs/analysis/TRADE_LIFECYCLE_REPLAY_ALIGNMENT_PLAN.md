# Trade Lifecycle Replay and Pattern Retrieval Alignment Plan

_Drafted: 2026-04-24_

This document defines a complementary workstream for trade lifecycle provenance,
queryability, replay, and pattern retrieval. It is intentionally not a parallel
roadmap to the main phase plan. Its purpose is to strengthen the evidence,
memory, and audit foundations required by the existing roadmap in:

- `docs/analysis/AI_LED_PORTFOLIO_PHASE_PLAN.md`
- `docs/analysis/AGENTIC_SOTA_ALIGNMENT_PLAN.md`
- `docs/analysis/PROMPT_AUDIT_2026-04-23.md`

The immediate user need behind this workstream is:

1. query all entry triggers that actually fired and opened positions
2. pair them with the associated exit that closed the position
3. replay the trade with its full setup, signal, fill, and event chronology
4. backtest or retrieve similar trigger patterns later

---

## 1. Why This Is a Companion Workstream

This plan should be read as an enabling layer for existing runbooks, not as a
replacement or reorder of them.

### 1.1 Alignment with the Full Phase Plan

The main dependency order remains the one defined in
`docs/analysis/AI_LED_PORTFOLIO_PHASE_PLAN.md`:

- Phase 5 — prompt hygiene
- Phase 6 — R88, R90
- Phase 7 — R91, R93, R97
- Phase 8/9 — R89, R92, R94, R95
- Phase 10 — R96

This lifecycle workstream fits into that order as follows:

- It must respect **Phase 5 prompt hygiene** before adding new prompt-facing
  provenance fields, because prompt identity and schema identity need to be
  stable before replay packs can be trusted.
- It strengthens **Phase 7 memory closure** by giving R91 reflexion and episode
  memory higher-quality trade outcome linkage.
- It strengthens **Phase 9 operator transparency** by giving R95 a lifecycle
  object to inspect, not just a plan-level confidence report.
- It de-risks **Phase 10 trigger architecture** by providing an acceptance and
  replay substrate for R96 trigger identity and registry continuity.

### 1.2 Alignment with the SOTA Plan

This workstream most directly supports:

- **R91 — Reflexion Memory**
  Better linkage from plan -> signal -> fill -> closed lifecycle produces better
  loser/winner episode summaries.
- **R95 — Aggregate Confidence + Ops Surface**
  A `plan-audit` view is substantially more useful when anchored to an actual
  trade lifecycle with replayable evidence.
- **R96 — Trigger Catalog + Incremental Trigger Registry**
  Stable lifecycle IDs and replay packs are the audit harness needed to verify
  that trigger identity survives replans and open positions.

It also complements R88/R90/R97 by making plan-level evidence connect to actual
executed behavior rather than only to the generated plan artifact.

### 1.3 Alignment with the Prompt Audit

The Prompt Audit correctly identifies prompt-surface drift and provenance gaps.
Those findings are directly relevant here:

- the live strategist prompt must have one source of truth
- schema identity must be canonical
- prompt/template injection paths must not be ambiguous

Without those fixes, a replay artifact cannot confidently answer:
"which exact prompt + schema + template cohort produced this trade?"

For that reason, this workstream is explicitly downstream of the prompt hygiene
cleanup and should consume its registry/provenance outputs rather than invent a
second prompt provenance mechanism.

---

## 2. Current Repo Foundations

The repo already contains most of the raw pieces needed for this workstream:

- `SignalEvent` and `signal_ledger` for trigger-time setup records
- `SetupEvent` and `setup_event_ledger` for frozen feature snapshots
- `ExecutionLedgerWorkflow` transaction and position metadata
- `TradeSet` / `TradeLeg` for proper lifecycle modeling
- append-only ops `EventStore` for chronological event reconstruction
- backtest playback endpoints for chart/event/state replay
- `episode_memory` for closed-trade memory records

The missing piece is not raw capture. The missing piece is a first-class
**lifecycle identity and read model** joining these stores together.

---

## 3. Architectural Gaps This Workstream Closes

### Gap A — No canonical lifecycle ID across setup, signal, fill, and close

Today the system has several identifiers:

- `setup_event_id`
- `signal_id`
- event `correlation_id`
- implicit position identity
- `TradeSet.set_id`

But they are not the same durable key across the lifecycle. This makes exact
entry/exit pairing and replay harder than it needs to be.

### Gap B — Paper/live trade-set query is reconstructed heuristically

The current paper trading trade set endpoint uses FIFO buy/sell pairing. That is
adequate for simple summary views, but it is not a durable source of truth for:

- partial exits
- shorts
- reversals
- exact exit provenance
- cross-plan exit binding analysis

### Gap C — Replay is run-centric, not lifecycle-centric

Backtest playback is useful, but the primary replay surfaces today are:

- run-level playback
- event-chain queries
- trade tables

What is still missing is a single lifecycle payload answering:

"show me everything that happened for this opened and later closed thesis."

### Gap D — Similar-pattern retrieval has data, but not a dedicated query model

`setup_event_ledger` and `episode_memory` already contain most of what is needed
for retrieval of similar setups, but there is not yet a dedicated retrieval
surface over normalized lifecycle records.

---

## 4. Proposed Work Packages

This workstream is intentionally split into small packages that map onto the
existing roadmap rather than competing with it.

### W1 — Prompt and Plan Provenance Hygiene Hooks

**Depends on:** Prompt Audit Phase 1.

Before lifecycle replay is extended, ensure `plan_generated` and related events
carry stable prompt provenance:

- `prompt_id`
- `prompt_hash`
- `schema_id`
- `template_id`
- `strategy_template_version`

This should reuse the prompt registry/manifest direction from
`docs/analysis/PROMPT_AUDIT_2026-04-23.md` rather than creating a parallel
provenance registry.

**Why it exists:** replay without prompt provenance weakens R88, R90, and R95.

### W2 — Canonical Lifecycle Identity

Add one durable lifecycle key, preferably `trade_set_id`, and thread it through:

- `SetupEvent`
- `SignalEvent`
- fill payloads
- execution ledger transaction history
- position metadata
- close/episode memory records
- ops event payloads

At minimum, a closed trade should be queryable by a single ID that allows
joining setup, signal, fills, and exits without heuristic matching.

**Why it exists:** this is the key prerequisite for R91 and R96-quality audit.

### W3 — Native Trade Lifecycle Materialization for Paper/Live

Materialize `TradeSet`/`TradeLeg` directly during execution instead of
reconstructing them later from FIFO fill pairing.

Backtests already expose `TradeSet`-shaped data. Paper/live should converge on
that model so all environments share the same lifecycle abstraction.

Deliverable:

- canonical live/paper `trade_sets` query path
- each leg stores `trigger_id`, category, reason, stop/target context, and
  optional `signal_id`

**Why it exists:** it removes reconstruction loss from the read path.

### W4 — Lifecycle Replay Pack

Create a lifecycle-centric replay payload:

- trade lifecycle header
- entry/exit timestamps and prices
- linked setup record
- linked signal record
- linked fill legs
- linked plan metadata
- linked event timeline
- candle/indicator slices for the entry/exit window

Suggested endpoints:

- `GET /trade-lifecycles`
- `GET /trade-lifecycles/{trade_set_id}`
- `GET /trade-lifecycles/{trade_set_id}/replay`

This complements, not replaces, the run-level playback endpoints and directly
supports the operator transparency goals of R95.

### W5 — Similar-Pattern Retrieval MVP

Start with a constrained retrieval problem:

- same symbol
- same timeframe
- same playbook or trigger family
- same setup state or hypothesis type

Use:

- `setup_event_ledger.feature_snapshot`
- `signal_ledger`
- `episode_memory`

to rank similar historical episodes and surface:

- closest prior winners
- closest prior losers
- common failure modes
- average outcome stats

This should feed R91 reflexion memory and later provide an operator-facing
"show me similar historical lifecycles" surface.

Do not start with cross-vehicle or cross-timeframe matching. That should come
later, after normalized features and lifecycle IDs are stable.

### W6 — R96 Acceptance Harness

Use lifecycle replay as the acceptance harness for trigger registry work:

- verify open positions retain the correct exit binding through replans
- verify canonical trigger IDs map cleanly into lifecycle records
- compare pre-R96 and post-R96 lifecycle replay outputs for parity

This gives R96 a concrete audit substrate beyond trigger-engine unit tests.

---

## 5. Recommended Ordering Relative to Existing Runbooks

| Companion Package | Best Phase Window | Why |
|---|---|---|
| W1 prompt/plan provenance hooks | Phase 5 | Consumes prompt hygiene work; avoids ambiguous replay provenance |
| W2 canonical lifecycle identity | Phase 6 | Safe to add before memory/uncertainty layers depend on it |
| W3 native paper/live trade set materialization | Phase 6.5 / early 7 | Enables higher-quality episode and reflexion records |
| W4 lifecycle replay pack | Phase 7–9 | Most useful once memory and plan audit surfaces exist |
| W5 similar-pattern retrieval MVP | Phase 7 | Directly supports R91 reflexion memory |
| W6 R96 acceptance harness | Phase 9–10 | Becomes critical when trigger identity mutates incrementally |

This ordering preserves the main roadmap in
`docs/analysis/AI_LED_PORTFOLIO_PHASE_PLAN.md` and fills its evidence-plane
gaps rather than moving its milestones.

---

## 6. Concrete Deliverables

### Documentation Deliverables

- this alignment document
- explicit references from the phase plan, SOTA plan, and prompt audit back to
  this document

### Data Model Deliverables

- `trade_set_id` or equivalent lifecycle key
- `signal_ledger` persistence extended to retain currently dropped provenance
- lifecycle linkage from setup -> signal -> fills -> close

### API Deliverables

- lifecycle list endpoint
- lifecycle detail endpoint
- lifecycle replay endpoint
- later: similar-pattern retrieval endpoint

### Verification Deliverables

- tests showing exact entry/exit provenance survives paper/live execution
- tests showing replay pack is reproducible for a closed trade
- R96-specific tests verifying trigger registry changes preserve lifecycle
  continuity under replan

---

## 7. Non-Goals

To keep this workstream aligned with the main roadmap, the following are
explicitly out of scope for the first implementation:

- a new independent strategist roadmap
- cross-vehicle pattern retrieval on day one
- cross-timeframe generalized replay before normalized feature contracts exist
- replacing the backtest playback stack
- adding prompt-surface complexity before Prompt Audit cleanup lands

---

## 8. Acceptance Criteria

This companion workstream should be considered successful when all of the
following are true:

1. An operator can query every entry that actually opened a position and see the
   exact exit that closed it without FIFO heuristic ambiguity.
2. A closed lifecycle can be replayed from a single lifecycle ID with candles,
   fills, setup/signal provenance, and event chronology.
3. R91 reflexion summaries can cite a stable closed lifecycle rather than
   reconstructing one from partial evidence.
4. R95 `plan-audit` can link from a plan to one or more executed lifecycles.
5. R96 trigger-registry work can be regression-tested using lifecycle replay
   packs rather than trigger-level inspection alone.

---

## 9. Cross-References

- Main roadmap and dependency order:
  `docs/analysis/AI_LED_PORTFOLIO_PHASE_PLAN.md`
- SOTA/runbook rationale for R88–R97:
  `docs/analysis/AGENTIC_SOTA_ALIGNMENT_PLAN.md`
- Prompt provenance and prompt-surface cleanup:
  `docs/analysis/PROMPT_AUDIT_2026-04-23.md`
