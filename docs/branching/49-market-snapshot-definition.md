# Runbook 49: Market Snapshot Definition

## Purpose

Define standardized, timestamped snapshot contracts that become the single source of
truth for deterministic tick execution and policy-loop reasoning.

This runbook is the foundation for the FinAgent-inspired upgrades: the paper's value is
primarily in the **agent architecture** (multimodal reasoning + reflection + memory),
not in introducing a new pretrained base model. In this codebase, that means enforcing
an input contract before expanding reasoning loops.

Without a strict snapshot contract, the strategist can drift into inconsistent prompt
inputs (e.g., indicators present but no provenance, ad-hoc news text, no chart-structure
encoding), which makes memory retrieval, judge validation, and post-trade attribution
unreliable.

Important cadence alignment:

- the trigger engine runs from a **lightweight tick snapshot**
- the strategist/judge policy loop runs from a **heavier policy snapshot**
- do not build full multimodal policy objects every 1m bar unless a policy event fires

## Naming / Redundancy Guard (Required)

`MarketSnapshot` in this runbook should be treated as a **snapshot family / umbrella**
concept, not a mandate to replace every existing snapshot type in the repo.

Use the following rule to avoid redundancy:

- keep existing domain-specific snapshots where they already serve a distinct purpose
  (e.g., `IndicatorSnapshot`, position/fee snapshots, judge performance snapshots)
- introduce `TickSnapshot` / `PolicySnapshot` as orchestration-layer decision snapshots
- compose or reference existing `IndicatorSnapshot` data inside them rather than cloning
  all fields into a second parallel schema

## Non-Negotiable: Versioned, Atomic, Immutable Snapshots

For Phase 8 to be credible, snapshots must be:

- **Versioned** (`snapshot_version`) so schema changes are explicit and replay-safe
- **Atomic** (frozen at one decision boundary / bar) so all modalities reference the same state
- **Immutable** once handed to reasoning loops (strategist and judge must see the same object)

Hard guarantee:

- the strategist and judge receive the **same frozen snapshot payload** (or a byte-for-byte
  equivalent canonical serialization) identified by the same `snapshot_hash`
- no post-hoc mutation of snapshot fields is allowed after freeze/dispatch

This prevents memory contamination, non-reproducible rationales, and reflection drift.

## Scope

1. **`schemas/market_snapshot.py`** — new Pydantic models for multimodal snapshot
2. **`services/market_snapshot_builder.py`** — builder/normalizer for strategist/judge
3. **`trading_core/trigger_engine.py`** (and/or execution path) — consume `TickSnapshot`
4. **`agents/strategies/plan_provider.py`** — require `PolicySnapshot` for policy evaluation
5. **`agents/strategies/llm_client.py`** — consume policy snapshot sections in deterministic order
6. **`agents/judge_agent_client.py`** — include policy snapshot hash + staleness metadata in evals
7. **`vector_store/retriever.py`** — query context from policy snapshot fields (not ad-hoc dicts)
8. **`backtesting/llm_strategist_runner.py`** — construct snapshots during backtests
9. **`tools/paper_trading.py`** (or equivalent paper trading planner path) — construct snapshots
10. **`agents/event_emitter.py` / signal telemetry paths** — persist `snapshot_id`/`snapshot_hash`
11. **`tests/test_market_snapshot_schema.py`** and builder integration tests

## Out of Scope

- Raw news collection, scraping, or vendor integration (snapshot consumes pre-summarized text)
- Vision model training or image storage (visual inputs are encoded fingerprints/tags only)
- UI rendering of the snapshot (API/ops exposure can be a follow-up runbook)
- Strategy logic changes beyond switching inputs to the snapshot family (`TickSnapshot` / `PolicySnapshot`)

## Key Principle: Snapshot as Evidence Artifact

Snapshots are immutable evidence artifacts for execution/policy boundaries:

- It is **not** a direct trading command.
- It must be **timestamped and hashable**.
- It must preserve **input provenance** (data source, extractor version, generation time).
- They must be usable by trigger engine, strategist, judge, and post-trade analysis without reinterpretation.

## Two Snapshot Types (Required)

### A. `TickSnapshot` (Deterministic Execution Layer)

Lightweight. Built every tick/bar for the trigger engine and position manager.

Required characteristics:

- no LLM-only context
- no news/text digest
- no memory retrieval bundle
- no heavy multimodal encodings unless directly needed by deterministic triggers
- optimized for fast trigger/stop/target evaluation

Used by:

- trigger engine
- stop/target manager
- position manager
- episode/state transition logger

Implementation note:

- `TickSnapshot` should typically embed/reference the current `IndicatorSnapshot` (or a
  normalized subset) instead of redefining the full indicator field surface.

### B. `PolicySnapshot` (Event-Driven Policy Loop)

Heavier. Built only on policy events (not every bar) for strategist/judge evaluation.

Expected contents:

- regime fingerprint and regime transition metadata
- normalized volatility / multi-timeframe state
- derived signals relevant to playbook selection
- optional text/visual summaries (timestamped)
- memory retrieval bundle (or stable reference + summary)
- expectation / calibration metrics used for policy decisions

Used by:

- strategist (playbook selection / policy envelope)
- policy-level reflection
- judge validation at policy boundary

Strategist should only see `PolicySnapshot`, not raw per-bar tick snapshots.

Implementation note:

- `PolicySnapshot` should aggregate references/derived summaries from existing snapshot
  primitives (`IndicatorSnapshot`, regime fingerprint outputs, memory bundle summaries)
  rather than duplicating raw indicator schemas.

## Schema Contract (Required Sections)

### 1. Identity / Provenance

```python
class SnapshotProvenance(BaseModel):
    model_config = {"extra": "forbid"}
    snapshot_version: str                # schema version, e.g. "1.0"
    snapshot_kind: Literal["tick", "policy"]
    snapshot_id: str                     # uuid4
    snapshot_hash: str                   # sha256 over canonical JSON
    feature_pipeline_hash: str           # hash of deterministic feature derivation pipeline/config
    as_of_ts: datetime                   # market state timestamp (UTC)
    generated_at_ts: datetime            # builder timestamp (UTC)
    created_at_bar_id: str               # canonical bar key used to freeze the snapshot
    policy_event_id: str | None = None   # set only for policy snapshots
    parent_tick_snapshot_id: str | None = None
    symbol: str
    timeframe: str
    data_window_start_ts: datetime | None = None
    data_window_end_ts: datetime | None = None
    feature_pipeline_version: str | None = None
    news_digest_version: str | None = None
    visual_encoder_version: str | None = None
```

### Deterministic Feature Derivation Log (Required)

The snapshot must carry (or reference) a deterministic derivation manifest for all
derived features used in reasoning:

- transform/extractor name
- version/hash
- input window / bar IDs
- parameter values
- output field names

This can be embedded as a compact `feature_derivation_log` or stored separately with a
stable reference from the snapshot. The goal is reproducibility, not verbosity.

### 2. Numerical Data (raw-ish market state)

- Latest OHLCV and selected lookback aggregates
- Volume metrics (rolling mean, spike ratio, turnover proxies)
- Volatility metrics (ATR, realized vol, range expansion)
- Microstructure or execution context if available (spread/slippage proxy)

### 3. Derived Signals (deterministic features / regime flags)

- Regime classification + confidence
- Trend/volatility state flags
- Trigger-relevant identifiers (`compression_flag`, `breakout_confirmed`, etc.)
- HTF structure fields from Runbook 41/42 (when available)
- Screener rank / anomaly score (when Runbook 39 path is active)

For cross-symbol generalization and retrieval quality, derived signals should include
normalized/relative variants (percentiles, z-scores, anchor-relative distances), not
just raw values.

### 4. Text Signals (summarized, normalized)

Text inputs must be **summaries**, not raw feeds:

- `headline_summary` / `event_summary`
- `event_ts` or `window_ts`
- sentiment / impact label (if available)
- confidence / coverage count
- source provenance

All text entries must be tagged with a timestamp to prevent "floating context" in judge
reviews and to support stale-news filtering.

### 5. Visual Signals (encoded, not raw images)

Visual inputs should be stored as compact features:

- chart pattern fingerprints (hash/embedding/vector ref)
- structural tags (inside-bar cluster, compression box, impulse candle, trend channel)
- extractor version and source candle window

Do not place raw image bytes in the snapshot. Persist a reference or encoded summary.

### 6. Quality / Staleness

The snapshot must expose quality flags so the strategist/judge can act deterministically:

- `is_stale` (bool)
- `staleness_seconds`
- `missing_sections` (e.g., `["text_signals"]`)
- `quality_warnings` (e.g., delayed news digest, partial visual encoding)

### 7. Policy-Only Context (for `PolicySnapshot` only)

This section is absent from `TickSnapshot` by default:

- memory retrieval bundle (or `memory_bundle_id` + summary)
- expectation / calibration summary (hold P50/P90, MAE/MFE drift flags)
- policy event trigger metadata (`regime_change`, `position_closed`, `heartbeat`, etc.)

### 8. `normalized_features` Block (Required for `PolicySnapshot`)

To support cross-symbol memory retrieval and regime transition detection, `PolicySnapshot`
should include a first-class `normalized_features` section containing fields such as:

- rolling volatility percentile
- ATR percentile / z-score (normalized to symbol history)
- volume percentile / participation percentile
- relative position in rolling distribution (e.g., range expansion percentile)
- distance from HTF anchors in ATR units
- normalized regime-fingerprint components used by the transition detector

The `normalized_features` block should carry a versioned linkage to the regime fingerprint
vector contract (Runbook 55), for example:

- `regime_fingerprint_vector_version`
- `regime_fingerprint_feature_order_id` (or equivalent stable ordering/version key)
- explicit mapping/reference if `PolicySnapshot.normalized_features` is a superset of the
  Runbook 55 `numeric_vector`

Hard rule:

- memory similarity (Runbook 51) and regime-distance/transition logic (Runbook 55) must
  use `PolicySnapshot.normalized_features` (plus categorical state fields), not raw
  OHLCV/raw ATR/raw volume fields
- `PolicySnapshot.normalized_features` and Runbook 55 fingerprint vectors must be version-
  aligned or explicitly mapped; silent drift between normalized blocks is not allowed

## Time Normalization Rules (Cross-Timeframe Comparisons)

To avoid tautological or misleading cross-timeframe conditions (e.g., comparing raw ATR
values across horizons), all cross-timeframe volatility comparisons must be normalized to
a common horizon before comparison.

Minimum rule:

- **Do not compare raw ATR values from different timeframes directly** in strategist/judge
  logic or snapshot-derived invariants.
- When using ATR-like measures across horizons, normalize to a common horizon using a
  consistent scaling rule (e.g., ATR scales approximately with `sqrt(T)` as a practical
  heuristic), and record the normalization method in the derivation log.

This is not about perfect market microstructure physics; it is a correctness guardrail
against tautological conditions and invalid invariants.

## Implementation Steps

### Step 1: Add `schemas/market_snapshot.py`

Define:

- `TickSnapshot`
- `PolicySnapshot`
- `BaseSnapshot` (optional shared base)
- `SnapshotProvenance`
- `NumericalSignalBlock`
- `DerivedSignalBlock`
- `TextSignalDigest`
- `VisualSignalFingerprint`
- `SnapshotQuality`
- `FeatureDerivationEntry` / `FeatureDerivationLog`

Rules:

- `model_config = {"extra": "forbid"}`
- all timestamps are UTC datetimes
- stable field ordering for hash generation
- no free-form blobs without schema wrappers
- include `snapshot_version`, `feature_pipeline_hash`, `created_at_bar_id`
- include `snapshot_kind` and policy-only fields gated by `snapshot_kind=="policy"`

### Step 2: Build canonical hashing in `services/market_snapshot_builder.py`

The builder should:

1. Normalize values (types, timestamps, NaN handling)
2. Sort lists where ordering is semantically stable
3. Build/attach deterministic feature derivation log (including normalization methods)
4. Freeze the snapshot payload (immutable object or immutable canonical JSON blob)
5. Serialize to canonical JSON
6. Compute `snapshot_hash` (sha256)
7. Attach `quality` flags and `missing_sections`

This hash is the traceability anchor for memory retrieval and judge audits.

### Step 3: Split builders by cadence (`TickSnapshot` vs `PolicySnapshot`)

Implement distinct builder entrypoints:

- `build_tick_snapshot(...)` for every bar/tick
- `build_policy_snapshot(...)` for policy events only

`build_policy_snapshot(...)` may incorporate text/visual summaries and memory retrieval
references; `build_tick_snapshot(...)` should stay lean.

### Step 4: Make strategist invocation require `PolicySnapshot`

Update strategist plan generation entrypoints to accept a `PolicySnapshot` instead of
loosely structured dicts. Reject or skip planning when:

- snapshot is stale beyond configured limit
- required numerical/derived blocks are missing
- snapshot hash cannot be computed

Optional inputs (text/visual) may be absent, but must be explicitly marked absent.

Trigger engine paths should consume `TickSnapshot` only and must not depend on policy-only
fields.

### Step 5: Enforce atomic handoff to strategist and judge

The orchestration layer must freeze the snapshot once per decision boundary (tick or policy event) and pass the same
frozen payload to all reasoning components involved in that boundary.

Required behavior:

- strategist and judge references point to the same `snapshot_id` / `snapshot_hash`
- any attempted mutation after freeze raises or creates a new snapshot ID/hash
- delayed judge evaluations must reference the original snapshot (not a regenerated one)

### Step 6: Thread snapshot metadata through judge + telemetry

Every strategist proposal and judge evaluation should record:

- `snapshot_id`
- `snapshot_hash`
- `snapshot_as_of_ts`
- `snapshot_staleness_seconds`
- `snapshot_version`
- `snapshot_kind`
- `feature_pipeline_hash`
- `created_at_bar_id`

This enables later audits: "What exact evidence was available when this decision was made?"

### Step 7: Backtest and paper-trading parity

Backtests and paper trading must build schema-compatible `TickSnapshot` and
`PolicySnapshot` objects, so trigger behavior and policy-loop reasoning are comparable
across environments.

## Acceptance Criteria

- [ ] `TickSnapshot` and `PolicySnapshot` schemas exist with explicit `snapshot_kind`
- [ ] Snapshot includes UTC timestamps and provenance/version metadata (`snapshot_version`, `feature_pipeline_hash`, `created_at_bar_id`)
- [ ] Snapshot includes deterministic feature derivation log or stable reference to it
- [ ] `snapshot_hash` computed from canonical serialization and persisted in decision telemetry
- [ ] Snapshot is frozen/immutable once dispatched; strategist and judge receive identical snapshot payload/hash for the same policy event
- [ ] Trigger engine consumes `TickSnapshot`; strategist/judge consume `PolicySnapshot`
- [ ] `TickSnapshot` excludes heavy policy-only fields (news/text/memory bundle by default)
- [ ] `PolicySnapshot` can carry memory bundle reference/summary and expectation calibration context
- [ ] `PolicySnapshot` includes normalized percentile/z-score style features for cross-symbol retrieval and regime-distance comparisons
- [ ] Memory similarity and regime-distance logic are contractually limited to `PolicySnapshot.normalized_features` + categorical state fields (not raw market values)
- [ ] `PolicySnapshot.normalized_features` is version-linked/mapped to Runbook 55 regime-fingerprint vector definitions (no silent normalized-block drift)
- [ ] Strategist policy evaluation requires a `PolicySnapshot` input
- [ ] Missing optional modalities are explicit (`missing_sections`), not silently omitted
- [ ] Staleness checks block/skip strategist invocation when thresholds are exceeded
- [ ] Cross-timeframe volatility comparisons in snapshot-derived features/invariants are normalized to a common horizon and derivation method is recorded
- [ ] Judge evaluations record snapshot provenance and staleness
- [ ] Backtest and paper-trading planner paths construct schema-compatible snapshots

## Test Plan

```bash
# Schema validation
uv run pytest tests/test_market_snapshot_schema.py -vv

# Builder normalization / hashing
uv run pytest tests/test_market_snapshot_builder.py -vv

# Tick vs policy snapshot shape and field gating
uv run pytest -k "tick_snapshot or policy_snapshot" -vv

# normalized_features block contract (similarity / regime-distance inputs)
uv run pytest -k "policy_snapshot and normalized_features" -vv

# normalized_features <-> regime fingerprint vector version alignment
uv run pytest -k "policy_snapshot and fingerprint_vector_version" -vv

# Cross-timeframe normalization guardrails
uv run pytest -k "market_snapshot and normalization" -vv

# Strategist integration (reject stale/malformed policy snapshots)
uv run pytest -k "policy_snapshot and plan_provider" -vv

# Trigger engine integration (tick snapshot only)
uv run pytest -k "tick_snapshot and trigger_engine" -vv

# Judge telemetry includes identical snapshot provenance/hash
uv run pytest -k "judge and snapshot_hash" -vv

# Regression (planning/backtest)
uv run pytest tests/test_plan_provider.py tests/test_llm_strategist_runner.py -vv
```

## Human Verification Evidence

```text
TODO:
1. Run one policy evaluation event (not every bar) in paper trading and inspect decision telemetry.
2. Confirm snapshot_version/feature_pipeline_hash/created_at_bar_id are present.
3. Confirm strategist and judge reference the same snapshot_id/snapshot_hash for the same tick.
4. Confirm snapshot_id/snapshot_hash are stable for identical inputs.
5. Confirm missing text/visual modalities are marked in missing_sections (not silently absent).
6. Confirm `TickSnapshot` objects do not include text/news or memory bundle payloads by default.
7. Simulate stale policy snapshot (> threshold) and verify strategist skips with explicit reason.
8. Inspect one cross-timeframe volatility invariant and confirm normalization method is
   recorded in the derivation log (no raw ATR timeframe comparison).
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-22 | Runbook created — tick/policy snapshot contracts for deterministic execution and strategist/judge inputs | Codex |

## Test Evidence (append results before commit)

```text
TODO
```

## Worktree Setup

```bash
git worktree add -b feat/market-snapshot-definition ../wt-market-snapshot main
cd ../wt-market-snapshot

# When finished
git worktree remove ../wt-market-snapshot
```

## Git Workflow

```bash
git checkout -b feat/market-snapshot-definition

git add schemas/market_snapshot.py \
  services/market_snapshot_builder.py \
  agents/strategies/plan_provider.py \
  agents/strategies/llm_client.py \
  agents/judge_agent_client.py \
  vector_store/retriever.py \
  backtesting/llm_strategist_runner.py \
  tools/paper_trading.py \
  tests/test_market_snapshot_schema.py \
  tests/test_market_snapshot_builder.py \
  docs/branching/49-market-snapshot-definition.md \
  docs/branching/README.md

uv run pytest tests/test_market_snapshot_schema.py tests/test_market_snapshot_builder.py -vv

# Commit only after Test Evidence + Human Verification Evidence are recorded
git commit -m "feat: add multimodal market snapshot contract for strategist and judge (Runbook 49)"
```
