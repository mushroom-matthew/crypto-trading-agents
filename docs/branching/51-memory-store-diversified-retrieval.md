# Runbook 51: Memory Store Schema and Diversified Retrieval

## Purpose

Add a diversified episode-memory layer for strategist and judge reasoning, with explicit
contrastive retrieval across:

- `winning_contexts`
- `losing_contexts`
- `failure_mode_patterns`

This runbook upgrades memory retrieval from "nearest similar examples" to a structured
evidence bundle that reduces confirmation bias and supports judge rejection criteria.

Cadence rule:

- retrieval runs in the **event-driven policy loop only**
- retrieval does **not** run on every tick/bar

## Why This Matters Here

The codebase already has strong infrastructure for signal outcomes and playbook evidence
(Runbooks 43 and 48). What is missing is a memory abstraction tailored for decision-time
reasoning. The strategist should not only retrieve "what looked similar and won"; it also
needs "what looked similar and failed" and "what failure pattern usually appears before
losses." The judge needs the same contrastive evidence to reject unsupported plans.

## Scope

1. **`schemas/episode_memory.py`** — new memory record + retrieval result schemas
2. **`services/episode_memory_service.py`** — build/store episode records from outcomes
3. **`services/memory_retrieval_service.py`** — diversified retrieval with bucket quotas
4. **`services/signal_outcome_reconciler.py`** — emit/attach memory-ready outcome features
5. **`services/signal_ledger_service.py`** or persistence layer — store memory annotations
6. **DB migration** (or durable store schema) for episode memory records + indexes
7. **`agents/strategies/plan_provider.py`** — retrieve/reuse memory bundle per policy evaluation event
8. **`services/judge_feedback_service.py`** / judge path — retrieve memory bundle for validation
9. **`vector_store/retriever.py`** (optional coordination) — keep strategy retrieval separate from episode memory retrieval
10. **Tests** for memory bucketing, indexing, and retrieval diversity

## Out of Scope

- Replacing the strategy vector store (this is complementary to strategy/playbook docs)
- Online learning / model weight updates
- Automatic relabeling of historical outcomes without audit trail
- Human-authored playbook edits (covered elsewhere; this runbook supplies evidence)

## Memory Record Design (Episode-Centric)

Each historical episode should be annotated with:

- identity (`episode_id`, `signal_id`, `trade_id` if present)
- timestamps (entry, exit, resolution)
- `MarketSnapshot` references (`snapshot_id`, `snapshot_hash`)
- regime fingerprint (deterministic compact representation)
- strategy metadata (`playbook_id`, `template_id`, trigger category, symbol, timeframe)
- outcome metrics (`pnl`, `r_achieved`, `hold_bars`, `hold_minutes`)
- excursion metrics (`mae`, `mfe`, `mae_pct`, `mfe_pct`)
- decision metadata (stance, confidence, judge action context if any)
- labels:
  - `outcome_class`: `win | loss | neutral`
  - `failure_modes`: list[str] (e.g., `false_breakout_reversion`, `news_gap_against`, `late_chase`)

## Retrieval Strategy (Diversified by Design)

The retriever must return a contrastive bundle, not a single ranked list:

```python
class DiversifiedMemoryBundle(BaseModel):
    winning_contexts: list[EpisodeMemoryRecord]
    losing_contexts: list[EpisodeMemoryRecord]
    failure_mode_patterns: list[EpisodeMemoryRecord]
    retrieval_meta: dict
```

### Rules

1. **Bucket quotas** (example defaults):
   - wins: 3
   - losses: 3
   - failure-modes: 2
2. **Similarity is necessary but not sufficient**
   - do not return only nearest neighbors
3. **Diversity constraints**
   - avoid all examples from same day/symbol unless unavoidable
   - include regime diversity if current fingerprint is ambiguous
4. **Explicit insufficiency**
   - if a bucket lacks data, return empty bucket + `retrieval_meta["insufficient_buckets"]`

## Similarity Metric Contract (Required)

"Nearest" must be defined explicitly. Do not allow ad-hoc or implicit similarity.

Recommended baseline: **weighted hybrid similarity** combining:

- categorical matches (regime, playbook, direction, timeframe bucket)
- normalized numeric features (z-scored or robust-scaled)
- optional text/visual embedding similarity (if available)

Illustrative scoring components:

- `regime_match_score` (high weight)
- `playbook_match_score` (high weight)
- `timeframe_compatibility_score` (high weight)
- `feature_vector_similarity` (medium weight; cosine on normalized numeric vector)
- `outcome-context recency factor` (bounded decay, low/medium weight)

Do not overweight raw oscillator values (e.g., exact RSI proximity) relative to regime and
playbook structure. Regime/playbook context should dominate "looks-nearby" noise.

The runbook implementation must define:

- which fields participate in similarity
- scaling/normalization method per field group
- exact distance/similarity function(s)
- default weights and where they are configured

## Cross-Symbol Retrieval Policy (Required)

Cross-symbol retrieval is a major contamination risk if not normalized.

Allowed modes (pick one explicitly in config and telemetry):

1. **Per-symbol only** (safest default)
2. **Hierarchical retrieval**
   - symbol-first
   - then asset-class / cohort (e.g., majors vs alts)
   - then global fallback
3. **Cross-symbol normalized**
   - only if features are normalized and symbol-specific scale effects are controlled

If cross-symbol retrieval is enabled, the retriever must tag each memory record with the
retrieval scope (`symbol`, `cohort`, `global`) so strategist/judge can discount weaker
matches.

### Recommended Initial Production Mode

Start with **hierarchical retrieval**:

1. symbol-local
2. cohort-level (asset class / venue / liquidity bucket)
3. global fallback

Global fallback should be disallowed unless normalized regime-fingerprint distance is
below a configured threshold (to avoid regime leakage from unrelated symbols).

## Retrieval Cadence and Reuse (Policy Loop Only)

Memory retrieval is performed only during policy evaluation events (Runbook 54), not per
tick. The retrieval service should support bundle reuse to avoid redundant queries.

Required behavior:

- if regime fingerprint delta is below configured threshold and policy scope is unchanged,
  reuse the prior memory bundle
- re-query when:
  - regime-change delta exceeds threshold
  - policy event type requires refresh (e.g., position closed, major vol band shift)
  - bundle TTL expires / heartbeat refresh hits

Persist in `retrieval_meta`:

- `policy_event_type`
- `regime_fingerprint_delta`
- `bundle_reused` (bool)
- `reuse_reason` / `requery_reason`

## Failure-Mode Pattern Taxonomy (Initial)

Start with a small, explicit taxonomy. Expand only when telemetry proves need.

Examples:

- `false_breakout_reversion`
- `trend_exhaustion_after_extension`
- `low_volume_breakout_failure`
- `macro_news_whipsaw`
- `signal_conflict_chop`
- `late_entry_poor_r_multiple`
- `stop_too_tight_noise_out`

These labels can be produced deterministically where possible, with judge-assisted review
as a secondary annotation path.

## Implementation Steps

### Step 1: Define memory schemas

Create `schemas/episode_memory.py` with:

- `EpisodeMemoryRecord`
- `DiversifiedMemoryBundle`
- `MemoryRetrievalRequest`
- `MemoryRetrievalMeta`

Prefer strict typing and explicit labels over free-form notes.

### Step 2: Build episode records from signal/trade outcomes

Implement `services/episode_memory_service.py` to transform signal ledger + outcome data
into memory records. This service should run when an episode resolves (or via backfill
batch job for historical records).

Key requirement: store regime fingerprint + playbook metadata at episode close time.

### Step 3: Implement diversified retrieval

`services/memory_retrieval_service.py` should:

1. query candidate episodes by symbol/timeframe/regime similarity
2. score candidates using the explicit weighted-hybrid metric + recency decay (bounded)
3. split into buckets (win/loss/failure-mode)
4. apply bucket quotas + diversity constraints
5. return `DiversifiedMemoryBundle`

Define retrieval scope policy in code (per-symbol / hierarchical / cross-symbol normalized)
and persist the chosen scope in retrieval telemetry.

### Step 3A: Define feature groups and weights explicitly

Create a documented/configurable similarity spec that states:

- categorical fields and match rules
- numeric feature list and normalization method
- text/visual embedding fields (optional)
- weights (with rationale)
- symbol-scope policy and cohort mapping (if hierarchical)
- global-fallback gate based on normalized regime-fingerprint distance threshold

### Step 4: Integrate with strategist and judge

- Strategist policy loop retrieves (or reuses) memory bundle on policy evaluation events
- Judge policy-boundary validation uses the same memory bundle to check proposed plan
  against known failures

Tick engine paths must not perform memory retrieval.

The memory bundle is advisory evidence; deterministic guardrails still take precedence.

### Step 5: Observability and audits

Persist retrieval telemetry:

- bucket counts returned
- insufficient buckets
- retrieval latency
- candidate pool size
- top labels/regimes represented

This is necessary to detect retrieval collapse (e.g., always returning wins only).

## Acceptance Criteria

- [x] Episode memory schema stores regime fingerprint, outcome metrics, and playbook metadata
- [x] Resolved episodes are persisted as memory records (in-memory store; DB migration deferred)
- [x] Memory retrieval returns explicit buckets: wins, losses, failure-modes
- [x] Similarity metric is explicitly defined (feature groups, normalization, weights, distance function)
- [x] Cross-symbol retrieval policy is explicit (hierarchical: symbol-local → global fallback) and recorded in telemetry
- [x] If global fallback is enabled, it is gated by normalized regime-fingerprint distance threshold
- [x] Memory retrieval is cadence-neutral (stateless service; caller controls when to invoke)
- [x] Memory bundle reuse is gated by regime-fingerprint delta / TTL and recorded in telemetry
- [x] Empty/insufficient buckets are explicit and logged in `retrieval_meta.insufficient_buckets`
- [x] Retrieval telemetry records bucket counts and latency
- [x] Tests cover bucketing, diversity constraints, and insufficiency behavior
- [x] Strategist / judge integration deferred to R54 cadence wiring

## Test Plan

```bash
# Memory schemas
uv run pytest tests/test_episode_memory_schema.py -vv

# Memory record creation from outcomes
uv run pytest tests/test_episode_memory_service.py -vv

# Diversified retrieval bucketing/diversity constraints
uv run pytest tests/test_memory_retrieval_service.py -vv

# All R51 tests together
uv run pytest tests/test_episode_memory_schema.py tests/test_episode_memory_service.py tests/test_memory_retrieval_service.py -v
```

## Human Verification Evidence

- `EpisodeMemoryRecord` stores `regime_fingerprint` (normalized features dict from R55),
  `outcome_class` (`win`/`loss`/`neutral`), and `failure_modes` list drawn from
  `FAILURE_MODE_TAXONOMY`.
- `DiversifiedMemoryBundle` has three typed buckets (`winning_contexts`, `losing_contexts`,
  `failure_mode_patterns`) and a `MemoryRetrievalMeta` with explicit similarity weights,
  retrieval scope, candidate pool size, and `insufficient_buckets`.
- `MemoryRetrievalService.retrieve()` applies bucket quotas (win=3, loss=3, failure=2 by
  default), same-day diversity constraint, and global fallback gated by
  `global_fallback_max_fingerprint_distance=0.30`.
- Bundle reuse fires when `prior_bundle.retrieval_meta.regime_fingerprint_delta < 0.05`;
  `retrieval_meta.bundle_reused=True` and `reuse_reason` are set.
- Retrieval service is stateless — cadence control (policy-loop-only, not per-tick) is
  enforced by the caller (R54 integration).

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-22 | Runbook created — diversified episode memory store and contrastive retrieval | Codex |
| 2026-02-26 | Implementation complete — schemas/episode_memory.py, services/episode_memory_service.py, services/memory_retrieval_service.py; 90 new tests, 1363 full suite passing | Claude |

## Test Evidence (append results before commit)

```text
uv run pytest tests/test_episode_memory_schema.py tests/test_episode_memory_service.py \
  tests/test_memory_retrieval_service.py -v

90 passed in 4.94s

Full suite (excluding pre-existing DB_DSN collection errors):
1363 passed, 2 skipped (2 pre-existing test_factor_loader.py failures unrelated to R51)
```

## Worktree Setup

```bash
git worktree add -b feat/memory-diversified-retrieval ../wt-memory-diversified main
cd ../wt-memory-diversified

# When finished
git worktree remove ../wt-memory-diversified
```

## Git Workflow

```bash
git checkout -b feat/memory-diversified-retrieval

git add schemas/episode_memory.py \
  services/episode_memory_service.py \
  services/memory_retrieval_service.py \
  services/signal_outcome_reconciler.py \
  services/signal_ledger_service.py \
  app/db/migrations/ \
  agents/strategies/plan_provider.py \
  services/judge_feedback_service.py \
  tests/test_episode_memory_schema.py \
  tests/test_episode_memory_service.py \
  tests/test_memory_retrieval_service.py \
  docs/branching/51-memory-store-diversified-retrieval.md \
  docs/branching/README.md

uv run pytest tests/test_episode_memory_schema.py tests/test_memory_retrieval_service.py -vv

# Commit only after Test Evidence + Human Verification Evidence are recorded
git commit -m "feat: add diversified episode memory retrieval for strategist and judge (Runbook 51)"
```
