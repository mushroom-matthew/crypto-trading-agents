# Runbook 55: Regime Fingerprint and Transition Detector

## Purpose

Define a deterministic, inspectable regime fingerprint + transition detector that emits
`regime_state_changed` policy events for the event-driven policy loop (Runbook 54).

This is the keystone for stable policy cadence:

- if it over-triggers, policy thrashes
- if it under-triggers, policy goes stale

The detector must be deterministic, bounded, decomposable, and replayable. No LLMs.

## Core Design Decision (Required)

Use **both** detector scopes:

1. **Symbol-local detector** (primary)
   - drives policy cadence for that symbol
2. **Cohort/shared detector** (secondary)
   - informs retrieval priors and playbook weighting, not direct symbol policy cadence

This prevents false coupling like "BTC changed regime, therefore SOL must re-evaluate"
while still allowing cross-symbol generalization signals to inform reasoning.

## Scope

1. **`schemas/regime_fingerprint.py`** — `RegimeFingerprint`, transition events, telemetry payloads
2. **`services/regime_transition_detector.py`** — state machine + distance metric + hysteresis
3. **`trading_core/regime_classifier.py`** — map/extend outputs into stable vocabulary and confidence inputs
4. **`agents/analytics/indicator_snapshots.py`** (or snapshot builder path) — build normalized fingerprint inputs
5. **`services/market_snapshot_builder.py`** — include normalized fingerprint components in `PolicySnapshot`
6. **Policy loop orchestration** (`agents/strategies/plan_provider.py`, workflow/service scheduler) — consume transition events
7. **`services/memory_retrieval_service.py`** — use cohort/shared regime state as retrieval prior weighting only
8. **Telemetry/events** — persist transition decisions and suppression reasons
9. **Tests** — deterministic replay, hysteresis, cooldown, shock override, under/over-trigger cases

## Out of Scope

- LLM-based regime classification or transition detection
- Playbook editing or strategist logic changes beyond consuming transition events
- Tick-level trigger engine behavior (detector feeds policy loop only)
- ML training for learned transition thresholds (follow-up, if ever)

## Hard Constraints (Non-Negotiable)

### 1. Normalized-Only Fingerprint Components

`RegimeFingerprint` must contain only:

- categorical state labels
- normalized / percentile / z-scored numeric features
- confidence values
- versioned metadata

Forbidden in the fingerprint distance vector:

- raw price levels
- raw ATR values
- raw volume
- raw RSI values
- symbol-specific absolute scalars

Reason: raw symbol-scale values corrupt cohort/global comparability and transition quality.

### 2. Stable Vocabulary (Must Be Fixed Now)

The detector must operate on a fixed categorical vocabulary.

Recommended canonical enums:

- `trend_state`: `up | down | sideways`
- `vol_state`: `low | mid | high | extreme`
- `structure_state`: `compression | expansion | mean_reverting | breakout_active | breakdown_active | neutral`

If existing code uses different labels (e.g., `uptrend`, `downtrend`, `normal`), add a
deterministic mapping layer and log the mapping version.

### 3. Bounded, Decomposable, Inspectable Distance

`regime_fingerprint_distance(curr, prev)` must be:

- bounded (`[0, 1]` recommended)
- decomposable into per-component contributions
- inspectable in telemetry (weights, deltas, threshold decisions)

### 4. Asymmetric Hysteresis + Debounce

The detector must use asymmetric thresholds and timing controls:

- `ENTER` threshold (higher)
- `EXIT` / persistence threshold (lower)
- minimum dwell time in a regime
- cooldown after transition

This matters more than exact weights for preventing oscillation in chop.

### 5. HTF-Close Gating by Default

Regime reassessment eligibility should default to HTF bar closes (e.g., 5m/15m for 1m
execution systems), unless a documented shock override fires.

This aligns policy updates with structure, not noise.

## `RegimeFingerprint` Schema Contract

## Required Fields

```python
class RegimeFingerprint(BaseModel):
    model_config = {"extra": "forbid"}

    fingerprint_version: str
    schema_version: str

    symbol: str
    scope: Literal["symbol", "cohort"]
    cohort_id: str | None = None

    as_of_ts: datetime
    bar_id: str
    source_timeframe: str              # timeframe used to evaluate transition (HTF by default)

    # Stable categorical states
    trend_state: Literal["up", "down", "sideways"]
    vol_state: Literal["low", "mid", "high", "extreme"]
    structure_state: Literal[
        "compression",
        "expansion",
        "mean_reverting",
        "breakout_active",
        "breakdown_active",
        "neutral",
    ]

    # Confidence values (normalized)
    trend_confidence: float = Field(ge=0.0, le=1.0)
    vol_confidence: float = Field(ge=0.0, le=1.0)
    structure_confidence: float = Field(ge=0.0, le=1.0)
    regime_confidence: float = Field(ge=0.0, le=1.0)

    # Normalized numeric components only (examples)
    vol_percentile: float = Field(ge=0.0, le=1.0)
    atr_percentile: float = Field(ge=0.0, le=1.0)
    volume_percentile: float = Field(ge=0.0, le=1.0)
    range_expansion_percentile: float = Field(ge=0.0, le=1.0)
    realized_vol_z: float
    distance_to_htf_anchor_atr: float
    trend_strength_z: float | None = None

    # Vector actually used for distance (explicit and replayable)
    numeric_vector: list[float]
    numeric_vector_feature_names: list[str]
```

## Fingerprint Construction Rules

- `numeric_vector` must contain only normalized features
- field ordering for `numeric_vector_feature_names` is fixed by version
- all components used by distance must be present in telemetry
- missing features must be explicit (`None` + imputation policy metadata), not silent

## Distance Contract: `regime_fingerprint_distance(curr, prev)`

## Output Contract

```python
class RegimeDistanceResult(BaseModel):
    distance_value: float                 # bounded [0, 1]
    threshold_enter: float
    threshold_exit: float
    threshold_used: float
    threshold_type: Literal["enter", "exit"]
    component_contributions: dict[str, float]
    component_deltas: dict[str, float | str]
    weights: dict[str, float]
    confidence_delta: float
    suppressed_by_hysteresis: bool = False
    suppressed_by_cooldown: bool = False
    suppressed_by_min_dwell: bool = False
```

## Distance Formula Requirements

The implementation must combine:

1. **Categorical mismatch terms** (weighted)
   - `trend_state` changed
   - `vol_state` changed
   - `structure_state` changed
2. **Normalized numeric distance**
   - weighted L1/L2/cosine over `numeric_vector` (choose and document)
3. **Smoothed confidence delta**
   - classifier confidence change (bounded contribution)

Constraints:

- final output must be bounded
- component contributions must sum (or approximately sum) to `distance_value`
- weights and formula version must be emitted in telemetry

## Transition Detector State Machine

## State

Per detector scope (`symbol`, and optionally `cohort`):

- `current_regime_fingerprint`
- `last_transition_ts`
- `current_regime_entered_ts`
- `cooldown_until_ts`
- `pending_candidate_regime` (optional)
- hysteresis/debounce counters

## Transition Logic (Required)

1. Build current fingerprint on eligible evaluation boundary (HTF close by default)
2. Compute `RegimeDistanceResult(curr, prev)`
3. Apply hysteresis:
   - use `ENTER` threshold for proposed regime transition
   - use `EXIT` / persistence threshold for reversion / stability checks
4. Apply `min_dwell` guard
5. Apply cooldown guard
6. Apply optional shock override
7. Emit decision:
   - `regime_state_changed = True/False`
   - `reason_code`
   - transition payload + suppression metadata

## Required Reason Codes

Examples (extend as needed):

- `distance_enter_threshold_crossed`
- `distance_below_threshold`
- `suppressed_hysteresis`
- `suppressed_min_dwell`
- `suppressed_cooldown`
- `shock_override_volatility_jump`
- `htf_gate_not_ready`
- `missing_required_features`

## Shock Override Path (Required)

Allow a controlled bypass of HTF-close gating and/or hysteresis for documented shocks,
for example:

- volatility percentile band jump
- realized vol z-score spike beyond threshold

Shock override must still respect cooldown repetition controls to avoid repeated firing.

## HTF-Close Gating (Default Behavior)

The detector should evaluate transitions on HTF closes by default:

- 1m execution -> assess on 5m or 15m closes
- 5m execution -> assess on 15m/1h closes

Exceptions:

- shock override
- explicit operator-triggered diagnostic evaluation (telemetry tagged)

## Telemetry (Non-Negotiable)

Every detector evaluation (fired or not) must emit:

- detector scope (`symbol` / `cohort`)
- `symbol`, `cohort_id`
- current + prior fingerprint versions
- `distance_value`
- `threshold_used` and threshold type
- per-component contributions and deltas
- weights / formula version
- hysteresis state
- dwell time and cooldown counters
- `regime_state_changed`
- `reason_code`
- `htf_gate_eligible` / `shock_override_used`

If you cannot explain why a transition fired (or did not fire), the detector is not
operationally acceptable.

## Integration Model (Runbooks 54 / 51 / 49)

### Policy Loop Integration (Runbook 54)

- symbol-local detector emits policy events consumed by the policy loop
- no per-tick LLM calls are introduced
- policy loop remains event-driven + heartbeat

### Memory Retrieval Integration (Runbook 51)

- cohort detector state influences retrieval priors/weights
- cohort/global regime is advisory for retrieval only
- symbol-local detector remains authoritative for symbol policy cadence

### Snapshot Integration (Runbook 49)

- `PolicySnapshot` must include normalized regime-fingerprint components and detector outputs
- `TickSnapshot` must not carry heavy transition diagnostics unless explicitly required for logging
- `PolicySnapshot.normalized_features` must be version-linked/mapped to the detector's
  `numeric_vector_feature_names` contract (`fingerprint_version` / feature-order version)

## Implementation Steps

### Step 1: Define schemas in `schemas/regime_fingerprint.py`

Create:

- `RegimeFingerprint`
- `RegimeDistanceResult`
- `RegimeTransitionDecision`
- `RegimeTransitionTelemetryEvent`
- `RegimeTransitionDetectorState`

### Step 2: Add vocabulary mapping layer

Map existing classifier outputs to canonical enums and persist a mapping version.

Examples:

- `uptrend` -> `up`
- `downtrend` -> `down`
- `normal` -> `mid`

### Step 3: Build normalized fingerprint constructor

Construct fingerprints from `PolicySnapshot`-compatible normalized fields only.
Reject construction if required normalized fields are missing (emit telemetry).

### Step 4: Implement bounded distance function

Implement and document:

- feature groups
- weights
- bounded output normalization
- component contribution reporting

### Step 5: Implement asymmetric hysteresis state machine

Add:

- `enter_threshold`
- `exit_threshold`
- `min_dwell`
- `cooldown`
- shock override rules
- HTF-close gating

### Step 6: Emit transition events and integrate policy loop

Emit `regime_state_changed` policy event payloads with reason codes and telemetry fields.
Policy loop consumes these events (Runbook 54).

### Step 7: Wire cohort detector for retrieval priors (optional first release)

If implementing both scopes incrementally:

- ship symbol-local detector first (policy cadence)
- add cohort detector second (retrieval weighting only)

## Acceptance Criteria

- [ ] `RegimeFingerprint` schema exists and is versioned
- [ ] Fingerprint distance vector contains normalized-only features (no raw symbol-scale fields)
- [ ] Fingerprint vector version/order is published and linkable from `PolicySnapshot.normalized_features`
- [ ] Categorical vocabulary is fixed and mapped deterministically from current classifier outputs
- [ ] `regime_fingerprint_distance(curr, prev)` returns bounded output with decomposed contributions and weights
- [ ] Transition detector uses asymmetric hysteresis (`enter` vs `exit`) plus min-dwell and cooldown
- [ ] Detector supports HTF-close gating by default and shock override with repetition controls
- [ ] Detector emits `regime_state_changed` with explicit `reason_code` and suppression metadata
- [ ] Telemetry is emitted for fired and non-fired evaluations with distance/threshold/component details
- [ ] `PolicySnapshot` integration preserves version alignment between normalized feature block and fingerprint vector contract
- [ ] Symbol-local detector drives policy cadence; cohort detector (if enabled) is advisory for retrieval priors only
- [ ] Same input series produces identical transition decisions (deterministic replay invariance)
- [ ] No per-tick LLM calls are introduced by detector integration

## Test Plan (Non-Negotiable)

```bash
# Schema + vocabulary mapping
uv run pytest tests/test_regime_fingerprint_schema.py -vv
uv run pytest tests/test_regime_vocabulary_mapping.py -vv

# Distance metric (bounded, decomposable, deterministic)
uv run pytest tests/test_regime_fingerprint_distance.py -vv

# Transition detector state machine: hysteresis / dwell / cooldown
uv run pytest tests/test_regime_transition_detector.py -vv

# Oscillation prevention around boundary (chop)
uv run pytest -k "regime_transition and hysteresis" -vv

# Shock override fires once, not repeatedly
uv run pytest -k "regime_transition and shock_override" -vv

# Under-trigger guard (real regime change not suppressed)
uv run pytest -k "regime_transition and under_trigger" -vv

# Deterministic replay invariance
uv run pytest -k "regime_transition and replay" -vv

# Policy-loop integration (no per-tick strategist/judge calls)
uv run pytest -k "policy_loop and regime_state_changed" -vv
```

## Human Verification Evidence

```text
TODO:
1. Run a symbol through a choppy boundary scenario and confirm hysteresis/min_dwell prevent
   oscillation (suppression reasons emitted).
2. Trigger a volatility shock override and confirm exactly one transition event fires,
   followed by cooldown suppression on subsequent bars.
3. Inspect non-fired transition telemetry and confirm distance, threshold, component
   contributions, and suppression reason are present.
4. Confirm policy-loop reevaluation occurs on HTF close / transition event without any
   per-tick LLM calls.
5. If cohort detector enabled, confirm cohort regime affects retrieval weighting only and
   does not directly force symbol policy reevaluation.
```

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-22 | Runbook created — deterministic regime fingerprint + transition detector for policy-loop cadence | Codex |

## Test Evidence (append results before commit)

```text
TODO
```

## Worktree Setup

```bash
git worktree add -b feat/regime-fingerprint-transition-detector ../wt-regime-transition main
cd ../wt-regime-transition

# When finished
git worktree remove ../wt-regime-transition
```

## Git Workflow

```bash
git checkout -b feat/regime-fingerprint-transition-detector

git add schemas/regime_fingerprint.py \
  services/regime_transition_detector.py \
  trading_core/regime_classifier.py \
  agents/analytics/indicator_snapshots.py \
  services/market_snapshot_builder.py \
  agents/strategies/plan_provider.py \
  services/memory_retrieval_service.py \
  tests/test_regime_fingerprint_schema.py \
  tests/test_regime_vocabulary_mapping.py \
  tests/test_regime_fingerprint_distance.py \
  tests/test_regime_transition_detector.py \
  docs/branching/55-regime-fingerprint-transition-detector.md \
  docs/branching/54-reasoning-agent-cadence-rules.md \
  docs/branching/49-market-snapshot-definition.md \
  docs/branching/51-memory-store-diversified-retrieval.md \
  docs/branching/README.md

uv run pytest tests/test_regime_fingerprint_schema.py tests/test_regime_fingerprint_distance.py tests/test_regime_transition_detector.py -vv

# Commit only after Test Evidence + Human Verification Evidence are recorded
git commit -m "feat: add deterministic regime fingerprint transition detector (Runbook 55)"
```
