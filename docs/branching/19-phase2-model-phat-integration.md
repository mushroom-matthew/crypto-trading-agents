# Branch: phase2-model-phat-integration

## Purpose
Define how model-generated directional belief (`p_hat`) is produced and supplied to the deterministic policy engine as `signal_strength` only.

This runbook is a production contract for real-capital operation.

## Source Plans
- docs/analysis/CONTINUOUS_POLICY_PIVOT_PLAN.md (Phase 2)

## Prerequisites (must be green before implementation)
- Phase 1 deterministic policy integration merged and stable (turnover/trade count/drawdown within accepted bands vs baseline).
- Learning-book isolation operational (runbooks 09-12).

## 1) Scope and Non-Scope

### In Scope
- Producing calibrated `p_hat` under a versioned model contract.
- Mapping `p_hat` to `signal_strength` for policy magnitude control.
- Defining label/horizon semantics and feature contracts for deterministic inference.
- Model artifact/version governance, promotion criteria, rollback path.
- Per-bar telemetry for model attribution, calibration, and fallback.

### Out of Scope / Prohibited
- Trigger creation/discovery.
- Direction or permission decisions.
- Trade initiation, execution control, or order behavior.
- Mid-plan strategy edits outside replan flow.
- Policy math changes (owned by Phase 1 contract).

## 2) Position in the Plan Lifecycle

Canonical flow:

`Strategist -> Plan (optional model_binding) -> Triggers -> Signal Source -> Policy -> Execution`

Binding contract:
- Model binding occurs only at plan creation or replan.
- Model binding is immutable while a plan is active.
- No per-bar model swapping.

## 3) Trigger-Model-Policy Authority Contract (MANDATORY)

Authority boundaries:
- Trigger state is authoritative for permission and direction.
- Policy enforces exposure bounds and overrides.
- Model provides magnitude signal only.

Hard rules:
- `p_hat` modulates magnitude only; it never grants permission and never sets direction.
- If `trigger_state = inactive`, policy output remains `target_weight = 0` regardless of `p_hat`.
- If `trigger_state = exit_only`, model input is ignored for new exposure and policy decays to flat.

Directional-safe `p_hat -> signal_strength` mapping:
- `long_allowed`: `signal_strength = clamp(2 * (p_hat - 0.5), 0, 1)`
- `short_allowed`: `signal_strength = clamp(2 * (0.5 - p_hat), 0, 1)`
- `inactive/exit_only`: `signal_strength = 0`

This mapping guarantees model output cannot invert trigger direction.

## 4) Model Binding Contract (Plan-Level)

`model_binding` must be present in the plan when `signal_source = model`.

Required fields:
- `signal_source`: `proxy | model`
- `model_id`
- `model_version_hash`
- `feature_version`
- `label_definition`
- `horizon_id` (must match `PolicyConfig.horizon_id`)
- `calibration_version`
- `training_window`
- `data_cutoff_ts`
- `fallback_signal_source` (typically `proxy`)

Immutability:
- Binding values cannot change intra-plan.
- Any model change requires a replan and a new `plan_id`.

## 5) Signal Semantics Contract (Label, Features, Calibration)

### Label & Horizon definition
- `horizon_bars` and `horizon_id` must be explicit and stored with the artifact.
- `label_definition` (return sign/threshold, aggregation rules) must be explicit.
- Artifact horizon must match `PolicyConfig.horizon_id`.

### Feature contract
- Exact feature set and ordering are versioned (`feature_version` hash).
- Feature normalization and missing-data handling are fixed by version.
- Policy layer never consumes features directly; it receives only `signal_strength`.

### Model architecture discipline
- Start with auditable baseline models (logistic/linear/tree) before complex sequence models.
- Inference must be deterministic; training must be reproducible from metadata.
- No agent-driven auto-architecture changes in production plans.

### Calibration pipeline (first-class)
- Bound calibration method/version (e.g., Platt/temperature) is mandatory.
- Held-out calibration evaluation is required.
- Required calibration metrics: `Brier`, `ECE`, reliability curves.
- Uncalibrated artifacts are invalid for promotion.

## 6) Online Signal Production Pipeline (Deterministic)

Per bar, per symbol:
1. Load bound artifact by `model_version_hash`.
2. Build features using `feature_version`.
3. Run deterministic inference to produce `p_hat_raw`.
4. Apply bound calibration transform to produce `p_hat_calibrated`.
5. Convert to `signal_strength` using section 3 mapping.
6. Pass only `signal_strength` to policy engine.

Failure handling:
- On artifact/feature/calibration failure, fail closed to configured fallback source.
- Emit explicit fallback reason code and persist it with decision telemetry.

## 7) Evaluation, Promotion, and Rollback Contract

Promotion requires all gates:
- Calibration improvement (`Brier`, `ECE` and reliability curve checks).
- No material regression in churn/slippage/drawdown vs proxy baseline.
- Learning-book isolation compliance for model-sourced trades.

Rollback contract:
- Replan with `signal_source = proxy` is sufficient to disable model usage.
- No policy or trigger contract changes required for rollback.
- Canonical switch:
  ```
  signal_source := proxy | model_vX
  ```

## 8) Telemetry & Persistence

Persist per bar, per symbol (linked to `plan_id` and `trade_set`):
- `trigger_state`
- `signal_source`
- `model_id`, `model_version_hash`, `feature_version`, `calibration_version`
- `p_hat_raw`, `p_hat_calibrated`
- `signal_strength`
- fallback/applied override reason codes
- downstream `target_weight_raw/final` linkage for attribution

Contract:
- Every bar must have a model-signal decision record, including fallback cases.
- Judge must be able to attribute outcomes separately to trigger quality, policy behavior, and signal source quality.

## 9) Acceptance Criteria

- `p_hat` modulates magnitude only, never permission or direction.
- Model binding changes only at plan creation or replan.
- No model-generated exposure is possible without trigger permission.
- Model failures degrade safely via explicit fallback to proxy signal source.
- Phase 2 remains optional and fully reversible without changing trigger or policy contracts.
- Model artifacts include complete metadata (`model_version_hash`, `feature_version`, `label_definition`, `horizon_id`, calibration metadata, training window).
- Promotion gates enforce no-ship on materially worse churn/slippage/drawdown.

## Data & Compute Notes
- Data: multi-regime OHLCV with funding/spread/VWAP where available.
- Minimum history: 6-12 months per timeframe for initial promotion.
- Compute: CPU acceptable for 15m/1h and few symbols; GPU recommended for 1m/multi-asset.

## Critical Decision (lock early)
- Choose v1 horizon/cadence explicitly; default recommendation is 15m unless microstructure edge is proven.

## Relationship to Phase 1
- Phase 1 is mandatory.
- Phase 2 is optional and reversible.
- Only one datum crosses the boundary: `signal_strength`.
- Trigger permission remains authoritative at all times.

## Key Files
- `models/` or `trading_core/models/` (artifact load + inference path)
- `trading_core/signal_source.py` (model/proxy signal interface)
- `agents/strategies/policy_engine.py` (Phase 1 consumer)
- `backtesting/llm_strategist_runner.py` (binding + evaluation wiring)
- `ops_api/event_store.py` (model signal telemetry persistence)
- `schemas/` (ModelBinding + signal telemetry contracts)

## Test Plan (required before commit)
- `uv run pytest tests/test_model_binding_contract.py -vv`
- `uv run pytest tests/test_phat_to_signal_mapping.py -vv`
- `uv run pytest tests/test_model_signal_fallback.py -vv`
- `uv run pytest tests/test_policy_model_boundary.py -vv`
- `uv run pytest tests/test_model_telemetry_persistence.py -vv`

If tests cannot be run locally, obtain user-run output and paste it in Test Evidence before committing.

## Human Verification (required)
- Confirm a plan with `signal_source=model` uses a fixed model artifact across all bars.
- Confirm trigger-disabled periods still force `target_weight_final = 0` even with high `p_hat`.
- Induce a model load or feature failure and confirm fallback to proxy with reason logging.
- Compare model vs proxy runs and verify attribution fields are complete.

## Worktree Setup (recommended for parallel agents)
Use a linked worktree so multiple branches can be worked on in parallel from one clone.

```bash
git fetch
git worktree add -b phase2-model-phat-integration ../wt-phase2-model-phat-integration main
cd ../wt-phase2-model-phat-integration

# When finished (after merge)
git worktree remove ../wt-phase2-model-phat-integration
```

## Git Workflow (explicit)
```bash
git checkout main
git pull
git checkout -b phase2-model-phat-integration

git status
git diff

git add trading_core models backtesting ops_api schemas tests docs/branching/19-phase2-model-phat-integration.md docs/branching/README.md

uv run pytest tests/test_model_binding_contract.py -vv
uv run pytest tests/test_phat_to_signal_mapping.py -vv
uv run pytest tests/test_model_signal_fallback.py -vv
uv run pytest tests/test_policy_model_boundary.py -vv

# Commit only after Test Evidence + Human Verification Evidence are recorded
git commit -m "Policy pivot phase2: model p_hat signal-source integration contract"
```

## Change Log (update during implementation)
- YYYY-MM-DD: Summary of changes, files touched, and decisions.

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)
