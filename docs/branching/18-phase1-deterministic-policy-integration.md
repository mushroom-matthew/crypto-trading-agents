# Branch: phase1-deterministic-policy-integration

## Purpose
Define the deterministic policy layer for position expression while a strategy plan is active, with strict trigger gating and plan-scoped parameters.

This runbook is a production contract for real-capital operation.

## Source Plans
- docs/analysis/CONTINUOUS_POLICY_PIVOT_PLAN.md (Phase 1)

## Prerequisites (must be green before implementation)
- Emergency exits correct and auditable (runbooks 03-06).
- Judge robustness complete (runbooks 13, 16).
- Risk correctness complete (runbooks 14, 15).
- No-change replan guard active with suppression metrics (`X-policy-pivot-phase0.md`).
- Learning-book isolation operational (runbooks 09-12).
- Strategist simplification merged (runbook 01).

## 1) Scope and Non-Scope

### In Scope
- Mapping `trigger_state + signal_strength + vol_hat` to deterministic `target_weight`.
- Smoothing, bounding, and risk-capping exposure while a plan is active.
- Persisting policy decisions and overrides per bar for audit and attribution.

### Out of Scope / Deferred
- Trigger creation/discovery logic.
- Plan selection, replacement, or replan policy.
- Model training and calibration.
- Execution mechanics (order type, routing, slicing, venue adapter behavior).

## 2) Position of the Policy in the Plan Lifecycle

Canonical flow:

`Strategist -> Plan -> Triggers -> Policy -> Execution`

Contract:
- Policy runs continuously on each decision bar while a plan is active.
- Policy is trigger-gated at all times.
- `PolicyConfig` is fixed for the full plan lifecycle (until replan or plan expiry).
- There is no per-bar LLM trade decision path.

## 3) Trigger-Policy Contract (MANDATORY)

`trigger_state in {inactive, long_allowed, short_allowed, exit_only}`

Rules:
- `inactive` -> `target_weight = 0`
- `long_allowed` -> `target_weight >= 0`
- `short_allowed` -> `target_weight <= 0`
- `exit_only` -> `target_weight` must move monotonically toward `0`

Hard invariants:
- Policy must never create exposure when trigger permission is absent.
- Policy must never override trigger direction.
- If no trigger is active for a symbol/direction, `target_weight` is exactly `0`.

## 4) Policy Definition (Canonical Math)

Deterministic transformation:

`(signal_strength, vol_hat, policy_config, risk_overrides, position_state) -> target_weight`

Where direction is inherited from `trigger_state`:
- `dir = +1` for `long_allowed`
- `dir = -1` for `short_allowed`

Ordered evaluation (must be implemented in this order):

1. **Trigger gate**
   - If `trigger_state = inactive`: `target_weight_raw = 0`, `target_weight_final = 0` and stop.
   - If `trigger_state = exit_only`: set policy raw target to `0` and apply monotone decay to current position (step 6).
2. **Signal sanitation**
   - `s = clamp(signal_strength, 0, 1)`
3. **Deadband**
   - `s_db = 0` if `s < tau`, else `(s - tau) / (1 - tau)`
4. **Volatility scaling**
   - `vol_scale = clamp(vol_target / max(vol_hat, eps), 0, 1)`
5. **Raw magnitude + bounds**
   - `m_raw = s_db * vol_scale`
   - `m_bounded = 0` if `m_raw = 0` else `clamp(max(m_raw, w_min), w_min, w_max)`
   - `target_weight_raw = dir * m_bounded`
6. **Smoothing / monotone exit path**
   - Let `alpha = alpha_by_horizon[horizon_id]`, fixed deterministic lookup.
   - For `long_allowed/short_allowed`:  
     `target_weight_policy = (1 - alpha) * current_weight + alpha * target_weight_raw`
   - For `exit_only`:  
     `target_weight_policy = (1 - alpha) * current_weight`  
     (guarantees monotone convergence of `|target_weight|` toward `0` for `0 < alpha <= 1`)
7. **Risk caps**
   - `target_weight_capped = apply_risk_caps(target_weight_policy, risk_overrides)`
8. **Override precedence**
   - Apply section 6 in strict order to produce `target_weight_final`.

No randomness, sampling, or non-deterministic branching is allowed.
This definition is versioned and immutable per run.

## 5) PolicyConfig Contract (Plan-Level)

`PolicyConfig` is embedded in the plan and immutable intra-plan.

Required fields:
- `tau`: deadband threshold, `0 <= tau < 1`
- `vol_target`: volatility target used for scale normalization
- `w_min`: minimum non-zero absolute weight
- `w_max`: maximum absolute weight
- `horizon_id`: horizon key used for deterministic smoothing constants
- `symbol_allowlist`: symbols eligible for policy output under this plan
- `stand_down_state`: explicit stand-down state payload for forced neutralization

`stand_down_state` required members:
- `stand_down_until_ts`
- `stand_down_reason`
- `stand_down_source` (`judge | ops | risk_engine`)

Validation invariants:
- `0 <= w_min <= w_max <= 1`
- `symbol` not in `symbol_allowlist` -> `target_weight_final = 0`

## 6) Overrides & Precedence

Strict precedence:

`emergency_exit > stand_down > risk caps > policy output`

Semantics:
- `emergency_exit`: bypass policy math, force immediate flatten (`target_weight_final = 0`).
- `stand_down`: force neutral exposure while state is active (`target_weight_final = 0`).
- `risk caps`: clamp policy output to enforced limits.
- `policy output`: deterministic result from section 4.

All applied overrides must emit explicit reason codes.

## 7) Telemetry & Persistence

Persist per bar, per symbol (linked to `plan_id` and `trade_set`):
- `trigger_state`
- `signal_strength`
- `signal_deadbanded`
- `vol_hat`
- `vol_scale`
- `target_weight_raw`
- `target_weight_policy` (after smoothing)
- `target_weight_capped` (after risk caps)
- `target_weight_final`
- `delta_weight = target_weight_final - current_weight`
- applied override reasons and precedence tier
- `policy_config` version/hash

Contract:
- Every bar must produce a decision record, even when no trade occurs.
- Records must be replayable for deterministic audit.
- Judge must be able to separate trigger quality from policy behavior.

## 8) Acceptance Criteria

- Policy never creates exposure without trigger permission.
- Exposure magnitude is explainable independently of trigger logic.
- Judge can attribute outcomes to trigger quality vs policy behavior.
- No safety regressions from Phase 0 invariants.
- Replay determinism: same inputs produce same `target_weight_final`.
- Reduced churn vs baseline with comparable risk envelope.
- Every bar emits a complete decision record, including no-trade bars.

## 9) Delta-Weight Execution Interface (Boundary Contract)
Policy emits intent only:
- `delta_weight = target_weight_final - current_weight`
- ignore rebalance if `abs(delta_weight) <= epsilon_w`
- ignore rebalance if resulting notional is below `min_trade_notional`
- enforce `rebalance_cooldown_bars` between non-emergency rebalances

Execution owns order mechanics:
- order type selection, slicing, routing, and venue adapter behavior are out of scope.

## Relationship to Phase 2
- Phase 1 is mandatory.
- Phase 2 is optional and reversible.
- The only boundary input from Phase 2 is `signal_strength`.
- Trigger permission remains authoritative at all times.

## Key Files
- `agents/strategies/policy_engine.py` (new deterministic policy module)
- `agents/strategies/trigger_engine.py` (trigger-state handoff)
- `agents/strategies/risk_engine.py` (caps + stand-down interface)
- `backtesting/llm_strategist_runner.py` (policy integration in simulation path)
- `ops_api/event_store.py` (policy decision persistence)
- `schemas/` (PolicyConfig and DecisionRecord contracts)

## Test Plan (required before commit)
- `uv run pytest tests/test_policy_engine.py -vv`
- `uv run pytest tests/test_trigger_policy_contract.py -vv`
- `uv run pytest tests/test_policy_override_precedence.py -vv`
- `uv run pytest tests/test_policy_telemetry_persistence.py -vv`
- `uv run pytest tests/test_llm_strategist_runner.py -k policy -vv`

If tests cannot be run locally, obtain user-run output and paste it in Test Evidence before committing.

## Human Verification (required)
- Run a backtest with trigger inactivity periods and confirm `target_weight_final = 0` on all inactive bars.
- Run a backtest with `exit_only` and confirm monotone convergence toward flat.
- Trigger emergency exit and confirm policy math is bypassed with flattening.
- Confirm telemetry records include `plan_id`, `trade_set`, and override reasons.

## Worktree Setup (recommended for parallel agents)
Use a linked worktree so multiple branches can be worked on in parallel from one clone.

```bash
git fetch
git worktree add -b phase1-deterministic-policy-integration ../wt-phase1-deterministic-policy-integration main
cd ../wt-phase1-deterministic-policy-integration

# When finished (after merge)
git worktree remove ../wt-phase1-deterministic-policy-integration
```

## Git Workflow (explicit)
```bash
git checkout main
git pull
git checkout -b phase1-deterministic-policy-integration

git status
git diff

git add agents/strategies backtesting ops_api schemas tests docs/branching/18-phase1-deterministic-policy-integration.md docs/branching/README.md

uv run pytest tests/test_policy_engine.py -vv
uv run pytest tests/test_trigger_policy_contract.py -vv
uv run pytest tests/test_policy_override_precedence.py -vv
uv run pytest tests/test_policy_telemetry_persistence.py -vv

# Commit only after Test Evidence + Human Verification Evidence are recorded
git commit -m "Policy pivot phase1: deterministic trigger-gated policy contract"
```

## Change Log (update during implementation)
- YYYY-MM-DD: Summary of changes, files touched, and decisions.

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit when required)
