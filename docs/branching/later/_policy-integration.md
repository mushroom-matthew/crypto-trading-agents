# Branch: policy-integration (deferred)

## Purpose
Define, implement, and govern the deterministic trading policy layer that converts a directional belief and risk context into a target portfolio weight, independent of how that belief is produced.

This runbook establishes the canonical "inner loop" of trading behavior. Once in place, all strategies (LLM-driven, rules-based, or ML) must express intent through this policy.

## Source Plans
- docs/analysis/CONTINUOUS_POLICY_PIVOT_PLAN.md (Phase 1)

## Prerequisites (must all be green before starting)
- Emergency exits correct and auditable (runbooks 03-06)
- Judge robustness (runbooks 13, 16)
- Risk correctness (runbooks 14, 15)
- No-change replan guard active with metrics (later/_policy-pivot-phase0.md)
- Learning Book isolation wired (runbooks 09-12)
- Strategist simplification merged (runbook 01)

## Scope

### 1. Policy Definition (Authoritative Math)
The exact mathematical transformation:
```
(p_hat, vol_hat, policy_config, risk_overrides, position_state) -> target_weight
```
Including:
- Probability centering (2*p_hat - 1)
- Deadband logic (tau)
- Volatility scaling (sigma_star / vol_hat)
- Hard bounds (w_min / w_max)
- Ordering of operations (deadband before scaling, bounds last)

This definition is versioned and immutable per run.

### 2. PolicyConfig Contract
The only interface the Strategist/Judge may control:
- tau (deadband)
- vol_target
- w_min / w_max
- horizon identifier
- symbol allowlist
- stand_down state

Explicitly excludes: order types, triggers, stop placement, model architecture, feature logic.

### 3. Stand-Down & Override Semantics
How policy output is overridden (not when):
- stand_down -> target_weight := 0
- cooldown behavior (stand_down_until_ts)
- Precedence ordering: emergency exit > stand_down > risk caps > policy output

All overrides must be explicit, logged, and emit block events.

### 4. Delta-Weight Execution Interface
How policy output translates to execution intent (without owning execution):
- delta_w = target_w - current_w
- min_trade thresholds
- rebalance bands
- cooldowns between rebalances

Excludes: order slicing, post-only vs market logic, exchange adapters.

### 5. Deterministic Telemetry (Mandatory)
Per bar, per symbol:
- p_hat (raw input)
- centered signal
- deadbanded signal
- vol_hat
- target_weight_raw
- target_weight_final
- delta_weight
- override_reason(s)

This telemetry is the audit spine for all later model evaluation.

## Key Files
- New: policy engine module (TBD, likely `agents/strategies/policy_engine.py`)
- backtesting/llm_strategist_runner.py (integration point)
- agents/strategies/risk_engine.py (override/cap interface)
- ops_api/event_store.py (telemetry persistence)

## Acceptance Criteria
- Deterministic replay: same inputs -> same outputs
- Reduced churn vs baseline
- Emergency exits still obey Phase 0 guards
- Full DB persistence and UI visibility
- Every bar produces a complete decision record, even if no trade

## Out of Scope
- Train models
- Define features
- Choose prediction horizons
- Tune hyperparameters automatically
- Decide trade direction
- Perform backtest optimization

This runbook assumes p_hat is an input, not something to be derived here.

## Relationship to Model Integration
Interacts at exactly one seam: `p_hat -> Policy Engine -> target_weight`. Nothing else crosses the boundary.

## Change Log
- 2026-01-29: Initial scope definition from agent conversation.
