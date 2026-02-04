# Branch: model-integration (deferred)

## Purpose
Define how directional belief sources (p_hat) are produced, calibrated, versioned, evaluated, and safely swapped without changing policy or execution behavior.

This runbook is about signal quality, not trading behavior.

## Source Plans
- docs/analysis/CONTINUOUS_POLICY_PIVOT_PLAN.md (Phase 2)

## Prerequisites
- Policy integration runbook merged and stable (turnover, trade count, drawdown within acceptable bands vs baseline)
- Learning Book isolation operational (runbooks 09-12)

## Scope

### 1. Label & Horizon Definition
What "direction" means:
- Horizon in bars (e.g., next 15m bar)
- Label definition (sign of return, thresholded return, etc.)
- Aggregation rules if multi-bar
- Stored with the model artifact; must match policy horizon

### 2. Feature Contract
- Exact feature set (e.g., 64x12)
- Feature normalization
- Missing-data handling
- Feature version hash

The policy layer never sees features, only p_hat.

### 3. Model Architecture (Pluggable)
May include:
- Baseline (logistic / linear / tree)
- TCN or other sequence models

Enforces:
- Small, auditable architectures
- Deterministic inference
- Reproducible training

No auto-architecture or agent-driven modeling.

### 4. Calibration Pipeline (First-Class)
- Calibration method (Platt, temperature)
- Held-out calibration set
- Metrics: Brier score, ECE, reliability curves

Uncalibrated models are invalid for promotion.

### 5. Artifact & Versioning Discipline
Each model artifact stores:
- model hash
- feature version
- label definition
- horizon
- calibration params
- training window

Backtests must reference these explicitly.

### 6. Promotion & Rollback Rules
Objective gates:
- Calibration improvement
- No regression in churn, slippage, drawdown
- Learning-book isolation compliance

Rollback is a one-line config change:
```
signal_source := proxy | model_vX
```

## Key Files
- New: model training/inference module (TBD)
- New: calibration pipeline (TBD)
- backtesting/llm_strategist_runner.py (signal source swap point)
- Policy engine module (consumer of p_hat)

## Acceptance Criteria
- Model artifact includes all required metadata
- Calibration metrics (Brier, ECE) computed and stored
- Promotion gate enforced: no ship if churn/slippage worsen materially
- Rollback to proxy p_hat is one config change
- Learning-book isolation tags propagated for all model-sourced trades

## Out of Scope
- Define target weights (policy integration owns this)
- Modify policy math
- Control execution
- Override risk caps
- Decide when to trade

A model that "needs" special handling belongs in research, not production.

## Data & Compute Notes
- Data: multi-regime OHLCV; funding/spread/VWAP where possible
- Minimum: 6-12 months per timeframe
- Compute: CPU ok for 15m/1h and few symbols; GPU recommended for 1m or multi-asset

## Critical Decision (lock early)
- Choose horizon/cadence for v1: 15m recommended unless microstructure edge is proven

## Change Log
- 2026-01-29: Initial scope definition from agent conversation.
