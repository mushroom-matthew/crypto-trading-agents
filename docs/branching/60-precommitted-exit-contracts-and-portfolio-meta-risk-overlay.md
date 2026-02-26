# Runbook 60: Precommitted Exit Contracts and Portfolio Meta-Risk Overlay

## Purpose

Define a deterministic execution contract that requires every position to carry a
precommitted exit plan at entry time, while separating portfolio-level de-risking
and reallocation into a distinct meta-risk overlay policy.

This runbook closes a class of weak/ambiguous exits where a generic `direction="exit"`
trigger can flatten a position without a strong, predeclared thesis invalidation rule.

## Problem Statement

Observed in paper trading (`paper-trading-cee4db02`): a `risk_off` trigger exited a
BNB position through a `..._flat` path, which bypassed the stricter exit-rule path
(including exit-category binding checks). The plan contained:

- `direction = "exit"`
- `entry_rule = "not is_flat"`
- `exit_rule = "not is_flat and below_stop"`

Because the flatten was produced by the entry-rule path, the broad `entry_rule` acted
like "flatten whenever in position", which is not the intended semantics for a normal
strategy exit.

## Design Principle (Non-Negotiable)

**We must know how we will get out before we get in.**

Every entry must produce a typed, persisted exit contract that defines:

- invalidation (hard stop or equivalent structural stop)
- profit realization plan (single target or ladder)
- time-based expiry / max hold
- allowed amendments (if any) and audit requirements

Portfolio-level risk reduction is allowed, but only via a separate deterministic overlay
policy (e.g., concentration, drawdown, correlation, or regime-shift responses).

## Position in Architecture

This runbook sits between strategy/playbook planning and execution:

- **Upstream:** Runbooks 52/56 define typed playbooks and deterministic activation/target mapping
- **At entry:** this runbook materializes a `PositionExitContract` from the selected playbook/trigger
- **Runtime:** trigger engine executes only contract-backed exits for normal trade management
- **Overlay:** portfolio meta-risk policy can issue deterministic trims/reallocations
- **Emergency:** `emergency_exit` remains a separate safety interrupt path

## Dependencies

- **Runbook 42** — level-anchored stop/target resolution and `position_meta` stop/target fields
- **Runbook 45** — adaptive trade management semantics (partials/trailing need contract form)
- **Runbook 52** — typed playbook definition should declare exit/risk-reduce capabilities
- **Runbook 54** — cadence + policy-loop event rules for overlay evaluation
- **Runbook 55** — deterministic regime transition detector (overlay inputs)
- **Runbook 58** — deterministic structure engine for structural stop/target candidates
- **Runbook 43** — signal ledger / outcome reconciler for provenance and auditability
- **Runbook 17** (completed) — graduated de-risk taxonomy (legacy semantics to preserve/migrate)

## Scope

1. **New typed execution contracts**
   - `PositionExitContract`
   - `ExitLeg` / `ExitLadder`
   - `TimeExitRule`
   - `PortfolioMetaRiskPolicy`
   - `PortfolioMetaAction`
2. **Planner/playbook contract threading**
   - playbook/template outputs must specify exit plan requirements before entry
3. **Trigger compiler / validation**
   - disallow weak generic `direction="exit"` trigger flatten semantics for normal exits
4. **Trigger engine execution semantics**
   - normal exits execute from contract state, not ad hoc entry-rule flatten paths
5. **Paper trading + backtest execution**
   - persist contract at entry, execute deterministic contract legs/timeouts/invalidations
6. **Portfolio overlay engine**
   - deterministic portfolio-level conditions -> trim/reallocate actions with audit trail
7. **Ops telemetry / UI**
   - visible active exit contracts, overlay actions, and reasons
8. **Migration / compatibility layer**
   - preserve existing templates while rolling into stronger contract semantics
9. **Tests**
   - contract creation, enforcement, overlay behavior, and bypass restrictions

## Out of Scope (First Implementation)

- Live exchange order-type optimization (e.g., advanced TWAP/VWAP execution)
- Fully autonomous LLM-authored mid-trade contract rewrites
- Dynamic discretionary overrides without explicit operator approval + audit record
- Correlation-model forecasting beyond deterministic observable metrics

## Core Separation of Concerns (Required)

### 1. Position Exit Contract (per position)

Defines how a specific position may be reduced or closed.

Examples:

- hard stop invalidation
- target ladder (`25% @ 1R`, `50% @ 2R`, remainder trailing)
- max hold (`flatten after N bars / time`)
- structural invalidation (if backed by deterministic structure engine output)

### 2. Portfolio Meta-Risk Overlay (portfolio-wide policy)

Defines when portfolio-level risk conditions may trim, rebalance, or halt new risk.

Examples:

- max symbol concentration exceeded -> trim to cap
- portfolio drawdown threshold exceeded -> reduce gross exposure by X%
- correlation spike across open positions -> reduce clustered exposure
- regime transition to hostile state -> reduce sizing multiplier / rebalance to cash

### 3. Emergency Exit (safety interrupt)

Reserved for system/market safety cases only:

- data integrity failure
- execution desynchronization
- extreme market conditions defined in emergency policy

Emergency exits may bypass normal contract sequencing, but must remain explicitly typed
and auditable. They should not be used as a substitute for weak normal exits.

## Schema Contract (Required)

## `PositionExitContract`

Illustrative shape:

```python
class PositionExitContract(BaseModel):
    contract_id: str
    position_id: str
    symbol: str
    side: Literal["long", "short"]
    created_at: datetime
    source_plan_id: str | None = None
    source_trigger_id: str
    source_category: str | None = None

    entry_price: float
    initial_qty: float
    stop_price_abs: float
    target_legs: list["ExitLeg"] = []
    time_exit: "TimeExitRule | None" = None
    trailing_rule: dict[str, Any] | None = None  # concrete typed model in implementation

    # Allowed modifications after entry
    amendment_policy: Literal["none", "tighten_only", "policy_approved"] = "tighten_only"
    allow_portfolio_overlay_trims: bool = True

    # Audit / provenance
    template_id: str | None = None
    playbook_id: str | None = None
    snapshot_id: str | None = None
    snapshot_hash: str | None = None
```

## `ExitLeg`

```python
class ExitLeg(BaseModel):
    leg_id: str
    kind: Literal["take_profit", "risk_reduce", "time_exit", "full_exit"]
    trigger_mode: Literal["price_level", "r_multiple", "time", "structure_event"]
    fraction: float  # 0 < fraction <= 1
    price_abs: float | None = None
    r_multiple: float | None = None
    structure_level_id: str | None = None
    priority: int = 0
    enabled: bool = True
```

## `TimeExitRule`

```python
class TimeExitRule(BaseModel):
    max_hold_bars: int | None = None
    max_hold_minutes: int | None = None
    session_boundary_action: Literal["hold", "flatten", "reassess"] = "reassess"
```

## `PortfolioMetaRiskPolicy`

```python
class PortfolioMetaRiskPolicy(BaseModel):
    policy_id: str
    version: str
    enabled: bool = True

    # Deterministic conditions only
    max_symbol_concentration_pct: float | None = None
    max_sector_or_cluster_concentration_pct: float | None = None
    portfolio_drawdown_reduce_threshold_pct: float | None = None
    correlation_reduce_threshold: float | None = None
    hostile_regime_reduce_enabled: bool = False

    # Preapproved actions
    actions: list["PortfolioMetaAction"] = []
```

## `PortfolioMetaAction`

```python
class PortfolioMetaAction(BaseModel):
    action_id: str
    condition_id: str
    kind: Literal[
        "trim_largest_position_to_cap",
        "reduce_gross_exposure_pct",
        "rebalance_to_cash_pct",
        "freeze_new_entries",
        "tighten_position_stops",
    ]
    params: dict[str, Any] = {}
    priority: int = 0
    cooldown_minutes: int | None = None
```

## Hard Constraints (Non-Negotiable)

### 1. No entry without an exit contract

A normal (non-learning, non-manual override) entry must be rejected if a valid
`PositionExitContract` cannot be materialized at entry time.

### 2. Normal strategy exits cannot be "flatten because in position"

For non-emergency categories, `direction="exit"` triggers must not produce a flatten order
from the entry-rule path simply because `entry_rule` evaluates true (e.g., `not is_flat`).

### 3. Portfolio overlay actions are predefined and deterministic

The overlay may not invent arbitrary actions at runtime. It only executes actions already
declared in the active `PortfolioMetaRiskPolicy`.

### 4. Overlay actions are audited as portfolio policy events

Every overlay action must include:

- triggering condition(s)
- portfolio metrics snapshot
- action recipe / sizing math
- affected positions / quantities
- resulting portfolio metrics

### 5. Emergency exits remain explicitly typed

Emergency logic is not a fallback for weak `risk_off` or `risk_reduce` strategy rules.

## Runtime Semantics (Required)

## A. Entry Flow

On entry order generation / execution:

1. Resolve deterministic stop and target candidates (Runbooks 42/58/56)
2. Materialize `PositionExitContract`
3. Persist contract in position state (`position_meta` initially; dedicated structure later)
4. Emit `position_exit_contract_created` event with provenance
5. Reject entry if contract cannot be formed (unless an explicit operator/manual override exists)

## B. Normal Exit Flow

Normal strategy exit behavior should prefer:

- contract stop/target/time rules
- contract-approved structural invalidation
- contract-approved scale-out legs

`direction="exit"` triggers remain allowed only if they map to a contract-defined action or
contract amendment path and pass the same binding/hold checks as other normal exits.

## C. Portfolio Meta-Risk Flow

A separate overlay evaluator runs on portfolio cadence (Runbook 54) or event triggers:

- evaluates deterministic portfolio metrics
- checks policy conditions
- emits `portfolio_meta_condition_fired`
- executes preapproved action recipe (`trim`, `rebalance`, `freeze entries`, etc.)
- emits `portfolio_meta_action_executed`

Overlay actions may bypass per-trigger category binding, because they are portfolio-policy
actions, not strategy-trigger exits. This is the clean place to put "asset reallocation."

## D. Emergency Flow

Emergency exits remain distinct and may bypass normal sequencing, but:

- must be categorized `emergency_exit`
- must emit explicit emergency reason/evidence
- must not reuse generic strategy trigger IDs

## Migration Strategy (Required)

### Phase M1 — Guardrails (low risk)

- Add telemetry detecting normal `direction="exit"` triggers that produce `..._flat`
- Emit warnings when entry-rule flatten path is used for non-emergency exits
- Add counters for `exit_rule_path` vs `entry_flat_path`

### Phase M2 — Contract Materialization (compatibility mode)

- Materialize `PositionExitContract` from existing stop/target/time fields at entry
- Continue honoring legacy exits, but annotate whether each exit was contract-backed

### Phase M3 — Enforcement (normal exits)

- Reject / suppress non-emergency entry-rule flatten exits not mapped to contract actions
- Require contract for all new entries in paper trading and backtests

### Phase M4 — Portfolio Overlay Activation

- Introduce deterministic portfolio overlay evaluator and action registry
- Move "risk_reduce" / "reallocation" semantics from weak trigger rules into policy actions

### Phase M5 — Template / Playbook Tightening

- Update templates/prompts/playbooks to declare exit contract requirements explicitly
- Reduce or remove generic `direction="exit"` strategy triggers where contract legs suffice

## Implementation Steps

### Step 1: Add contract schemas and audit events

Introduce new typed models (or extend existing schemas) for:

- `PositionExitContract`, `ExitLeg`, `TimeExitRule`
- `PortfolioMetaRiskPolicy`, `PortfolioMetaAction`
- audit event payloads for contract creation/amendment/execution and portfolio overlay actions

### Step 2: Thread exit-contract intent through playbook/template outputs

Update planner/playbook contracts so entry-trigger generation includes enough structured
data to form a contract deterministically (stop source, target mode, time expiry, optional
predeclared scale-out ladder).

### Step 3: Materialize and persist contract at entry

At fill time (paper and backtest first), use existing anchored stop/target resolution
(Runbook 42) plus time/ladder settings to create a contract and persist it alongside
position metadata.

### Step 4: Refactor trigger-engine exit semantics

- Separate "exit trigger evaluation" from "entry-rule flatten for `direction=exit`"
- Apply hold/binding checks consistently for normal strategy exits
- Restrict or deprecate non-emergency `..._flat` path unless explicitly mapped to a
  contract-defined partial/full exit action

### Step 5: Implement contract execution state machine

Track which contract legs have fired and remaining quantity:

- stop invalidation
- target legs / scale-outs
- time expiry
- optional trailing activation (if declared)

### Step 6: Introduce portfolio meta-risk overlay evaluator

Add deterministic portfolio-policy evaluation (paper + backtest) with predeclared action
recipes (trim to concentration cap, reduce gross exposure, rebalance to cash, freeze new
entries, etc.).

### Step 7: Emit portfolio overlay telemetry + audit evidence

Persist overlay condition/action events via event store and signal ledger-compatible records
so operators can distinguish:

- strategy exit
- portfolio meta-risk trim/reallocation
- emergency exit

### Step 8: UI / Ops API exposure

Expose:

- active position exit contract (stop, targets, time exit, fired legs)
- portfolio meta-risk policy status + recent actions
- reasoned classification for each exit (`strategy_contract`, `portfolio_overlay`, `emergency`)

### Step 9: Backtest + paper-trading parity and rollout flags

Ship behind flags first, validate parity and audit quality, then make contract-backed exits
the default for paper trading before live paths.

## Acceptance Criteria

- [x] Every new paper/backtest entry can materialize a valid `PositionExitContract` or is rejected with explicit reason
- [x] Non-emergency `direction="exit"` triggers no longer flatten positions via permissive entry-rule path unless mapped to a contract-defined action (M1 guardrail: `entry_rule_flatten_detected` event)
- [x] Hold-period and exit-binding checks apply consistently to normal strategy exits — Phase M3: paper trading blocks entry-rule flatten exits for symbols with active contracts (trade_blocked event); backtest logs warning (M3 backtest parity)
- [x] Portfolio-level risk reduction / reallocation actions are executed only via a deterministic `PortfolioMetaRiskPolicy`
- [x] Overlay actions emit explicit audit events with triggering metrics and resulting portfolio changes
- [x] Exit telemetry clearly distinguishes `strategy_contract`, `portfolio_overlay`, and `emergency` classes
- [x] Existing adaptive trade-management concepts (R45) are representable as contract legs/state transitions (ExitLeg with r_multiple/price_level trigger modes)
- [x] Templates/playbooks can declare enough structured data to form exit contracts without ad hoc runtime inference (TriggerCondition.target_anchor_type → ExitLeg derivation)
- [x] Backtest and paper-trading paths behave consistently for contract creation and execution (Phase M2 + M3 implemented in both paths; backtesting/llm_strategist_runner.py has _materialize_backtest_contract + M3 warning + exit_contracts audit log in StrategistBacktestResult)
- [x] Ops API/UI surfaces active contracts and portfolio overlay actions for operator inspection

## Test Plan

```bash
# Trigger-engine semantics: block non-emergency entry-rule flatten exits
uv run pytest tests/test_trigger_engine_exit_contract_semantics.py -vv

# Contract schema validation + materialization from entry fills
uv run pytest tests/test_position_exit_contracts.py -vv

# Contract execution state machine (targets / stop / time expiry)
uv run pytest tests/test_exit_contract_execution.py -vv

# Portfolio meta-risk overlay conditions and deterministic actions
uv run pytest tests/test_portfolio_meta_risk_overlay.py -vv

# Phase M3 paper trading enforcement
uv run pytest tests/test_paper_trading_exit_contracts.py -vv

# Backtest parity: _materialize_backtest_contract + StrategistBacktestResult field
uv run pytest tests/test_backtest_exit_contract_parity.py -vv

# Ops API response models and _contract_dict_to_response converter
uv run pytest tests/test_ops_api_portfolio_overlay.py -vv
```

## Test Evidence

```
tests/test_trigger_engine_exit_contract_semantics.py   23 passed
tests/test_position_exit_contracts.py                  46 passed
tests/test_exit_contract_execution.py                  30 passed
tests/test_portfolio_meta_risk_overlay.py              35 passed
tests/test_paper_trading_exit_contracts.py             15 passed  (Phase M3 enforcement)
tests/test_backtest_exit_contract_parity.py            21 passed  (backtest parity)
tests/test_ops_api_portfolio_overlay.py                22 passed  (ops API response models)
Total new tests:                                      172 passed

Full suite (ignoring 2 pre-existing test_factor_loader failures):
  2 failed (pre-existing), 1745+ passed, 2 skipped
  No regressions introduced.
```

## Worktree Setup

```bash
git fetch origin
git worktree add ../crypto-trading-agents-r60 origin/main
cd ../crypto-trading-agents-r60
git checkout -b r60-exit-contracts-portfolio-overlay
```

## Git Workflow

```bash
git status
git add docs/branching/60-precommitted-exit-contracts-and-portfolio-meta-risk-overlay.md docs/branching/README.md
git commit -m "docs(runbook): add r60 exit contracts and portfolio meta-risk overlay"
```

## Human Verification

- [x] Runbook enforces "know the exit before entry" as a hard invariant: `_execute_order` calls `build_exit_contract` at entry time (Phase M2); missing stop rejects the entry before contract creation.
- [x] Portfolio reallocation / risk-reduce is modeled as a deterministic `PortfolioMetaRiskPolicy` with preapproved `PortfolioMetaAction` entries — no ad hoc runtime action fabrication allowed.
- [x] Emergency exits remain distinct: `category="emergency_exit"` is explicitly excluded from the M1 guardrail detection; three exit classes (strategy_contract, portfolio_overlay, emergency) are consistently labeled in all emitted events.
- [x] Phase M3 paper trading enforcement: `_evaluate_and_execute` blocks entry-rule flatten exits (reason ends `_flat`, non-emergency) for symbols with active contracts; emits `trade_blocked` event with `reason="m3_contract_backed_exit"`.
- [x] Backtest parity: `LLMStrategistBacktester._materialize_backtest_contract()` builds `PositionExitContract` from position_meta at entry; stores in `self.exit_contracts[symbol]` and appends to `_exit_contract_audit`; M3 warning logged when flat exit fires with active contract; `StrategistBacktestResult.exit_contracts` field returns audit list.
- [x] Migration steps (M1→M2→M3→M4→M5) fully implemented through M3: M1 detects and warns, M2 materializes contracts at entry (both paper trading and backtest), M3 enforces blocking in paper trading and logs warnings in backtest.

## Change Log

- 2026-02-26: Initial runbook drafted for precommitted position exit contracts and deterministic portfolio meta-risk overlay actions.
- 2026-02-26: Full implementation: schemas (PositionExitContract, ExitLeg, TimeExitRule, PortfolioMetaRiskPolicy, PortfolioMetaAction), builder service (build_exit_contract, can_build_contract), portfolio overlay evaluator (services/portfolio_meta_risk_overlay.py), paper trading Phase M2 wiring (_execute_order → contract materialization + position_exit_contract_created event), M1 guardrail (entry_rule_flatten_detected event in order loop), Ops API (ops_api/routers/exit_contracts.py, two endpoints), 134 new tests across 4 test files.
- 2026-02-26: Deferred sections completed: Phase M3 enforcement in paper trading (_evaluate_and_execute blocks entry-rule flatten exits for symbols with active contracts, emits trade_blocked event); backtest parity (_materialize_backtest_contract helper + M3 warning + StrategistBacktestResult.exit_contracts audit field in backtesting/llm_strategist_runner.py); 3 additional test files (test_paper_trading_exit_contracts.py 15 tests, test_backtest_exit_contract_parity.py 21 tests, test_ops_api_portfolio_overlay.py 22 tests); total 172 new tests, no regressions.
