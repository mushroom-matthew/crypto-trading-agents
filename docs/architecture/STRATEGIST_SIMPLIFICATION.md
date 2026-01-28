# Strategist Simplification Proposal

## Problem Statement

The current LLM strategist architecture has several issues:

1. **Over-prompting**: 10+ prescriptive requirements force template generation
2. **Risk redundancy**: LLM outputs risk_constraints but user settings override them anyway
3. **Forced triggers**: No path for "no trades warranted" stance
4. **Monolithic prompt**: Strategic knowledge embedded instead of retrieved

## Current Flow (Problematic)

```
User Config (risk limits) ──┐
                            ├──> LLM Prompt (huge, prescriptive)
Strategic Rules (embedded) ─┘
                            │
                            ▼
                      LLM generates
                      - risk_constraints (ignored)
                      - 5+ required triggers
                      - sizing_rules
                            │
                            ▼
             _merge_plan_risk_constraints()
             overwrites LLM risk choices with user config
                            │
                            ▼
                     Plan Execution
```

## Proposed Flow (Simplified)

```
User Config ─────────────────────────────────────────┐
                                                     │
Market State ──┐                                     │
               │                                     │
               ▼                                     │
       Regime Classifier                             │
       (deterministic or ML)                         │
               │                                     │
               ├── regime: bull/bear/range/volatile  │
               │                                     │
               ▼                                     │
       Vector Store Query                            │
       (regime-specific strategies)                  │
               │                                     │
               ▼                                     │
       Simple LLM Prompt                             │
       "Given regime X and retrieved strategies,    │
        decide: trade or wait?"                     │
               │                                     │
               ├── Option A: Generate triggers      │
               ├── Option B: Set regime alerts      │
               └── Option C: Hold current position  │
                            │                        │
                            ▼                        │
                     Risk Engine ◄───────────────────┘
                     (enforces user limits, not LLM choices)
```

## Key Changes

### 1. Remove Risk Constraints from LLM Output

The LLM should NOT output `risk_constraints`. User config should be the single source of truth.

**Before** (schema):
```python
class StrategyPlan:
    triggers: List[TriggerCondition]
    risk_constraints: RiskConstraint  # LLM outputs, then overridden
    sizing_rules: List[PositionSizingRule]
```

**After**:
```python
class StrategyPlan:
    triggers: List[TriggerCondition]  # Can be empty
    sizing_hints: List[SizingHint]    # Suggestions, not rules
    regime_assessment: RegimeAssessment
    stance: Literal["active", "defensive", "wait"]
```

### 2. Allow Empty Triggers with Regime Alerts

New output option when conditions don't warrant trading:

```python
class RegimeAssessment:
    regime: Literal["bull", "bear", "range", "volatile", "uncertain"]
    confidence: float  # 0-1
    rationale: str

class RegimeAlert:
    """Condition that would change the assessment."""
    indicator: str
    threshold: float
    direction: Literal["above", "below"]
    interpretation: str  # "RSI > 70 would signal overbought"

class StrategyPlan:
    triggers: List[TriggerCondition]  # Can be []
    regime: RegimeAssessment
    alerts: List[RegimeAlert]  # Watch conditions
    next_assessment_trigger: str  # When to re-evaluate
```

### 3. Vector Store for Strategic Knowledge

Instead of embedding all strategic rules in the prompt:

```
prompts/
  llm_strategist_core.txt     # Simple, general prompt

vector_store/
  regime_strategies/
    bull_trending.md          # Strategies for bull markets
    bear_defensive.md         # Strategies for bear markets
    range_mean_revert.md      # Range-bound strategies
    volatile_breakout.md      # High-vol strategies

  indicator_playbooks/
    rsi_divergence.md
    macd_crossover.md
    bollinger_squeeze.md
```

The prompt becomes:
```
You are a crypto strategist. Current regime: {regime}

Retrieved strategies for this regime:
{vector_store_results}

Market state:
{indicators}

Decide:
1. Generate triggers if high-confidence opportunities exist
2. Set regime alerts if conditions are developing
3. Wait if conditions are unclear

You are NOT required to generate triggers. Quality over quantity.
```

### 4. Simplified Prompt Structure

**Current** (164 lines with 10+ requirements):
- Multi-layer synthesis requirements
- Mandatory volatility gating
- Trigger diversity rules
- BTC/ETH asymmetry rules
- Composite logic requirements
- Confidence grading rules
- ...etc

**Proposed** (~30 lines):
```
You are a crypto swing strategist.

INPUTS:
- Regime: {regime} (from classifier)
- Indicators: {indicators}
- Retrieved strategies: {vector_results}
- User risk budget: {daily_risk_budget_pct}%

OUTPUTS (JSON):
{
  "stance": "active" | "defensive" | "wait",
  "regime_assessment": {
    "regime": "...",
    "confidence": 0.0-1.0,
    "rationale": "..."
  },
  "triggers": [...],  // Empty if stance=wait
  "alerts": [...],    // Regime change indicators
  "rationale": "..."
}

GUIDELINES:
- Only generate triggers with high conviction
- Empty triggers is acceptable when conditions are unclear
- Alerts help detect regime changes for re-evaluation
- Risk enforcement is handled downstream, not by you
```

## Implementation Phases

### Phase 1: Allow Empty Triggers
- Update schema to make triggers optional
- Update prompt to explicitly allow "wait" stance
- Add regime_alerts output field

### Phase 2: Remove Risk Redundancy
- Remove risk_constraints from LLM output
- User config becomes single source of truth
- Simplify _merge_plan_risk_constraints to no-op

### Phase 3: Vector Store Integration
- Extract strategic knowledge into markdown docs
- Implement RAG pipeline for regime-specific retrieval
- Simplify prompt to core decision logic

### Phase 4: Regime Classifier
- Deterministic classifier from indicators
- Pre-computes regime before LLM call
- LLM focuses on trigger selection, not regime detection

## Benefits

1. **Simpler LLM task**: Decide trade/wait, not encode all trading rules
2. **No redundancy**: User config is authoritative for risk
3. **Quality over quantity**: Empty triggers acceptable
4. **Maintainable strategies**: Update vector store, not prompt
5. **Transparent choices**: LLM rationale visible without noise
6. **Faster iteration**: Test new strategies via documents

## Open Questions

1. How to handle transition from current plans to new format?
2. Should regime classifier be ML or rule-based?
3. How to version vector store strategies?
4. How to measure strategy retrieval quality?
