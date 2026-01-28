# Branch: strategist-simplification

## Priority: 1 (Top Priority)

## Purpose
Complete architectural overhaul of the LLM strategist to eliminate over-prompting, remove redundant risk handling, enable "no trade" stances with regime alerts, and implement vector-store-based strategy retrieval.

## Source Plans
- docs/architecture/STRATEGIST_SIMPLIFICATION.md (comprehensive proposal)

## Problem Statement
The current strategist architecture has critical issues:
1. **Over-prompting**: 164-line prompt with 10+ prescriptive rules forces template generation instead of strategic reasoning
2. **Risk redundancy**: LLM outputs `risk_constraints` which are then overwritten by user config (wasted tokens, confusing)
3. **Forced triggers**: Prompt requires "at least five triggers" - no path for "conditions don't warrant trading"
4. **Monolithic prompt**: Strategic knowledge embedded rather than retrieved from a knowledge base
5. **No visibility**: User can't see what choices the LLM made vs what was overridden

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER CONFIG                                     │
│  - risk_limits (authoritative)                                              │
│  - symbols, timeframes                                                       │
│  - daily_risk_budget_pct                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          REGIME CLASSIFIER                                   │
│  Input: indicators, price action                                            │
│  Output: regime (bull/bear/range/volatile/uncertain), confidence            │
│  Implementation: Deterministic rules or lightweight ML                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VECTOR STORE (RAG)                                  │
│  Query: regime + market conditions                                          │
│  Returns: Relevant strategy documents for current regime                    │
│  Content:                                                                    │
│    - regime_strategies/bull_trending.md                                     │
│    - regime_strategies/bear_defensive.md                                    │
│    - regime_strategies/range_mean_revert.md                                 │
│    - indicator_playbooks/rsi_divergence.md                                  │
│    - indicator_playbooks/macd_crossover.md                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SIMPLIFIED LLM STRATEGIST                              │
│  Prompt: ~40 lines (not 164)                                                │
│  Inputs:                                                                     │
│    - Regime classification (pre-computed)                                   │
│    - Retrieved strategy documents                                           │
│    - Current indicators                                                      │
│    - User risk budget (for awareness, not output)                           │
│  Outputs:                                                                    │
│    - stance: "active" | "defensive" | "wait"                                │
│    - triggers: [...] (CAN BE EMPTY)                                         │
│    - regime_alerts: [...] (conditions to watch)                             │
│    - sizing_hints: [...] (advisory, not enforced)                           │
│    - rationale: "..." (visible to user)                                     │
│  NOT output: risk_constraints (removed)                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RISK ENGINE                                        │
│  Source of truth: USER CONFIG (not LLM)                                     │
│  Enforces: max_position_risk_pct, daily_budget, exposure limits             │
│  LLM sizing_hints: Advisory only, can be overridden                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRIGGER ENGINE                                      │
│  Handles: Empty trigger plans gracefully                                    │
│  Monitors: Regime alerts for re-assessment triggers                         │
│  Reports: Which triggers fired, why, LLM rationale                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Full Scope

### 1. Schema Changes
- [ ] Add `stance` field to StrategyPlan: `"active" | "defensive" | "wait"`
- [ ] Add `RegimeAlert` schema for regime-change indicators
- [ ] Add `RegimeAssessment` schema with regime, confidence, rationale
- [ ] Add `SizingHint` schema (advisory, not enforced)
- [ ] Remove `risk_constraints` from StrategyPlan output
- [ ] Make `triggers` explicitly allow empty list

### 2. Regime Classifier
- [ ] Create `trading_core/regime_classifier.py`
- [ ] Implement deterministic regime classification from indicators:
  - Bull: price > SMA, RSI > 50, MACD positive
  - Bear: price < SMA, RSI < 50, MACD negative
  - Range: BBW < threshold, ADX < 25
  - Volatile: ATR spike, BBW expansion
  - Uncertain: conflicting signals
- [ ] Add confidence score (0-1) based on signal agreement
- [ ] Integrate into strategist runner (pre-LLM call)

### 3. Vector Store Infrastructure
- [ ] Create `vector_store/` directory structure
- [ ] Create embedding pipeline for strategy documents
- [ ] Implement RAG retrieval by regime + indicators
- [ ] Create initial strategy documents:
  - `regime_strategies/bull_trending.md`
  - `regime_strategies/bear_defensive.md`
  - `regime_strategies/range_mean_revert.md`
  - `regime_strategies/volatile_breakout.md`
  - `regime_strategies/uncertain_wait.md`
  - `indicator_playbooks/rsi_extremes.md`
  - `indicator_playbooks/macd_divergence.md`
  - `indicator_playbooks/bollinger_squeeze.md`
  - `indicator_playbooks/support_resistance.md`

### 4. Prompt Simplification
- [ ] Create new `prompts/llm_strategist_simple.txt` (~40 lines)
- [ ] Remove all prescriptive rules (5 triggers requirement, etc.)
- [ ] Add placeholder for retrieved strategy context
- [ ] Add explicit "wait is acceptable" guidance
- [ ] Remove risk_constraints from expected output
- [ ] Add rationale requirement for transparency

### 5. Risk Redundancy Removal
- [ ] Remove `risk_constraints` from LLM output handling
- [ ] Delete `_merge_plan_risk_constraints` or make it no-op
- [ ] User config becomes single source of truth
- [ ] Add `sizing_hints` field for LLM suggestions (advisory)
- [ ] Update plan validation to not require risk_constraints

### 6. Empty Trigger Handling
- [ ] Update `plan_provider.py` to handle empty triggers
- [ ] Update `trigger_engine.py` to handle plans with no triggers
- [ ] Update `llm_strategist_runner.py` to track "wait" days
- [ ] Add metrics for stance distribution (active/defensive/wait)

### 7. Regime Alert Monitoring
- [ ] Create alert evaluation in trigger engine
- [ ] When alert condition met, flag for re-assessment
- [ ] Add alert firing to event stream
- [ ] UI exposure of active alerts (deferred to separate branch)

### 8. Observability & Transparency
- [ ] Log LLM rationale to event store
- [ ] Track stance distribution in backtest metrics
- [ ] Add "decisions explained" to backtest summary
- [ ] Make sizing_hints vs actual sizing visible

## Key Files

### New Files
- trading_core/regime_classifier.py
- vector_store/embeddings.py
- vector_store/retriever.py
- vector_store/strategies/bull_trending.md
- vector_store/strategies/bear_defensive.md
- vector_store/strategies/range_mean_revert.md
- vector_store/strategies/volatile_breakout.md
- vector_store/strategies/uncertain_wait.md
- vector_store/playbooks/rsi_extremes.md
- vector_store/playbooks/macd_divergence.md
- vector_store/playbooks/bollinger_squeeze.md
- prompts/llm_strategist_simple.txt

### Modified Files
- schemas/llm_strategist.py (new schemas, remove risk_constraints output)
- prompts/llm_strategist_prompt.txt (simplify or replace)
- agents/strategies/plan_provider.py (handle empty triggers, use regime)
- agents/strategies/trigger_engine.py (handle empty plans, alert monitoring)
- agents/strategies/llm_client.py (new output parsing)
- backtesting/llm_strategist_runner.py (integrate classifier, vector store)
- services/strategist_plan_service.py (remove _merge_plan_risk_constraints)

## Dependencies / Coordination
- None - this is foundational work
- Current branch (comp-audit-ui-trade-stats) should merge first
- Subsequent branches should wait for this to merge

## Acceptance Criteria

### Must Have
- [ ] LLM can return `{"triggers": [], "stance": "wait", "regime_alerts": [...]}` without error
- [ ] Backtest runs successfully with plans that have zero triggers
- [ ] User risk config is the single source of truth (no LLM overrides)
- [ ] Prompt is under 50 lines
- [ ] Regime classifier produces sensible classifications
- [ ] At least 5 strategy documents in vector store

### Should Have
- [ ] Regime alerts fire and trigger re-assessment
- [ ] Backtest metrics include stance distribution
- [ ] LLM rationale is logged and visible
- [ ] Sizing hints visible separately from enforced sizing

### Nice to Have
- [ ] Vector store retrieval quality metrics
- [ ] A/B comparison: old prompt vs simplified

## Test Plan (required before commit)

```bash
# Schema validation
uv run pytest tests/test_llm_strategist_schema.py -vv

# Regime classifier
uv run pytest tests/test_regime_classifier.py -vv

# Vector store retrieval
uv run pytest tests/test_vector_store.py -vv

# Plan provider with empty triggers
uv run pytest tests/test_plan_provider.py -vv

# Trigger engine with empty plans
uv run pytest tests/test_trigger_engine.py -vv

# Full backtest integration
uv run pytest tests/test_llm_strategist_runner.py -vv

# Import verification
uv run python -c "
from schemas.llm_strategist import StrategyPlan, RegimeAlert, RegimeAssessment
from trading_core.regime_classifier import classify_regime
print('Schema OK')
"
```

## Human Verification (required)

1. Run a backtest with the simplified prompt
2. Verify the LLM chooses "wait" stance in at least some market conditions
3. Verify backtest completes without errors when plan has no triggers
4. Verify regime classifier produces sensible outputs
5. Verify retrieved strategies match the regime
6. Confirm user risk config is not overridden by LLM

## Worktree Setup

```bash
git fetch
git worktree add -b strategist-simplification ../wt-strategist-simplification main
cd ../wt-strategist-simplification

# When finished
git worktree remove ../wt-strategist-simplification
```

## Git Workflow

```bash
git checkout main && git pull
git checkout -b strategist-simplification

# Work in phases, commit incrementally:
# Commit 1: Schema changes
# Commit 2: Regime classifier
# Commit 3: Vector store infrastructure
# Commit 4: Prompt simplification
# Commit 5: Risk redundancy removal
# Commit 6: Integration and tests

git status && git diff
git add <files>
git commit -m "Strategist: <phase description>"
```

## Change Log (update during implementation)
- YYYY-MM-DD: Summary of changes, files touched, and decisions.

## Test Evidence (append results before commit)

## Human Verification Evidence (append results before commit)

---

## Design Details

### RegimeAlert Schema
```python
class RegimeAlert(SerializableModel):
    """Condition that would trigger strategy re-assessment."""
    indicator: str                           # e.g., "rsi_14"
    threshold: float                         # e.g., 70.0
    direction: Literal["above", "below", "crosses"]
    symbol: str                              # e.g., "BTC-USD"
    interpretation: str                      # Human-readable meaning
    priority: Literal["high", "medium", "low"] = "medium"
```

### RegimeAssessment Schema
```python
class RegimeAssessment(SerializableModel):
    """Pre-computed regime classification."""
    regime: Literal["bull", "bear", "range", "volatile", "uncertain"]
    confidence: float                        # 0.0 - 1.0
    primary_signals: List[str]               # What drove the classification
    conflicting_signals: List[str]           # Why confidence may be lower
```

### SizingHint Schema
```python
class SizingHint(SerializableModel):
    """Advisory sizing suggestion from LLM (not enforced)."""
    symbol: str
    suggested_risk_pct: float               # LLM's suggestion
    rationale: str                          # Why this sizing
    # Note: Actual sizing comes from user config + risk engine
```

### Updated StrategyPlan
```python
class StrategyPlan(SerializableModel):
    plan_id: str
    run_id: str | None = None
    generated_at: datetime
    valid_until: datetime

    # Regime (pre-computed, not from LLM)
    regime_assessment: RegimeAssessment

    # LLM decisions
    stance: Literal["active", "defensive", "wait"] = "active"
    triggers: List[TriggerCondition] = Field(default_factory=list)  # CAN BE EMPTY
    regime_alerts: List[RegimeAlert] = Field(default_factory=list)
    sizing_hints: List[SizingHint] = Field(default_factory=list)   # Advisory only

    # Transparency
    global_view: str | None = None
    rationale: str | None = None            # Why this stance/triggers
    retrieved_strategies: List[str] = Field(default_factory=list)  # What was retrieved

    # REMOVED: risk_constraints (user config is authoritative)
    # REMOVED: sizing_rules (replaced by advisory sizing_hints)
```

### Simplified Prompt (~40 lines)
```
You are a crypto swing strategist. Your job is to decide whether to trade.

CONTEXT PROVIDED:
- Regime: {regime_assessment.regime} (confidence: {regime_assessment.confidence})
- Indicators: {indicators_summary}
- Retrieved strategies for this regime:
{retrieved_strategy_content}

USER CONSTRAINTS (for your awareness - enforced downstream):
- Daily risk budget: {daily_risk_budget_pct}%
- Max position risk: {max_position_risk_pct}%

YOUR DECISION (JSON):
{
  "stance": "active" | "defensive" | "wait",
  "triggers": [...],        // Empty list is acceptable
  "regime_alerts": [...],   // Conditions to watch
  "sizing_hints": [...],    // Your suggestions (advisory)
  "rationale": "..."        // Explain your decision
}

GUIDELINES:
- "wait" is a valid choice when conditions are unclear
- Quality over quantity - only high-conviction triggers
- Regime alerts help detect when to re-assess
- Your sizing_hints are suggestions; user config is enforced
- Be transparent in your rationale
```

### Regime Classifier Rules (Initial)
```python
def classify_regime(indicators: Dict) -> RegimeAssessment:
    signals = []
    conflicts = []

    # Trend signals
    if close > sma_medium:
        signals.append("price_above_sma")
    else:
        signals.append("price_below_sma")

    # Momentum signals
    if rsi > 50:
        signals.append("rsi_bullish")
    elif rsi < 50:
        signals.append("rsi_bearish")

    # Volatility signals
    if atr / close > 0.03:  # 3% ATR
        signals.append("high_volatility")

    if bbw < 0.02:  # Bollinger squeeze
        signals.append("low_volatility")

    # Classification logic
    bull_signals = {"price_above_sma", "rsi_bullish", "macd_positive"}
    bear_signals = {"price_below_sma", "rsi_bearish", "macd_negative"}

    bull_count = len(set(signals) & bull_signals)
    bear_count = len(set(signals) & bear_signals)

    if "high_volatility" in signals:
        regime = "volatile"
        confidence = 0.7
    elif bull_count >= 2 and bear_count == 0:
        regime = "bull"
        confidence = bull_count / 3
    elif bear_count >= 2 and bull_count == 0:
        regime = "bear"
        confidence = bear_count / 3
    elif "low_volatility" in signals:
        regime = "range"
        confidence = 0.6
    else:
        regime = "uncertain"
        confidence = 0.3
        conflicts = [s for s in signals if s not in bull_signals | bear_signals]

    return RegimeAssessment(
        regime=regime,
        confidence=confidence,
        primary_signals=signals,
        conflicting_signals=conflicts
    )
```

### Vector Store Structure
```
vector_store/
├── embeddings/
│   └── strategy_embeddings.pkl      # Pre-computed embeddings
├── strategies/
│   ├── bull_trending.md             # Trend-following in bull markets
│   ├── bear_defensive.md            # Capital preservation in bear
│   ├── range_mean_revert.md         # Mean reversion in ranges
│   ├── volatile_breakout.md         # Breakout plays in high vol
│   └── uncertain_wait.md            # When to sit out
├── playbooks/
│   ├── rsi_extremes.md              # RSI overbought/oversold
│   ├── macd_divergence.md           # MACD signal divergence
│   ├── bollinger_squeeze.md         # Volatility compression
│   ├── support_resistance.md        # S/R level plays
│   └── trend_continuation.md        # Riding trends
└── retriever.py                     # RAG implementation
```
