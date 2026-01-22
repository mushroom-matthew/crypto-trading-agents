# Repo Analysis: 100% Trigger Churn on Replan

## Problem Statement

Every replan completely replaces all triggers (0 unchanged). This prevents learning which triggers work and creates instability in trading behavior.

**Evidence from backtest-4722e226**:
```
| Plan | Added | Removed | Changed | Unchanged |
|------|-------|---------|---------|-----------|
| 1→2  | 6     | 5       | 0       | 0         |
| 2→3  | 6     | 6       | 0       | 0         |
| 3→4  | 6     | 6       | 0       | 0         |
```

---

## Root Cause: Stateless Plan Generation

The LLM strategist receives **zero context about previous triggers** when replanning.

### 1. LLMInput Has No Previous Plan Field

**File**: `schemas/llm_strategist.py:91-98`

```python
class LLMInput(SerializableModel):
    """Structured payload that is serialized and sent to the LLM strategist."""

    portfolio: PortfolioState
    assets: List[AssetState]
    risk_params: Dict[str, Any]
    global_context: Dict[str, Any] = Field(default_factory=dict)
    market_structure: Dict[str, Any] = Field(default_factory=dict)
    # ❌ NO previous_plan field
    # ❌ NO prior_triggers field
    # ❌ NO trigger_performance field
```

**Impact**: The LLM only sees current market state, not what worked before.

---

### 2. Strategist Prompt Has No Continuity Instructions

**File**: `prompts/llm_strategist_prompt.txt`

The 84-line prompt instructs on:
- ✅ Multi-timeframe synthesis
- ✅ Volatility gating
- ✅ Trigger diversity
- ✅ Confidence grading

But **NEVER mentions**:
- ❌ Preserving working triggers
- ❌ Maintaining trigger ID stability
- ❌ Using trigger performance history
- ❌ Updating vs replacing triggers

Line 8-10 reinforces statelessness:
> "Data usage rules: Use only precomputed indicator fields and context provided in the input."

---

### 3. Plan Generation Ignores Prior Plan

**File**: `agents/strategies/plan_provider.py:159-214`

```python
def get_plan(self, run_id, plan_date, llm_input, ...):
    # Cache lookup by hash of CURRENT LLMInput
    cache_path = self._cache_path(run_id, plan_date, llm_input)
    cached = self._load_cached(cache_path)
    if cached:
        return cached  # Only if EXACT same input

    # Generate fresh plan - NO previous plan passed
    plan = self.llm_client.generate_plan(
        llm_input,  # ❌ No previous_plan parameter
        prompt_template=resolved_prompt,
        ...
    )
```

**Impact**: If any indicator changes (which always happens), cache misses and LLM generates from scratch.

---

### 4. LLM Call Excludes Prior Context

**File**: `agents/strategies/llm_client.py:184-192`

```python
completion = self.client.responses.create(
    model=self.model,
    input=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": llm_input.to_json()},  # ❌ Only current state
    ],
    ...
)
```

**Impact**: The LLM has no way to know what triggers existed before.

---

### 5. Trigger IDs Are LLM-Generated (Non-Deterministic)

**File**: `schemas/llm_strategist.py:112-129`

```python
class TriggerCondition(SerializableModel):
    id: str  # LLM generates this - no stability guarantee
    symbol: str
    entry_rule: str
    exit_rule: str
    ...
```

The LLM invents IDs like:
- Plan 1: `tc_long_1`
- Plan 2: `btc_trend_continuation_long_1`
- Plan 3: `tc_long_1` (again)

**No deterministic hashing or matching algorithm exists.**

---

## Proposed Fixes

### Fix 1: Add Previous Plan to LLMInput

**File**: `schemas/llm_strategist.py`

```python
class LLMInput(SerializableModel):
    portfolio: PortfolioState
    assets: List[AssetState]
    risk_params: Dict[str, Any]
    global_context: Dict[str, Any] = Field(default_factory=dict)
    market_structure: Dict[str, Any] = Field(default_factory=dict)
    # NEW FIELDS
    previous_triggers: List[TriggerSummary] = Field(default_factory=list)
    trigger_performance: Dict[str, TriggerStats] = Field(default_factory=dict)

class TriggerSummary(SerializableModel):
    id: str
    symbol: str
    timeframe: str
    direction: str
    category: str
    entry_rule: str
    exit_rule: str

class TriggerStats(SerializableModel):
    trigger_id: str
    attempts: int
    executed: int
    blocked: int
    win_rate: float
    avg_pnl: float
```

### Fix 2: Update Strategist Prompt

**File**: `prompts/llm_strategist_prompt.txt`

Add section:
```
## TRIGGER CONTINUITY

When replanning, the previous plan's triggers are provided in `previous_triggers`.

Rules for trigger updates:
1. PRESERVE triggers with win_rate > 50% - keep the same ID and rules
2. MODIFY triggers with win_rate 30-50% - keep ID, adjust parameters
3. REPLACE triggers with win_rate < 30% - generate new trigger
4. KEEP emergency_exit triggers unless explicitly broken

When preserving a trigger, use the EXACT same trigger ID from previous_triggers.
```

### Fix 3: Add Trigger Matching Algorithm

**File**: `agents/strategies/trigger_engine.py` (new function)

```python
def match_triggers(old_triggers: List[TriggerCondition],
                   new_triggers: List[TriggerCondition]) -> Dict[str, str]:
    """Match new triggers to old by semantic similarity."""
    def signature(t):
        return (t.symbol, t.timeframe, t.direction, t.category)

    matches = {}
    old_by_sig = {signature(t): t for t in old_triggers}

    for new_t in new_triggers:
        sig = signature(new_t)
        if sig in old_by_sig:
            # Semantic match - reuse old ID
            matches[new_t.id] = old_by_sig[sig].id

    return matches
```

### Fix 4: Deterministic Trigger ID Generation

**File**: `schemas/llm_strategist.py`

```python
class TriggerCondition(SerializableModel):
    @property
    def deterministic_id(self) -> str:
        """Generate stable ID from trigger content."""
        content = f"{self.symbol}:{self.timeframe}:{self.direction}:{self.category}"
        content += f":{self.entry_rule}:{self.exit_rule}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
```

---

## Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `schemas/llm_strategist.py:91-98` | Add `previous_triggers`, `trigger_performance` to LLMInput | P0 |
| `prompts/llm_strategist_prompt.txt` | Add trigger continuity instructions | P0 |
| `agents/strategies/plan_provider.py:159-214` | Pass previous plan to LLM | P0 |
| `agents/strategies/llm_client.py:184-192` | Include previous triggers in prompt | P0 |
| `agents/strategies/trigger_engine.py` | Add trigger matching algorithm | P1 |
| `backtesting/llm_strategist_runner.py` | Track trigger performance for replan | P1 |

---

## Validation Criteria

After fix, verify:
- [ ] Replans show `unchanged > 0` when triggers performed well
- [ ] Trigger IDs remain stable across replans for preserved triggers
- [ ] Trigger performance data is passed to LLM
- [ ] Prompt instructs LLM to preserve working triggers
