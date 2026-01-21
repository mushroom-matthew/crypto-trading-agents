# Repo Analysis: Judge Can Disable All Categories

## Problem Statement

The judge can disable ALL trading categories, making trading impossible. This is a self-defeating constraint with no validation.

**Evidence from backtest-4722e226**:
```json
"disabled_categories": [
  "trend_continuation",
  "reversal",
  "volatility_breakout",
  "mean_reversion",
  "emergency_exit",
  "other"
]
```

This is **every possible category** - the system paralyzed itself.

---

## Root Cause: No Validation at Any Layer

### 1. Schema Has No Constraint on disabled_categories

**File**: `schemas/judge_feedback.py:22-32`

```python
class JudgeConstraints(SerializableModel):
    """Machine-readable knobs that the executor must enforce."""

    max_trades_per_day: Optional[int] = Field(default=None, ge=0)
    min_trades_per_day: Optional[int] = Field(default=None, ge=0)
    max_triggers_per_symbol_per_day: Optional[int] = Field(default=None, ge=0)
    symbol_risk_multipliers: Dict[str, float] = Field(default_factory=dict)
    disabled_trigger_ids: List[str] = Field(default_factory=list)
    disabled_categories: List[str] = Field(default_factory=list)  # ❌ NO VALIDATION
    risk_mode: Literal["normal", "conservative", "emergency"] = "normal"
```

**Problem**: `disabled_categories` is a simple `List[str]` with:
- ❌ No max length check
- ❌ No validation against category enum
- ❌ No check that at least one remains

---

### 2. Heuristic Generation Can Suggest All Categories

**File**: `services/judge_feedback_service.py:258-269`

```python
# Category-specific feedback
for cat, stats in (trade_metrics.category_stats or {}).items():
    cat_wr = stats.get("win_rate", 0)
    cat_count = stats.get("count", 0)
    if cat_count >= 3 and cat_wr < 0.3:
        analysis.suggested_strategist_constraints.setdefault("vetoes", []).append(
            f"Disable {cat} triggers (win rate {cat_wr * 100:.0f}% over {cat_count} trades)."
        )
```

**Problem**: If ALL categories have < 30% win rate, ALL get added to vetoes. No guard against this.

---

### 3. LLM Response Parsed Without Validation

**File**: `services/judge_feedback_service.py:589`

```python
feedback = JudgeFeedback.model_validate_json(json_block)
```

**Problem**: The LLM can return any list of categories. Pydantic validates the type (`List[str]`) but not the content or count.

---

### 4. Fallback Heuristics Pass Through Suggestions

**File**: `services/judge_feedback_service.py:618-620`

```python
constraints = JudgeConstraints(
    ...
    disabled_categories=heuristics.suggested_constraints.get("disabled_categories", []),
)
```

**Problem**: Directly passes through whatever was suggested, no filtering.

---

### 5. Judge Prompt Has No Guard Instruction

**File**: `prompts/llm_judge_prompt.txt:94`

```
"disabled_categories": ["trend_continuation", "reversal", "volatility_breakout", "mean_reversion", "emergency_exit", "other"]
```

The prompt shows all 6 categories as examples but **never says**:
- ❌ "Never disable all categories"
- ❌ "Always keep at least one entry category"
- ❌ "Emergency_exit should rarely be disabled"

---

### 6. Execution Engine Blindly Enforces

**File**: `trading_core/execution_engine.py:243-252`

```python
if constraints and not is_emergency_exit:
    if trigger.trigger_id in constraints.disabled_trigger_ids:
        # Skip
    if trigger.category and trigger.category in constraints.disabled_categories:
        state.log_skip(BlockReason.CATEGORY.value)
        return TradeEvent(
            timestamp=bar_timestamp,
            trigger_id=trigger.trigger_id,
            symbol=trigger.symbol,
            action="skipped",
            reason=BlockReason.CATEGORY.value,
            detail=f"Category {trigger.category} disabled by judge",
        )
```

**Problem**: No fallback. If all categories disabled:
1. Every trigger hits this check
2. Every trigger is skipped
3. Nothing trades
4. Next judge eval sees 0 trades → score drops → more categories disabled

**Self-reinforcing failure loop.**

---

## Available Categories

**File**: `schemas/llm_strategist.py:102-109`

```python
TriggerCategory = Literal[
    "trend_continuation",
    "reversal",
    "volatility_breakout",
    "mean_reversion",
    "emergency_exit",
    "other",
]
```

Exactly 6 categories. Entry categories (first 4) vs exit category (1) vs catch-all (1).

---

## Proposed Fixes

### Fix 1: Add Pydantic Validator to Schema

**File**: `schemas/judge_feedback.py`

```python
from pydantic import field_validator

ENTRY_CATEGORIES = {"trend_continuation", "reversal", "volatility_breakout", "mean_reversion"}
ALL_CATEGORIES = ENTRY_CATEGORIES | {"emergency_exit", "other"}

class JudgeConstraints(SerializableModel):
    disabled_categories: List[str] = Field(default_factory=list)

    @field_validator("disabled_categories")
    @classmethod
    def validate_disabled_categories(cls, v: List[str]) -> List[str]:
        # Validate category names
        invalid = set(v) - ALL_CATEGORIES
        if invalid:
            raise ValueError(f"Invalid categories: {invalid}")

        # Ensure at least one entry category remains
        disabled_entry = set(v) & ENTRY_CATEGORIES
        if disabled_entry == ENTRY_CATEGORIES:
            raise ValueError(
                "Cannot disable all entry categories. "
                f"At least one of {ENTRY_CATEGORIES} must remain enabled."
            )

        return v
```

### Fix 2: Add Guard to Heuristic Generation

**File**: `services/judge_feedback_service.py:258-269`

```python
# After building suggested disabled categories
suggested_disabled = analysis.suggested_constraints.get("disabled_categories", [])
entry_cats = {"trend_continuation", "reversal", "volatility_breakout", "mean_reversion"}

# Ensure at least one entry category remains
if set(suggested_disabled) >= entry_cats:
    # Keep the category with highest win rate
    best_cat = max(
        entry_cats,
        key=lambda c: (trade_metrics.category_stats or {}).get(c, {}).get("win_rate", 0)
    )
    suggested_disabled = [c for c in suggested_disabled if c != best_cat]
    analysis.observations.append(
        f"Preserved {best_cat} (best performing) to ensure trading can continue."
    )

analysis.suggested_constraints["disabled_categories"] = suggested_disabled
```

### Fix 3: Update Judge Prompt

**File**: `prompts/llm_judge_prompt.txt`

Add constraint section:
```
## CONSTRAINT LIMITS

CRITICAL: You must NEVER disable all entry categories. At least ONE of these must remain enabled:
- trend_continuation
- reversal
- volatility_breakout
- mean_reversion

If all categories are performing poorly, keep the BEST performing one enabled and focus
recommendations on improving trigger quality rather than disabling everything.

The emergency_exit category should almost NEVER be disabled - it's a safety mechanism.
```

### Fix 4: Add Execution Engine Fallback

**File**: `trading_core/execution_engine.py:243-252`

```python
# Before applying constraints, validate
def _validate_constraints(constraints: JudgeConstraints) -> JudgeConstraints:
    """Ensure constraints don't disable all trading."""
    if not constraints:
        return constraints

    entry_cats = {"trend_continuation", "reversal", "volatility_breakout", "mean_reversion"}
    disabled = set(constraints.disabled_categories or [])

    if disabled >= entry_cats:
        logger.warning(
            "Judge disabled all entry categories - removing least disabled category"
        )
        # Remove one category from disabled list
        constraints.disabled_categories = [
            c for c in constraints.disabled_categories
            if c not in entry_cats
        ][:len(entry_cats) - 1]

    return constraints
```

### Fix 5: Add Observability for This Failure Mode

**File**: `agents/event_emitter.py` or execution engine

```python
async def emit_constraint_warning(
    disabled_categories: List[str],
    run_id: str,
):
    entry_cats = {"trend_continuation", "reversal", "volatility_breakout", "mean_reversion"}
    disabled_entry = set(disabled_categories) & entry_cats

    if len(disabled_entry) >= 3:  # Warning threshold
        await emit_event(
            "constraint_warning",
            {
                "warning": "High number of disabled entry categories",
                "disabled_entry_categories": list(disabled_entry),
                "remaining_entry_categories": list(entry_cats - disabled_entry),
            },
            source="judge",
            run_id=run_id,
        )
```

---

## Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `schemas/judge_feedback.py:22-32` | Add Pydantic validator | P0 |
| `prompts/llm_judge_prompt.txt` | Add constraint limits instruction | P0 |
| `services/judge_feedback_service.py:258-269` | Add guard in heuristic generation | P0 |
| `trading_core/execution_engine.py:243-252` | Add fallback validation | P1 |
| `agents/event_emitter.py` | Add warning event | P2 |

---

## Validation Criteria

After fix, verify:
- [ ] Schema rejects `disabled_categories` with all entry categories
- [ ] Heuristics never suggest disabling all entry categories
- [ ] LLM prompt explicitly forbids disabling all categories
- [ ] Execution engine logs warning if constraint nearly disables all
- [ ] At least one entry category is always available for trading
