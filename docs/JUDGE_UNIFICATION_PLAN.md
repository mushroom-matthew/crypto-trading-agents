# Judge Agent Unification Plan

## Executive Summary

The current system has **three distinct judge implementations** that are fundamentally misaligned:

| Component | Location | Method | Uses LLM? |
|-----------|----------|--------|-----------|
| Live Judge Agent | `agents/judge_agent_client.py` | OpenAI GPT call via `analyze_decision_quality()` | Yes |
| Trading Core Judge | `trading_core/judge_agent.py` | Deterministic risk checks (`evaluate_intents()`) | No |
| Backtest Judge | `backtesting/llm_strategist_runner.py` | Hardcoded heuristics in `_judge_feedback()` | No |

**The Problem**: When running backtests without shims, the judge behavior is completely deterministic and heuristic-based, meaning backtest results don't reflect live trading behavior.

---

## Solution: Heuristics as Context for LLM

Following the pattern established by `StrategyPlanProvider` and `LLMClient`, we will:

1. **Compute deterministic heuristics first** - The existing `_judge_feedback()` heuristics become *input context* for the LLM
2. **Pass heuristics + performance data to LLM** - The LLM can refine, override, or validate the heuristic assessment
3. **Use transport pattern for shimming** - When `use_judge_shim=True`, the transport returns a deterministic response based on heuristics

This approach:
- Preserves the computational work in the heuristics
- Gives the LLM structured pre-analysis to reason about
- Enables fast deterministic backtests via shim
- Matches live judge behavior when not shimming

---

## Architecture

### New Component: `JudgeFeedbackService`

```
JudgeFeedbackService
├── compute_heuristics(summary, trade_metrics) -> HeuristicAnalysis
├── build_prompt(heuristics, context) -> str
├── generate_feedback(summary, trade_metrics, context) -> JudgeFeedback
│   ├── If transport (shim): return deterministic from heuristics
│   └── Else: call LLM with heuristics as context
└── transport: Callable[[str], str] | None  # Shim injection point
```

### Data Flow

```
Performance Summary + Trade Metrics
         │
         ▼
┌─────────────────────────────────┐
│  compute_heuristics()           │
│  - Score calculation            │
│  - Win rate analysis            │
│  - Emergency exit detection     │
│  - Category breakdown           │
│  - Constraint recommendations   │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  HeuristicAnalysis              │
│  - base_score: float            │
│  - score_adjustments: list      │
│  - observations: list[str]      │
│  - suggested_constraints: dict  │
│  - red_flags: list[str]         │
└─────────────────────────────────┘
         │
         ├──── use_judge_shim=True ────► Deterministic JudgeFeedback
         │
         └──── use_judge_shim=False ───► LLM Call
                                              │
                                              ▼
                                    ┌─────────────────────────────────┐
                                    │  LLM Prompt includes:           │
                                    │  - Performance summary          │
                                    │  - Trade metrics                │
                                    │  - HEURISTIC PRE-ANALYSIS       │
                                    │  - Strategy context             │
                                    └─────────────────────────────────┘
                                              │
                                              ▼
                                        JudgeFeedback
```

---

## Implementation Plan

### Phase 1: Create JudgeFeedbackService

**File**: `services/judge_feedback_service.py`

```python
"""Judge feedback service following LLMClient pattern."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from pydantic import ValidationError

from agents.langfuse_utils import langfuse_span
from agents.llm.client_factory import get_llm_client
from schemas.judge_feedback import JudgeFeedback, JudgeConstraints, DisplayConstraints
from trading_core.trade_quality import TradeMetrics


class JudgeTransport(Protocol):
    """Protocol for judge response transport (enables shimming)."""
    def __call__(self, payload: str) -> str: ...


@dataclass
class HeuristicAnalysis:
    """Deterministic pre-analysis computed from metrics."""

    base_score: float = 50.0
    score_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    suggested_constraints: Dict[str, Any] = field(default_factory=dict)
    suggested_strategist_constraints: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_score(self) -> float:
        score = self.base_score
        for adj in self.score_adjustments:
            score += adj.get("delta", 0.0)
        return max(0.0, min(100.0, round(score, 1)))

    def to_prompt_section(self) -> str:
        """Format as context for LLM prompt."""
        lines = [
            "HEURISTIC PRE-ANALYSIS (deterministic metrics-based assessment):",
            f"- Computed base score: {self.base_score}",
            f"- Score adjustments applied:",
        ]
        for adj in self.score_adjustments:
            lines.append(f"  - {adj.get('reason', 'unknown')}: {adj.get('delta', 0):+.1f}")
        lines.append(f"- Final heuristic score: {self.final_score}")

        if self.observations:
            lines.append("\nOBSERVATIONS:")
            for obs in self.observations:
                lines.append(f"  - {obs}")

        if self.red_flags:
            lines.append("\nRED FLAGS (require attention):")
            for flag in self.red_flags:
                lines.append(f"  - {flag}")

        if self.suggested_constraints:
            lines.append("\nSUGGESTED MACHINE CONSTRAINTS:")
            for key, val in self.suggested_constraints.items():
                if val is not None:
                    lines.append(f"  - {key}: {val}")

        if self.suggested_strategist_constraints:
            lines.append("\nSUGGESTED STRATEGIST GUIDANCE:")
            for key, val in self.suggested_strategist_constraints.items():
                if val:
                    lines.append(f"  - {key}: {val}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_score": self.base_score,
            "final_score": self.final_score,
            "score_adjustments": self.score_adjustments,
            "observations": self.observations,
            "red_flags": self.red_flags,
            "suggested_constraints": self.suggested_constraints,
            "suggested_strategist_constraints": self.suggested_strategist_constraints,
        }


class JudgeFeedbackService:
    """Generate judge feedback using heuristics as context for LLM."""

    def __init__(
        self,
        transport: JudgeTransport | None = None,
        model: str | None = None,
    ) -> None:
        self.transport = transport
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = None
        self.last_generation_info: Dict[str, Any] = {}

    @property
    def client(self):
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def compute_heuristics(
        self,
        summary: Dict[str, Any],
        trade_metrics: TradeMetrics | None = None,
    ) -> HeuristicAnalysis:
        """Compute deterministic heuristic analysis from metrics."""
        # ... (existing _judge_feedback logic refactored here)
        pass

    def generate_feedback(
        self,
        summary: Dict[str, Any],
        trade_metrics: TradeMetrics | None = None,
        strategy_context: Dict[str, Any] | None = None,
    ) -> JudgeFeedback:
        """Generate judge feedback, using LLM when transport is None."""

        # Always compute heuristics first
        heuristics = self.compute_heuristics(summary, trade_metrics)

        # Build payload for transport/LLM
        payload = {
            "summary": summary,
            "trade_metrics": trade_metrics.to_dict() if trade_metrics else None,
            "heuristics": heuristics.to_dict(),
            "strategy_context": strategy_context,
        }
        payload_json = json.dumps(payload)

        if self.transport:
            # Shim path: transport returns deterministic response
            raw = self.transport(payload_json)
            feedback = JudgeFeedback.model_validate_json(raw)
            self.last_generation_info = {"source": "transport"}
            return feedback

        # LLM path: call with heuristics as context
        return self._call_llm(summary, trade_metrics, heuristics, strategy_context)

    def _call_llm(
        self,
        summary: Dict[str, Any],
        trade_metrics: TradeMetrics | None,
        heuristics: HeuristicAnalysis,
        strategy_context: Dict[str, Any] | None,
    ) -> JudgeFeedback:
        """Call LLM with heuristics as additional context."""
        # ... (build prompt with heuristics section, call LLM, parse response)
        pass
```

### Phase 2: Add Judge Shim Transport

**File**: `backtesting/llm_shim.py` (add to existing)

```python
def make_judge_shim_transport() -> Callable[[str], str]:
    """Return a transport callable that emits deterministic judge feedback."""

    def _transport(payload: str) -> str:
        data = json.loads(payload)
        heuristics = data.get("heuristics", {})

        # Use heuristic analysis directly for deterministic response
        score = heuristics.get("final_score", 50.0)
        observations = heuristics.get("observations", [])
        suggested_constraints = heuristics.get("suggested_constraints", {})
        suggested_strategist = heuristics.get("suggested_strategist_constraints", {})

        feedback = {
            "score": score,
            "notes": " ".join(observations[:5]) if observations else "Shim judge response.",
            "constraints": {
                "max_trades_per_day": suggested_constraints.get("max_trades_per_day"),
                "max_triggers_per_symbol_per_day": suggested_constraints.get("max_triggers_per_symbol_per_day"),
                "risk_mode": suggested_constraints.get("risk_mode", "normal"),
                "disabled_trigger_ids": suggested_constraints.get("disabled_trigger_ids", []),
                "disabled_categories": suggested_constraints.get("disabled_categories", []),
            },
            "strategist_constraints": {
                "must_fix": suggested_strategist.get("must_fix", []),
                "vetoes": suggested_strategist.get("vetoes", []),
                "boost": suggested_strategist.get("boost", []),
                "regime_correction": suggested_strategist.get("regime_correction"),
                "sizing_adjustments": suggested_strategist.get("sizing_adjustments", {}),
            },
        }
        return json.dumps(feedback)

    return _transport
```

### Phase 3: Wire into LLMStrategistBacktester

**File**: `backtesting/llm_strategist_runner.py` (modify)

```python
# In __init__:
from services.judge_feedback_service import JudgeFeedbackService
from backtesting.llm_shim import make_judge_shim_transport

# Create judge service with transport if shimming
if use_judge_shim:
    judge_transport = make_judge_shim_transport()
else:
    judge_transport = None
self.judge_service = JudgeFeedbackService(transport=judge_transport, model=llm_model)

# Replace _judge_feedback method:
def _judge_feedback(
    self,
    summary: Dict[str, Any],
    trade_metrics: TradeMetrics | None = None,
) -> Dict[str, Any]:
    """Generate judge feedback using service."""
    strategy_context = {
        "risk_params": self.risk_params,
        "active_risk_limits": self.active_risk_limits.model_dump(),
        "pairs": self.pairs,
    }
    feedback = self.judge_service.generate_feedback(
        summary=summary,
        trade_metrics=trade_metrics,
        strategy_context=strategy_context,
    )
    return feedback.model_dump()
```

### Phase 4: Update Activities

**File**: `backtesting/activities.py` (already passes `use_judge_shim`)

No changes needed - the flag is already passed through.

---

## Files to Modify

| File | Changes |
|------|---------|
| `services/judge_feedback_service.py` | **NEW** - Service with transport pattern |
| `backtesting/llm_shim.py` | Add `make_judge_shim_transport()` |
| `backtesting/llm_strategist_runner.py` | Use `JudgeFeedbackService`, move heuristics to service |
| `prompts/llm_judge_prompt.txt` | Add section for heuristic pre-analysis |

---

## Migration Checklist

- [ ] Create `services/judge_feedback_service.py` with `JudgeFeedbackService`
- [ ] Extract heuristic computation from `_judge_feedback()` to `compute_heuristics()`
- [ ] Add `make_judge_shim_transport()` to `backtesting/llm_shim.py`
- [ ] Update `LLMStrategistBacktester` to use `JudgeFeedbackService`
- [ ] Update `llm_judge_prompt.txt` to include heuristic section
- [ ] Add tests comparing shim vs non-shim outputs
- [ ] Document config options

---

## Config Options

| Option | Default | Description |
|--------|---------|-------------|
| `use_judge_shim` | `False` | Use deterministic transport (fast, no LLM cost) |
| `judge_cadence_hours` | `4.0` | Hours between judge evaluations |
| `judge_replan_threshold` | `40.0` | Score below which to trigger replan |

---

## Testing Strategy

1. **Unit Tests**:
   - `test_judge_feedback_service.py` - Service computes valid heuristics
   - `test_judge_shim_transport.py` - Transport returns valid JudgeFeedback

2. **Integration Tests**:
   - Run backtest with `use_judge_shim=True` (baseline)
   - Run backtest with `use_judge_shim=False` (LLM path)
   - Compare score distributions and constraint patterns

3. **Regression Test**:
   - Record current heuristic outputs for 10 scenarios
   - After refactor, verify heuristic computation unchanged

---

## Architectural Notes

### Why Live and Backtest Judges Differ

The live judge (`agents/judge_agent_client.py`) and backtest judge (`JudgeFeedbackService`) are intentionally different:

| Aspect | Live Judge | Backtest Judge |
|--------|------------|----------------|
| **Purpose** | Long-term performance monitoring | Replan decisions during backtest |
| **Output** | Full evaluation report with 4 dimensions | JudgeFeedback only |
| **Context** | Temporal workflows (strategy specs, user prefs) | Strategy config from backtester |
| **Prompt** | Same `llm_judge_prompt.txt` | Same + heuristics section |
| **Actions** | Can update execution agent prompt | Triggers strategy replan |

### What's Shared

1. **Prompt Template**: Both use `prompts/llm_judge_prompt.txt`
2. **Schema**: Both produce/consume `JudgeFeedback` from LLM
3. **Constraints**: Same `JudgeConstraints` and `DisplayConstraints` schemas

### Remaining Gaps (Future Work)

1. **PerformanceAnalyzer**: Not used in backtest - could add for richer metrics
2. **4-Dimensional Scoring**: Backtest uses single score; could add weighted components
3. **Prompt Updates**: Backtest doesn't update prompts mid-run (would need workflow)

These gaps are acceptable because:
- Backtest only needs replan/constraint decisions, not full reports
- Adding PerformanceAnalyzer would slow backtests significantly
- Prompt updates during backtest would complicate determinism
