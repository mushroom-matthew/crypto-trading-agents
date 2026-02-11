# Branch: stance-diversity

## Purpose
The LLM strategist never uses `stance: defensive` or `stance: wait` despite the prompt allowing them and the judge repeatedly recommending defensive posture. All 21 plans in backtest ebf53879 used `stance: active`, even when the judge flagged bear/unclear regimes.

## Source Evidence
- Backtest `ebf53879`: 21 plans generated, 100% `stance: active`
- Judge regime corrections included:
  - "Market regime is unclear...shift to conservative, lower-frequency entries"
  - "BTC downtrend...favor exits/risk reductions and reduce new long entries"
  - "Treat as fragile uptrend...favor smaller sizing, tighter stops"
- Regime assessments varied: bull (6), mixed (11), bear (1), range (3) — but stance never changed
- The prompt says `wait: No triggers - market conditions unfavorable` but provides only one wait example vs extensive active examples

## Root Cause
1. **Prompt imbalance**: The prompt has detailed guidance for `active` stance with full trigger examples, but only a tiny `wait` example and no `defensive` example at all
2. **No judge→stance coupling**: The judge's `regime_correction` and `vetoes` are free-text fields that the LLM may or may not incorporate. There's no structured mechanism to force stance changes
3. **LLM bias toward action**: GPT-5 defaults to generating "useful" output (triggers) rather than empty/minimal output, even when the context suggests restraint

## Scope
1. **Add defensive stance example** to prompts with concrete trigger patterns
2. **Add structured stance hints** from judge to strategist (not just free text)
3. **Add stance tracking telemetry** so the judge can see stance distribution over time
4. **Consider rule-based stance override** when judge explicitly recommends it

## Out of Scope
- Changing the trigger engine behavior per stance (already handled — `wait` means empty triggers)
- Risk parameter changes (those flow through `risk_constraints`)

## Key Files
- `prompts/llm_strategist_simple.txt` — Add defensive example, strengthen stance guidance
- `prompts/strategy_plan_schema.txt` — Add defensive trigger patterns
- `backtesting/llm_strategist_runner.py` — Add stance history to judge snapshot
- `services/judge_feedback_service.py` — Add structured `recommended_stance` field
- `agents/strategies/prompt_builder.py` — Wire judge stance recommendation into strategist prompt

## Implementation Steps

### Step 1: Enrich stance guidance in prompts
Add to `llm_strategist_simple.txt`:
```
Stance guidance:
- active: Normal trading with entry and exit triggers. Use when regime is clear and
  indicators align.
- defensive: REDUCED exposure and TIGHTER stops. Use when:
  - Regime is unclear/mixed
  - Recent losses suggest poor signal quality
  - Multiple support/resistance breakdowns without resolution
  Defensive plans should:
  - Halve position sizing (target_risk_pct / 2)
  - Use only high-confidence (A-grade) triggers
  - Remove speculative entries (C-grade, reversal shorts)
  - Keep all exit and risk_reduce triggers active
- wait: NO triggers. Use when:
  - Conflicting signals across all timeframes
  - Extreme volatility without directional clarity
  - Judge has explicitly recommended waiting

IMPORTANT: You MUST use defensive or wait stance when conditions warrant it.
Using active stance during adverse conditions leads to unnecessary losses.
```

### Step 2: Add defensive example
Add a concrete defensive plan example to the prompt showing:
- Fewer triggers (3-4 instead of 7)
- Only A-grade confidence
- Halved `target_risk_pct` in sizing_rules
- No speculative categories (reversal, volatility_breakout)

### Step 3: Structured stance recommendation from judge
In `JudgeFeedbackService`, add a `recommended_stance` field:
```python
class JudgeConstraints:
    recommended_stance: Optional[str]  # "active", "defensive", "wait"
```
Wire this into the strategist prompt builder so the LLM sees: `JUDGE_RECOMMENDED_STANCE: defensive`

### Step 4: Stance distribution telemetry
Track stance usage in judge snapshot:
```python
"stance_history_last_5_plans": ["active", "active", "active", "active", "active"],
"stance_diversity_score": 0.0,  # 0 = all same, 1 = even distribution
```

## Test Plan
```bash
# Unit: prompt contains defensive example
python3 -c "
text = open('prompts/llm_strategist_simple.txt').read()
assert 'defensive' in text.lower() and 'MUST use defensive' in text
print('PASS: prompt contains defensive stance guidance')
"

# Unit: recommended_stance field in judge feedback
uv run pytest tests/test_judge_feedback.py -k recommended_stance -vv

# Integration: backtest with bear regime should produce at least 1 defensive/wait plan
```

## Test Evidence
```
tests/test_judge_death_spiral.py::TestStanceDiversity::test_recommended_stance_on_drawdown PASSED
tests/test_judge_death_spiral.py::TestStanceDiversity::test_recommended_stance_wait_on_low_quality PASSED
tests/test_judge_death_spiral.py::TestStanceDiversity::test_recommended_stance_active_when_healthy PASSED
tests/test_judge_death_spiral.py::TestStanceDiversity::test_stance_history_in_snapshot PASSED
tests/test_judge_death_spiral.py::TestStanceDiversity::test_stance_diversity_all_same PASSED
tests/test_judge_death_spiral.py::TestStanceDiversity::test_display_constraints_has_recommended_stance PASSED
```
All 6 stance diversity tests pass. Recommended stance computed from heuristics: `defensive` on drawdown (>30% emergency exits or >2% loss), `wait` on quality score <30, `active` otherwise. Stance history (last 5) and Shannon entropy diversity score tracked in snapshot. `DisplayConstraints` schema includes `recommended_stance` field. `judge_recommended_stance` wired into strategist prompt context.

Prompt verification: `llm_strategist_simple.txt` contains enriched stance guidance with concrete defensive example (fewer triggers, A-grade only, halved sizing).

## Acceptance Criteria
- [x] Prompt includes concrete defensive stance example with reduced triggers/sizing
- [x] Judge feedback includes structured `recommended_stance` field
- [x] Strategist prompt builder wires `JUDGE_RECOMMENDED_STANCE` into LLM context
- [x] Stance distribution tracked in judge snapshot
- [ ] Backtest with mixed/bear regimes produces at least 1 non-active stance plan — *requires validation backtest*

## Change Log
| Date | Change | Author |
|------|--------|--------|
| 2026-02-10 | Runbook created from backtest ebf53879 analysis | Claude |
| 2026-02-11 | Implemented: enriched stance guidance + defensive example in prompts, recommended_stance in DisplayConstraints schema, heuristic-based stance computation in judge_feedback_service.py, stance history + diversity score in snapshot, wired into strategist prompt context | Claude |

## Git Workflow
```bash
git checkout -b fix/stance-diversity
# ... implement changes ...
git add prompts/llm_strategist_simple.txt prompts/strategy_plan_schema.txt services/judge_feedback_service.py agents/strategies/prompt_builder.py
git commit -m "Improve stance diversity: defensive examples, judge stance hints, distribution tracking"
```
