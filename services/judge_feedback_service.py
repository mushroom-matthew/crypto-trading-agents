"""Judge feedback service following LLMClient transport pattern.

This service computes deterministic heuristics from performance metrics,
then either:
1. Uses a transport (shim) to return deterministic feedback based on heuristics
2. Calls the LLM with heuristics as context for richer evaluation
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from pydantic import ValidationError

from agents.langfuse_utils import langfuse_span
from agents.llm.client_factory import get_llm_client
from schemas.judge_feedback import JudgeFeedback, JudgeConstraints, DisplayConstraints
from trading_core.trade_quality import TradeMetrics

logger = logging.getLogger(__name__)


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
            "- Score adjustments applied:",
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
            for key, items in self.suggested_strategist_constraints.items():
                if items:
                    if isinstance(items, list):
                        for item in items:
                            lines.append(f"  - {key}: {item}")
                    elif isinstance(items, dict):
                        for sym, instr in items.items():
                            lines.append(f"  - {key}[{sym}]: {instr}")
                    else:
                        lines.append(f"  - {key}: {items}")

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


def _split_notes_and_json(response_text: str) -> tuple[str, str]:
    """Return (notes, json_str) from a two-section judge response."""
    if not isinstance(response_text, str):
        raise ValueError("Judge response must be text")
    if "JSON:" not in response_text:
        raise ValueError("Missing JSON section in judge response")
    notes_segment, json_segment = response_text.split("JSON:", 1)
    if "NOTES:" in notes_segment:
        notes_text = notes_segment.split("NOTES:", 1)[1].strip()
    else:
        notes_text = notes_segment.strip()
    start_idx = json_segment.find("{")
    end_idx = json_segment.rfind("}")
    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError("Could not locate JSON object in judge response")
    json_block = json_segment[start_idx : end_idx + 1]
    return notes_text, json_block


class JudgeFeedbackService:
    """Generate judge feedback using heuristics as context for LLM.

    Following the pattern from LLMClient:
    - When transport is provided (shim), returns deterministic feedback
    - When transport is None, calls LLM with heuristics as context
    """

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
        """Compute deterministic heuristic analysis from metrics.

        This extracts and structures all the heuristic logic that was previously
        embedded in _judge_feedback(), making it available as context for the LLM.
        """
        analysis = HeuristicAnalysis(base_score=50.0)
        return_pct = summary.get("return_pct", 0.0)
        trade_count = summary.get("trade_count", 0)

        # Return-based score adjustment
        return_delta = max(min(return_pct, 5.0), -5.0) * 4.0
        if abs(return_delta) > 0.1:
            analysis.score_adjustments.append({
                "reason": f"Return {return_pct:.2f}%",
                "delta": return_delta,
            })

        # Trade metrics analysis
        if trade_metrics:
            win_rate = trade_metrics.win_rate
            profit_factor = trade_metrics.profit_factor
            emergency_pct = trade_metrics.emergency_exit_pct
            max_consecutive_losses = trade_metrics.max_consecutive_losses
            quality_score = trade_metrics.quality_score

            analysis.observations.append(
                f"Deterministic quality score: {quality_score:.1f}/100"
            )
            analysis.observations.append(
                f"Win rate: {win_rate * 100:.1f}%, Profit factor: {profit_factor:.2f}"
            )

            # Win rate impact
            if win_rate >= 0.6:
                analysis.score_adjustments.append({
                    "reason": "Strong win rate (>=60%)",
                    "delta": 5.0,
                })
                analysis.observations.append("Strong win rate; strategy executing well.")
            elif win_rate < 0.4 and trade_count > 3:
                analysis.score_adjustments.append({
                    "reason": "Low win rate (<40%)",
                    "delta": -8.0,
                })
                analysis.red_flags.append("Low win rate (<40%); review trigger accuracy.")

            # Profit factor impact
            if profit_factor >= 1.5:
                analysis.score_adjustments.append({
                    "reason": "Good profit factor (>=1.5)",
                    "delta": 5.0,
                })
            elif profit_factor < 1.0 and trade_count > 3:
                analysis.score_adjustments.append({
                    "reason": "Poor profit factor (<1.0)",
                    "delta": -5.0,
                })
                analysis.red_flags.append("Profit factor <1.0; losses exceed gains.")

            # Emergency exit frequency
            if emergency_pct > 0.3:
                analysis.score_adjustments.append({
                    "reason": f"HIGH emergency exit rate ({emergency_pct * 100:.0f}%)",
                    "delta": -10.0,
                })
                analysis.red_flags.append(
                    f"HIGH emergency exit rate ({emergency_pct * 100:.0f}%); competing signals issue."
                )
            elif emergency_pct > 0.2:
                analysis.score_adjustments.append({
                    "reason": f"Elevated emergency exits ({emergency_pct * 100:.0f}%)",
                    "delta": -5.0,
                })
                analysis.observations.append(
                    f"Elevated emergency exits ({emergency_pct * 100:.0f}%); review signal priority."
                )

            # Consecutive losses
            if max_consecutive_losses >= 4:
                analysis.score_adjustments.append({
                    "reason": f"Streak of {max_consecutive_losses} consecutive losses",
                    "delta": -8.0,
                })
                analysis.red_flags.append(
                    f"Streak of {max_consecutive_losses} consecutive losses; possible regime mismatch."
                )

            # Build suggested constraints from trade metrics
            if emergency_pct > 0.25:
                analysis.suggested_strategist_constraints.setdefault("must_fix", []).append(
                    "HIGH emergency exit rate indicates competing signals. "
                    "Prioritize single highest-confidence trigger per symbol per timeframe."
                )
                analysis.suggested_strategist_constraints.setdefault("vetoes", []).append(
                    "Disable lower-confidence triggers when higher-grade trigger is active."
                )

            if win_rate < 0.4 and trade_count > 3:
                analysis.suggested_strategist_constraints.setdefault("must_fix", []).append(
                    "Win rate below 40%; review entry conditions for false signals."
                )

            # Category-specific feedback
            for cat, stats in (trade_metrics.category_stats or {}).items():
                cat_wr = stats.get("win_rate", 0)
                cat_count = stats.get("count", 0)
                if cat_count >= 3 and cat_wr < 0.3:
                    analysis.suggested_strategist_constraints.setdefault("vetoes", []).append(
                        f"Disable {cat} triggers (win rate {cat_wr * 100:.0f}% over {cat_count} trades)."
                    )
                elif cat_count >= 3 and cat_wr > 0.7:
                    analysis.suggested_strategist_constraints.setdefault("boost", []).append(
                        f"Favor {cat} triggers (win rate {cat_wr * 100:.0f}%)."
                    )

        # Trade count analysis
        if trade_count == 0:
            analysis.score_adjustments.append({
                "reason": "No trades executed",
                "delta": -5.0,
            })
            analysis.observations.append("No trades executed; confirm trigger sensitivity.")
            analysis.suggested_strategist_constraints.setdefault("must_fix", []).append(
                "Increase selectivity but avoid paralysis; at least one qualified trigger per day."
            )
        elif trade_count > 12:
            analysis.score_adjustments.append({
                "reason": "Elevated trade count (>12)",
                "delta": -3.0,
            })
            analysis.observations.append("Elevated trade count; monitor over-trading risk.")
            analysis.suggested_strategist_constraints.setdefault("vetoes", []).append(
                "Disable redundant scalp triggers bleeding cost."
            )
            analysis.suggested_strategist_constraints.setdefault("must_fix", []).append(
                "Cap total trades at 10 until judge score > 65."
            )
        else:
            analysis.observations.append("Trade cadence within expected bounds.")

        # Return-based observations and constraints
        if return_pct > 0.5:
            analysis.observations.append("Positive daily performance.")
            analysis.suggested_strategist_constraints.setdefault("boost", []).append(
                "Favor trend_continuation setups when volatility is orderly."
            )
            analysis.suggested_strategist_constraints["regime_correction"] = (
                "Lean pro-trend but protect gains with trailing exits."
            )
        elif return_pct < -0.5:
            analysis.observations.append("Drawdown detected; tighten stops.")
            analysis.suggested_strategist_constraints.setdefault("must_fix", []).append(
                "Tighten exits after 0.8% adverse move."
            )
            analysis.suggested_strategist_constraints.setdefault("vetoes", []).append(
                "Pause volatility_breakout triggers during drawdown."
            )
            analysis.suggested_strategist_constraints["regime_correction"] = (
                "Assume mixed regime until positive close recorded."
            )
        else:
            analysis.suggested_strategist_constraints["regime_correction"] = (
                "Maintain discipline until equity stabilizes."
            )

        # Position-based sizing adjustments
        positions = summary.get("positions_end", {})
        sizing_adjustments = {}
        for symbol in positions.keys():
            if return_pct < 0:
                sizing_adjustments[symbol] = "Cut risk by 25% until two winning days post drawdown."
            elif return_pct > 0.5:
                sizing_adjustments[symbol] = "Allow full allocation for grade A triggers only."
        if sizing_adjustments:
            analysis.suggested_strategist_constraints["sizing_adjustments"] = sizing_adjustments

        # Trigger budget analysis
        limit_stats = summary.get("limit_stats") or {}
        attempted = summary.get("attempted_triggers", 0)
        executed = summary.get("executed_trades", 0)
        trigger_budget = None
        if (attempted - executed) >= 10 or limit_stats.get("blocked_by_daily_cap", 0) >= 5:
            trigger_budget = 6
            analysis.suggested_constraints["max_triggers_per_symbol_per_day"] = trigger_budget
            analysis.suggested_strategist_constraints.setdefault("must_fix", []).append(
                "Limit strategist to <=6 high-conviction triggers per symbol until cadence improves."
            )

        # P&L breakdown analysis
        pnl_breakdown = summary.get("pnl_breakdown") or {}
        if pnl_breakdown:
            gross_pct = pnl_breakdown.get("gross_trade_pct", 0.0)
            fees_pct = pnl_breakdown.get("fees_pct", 0.0)
            flatten_pct = pnl_breakdown.get("flattening_pct", 0.0)
            if gross_pct > 0 and gross_pct + fees_pct + flatten_pct < 0:
                analysis.observations.append(
                    "Signals profitable pre-costs but fees/flattening erase gains."
                )
                analysis.suggested_strategist_constraints.setdefault("must_fix", []).append(
                    "Reduce churn; disable scalp triggers that only generate fee drag."
                )

        # Noisy trigger detection
        trigger_stats = summary.get("trigger_stats") or {}
        noisy_triggers = [
            trigger_id
            for trigger_id, stats in trigger_stats.items()
            if stats.get("executed", 0) == 0 and stats.get("blocked", 0) >= 5
        ]
        if noisy_triggers:
            analysis.suggested_strategist_constraints.setdefault("vetoes", []).append(
                f"Disable noisy triggers: {', '.join(noisy_triggers)}"
            )
            analysis.suggested_constraints["disabled_trigger_ids"] = noisy_triggers

        # Risk budget analysis
        risk_budget = summary.get("risk_budget")
        if risk_budget:
            used_pct = risk_budget.get("used_pct", 0.0)
            utilization_pct = risk_budget.get("utilization_pct", used_pct)
            if utilization_pct < 25 and return_pct > 0:
                analysis.observations.append(
                    "Risk budget underutilized despite gains; lean into high-conviction setups."
                )
                analysis.suggested_strategist_constraints.setdefault("boost", []).append(
                    "Increase size on grade A triggers until at least 30% of daily risk is deployed."
                )
            elif utilization_pct > 75 and return_pct < 0:
                analysis.observations.append(
                    "Risk budget heavily used on a losing day; tighten selectivity."
                )
                analysis.suggested_strategist_constraints.setdefault("must_fix", []).append(
                    "Stop firing marginal triggers once 75% of daily risk is consumed."
                )
                if trigger_budget is None:
                    analysis.suggested_constraints["max_triggers_per_symbol_per_day"] = 6
                else:
                    analysis.suggested_constraints["max_triggers_per_symbol_per_day"] = min(
                        trigger_budget, 6
                    )
            for symbol, pct in (risk_budget.get("symbol_usage_pct") or {}).items():
                if pct >= 70:
                    analysis.suggested_strategist_constraints.setdefault(
                        "sizing_adjustments", {}
                    )[symbol] = "Cut risk by 25% until per-symbol risk share normalizes."

        return analysis

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
        payload_json = json.dumps(payload, default=str)

        if self.transport:
            # Shim path: transport returns deterministic response
            raw = self.transport(payload_json)
            feedback = JudgeFeedback.model_validate_json(raw)
            self.last_generation_info = {"source": "transport", "heuristics": heuristics.to_dict()}
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
        prompt_template = self._load_prompt_template()

        # Format performance summary
        def _fmt_pct(value: float | None) -> str:
            if not isinstance(value, (int, float)):
                return "n/a"
            return f"{float(value) * 100:.2f}%"

        perf_lines = [
            f"Total return: {summary.get('return_pct', 0.0):.2f}%",
            f"Trade count: {summary.get('trade_count', 0)}",
            f"Winning trades: {summary.get('winning_trades', 0)}",
            f"Losing trades: {summary.get('losing_trades', 0)}",
        ]
        if trade_metrics:
            perf_lines.extend([
                f"Win rate: {_fmt_pct(trade_metrics.win_rate)}",
                f"Profit factor: {trade_metrics.profit_factor:.2f}",
                f"Quality score: {trade_metrics.quality_score:.1f}/100",
            ])
        performance_summary = "\n".join(perf_lines)

        # Format transaction summary (empty for backtest context)
        transaction_summary = "See trade metrics above for summary."

        def _truncate(text: str, limit: int = 160) -> str:
            if len(text) <= limit:
                return text
            return text[: limit - 3] + "..."

        fills_since_last_judge = summary.get("fills_since_last_judge") or []
        if fills_since_last_judge:
            fill_lines = []
            for fill in fills_since_last_judge[:20]:
                fill_lines.append(
                    "- {timestamp} {symbol} {side} qty={qty} price={price} trigger={trigger_id} pnl={pnl}".format(
                        timestamp=fill.get("timestamp"),
                        symbol=fill.get("symbol"),
                        side=fill.get("side"),
                        qty=fill.get("qty"),
                        price=fill.get("price"),
                        trigger_id=fill.get("trigger_id"),
                        pnl=fill.get("pnl"),
                    )
                )
            fill_details = "\n".join(fill_lines)
        else:
            fill_details = "No fills since last judge."

        trigger_attempts = summary.get("trigger_attempts") or {}
        if trigger_attempts:
            attempt_lines = []
            ranked = sorted(
                trigger_attempts.items(),
                key=lambda item: (item[1].get("attempted", 0), item[1].get("blocked", 0)),
                reverse=True,
            )
            for trigger_id, stats in ranked[:20]:
                reasons = stats.get("blocked_by_reason") or {}
                reason_str = ", ".join(f"{key}:{val}" for key, val in list(reasons.items())[:5])
                attempt_lines.append(
                    f"- {trigger_id}: attempted={stats.get('attempted', 0)} "
                    f"executed={stats.get('executed', 0)} blocked={stats.get('blocked', 0)}"
                    + (f" reasons={reason_str}" if reason_str else "")
                )
            trigger_attempts_text = "\n".join(attempt_lines)
        else:
            trigger_attempts_text = "No trigger attempts recorded."

        active_triggers = summary.get("active_triggers") or []
        if active_triggers:
            trigger_lines = []
            for trigger in active_triggers[:20]:
                entry_rule = _truncate(str(trigger.get("entry_rule", "")))
                exit_rule = _truncate(str(trigger.get("exit_rule", "")))
                trigger_lines.append(
                    "- {id} {symbol} {timeframe} {direction} {category} "
                    "conf={confidence} entry={entry_rule} exit={exit_rule}".format(
                        id=trigger.get("id"),
                        symbol=trigger.get("symbol"),
                        timeframe=trigger.get("timeframe"),
                        direction=trigger.get("direction"),
                        category=trigger.get("category"),
                        confidence=trigger.get("confidence"),
                        entry_rule=entry_rule,
                        exit_rule=exit_rule,
                    )
                )
            active_triggers_text = "\n".join(trigger_lines)
        else:
            active_triggers_text = "No active triggers provided."

        position_quality = summary.get("position_quality") or []
        if position_quality:
            position_lines = []
            for entry in position_quality[:10]:
                position_lines.append(
                    "- {symbol}: pnl={pnl:+.2f}% hold={hold:.1f}h risk={risk:.0f} underwater={underwater}".format(
                        symbol=entry.get("symbol"),
                        pnl=float(entry.get("unrealized_pnl_pct", 0.0)),
                        hold=float(entry.get("hold_hours", 0.0)),
                        risk=float(entry.get("risk_score", 0.0)),
                        underwater=bool(entry.get("is_underwater")),
                    )
                )
            position_quality_text = "\n".join(position_lines)
        else:
            position_quality_text = "No open positions."

        market_structure = summary.get("market_structure") or {}
        if market_structure:
            market_structure_text = json.dumps(market_structure, indent=2, default=str)
        else:
            market_structure_text = "No market structure provided."

        factor_exposures = summary.get("factor_exposures") or {}
        if factor_exposures:
            factor_exposures_text = json.dumps(factor_exposures, indent=2, default=str)
        else:
            factor_exposures_text = "No factor exposures provided."

        trigger_summary = (
            f"{len(active_triggers)} active triggers; {len(trigger_attempts)} trigger attempts since last judge."
            if active_triggers or trigger_attempts
            else "No trigger data available."
        )

        # Build the analysis prompt
        analysis_prompt = prompt_template.format(
            strategy_context=json.dumps(strategy_context, indent=2) if strategy_context else "No strategy context available.",
            risk_parameters=strategy_context.get("risk_params", "No risk parameters configured.") if strategy_context else "No risk parameters configured.",
            trigger_summary=trigger_summary,
            execution_settings="Default execution settings.",
            performance_summary=performance_summary,
            transaction_summary=transaction_summary,
            fill_details=fill_details,
            trigger_attempts=trigger_attempts_text,
            active_triggers=active_triggers_text,
            position_quality=position_quality_text,
            market_structure=market_structure_text,
            factor_exposures=factor_exposures_text,
            risk_state=json.dumps(summary.get("risk_state", {}), indent=2) if summary.get("risk_state") else "No risk state provided.",
            total_transactions=summary.get("trade_count", 0),
            buy_count=summary.get("buy_count", 0),
            sell_count=summary.get("sell_count", 0),
            symbols=", ".join(strategy_context.get("pairs", [])) if strategy_context else "Unknown",
        )

        # Append heuristic analysis section
        analysis_prompt += "\n\n" + heuristics.to_prompt_section()
        analysis_prompt += """

IMPORTANT: The heuristic pre-analysis above provides deterministic metrics-based assessment.
You should use this as a starting point but can adjust the score and constraints based on
your deeper analysis of strategy alignment, risk discipline, and execution quality.
Your final score can differ from the heuristic score if you have good reasons."""

        try:
            with langfuse_span("judge_feedback.backtest", metadata={"model": self.model}) as span:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert trading analyst evaluating algorithmic trading decisions. Provide objective, data-driven analysis.",
                        },
                        {"role": "user", "content": analysis_prompt},
                    ],
                    max_completion_tokens=800,
                )

                analysis_text = response.choices[0].message.content
                if span:
                    span.end(output=analysis_text)

            if not analysis_text:
                raise ValueError("Empty judge response")

            notes_text, json_block = _split_notes_and_json(analysis_text)
            feedback = JudgeFeedback.model_validate_json(json_block)
            # Preserve notes from LLM
            feedback.notes = notes_text

            self.last_generation_info = {
                "source": "llm",
                "model": self.model,
                "heuristics": heuristics.to_dict(),
                "raw_output": analysis_text,
            }
            return feedback

        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.warning("Judge feedback parse failed, falling back to heuristics: %s", exc)
            return self._feedback_from_heuristics(heuristics)
        except Exception as exc:
            logger.error("LLM judge call failed, falling back to heuristics: %s", exc)
            return self._feedback_from_heuristics(heuristics)

    def _feedback_from_heuristics(self, heuristics: HeuristicAnalysis) -> JudgeFeedback:
        """Build JudgeFeedback directly from heuristics (fallback)."""
        suggested = heuristics.suggested_strategist_constraints

        constraints = JudgeConstraints(
            max_trades_per_day=heuristics.suggested_constraints.get("max_trades_per_day"),
            max_triggers_per_symbol_per_day=heuristics.suggested_constraints.get(
                "max_triggers_per_symbol_per_day"
            ),
            risk_mode=heuristics.suggested_constraints.get("risk_mode", "normal"),
            disabled_trigger_ids=heuristics.suggested_constraints.get("disabled_trigger_ids", []),
            disabled_categories=heuristics.suggested_constraints.get("disabled_categories", []),
        )

        strategist_constraints = DisplayConstraints(
            must_fix=suggested.get("must_fix", []),
            vetoes=suggested.get("vetoes", []),
            boost=suggested.get("boost", []),
            regime_correction=suggested.get("regime_correction"),
            sizing_adjustments=suggested.get("sizing_adjustments", {}),
        )

        self.last_generation_info = {"source": "heuristics_fallback", "heuristics": heuristics.to_dict()}

        return JudgeFeedback(
            score=heuristics.final_score,
            notes=" ".join(heuristics.observations[:5]) if heuristics.observations else "Heuristic-based feedback.",
            constraints=constraints,
            strategist_constraints=strategist_constraints,
        )

    @staticmethod
    @lru_cache(1)
    def _load_prompt_template() -> str:
        """Load the judge prompt template from file."""
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "llm_judge_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        # Fallback minimal prompt
        return """You are an expert performance judge for a multi-asset crypto strategist.

PERFORMANCE SNAPSHOT:
{performance_summary}

TRADING STATISTICS:
- Total transactions: {total_transactions}
- Buy orders: {buy_count}
- Sell orders: {sell_count}
- Symbols traded: {symbols}

Respond with **two sections only**:

NOTES:
- Provide up to five concise sentences on strategy alignment, risk adherence, and execution quality.

JSON:
{{
  "score": <float 0-100>,
  "constraints": {{
    "max_trades_per_day": null,
    "max_triggers_per_symbol_per_day": null,
    "risk_mode": "normal",
    "disabled_trigger_ids": [],
    "disabled_categories": []
  }},
  "strategist_constraints": {{
    "must_fix": [],
    "vetoes": [],
    "boost": [],
    "regime_correction": null,
    "sizing_adjustments": {{}}
  }}
}}"""
