"""LLM as Judge agent for evaluating and improving execution agent performance."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

from temporalio.client import Client, RPCError, RPCStatusCode
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agents.workflows.judge_agent_workflow import JudgeAgentWorkflow
from agents.workflows.execution_ledger_workflow import ExecutionLedgerWorkflow
from pydantic import ValidationError

from tools.performance_analysis import PerformanceAnalyzer, format_performance_report
from tools.agent_logger import AgentLogger
from agents.utils import check_and_process_feedback, tool_result_data
from agents.constants import (
    ORANGE, CYAN, GREEN, RED, RESET, 
    DEFAULT_LOG_LEVEL, DEFAULT_OPENAI_MODEL,
    DEFAULT_TEMPORAL_ADDRESS, DEFAULT_TEMPORAL_NAMESPACE,
    JUDGE_AGENT, JUDGE_WF_ID, LEDGER_WF_ID
)
from agents.logging_utils import setup_logging
from agents.temporal_utils import connect_temporal
from agents.langfuse_utils import init_langfuse
from agents.llm.client_factory import get_llm_client
from agents.event_emitter import emit_event
from schemas.judge_feedback import JudgeFeedback, JudgeConstraints
from services.risk_adjustment_service import multiplier_from_instruction

logger = setup_logging(__name__, level="INFO")

init_langfuse()
openai_client = get_llm_client()


def _split_notes_and_json(response_text: str) -> Tuple[str, str]:
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


class JudgeAgent:
    """LLM as Judge agent for performance evaluation and prompt optimization."""
    
    def __init__(self, temporal_client: Client, mcp_session: ClientSession):
        self.temporal_client = temporal_client
        self.mcp_session = mcp_session
        self.performance_analyzer = PerformanceAnalyzer()
        self.agent_logger = AgentLogger("judge_agent", temporal_client)
        self.user_preferences = {}  # Cache user preferences
        
        # Evaluation criteria weights
        self.criteria_weights = {
            "returns": 0.30,
            "risk_management": 0.25,
            "decision_quality": 0.25,
            "consistency": 0.20
        }
        
        # Performance threshold for prompt updates
        self.update_threshold = 50.0  # Trigger prompt examination if score below this
        self.replan_score_threshold = float(os.environ.get("JUDGE_REPLAN_SCORE_THRESHOLD", "45"))
        self.replan_cooldown_seconds = int(os.environ.get("JUDGE_REPLAN_COOLDOWN_SECONDS", "3600"))
        self.last_replan_timestamp: Optional[int] = None
    
    async def initialize(self) -> None:
        """Initialize the judge agent."""
        # Ensure judge workflow exists
        try:
            handle = self.temporal_client.get_workflow_handle("judge-agent")
            await handle.describe()
        except RPCError as err:
            if err.status == RPCStatusCode.NOT_FOUND:
                await self.temporal_client.start_workflow(
                    JudgeAgentWorkflow.run,
                    id="judge-agent",
                    task_queue=os.environ.get("TASK_QUEUE", "mcp-tools"),
                )
                logger.info("Started judge agent workflow")
    
    async def should_evaluate(self) -> bool:
        """Check if it's time for a new evaluation."""
        try:
            # First check for immediate evaluation triggers
            handle = self.temporal_client.get_workflow_handle("judge-agent")
            recent_evaluations = await handle.query("get_evaluations", {"limit": 5, "since_ts": 0})  # Get last 5 evaluations
            
            # Check if there's a pending trigger request
            for evaluation in recent_evaluations:
                if (evaluation.get("type") == "immediate_trigger" and 
                    evaluation.get("status") == "trigger_requested"):
                    logger.info("Found immediate evaluation trigger, proceeding with evaluation")
                    return True
            
            # Check if ledger workflow exists and has data
            ledger_handle = self.temporal_client.get_workflow_handle(
                os.environ.get("LEDGER_WF_ID", "mock-ledger")
            )
            
            try:
                await ledger_handle.describe()
                # Check if there are any transactions to evaluate
                recent_transactions = await ledger_handle.query("get_transaction_history", {"since_ts": 0, "limit": 1})
                if not recent_transactions:
                    logger.info("No transactions found yet, skipping evaluation")
                    return False
            except Exception:
                logger.info("Ledger workflow not ready yet, skipping evaluation")
                return False
            
            # Check cooldown timing
            return await handle.query("should_trigger_evaluation", 0.167)  # 10 minute cooldown (0.167 hours)
        except Exception as exc:
            logger.debug("Failed to check evaluation timing: %s", exc)
            return False
    
    async def collect_performance_data(self, window_days: int = 7) -> Dict:
        """Collect comprehensive performance data for evaluation."""
        try:
            # Get performance metrics from ledger
            ledger_handle = self.temporal_client.get_workflow_handle(
                os.environ.get("LEDGER_WF_ID", "mock-ledger")
            )
            
            # Verify workflow exists and is running
            try:
                await ledger_handle.describe()
            except Exception as desc_exc:
                if "not found" in str(desc_exc).lower():
                    logger.info("Ledger workflow not found - no trading data available yet")
                    return {
                        "performance_metrics": {"total_trades": 0, "total_pnl": 0.0, "max_drawdown": 0.0, "win_rate": 0.0},
                        "risk_metrics": {"total_portfolio_value": 1000.0, "cash_ratio": 1.0, "max_position_concentration": 0.0, "num_positions": 0},
                        "transaction_history": [],
                        "evaluation_period_days": window_days,
                        "status": "no_data"
                    }
                else:
                    raise desc_exc
            
            performance_metrics = await ledger_handle.query("get_performance_metrics", window_days)
            risk_metrics = await ledger_handle.query("get_risk_metrics")
            transaction_history = await ledger_handle.query("get_transaction_history", {
                "since_ts": int((datetime.now(timezone.utc) - timedelta(days=window_days)).timestamp()),
                "limit": 500
            })
            
            return {
                "performance_metrics": performance_metrics,
                "risk_metrics": risk_metrics,
                "transaction_history": transaction_history,
                "evaluation_period_days": window_days,
                "status": "success"
            }
            
        except Exception as exc:
            logger.error("Failed to collect performance data: %s", exc)
            return {
                "performance_metrics": {"total_trades": 0, "total_pnl": 0.0, "max_drawdown": 0.0, "win_rate": 0.0},
                "risk_metrics": {"total_portfolio_value": 1000.0, "cash_ratio": 1.0, "max_position_concentration": 0.0, "num_positions": 0},
                "transaction_history": [],
                "evaluation_period_days": window_days,
                "status": "error",
                "error": str(exc)
            }
    
    async def collect_strategy_context(self) -> Dict[str, Any]:
        """Collect strategy configuration context from available workflows."""
        context = {
            "strategy_context": "No active strategy configuration found.",
            "risk_parameters": "No risk parameters configured.",
            "trigger_summary": "No active triggers.",
            "execution_settings": "Default execution settings.",
        }

        try:
            # Try to get strategy specs from StrategySpecWorkflow
            try:
                spec_handle = self.temporal_client.get_workflow_handle("strategy-spec")
                all_specs = await spec_handle.query("list_strategy_specs")
                if all_specs:
                    spec_lines = []
                    for key, spec in all_specs.items():
                        spec_lines.append(f"- {key}: mode={spec.get('mode')}, "
                                        f"entry_conditions={len(spec.get('entry_conditions', []))}, "
                                        f"exit_conditions={len(spec.get('exit_conditions', []))}")
                        if spec.get('risk'):
                            risk = spec['risk']
                            spec_lines.append(f"  Risk: max_fraction={risk.get('max_fraction_of_balance')}, "
                                            f"risk_per_trade={risk.get('risk_per_trade_fraction')}, "
                                            f"max_drawdown={risk.get('max_drawdown_pct')}%")
                    context["strategy_context"] = "\n".join(spec_lines)
            except Exception:
                pass  # Workflow not found

            # Try to get user preferences which may include strategy settings
            prefs = await self.get_user_preferences()
            if prefs:
                pref_lines = [
                    f"- Risk tolerance: {prefs.get('risk_tolerance', 'moderate')}",
                    f"- Trading style: {prefs.get('trading_style', 'balanced')}",
                    f"- Experience level: {prefs.get('experience_level', 'intermediate')}",
                ]
                if prefs.get('max_position_risk_pct'):
                    pref_lines.append(f"- Max position risk: {prefs.get('max_position_risk_pct')}%")
                if prefs.get('max_daily_loss_pct'):
                    pref_lines.append(f"- Max daily loss: {prefs.get('max_daily_loss_pct')}%")
                if prefs.get('min_hold_hours'):
                    pref_lines.append(f"- Min hold hours: {prefs.get('min_hold_hours')}")
                if prefs.get('walk_away_profit_target_pct'):
                    pref_lines.append(f"- Walk-away target: {prefs.get('walk_away_profit_target_pct')}%")
                context["execution_settings"] = "\n".join(pref_lines)

            # Try to get execution agent state for active triggers/settings
            try:
                exec_handle = self.temporal_client.get_workflow_handle("execution-agent")
                exec_state = await exec_handle.query("get_execution_state")
                if exec_state:
                    state_lines = []
                    if exec_state.get('max_trades_per_day'):
                        state_lines.append(f"- Max trades/day: {exec_state.get('max_trades_per_day')}")
                    if exec_state.get('max_triggers_per_symbol_per_day'):
                        state_lines.append(f"- Max triggers/symbol/day: {exec_state.get('max_triggers_per_symbol_per_day')}")
                    if exec_state.get('risk_mode'):
                        state_lines.append(f"- Risk mode: {exec_state.get('risk_mode')}")
                    if state_lines:
                        context["risk_parameters"] = "\n".join(state_lines)
            except Exception:
                pass  # Workflow not found

        except Exception as exc:
            logger.warning("Failed to collect strategy context: %s", exc)

        return context

    def _get_fallback_prompt(self) -> str:
        """Fallback prompt if template file is not found."""
        return """You are an expert performance judge for a multi-asset crypto strategist.

STRATEGY CONFIGURATION:
{strategy_context}

RISK PARAMETERS:
{risk_parameters}

ACTIVE TRIGGERS:
{trigger_summary}

EXECUTION SETTINGS:
{execution_settings}

PERFORMANCE SNAPSHOT:
{performance_summary}

RECENT TRANSACTIONS:
{transaction_summary}

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

    async def analyze_decision_quality(self, transaction_history: List[Dict], performance_snapshot: Dict[str, Any]) -> Dict:
        """Use LLM to analyze decision quality from transaction patterns."""

        if not transaction_history:
            feedback = JudgeFeedback(score=50.0)
            return {
                "notes": "No transactions to analyze; keep collecting executions before judging quality.",
                "decision_score": 50.0,
                "judge_feedback": feedback.model_dump(),
                "constraints": feedback.constraints.model_dump(),
                "strategist_constraints": feedback.strategist_constraints.model_dump(),
                "recommendations": ["Wait for new trades before drawing conclusions"],
            }

        # Collect strategy context for comprehensive evaluation
        strategy_context = await self.collect_strategy_context()

        # Prepare transaction summary for LLM analysis
        recent_transactions = transaction_history[:20]  # Last 20 transactions
        transaction_summary = "\n".join([
            f"Time: {datetime.fromtimestamp(tx['timestamp']).strftime('%Y-%m-%d %H:%M')} | "
            f"{tx['side']} {tx['quantity']:.2f} {tx['symbol']} @ ${tx['fill_price']:.2f} | "
            f"Cost: ${tx['cost']:.2f}"
            for tx in recent_transactions
        ])

        # Calculate basic statistics
        symbols = list(set(tx['symbol'] for tx in transaction_history))
        buy_count = len([tx for tx in transaction_history if tx['side'] == 'BUY'])
        sell_count = len([tx for tx in transaction_history if tx['side'] == 'SELL'])
        def _fmt_pct(value: float | None) -> str:
            if not isinstance(value, (int, float)):
                return "n/a"
            return f"{float(value) * 100:.2f}%"

        perf_lines = [
            f"Window: {performance_snapshot.get('start_date', 'n/a')} -> {performance_snapshot.get('end_date', 'n/a')}",
            f"Total return: {_fmt_pct(performance_snapshot.get('total_return'))}",
            f"Annualized return: {_fmt_pct(performance_snapshot.get('annualized_return'))}",
            f"Volatility: {_fmt_pct(performance_snapshot.get('volatility'))}",
            f"Sharpe ratio: {performance_snapshot.get('sharpe_ratio', 'n/a')}",
            f"Max drawdown: {_fmt_pct(performance_snapshot.get('max_drawdown'))}",
            f"Win rate: {_fmt_pct(performance_snapshot.get('win_rate'))}",
            f"Total trades: {performance_snapshot.get('total_trades', 0)}",
            f"Profit factor: {performance_snapshot.get('profit_factor', 'n/a')}",
        ]
        performance_summary = "\n".join(perf_lines)

        # Load enhanced prompt template
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "llm_judge_prompt.txt"
        try:
            prompt_template = prompt_path.read_text()
        except Exception:
            # Fallback to embedded prompt if file not found
            prompt_template = self._get_fallback_prompt()

        analysis_prompt = prompt_template.format(
            strategy_context=strategy_context.get("strategy_context", "No strategy context available."),
            risk_parameters=strategy_context.get("risk_parameters", "No risk parameters configured."),
            trigger_summary=strategy_context.get("trigger_summary", "No active triggers."),
            execution_settings=strategy_context.get("execution_settings", "Default execution settings."),
            performance_summary=performance_summary,
            transaction_summary=transaction_summary,
            total_transactions=len(transaction_history),
            buy_count=buy_count,
            sell_count=sell_count,
            symbols=', '.join(symbols),
            fill_details=strategy_context.get("fill_details", "No fill details available."),
            trigger_attempts=strategy_context.get("trigger_attempts", "No trigger attempt data."),
            active_triggers=strategy_context.get("active_triggers", "No active trigger data."),
            position_quality=strategy_context.get("position_quality", "No position quality data."),
            market_structure=strategy_context.get("market_structure", "No market structure data."),
            factor_exposures=strategy_context.get("factor_exposures", "No factor exposure data."),
            risk_state=strategy_context.get("risk_state", "No risk state data."),
        )
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert trading analyst evaluating algorithmic trading decisions. Provide objective, data-driven analysis."
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                max_completion_tokens=800
            )

            analysis_text = response.choices[0].message.content
            if not analysis_text:
                raise ValueError("Empty judge response")
            notes_text, json_block = _split_notes_and_json(analysis_text)
            feedback = JudgeFeedback.model_validate_json(json_block)
            self._apply_structured_constraints(feedback)
            decision_score = feedback.score if feedback.score is not None else 50.0
            display_constraints = feedback.strategist_constraints.model_dump()
            machine_constraints = feedback.constraints.model_dump()
            return {
                "notes": notes_text,
                "decision_score": decision_score,
                "judge_feedback": feedback.model_dump(),
                "constraints": machine_constraints,
                "strategist_constraints": display_constraints,
                "recommendations": display_constraints.get("must_fix", [])[:3],
            }
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            logger.error("Judge feedback JSON invalid: %s", exc)
            raise
        except Exception as exc:
            logger.error("Failed to analyze decision quality: %s", exc)
            raise
    
    async def generate_evaluation_report(self, performance_data: Dict) -> Dict:
        """Generate comprehensive evaluation report."""
        performance_metrics = performance_data.get("performance_metrics", {})
        risk_metrics = performance_data.get("risk_metrics", {})
        transaction_history = performance_data.get("transaction_history", [])
        
        # Generate detailed performance report
        performance_report = self.performance_analyzer.generate_performance_report(
            transactions=transaction_history,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics
        )
        
        performance_report_dict = {
            "start_date": performance_report.start_date.isoformat(),
            "end_date": performance_report.end_date.isoformat(),
            "total_return": performance_report.total_return,
            "annualized_return": performance_report.annualized_return,
            "volatility": performance_report.volatility,
            "sharpe_ratio": performance_report.sharpe_ratio,
            "max_drawdown": performance_report.max_drawdown,
            "total_trades": performance_report.total_trades,
            "win_rate": performance_report.win_rate,
            "avg_win": performance_report.avg_win,
            "avg_loss": performance_report.avg_loss,
            "profit_factor": performance_report.profit_factor,
            "var_95": performance_report.var_95,
            "max_position_size": performance_report.max_position_size,
            "avg_position_size": performance_report.avg_position_size,
            "decision_quality_score": performance_report.decision_quality_score,
            "risk_management_score": performance_report.risk_management_score,
            "consistency_score": performance_report.consistency_score,
            "overall_grade": performance_report.overall_grade
        }

        # Analyze decision quality using LLM
        decision_analysis = await self.analyze_decision_quality(transaction_history, performance_report_dict)

        # Calculate overall evaluation score
        return_score = min(100.0, max(0.0, (performance_report.annualized_return + 1) * 50))
        risk_score = performance_report.risk_management_score
        decision_score = decision_analysis.get("decision_score", 50.0)
        consistency_score = performance_report.consistency_score
        
        overall_score = (
            return_score * self.criteria_weights["returns"] +
            risk_score * self.criteria_weights["risk_management"] +
            decision_score * self.criteria_weights["decision_quality"] +
            consistency_score * self.criteria_weights["consistency"]
        )
        
        return {
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
            "evaluation_period_days": performance_data.get("evaluation_period_days", 7),
            "overall_score": overall_score,
            "component_scores": {
                "returns": return_score,
                "risk_management": risk_score,
                "decision_quality": decision_score,
                "consistency": consistency_score
            },
            "performance_report": performance_report_dict,
            "decision_analysis": decision_analysis,
            "strategist_constraints": decision_analysis.get("strategist_constraints", {}),
            "metrics": {
                "total_trades": len(transaction_history),
                "annualized_return": performance_report.annualized_return,
                "sharpe_ratio": performance_report.sharpe_ratio,
                "max_drawdown": performance_report.max_drawdown,
                "win_rate": performance_report.win_rate
            },
            "recommendations": decision_analysis.get("recommendations", [])
        }
    
    async def get_user_preferences(self) -> Dict:
        """Retrieve user preferences from judge workflow."""
        try:
            handle = self.temporal_client.get_workflow_handle("judge-agent")
            prefs = await handle.query("get_user_preferences")
            self.user_preferences = prefs
            return prefs
        except Exception as exc:
            logger.warning("Failed to get user preferences from judge workflow: %s", exc)
            return {}
    
    def parse_percentage(self, value: str, default: float) -> float:
        """Parse percentage string to float (e.g., '50%' -> 0.5)."""
        try:
            if isinstance(value, str) and value.endswith('%'):
                return float(value.rstrip('%')) / 100.0
            elif isinstance(value, (int, float)):
                return float(value) if value <= 1.0 else float(value) / 100.0
            else:
                return default
        except (ValueError, TypeError):
            return default
    
    def get_user_baseline_preferences(self) -> Dict[str, str]:
        """Get user's preference strings."""
        prefs = self.user_preferences
        
        baseline = {
            "risk_tolerance": prefs.get('risk_tolerance', 'moderate'),
            "trading_style": prefs.get('trading_style', 'balanced'),
            "experience_level": prefs.get('experience_level', 'intermediate')
        }
        
        logger.info("User preferences: risk_tolerance=%s, trading_style=%s, experience_level=%s",
                   baseline["risk_tolerance"], baseline["trading_style"], baseline["experience_level"])
        
        return baseline
    
    async def determine_context_updates(self, evaluation: Dict, user_feedback: List[str] = None) -> Optional[Dict]:
        """Determine if context updates are needed based on evaluation and user feedback."""
        overall_score = evaluation["overall_score"]
        performance_report = evaluation["performance_report"]
        component_scores = evaluation["component_scores"]
        
        # Get user preferences
        await self.get_user_preferences()
        user_prefs = self.get_user_baseline_preferences()
        
        # Check if we have user feedback that warrants an update
        if user_feedback:
            return {
                "update_type": "user_feedback_update",
                "reason": f"User provided direct feedback about agent performance",
                "context": {
                    "performance_score": overall_score,
                    "component_scores": component_scores,
                    "user_preferences": user_prefs,
                    "user_feedback": user_feedback
                },
                "changes": [
                    f"Incorporating user feedback into system prompt",
                    f"User preferences: {user_prefs['risk_tolerance']} risk, {user_prefs['trading_style']} style",
                    f"Feedback: {user_feedback[0][:100]}..." if user_feedback else "No feedback"
                ],
                "performance_report": performance_report
            }
        
        # Check if performance warrants prompt examination and update
        if overall_score < self.update_threshold:
            return {
                "update_type": "performance_based_update",
                "reason": f"Performance score ({overall_score:.1f}) below threshold ({self.update_threshold})",
                "context": {
                    "performance_score": overall_score,
                    "component_scores": component_scores,
                    "user_preferences": user_prefs
                },
                "changes": [
                    f"Examining system prompt due to performance score of {overall_score:.1f}",
                    f"User preferences: {user_prefs['risk_tolerance']} risk, {user_prefs['trading_style']} style",
                    "Agent will determine appropriate adjustments based on performance analysis"
                ],
                "performance_report": performance_report
            }
        
        # No updates needed if performance is above threshold
        return None
    
    async def _improve_system_prompt(self, current_prompt: str, performance_report: Dict, context: Dict) -> str:
        """Use LLM to improve the system prompt based on performance analysis."""
        
        # Extract user preferences from context if available
        user_preferences = {}
        if 'context' in context and isinstance(context['context'], dict):
            user_preferences = context['context'].get('user_preferences', {})
        elif 'user_preferences' in context:
            user_preferences = context.get('user_preferences', {})
        
        # Extract user feedback from context if available
        user_feedback = []
        if 'context' in context and isinstance(context['context'], dict):
            user_feedback = context['context'].get('user_feedback', [])
        
        # Build user preferences section for the prompt
        user_prefs_section = ""
        if user_preferences:
            user_prefs_section = f"""
USER PREFERENCES (CRITICAL - MUST BE REFLECTED IN THE PROMPT):
- Risk Tolerance: {user_preferences.get('risk_tolerance', 'moderate')}
- Trading Style: {user_preferences.get('trading_style', 'balanced')}
- Experience Level: {user_preferences.get('experience_level', 'intermediate')}

These preferences MUST be explicitly incorporated into the system prompt with specific behavioral guidelines that match the user's profile.
"""
        
        # Build user feedback section for the prompt
        user_feedback_section = ""
        if user_feedback:
            user_feedback_section = f"""
USER FEEDBACK (CRITICAL - ADDRESS THESE CONCERNS):
{chr(10).join(f"- {feedback}" for feedback in user_feedback)}

The user has provided direct feedback about the agent's performance. You MUST address these specific concerns in the improved system prompt by adding explicit guidelines that will change the agent's behavior accordingly.
"""
        
        improvement_prompt = f"""You are a prompt engineering expert tasked with improving a trading agent's system prompt based on performance analysis and user preferences.

CURRENT SYSTEM PROMPT:
```
{current_prompt}
```

PERFORMANCE ANALYSIS:
```json
{json.dumps(performance_report, indent=2)}
```
{user_prefs_section}{user_feedback_section}
CONTEXT & ISSUES IDENTIFIED:
- Update Type: {context.get('update_type', 'general_improvement')}
- Reason: {context.get('reason', 'Performance optimization needed')}
- Specific Changes Needed: {context.get('changes', [])}

INSTRUCTIONS:
1. Analyze the current prompt and performance data
2. If user preferences are provided (risk tolerance, trading style, experience level), these should HEAVILY influence how you rewrite the prompt
3. If user feedback is provided, you MUST address each piece of feedback with specific behavioral changes in the prompt
4. Make the prompt reflect the user's preferences through concrete behavioral changes, not just superficial mentions
5. The trading agent's behavior should clearly align with the user's stated risk tolerance, trading style, and experience level
6. Keep the prompt length reasonable (not too verbose)
7. Ensure the agent remains fully autonomous and doesn't ask for confirmation

CRITICAL REQUIREMENTS TO PRESERVE:
- Full autonomy (no confirmation requests)
- Memory-based analysis using ALL ticks from conversation history
- Incremental data fetching with cumulative memory
- Specific order execution format
- Action reporting (not suggestion reporting)

Return ONLY the improved system prompt, no explanations."""

        try:
            response = openai_client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
                messages=[{"role": "user", "content": improvement_prompt}]
            )
            
            improved_prompt = response.choices[0].message.content.strip()
            
            # Remove any markdown code blocks if present
            if improved_prompt.startswith("```") and improved_prompt.endswith("```"):
                lines = improved_prompt.split('\n')
                improved_prompt = '\n'.join(lines[1:-1])
            
            return improved_prompt
            
        except Exception as exc:
            logger.error("Failed to improve system prompt with LLM: %s", exc)
            # Fallback to current prompt if LLM fails
            return current_prompt
    
    async def update_prompt_for_user_preferences(self, user_preferences: Dict) -> bool:
        """Update execution agent system prompt based on user preference changes."""
        try:
            # Get current system prompt from execution agent workflow
            current_prompt = ""
            try:
                handle = self.temporal_client.get_workflow_handle("execution-agent")
                current_prompt = await handle.query("get_system_prompt")
                if not current_prompt:
                    print(f"{RED}[JudgeAgent] No current system prompt found - using fallback{RESET}")
                    # Import the fallback prompt from execution agent
                    from agents.execution_agent_client import SYSTEM_PROMPT
                    current_prompt = SYSTEM_PROMPT
            except Exception as query_exc:
                print(f"{RED}[JudgeAgent] Failed to query current system prompt: {query_exc}{RESET}")
                return False
            
            # Create context for user preference update
            context = {
                "update_type": "user_preferences_update",
                "reason": f"User preferences updated - adjusting trading parameters to match user's risk tolerance and trading style",
                "changes": [
                    f"Risk tolerance: {user_preferences.get('risk_tolerance', 'moderate')}",
                    f"Trading style: {user_preferences.get('trading_style', 'balanced')}",
                    f"Experience level: {user_preferences.get('experience_level', 'intermediate')}"
                ],
                "user_preferences": user_preferences,
                "context": {
                    "user_preferences": user_preferences
                },
                "performance_report": {
                    "trigger": "user_preferences_change",
                    "user_preferences": user_preferences
                }
            }
            
            # Use LLM to adapt the system prompt to new user preferences
            improved_prompt = await self._improve_system_prompt(
                current_prompt=current_prompt,
                performance_report=context["performance_report"],
                context=context
            )
            
            # Update execution agent workflow's system prompt
            try:
                await handle.signal("update_system_prompt", improved_prompt)
                print(f"{GREEN}[JudgeAgent] ✅ System prompt updated for user preferences{RESET}")
                print(f"{CYAN}[JudgeAgent] New prompt length: {len(improved_prompt)} chars{RESET}")
                
                # Log the preference update
                await self.agent_logger.log_action(
                    action_type="preference_update",
                    details={
                        "trigger": "user_preferences_change",
                        "user_preferences": user_preferences,
                        "changes": context["changes"]
                    },
                    result={
                        "success": True,
                        "prompt_length": len(improved_prompt)
                    }
                )
                
                return True
                
            except Exception as signal_exc:
                print(f"{RED}[JudgeAgent] Failed to signal execution agent with updated prompt: {signal_exc}{RESET}")
                return False
                
        except Exception as exc:
            print(f"{RED}[JudgeAgent] Error updating prompt for user preferences: {exc}{RESET}")
            await self.agent_logger.log_action(
                action_type="preference_update", 
                details={"user_preferences": user_preferences},
                result={"success": False, "error": str(exc)}
            )
            return False
    
    async def implement_context_update(self, update_spec: Dict) -> bool:
        """Implement the specified context update."""
        try:
            update_type = update_spec["update_type"]
            new_context = update_spec["context"]
            reason = update_spec["reason"]
            changes = update_spec["changes"]
            
            # Get current system prompt from execution agent workflow
            current_prompt = ""
            try:
                handle = self.temporal_client.get_workflow_handle("execution-agent")
                current_prompt = await handle.query("get_system_prompt")
                if not current_prompt:
                    print(f"{RED}[JudgeAgent] No current system prompt found in workflow{RESET}")
                    return False
            except Exception as query_exc:
                print(f"{RED}[JudgeAgent] Failed to query current system prompt: {query_exc}{RESET}")
                return False
            
            # Use LLM to improve the system prompt based on performance analysis
            improved_prompt = await self._improve_system_prompt(
                current_prompt=current_prompt,
                performance_report=update_spec.get("performance_report", {}),
                context=update_spec
            )
            
            # Update execution agent workflow's system prompt
            try:
                await handle.signal("update_system_prompt", improved_prompt)
                success = True
                
                print(f"{GREEN}[JudgeAgent] Updated execution agent system prompt: {update_type}{RESET}")
                print(f"{CYAN}[JudgeAgent] Reason: {reason}{RESET}")
                print(f"{CYAN}[JudgeAgent] New context: {new_context}{RESET}")
                for change in changes:
                    print(f"{CYAN}[JudgeAgent] - {change}{RESET}")
                
                print(f"{GREEN}[JudgeAgent] New System Prompt (length: {len(improved_prompt)} chars):{RESET}")
                print(f"{CYAN}{'='*80}{RESET}")
                
                # Display first few lines of the prompt
                prompt_lines = improved_prompt.split('\n')[:10]
                for line in prompt_lines:
                    print(f"{CYAN}{line}{RESET}")
                if len(improved_prompt.split('\n')) > 10:
                    print(f"{CYAN}... (truncated, {len(improved_prompt.split('\n')) - 10} more lines){RESET}")
                print(f"{CYAN}{'='*80}{RESET}")
                
            except Exception as signal_exc:
                print(f"{RED}[JudgeAgent] Failed to signal execution agent: {signal_exc}{RESET}")
                success = False
            
            # Log the context update
            await self.agent_logger.log_action(
                action_type="context_update",
                details={
                    "update_type": update_type,
                    "context": new_context,
                    "reason": reason,
                    "changes": changes
                },
                result={
                    "success": success,
                    "prompt_length": len(improved_prompt) if success else 0,
                    "error": None if success else "Failed to signal execution agent"
                }
            )
            
            return success
            
        except Exception as exc:
            logger.error("Failed to implement context update: %s", exc)
            return False
    
    async def record_evaluation(self, evaluation: Dict) -> None:
        """Record the evaluation in the judge workflow."""
        try:
            handle = self.temporal_client.get_workflow_handle("judge-agent")
            await handle.signal("record_evaluation", evaluation)
            logger.info("Recorded evaluation with score %.1f", evaluation["overall_score"])
            try:
                payload = {
                    "overall_score": evaluation.get("overall_score"),
                    "component_scores": evaluation.get("component_scores"),
                    "recommendations": evaluation.get("recommendations", []),
                    "context_update": evaluation.get("context_update"),
                    "strategy_replans": evaluation.get("strategy_replans", []),
                }
                await emit_event(
                    "plan_judged",
                    payload,
                    source="judge_agent",
                    run_id="judge-agent",
                    correlation_id=str(evaluation.get("overall_score", "")),
                )
            except Exception:
                logger.debug("Failed to emit plan_judged event", exc_info=True)
        except Exception as exc:
            logger.error("Failed to record evaluation: %s", exc)
    
    async def _mark_triggers_processed(self) -> None:
        """Mark any pending trigger requests as processed."""
        try:
            handle = self.temporal_client.get_workflow_handle("judge-agent")
            await handle.signal("mark_triggers_processed", {})
        except Exception as exc:
            logger.debug("Failed to mark triggers as processed: %s", exc)

    async def trigger_strategy_replan(self, evaluation: Dict) -> Optional[List[Dict[str, Any]]]:
        """Trigger deterministic strategy re-planning when performance degrades."""
        overall_score = evaluation.get("overall_score", 100.0)
        now_ts = int(datetime.now(timezone.utc).timestamp())
        if overall_score >= self.replan_score_threshold:
            return None
        if (
            self.last_replan_timestamp
            and now_ts - self.last_replan_timestamp < self.replan_cooldown_seconds
        ):
            logger.info("Skipping re-plan: cooldown active")
            return None

        broker_wf = os.environ.get("BROKER_WF_ID", "broker-agent")
        try:
            broker_handle = self.temporal_client.get_workflow_handle(broker_wf)
            symbols: list[str] = await broker_handle.query("get_symbols")
        except Exception as exc:
            logger.warning("Unable to fetch active symbols for re-plan: %s", exc)
            return None

        if not symbols:
            logger.info("No active symbols to re-plan")
            return None

        prefs = await self.get_user_preferences()
        risk_profile = prefs.get("risk_tolerance", "medium")
        trading_style = prefs.get("trading_style", "balanced")
        mode_map = {
            "aggressive": "breakout",
            "balanced": "trend",
            "conservative": "mean_revert",
        }
        mode = mode_map.get(trading_style, None)
        timeframe = os.environ.get("STRATEGY_TIMEFRAME", "15m")
        recommendations = evaluation.get("recommendations", [])
        notes = (
            f"Judge-triggered replan due to low score {overall_score:.1f}. "
            f"Top recommendations: {', '.join(recommendations[:2]) if recommendations else 'improve risk discipline'}"
        )

        replan_results: List[Dict[str, Any]] = []
        for symbol in symbols:
            payload = {
                "market": symbol,
                "timeframe": timeframe,
                "risk_profile": risk_profile,
                "mode": mode,
                "notes": notes,
            }
            try:
                result = await self.mcp_session.call_tool("plan_strategy", payload)
                replan_results.append(
                    {
                        "symbol": symbol,
                        "strategy": tool_result_data(result),
                    }
                )
                print(
                    f"{ORANGE}[JudgeAgent] 🔁 Strategy re-plan triggered for {symbol} (risk={risk_profile}){RESET}"
                )
            except Exception as exc:
                logger.error("Failed to plan strategy for %s: %s", symbol, exc)

        if replan_results:
            self.last_replan_timestamp = now_ts
            await self.agent_logger.log_action(
                action_type="strategy_replan",
                details={
                    "trigger": "judge_auto",
                    "threshold": self.replan_score_threshold,
                    "overall_score": overall_score,
                    "symbols": [entry["symbol"] for entry in replan_results],
                },
                result={"success": True},
            )
            return replan_results
        return None

    
    async def run_evaluation_cycle(self) -> None:
        """Run a complete evaluation cycle."""
        print(f"{ORANGE}[JudgeAgent] Starting evaluation cycle{RESET}")
        
        try:
            # Check for user feedback first
            user_feedback = await check_and_process_feedback(
                self.temporal_client,
                "judge-agent",
                agent_name="JudgeAgent",
                color_start=CYAN,
                color_end=RESET
            )
            
            # Collect performance data
            print(f"{CYAN}[JudgeAgent] Collecting performance data...{RESET}")
            performance_data = await self.collect_performance_data(window_days=7)
            
            if not performance_data:
                print(f"{RED}[JudgeAgent] Failed to collect performance data{RESET}")
                return
            
            # Handle the case where there's no trading data yet
            status = performance_data.get("status", "unknown")
            if status == "no_data":
                print(f"{CYAN}[JudgeAgent] No trading data available yet - skipping evaluation{RESET}")
                print(f"{CYAN}[JudgeAgent] Waiting for execution agent to start trading...{RESET}")
                return
            elif status == "error":
                print(f"{RED}[JudgeAgent] Error collecting data: {performance_data.get('error', 'Unknown error')}{RESET}")
                return
            
            # Generate evaluation report
            print(f"{CYAN}[JudgeAgent] Generating evaluation report...{RESET}")
            evaluation = await self.generate_evaluation_report(performance_data)
            
            # Display results
            print(f"{GREEN}[JudgeAgent] Evaluation completed{RESET}")
            print(f"{CYAN}[JudgeAgent] Overall Score: {evaluation['overall_score']:.1f}/100{RESET}")
            print(f"{CYAN}[JudgeAgent] Component Scores:{RESET}")
            for component, score in evaluation["component_scores"].items():
                print(f"{CYAN}[JudgeAgent]   {component}: {score:.1f}{RESET}")
            
            # Check for context updates (include user feedback if any)
            update_spec = await self.determine_context_updates(evaluation, user_feedback)
            if update_spec:
                print(f"{ORANGE}[JudgeAgent] Context update recommended: {update_spec['update_type']}{RESET}")
                success = await self.implement_context_update(update_spec)
                evaluation["context_update"] = {
                    "implemented": success,
                    "update_spec": update_spec
                }
            else:
                print(f"{GREEN}[JudgeAgent] No context updates needed{RESET}")
                evaluation["context_update"] = {"implemented": False}
            
            # Record evaluation
            await self.record_evaluation(evaluation)
            
            # Log comprehensive evaluation to workflow
            try:
                await self.agent_logger.log_summary(
                    summary_type="performance_evaluation",
                    data={
                        "evaluation_period_days": evaluation.get("evaluation_period_days", 7),
                        "overall_score": evaluation["overall_score"],
                        "component_scores": evaluation["component_scores"],
                        "performance_metrics": {
                            "total_trades": evaluation["metrics"]["total_trades"],
                            "annualized_return": evaluation["metrics"]["annualized_return"],
                            "sharpe_ratio": evaluation["metrics"]["sharpe_ratio"],
                            "max_drawdown": evaluation["metrics"]["max_drawdown"],
                            "win_rate": evaluation["metrics"]["win_rate"]
                        },
                        "decision_analysis": evaluation["decision_analysis"],
                        "context_update": evaluation.get("context_update", {"implemented": False}),
                        "recommendations": evaluation.get("recommendations", [])
                    },
                    performance_data=performance_data,
                    trigger_type=performance_data.get("trigger_type", "scheduled")
                )
                
                # Log context update action if one was implemented
                if evaluation.get("context_update", {}).get("implemented"):
                    await self.agent_logger.log_action(
                        action_type="context_update",
                        details=evaluation["context_update"]["update_spec"],
                        result={"success": True}
                    )
                    
            except Exception as log_error:
                logger.error(f"Failed to log evaluation: {log_error}")
            
            replan_info = await self.trigger_strategy_replan(evaluation)
            if replan_info:
                evaluation["strategy_replans"] = replan_info
                print(f"{ORANGE}[JudgeAgent] Triggered strategy re-planning for {len(replan_info)} market(s){RESET}")
            else:
                evaluation["strategy_replans"] = []
            
            # Mark any trigger requests as processed
            await self._mark_triggers_processed()
            
            # Display detailed report if requested
            if os.environ.get("JUDGE_VERBOSE", "false").lower() == "true":
                report_text = format_performance_report(evaluation["performance_report"])
                print(f"{CYAN}[JudgeAgent] Detailed Report:{RESET}")
                print(report_text)
            
        except Exception as exc:
            logger.error("Evaluation cycle failed: %s", exc)
            print(f"{RED}[JudgeAgent] Evaluation cycle failed: {exc}{RESET}")

    def _apply_structured_constraints(self, feedback: JudgeFeedback) -> None:
        constraints = feedback.constraints
        display = feedback.strategist_constraints
        for entry in display.must_fix or []:
            lowered = entry.lower()
            if "at least" in lowered and "per day" in lowered:
                constraints.min_trades_per_day = max(constraints.min_trades_per_day or 0, 1)
        for symbol, instruction in (display.sizing_adjustments or {}).items():
            lowered = instruction.lower()
            multiplier = multiplier_from_instruction(instruction)
            if multiplier is not None and multiplier < 1.0:
                constraints.symbol_risk_multipliers[symbol] = multiplier


async def _watch_judge_preferences(client: Client, current_preferences: dict, judge_agent: JudgeAgent) -> None:
    """Poll judge agent workflow for user preference updates."""
    wf_id = "judge-agent"
    while True:
        try:
            handle = client.get_workflow_handle(wf_id)
            prefs = await handle.query("get_user_preferences")
            
            # Check if preferences have changed
            if prefs != current_preferences:
                current_preferences.clear()
                current_preferences.update(prefs)
                judge_agent.user_preferences = dict(prefs)
                
                if prefs:
                    print(f"{GREEN}[JudgeAgent] ✅ User preferences updated: risk_tolerance={prefs.get('risk_tolerance', 'moderate')}, trading_style={prefs.get('trading_style', 'balanced')}, experience_level={prefs.get('experience_level', 'intermediate')}{RESET}")
                    
                    # IMMEDIATELY update execution agent system prompt to reflect new preferences
                    print(f"{ORANGE}[JudgeAgent] 🔄 Updating execution agent system prompt for new preferences...{RESET}")
                    try:
                        success = await judge_agent.update_prompt_for_user_preferences(prefs)
                        if success:
                            print(f"{GREEN}[JudgeAgent] ✅ Execution agent system prompt successfully updated{RESET}")
                        else:
                            print(f"{RED}[JudgeAgent] ❌ Failed to update execution agent system prompt{RESET}")
                    except Exception as update_exc:
                        print(f"{RED}[JudgeAgent] Error updating execution agent prompt: {update_exc}{RESET}")
                        
                else:
                    print(f"{CYAN}[JudgeAgent] No user preferences set - using defaults{RESET}")
                    
        except Exception as exc:
            # Silently continue if judge agent workflow not found or other issues
            pass
        
        await asyncio.sleep(3)  # Check every 3 seconds


async def run_judge_agent(server_url: str = "http://localhost:8080") -> None:
    """Run the judge agent with periodic evaluations."""
    mcp_url = server_url.rstrip("/") + "/mcp/"
    
    # Connect to Temporal
    temporal = await connect_temporal()
    
    # Connect to MCP server
    async with streamablehttp_client(mcp_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # Initialize judge agent
            judge = JudgeAgent(temporal, session)
            await judge.initialize()
            
            print(f"{GREEN}[JudgeAgent] Judge agent started{RESET}")
            
            # Start watching for user preference updates (similar to execution agent)
            current_preferences = {}
            _preferences_task = asyncio.create_task(_watch_judge_preferences(temporal, current_preferences, judge))
            
            print(f"{CYAN}[JudgeAgent] Waiting for trading activity before starting evaluations...{RESET}")
            
            # Main evaluation loop
            startup_delay = True
            while True:
                try:
                    # On startup, wait longer to let system stabilize
                    if startup_delay:
                        print(f"{CYAN}[JudgeAgent] Initial startup delay - waiting 10 minutes for system to stabilize{RESET}")
                        await asyncio.sleep(10 * 60)  # 10 minute initial delay
                        startup_delay = False
                    
                    # Check if evaluation is needed
                    if await judge.should_evaluate():
                        await judge.run_evaluation_cycle()
                    else:
                        # Log why we're not evaluating (but less frequently)
                        print(f"{CYAN}[JudgeAgent] Evaluation not needed - checking again in 5 minutes{RESET}")
                    
                    # Sleep for 5 minutes before checking again
                    await asyncio.sleep(5 * 60)
                    
                except KeyboardInterrupt:
                    print(f"{ORANGE}[JudgeAgent] Shutting down...{RESET}")
                    break
                except Exception as exc:
                    logger.error("Judge agent error: %s", exc)
                    # Sleep longer on error to avoid spam
                    await asyncio.sleep(5 * 60)


if __name__ == "__main__":
    asyncio.run(run_judge_agent(os.environ.get("MCP_SERVER", "http://localhost:8080")))
