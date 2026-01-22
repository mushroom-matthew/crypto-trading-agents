"""Execution agent workflow for receiving nudge signals."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple

from agents.execution_config import ExecutionGatingConfig

from temporalio import workflow


logger = logging.getLogger(__name__)


@dataclass
class SymbolDecisionState:
    """State tracked per symbol for execution decisions."""

    last_eval_price: Optional[float] = None
    last_eval_time: Optional[float] = None  # epoch seconds
    calls_in_current_window: int = 0
    current_window_start: Optional[float] = None
    last_position_side: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionAgentState:
    """Aggregated execution state keyed by trading symbol."""

    symbols: Dict[str, SymbolDecisionState] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {symbol: state.to_dict() for symbol, state in self.symbols.items()}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ExecutionAgentState":
        state = cls()
        if not data:
            return state
        for symbol, symbol_state in data.items():
            state.symbols[symbol] = SymbolDecisionState(
                last_eval_price=symbol_state.get("last_eval_price"),
                last_eval_time=symbol_state.get("last_eval_time"),
                calls_in_current_window=symbol_state.get("calls_in_current_window", 0),
                current_window_start=symbol_state.get("current_window_start"),
                last_position_side=symbol_state.get("last_position_side"),
            )
        return state

    def get_symbol_state(self, symbol: str) -> SymbolDecisionState:
        if symbol not in self.symbols:
            self.symbols[symbol] = SymbolDecisionState()
        return self.symbols[symbol]


@workflow.defn
class ExecutionAgentWorkflow:
    """Workflow that receives nudge signals for the execution agent."""

    def __init__(self) -> None:
        self.nudges: list[int] = []
        self.user_preferences: Dict = {}
        self.log_entries: List[Dict[str, Any]] = []
        self.decision_count = 0
        self.action_count = 0
        self.summary_count = 0
        self.system_prompt: str = ""  # Store the current system prompt
        self.user_feedback: List[Dict[str, Any]] = []  # Store user feedback messages
        self.execution_state = ExecutionAgentState()

    @workflow.signal
    def nudge(self, ts: int) -> None:
        self.nudges.append(ts)

    @workflow.signal  
    def set_user_preferences(self, preferences: Dict) -> None:
        """Update user trading preferences."""
        self.user_preferences.update(preferences)

    @workflow.query
    def get_nudges(self) -> list[int]:
        return list(self.nudges)
        
    @workflow.query
    def get_user_preferences(self) -> Dict:
        """Get current user preferences."""
        return dict(self.user_preferences)
    
    @workflow.signal
    def update_system_prompt(self, prompt: str) -> None:
        """Update the system prompt for the execution agent."""
        self.system_prompt = prompt
        workflow.logger.info(f"System prompt updated (length: {len(prompt)} chars)")
    
    @workflow.query
    def get_system_prompt(self) -> str:
        """Get the current system prompt."""
        return self.system_prompt

    @workflow.query
    def get_execution_state(self) -> Dict[str, Any]:
        """Return serialized execution gating state."""
        return self.execution_state.to_dict()

    @workflow.signal
    def update_execution_state(self, state_update: Dict[str, Dict[str, Any]]) -> None:
        """Merge execution state updates keyed by symbol."""
        for symbol, symbol_state in state_update.items():
            current = self.execution_state.get_symbol_state(symbol)
            for field_name, value in symbol_state.items():
                if hasattr(current, field_name):
                    setattr(current, field_name, value)
    
    @workflow.signal
    def add_user_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """Add user feedback to be incorporated into the agent's conversation."""
        feedback_entry = {
            **self._get_timestamp(),
            "feedback_id": f"feedback_{len(self.user_feedback) + 1}",
            "message": feedback_data.get("message", ""),
            "source": feedback_data.get("source", "user"),
            "processed": False
        }
        self.user_feedback.append(feedback_entry)
        workflow.logger.info(f"User feedback received: {feedback_data.get('message', '')[:100]}...")
    
    @workflow.query
    def get_pending_feedback(self) -> List[Dict[str, Any]]:
        """Get unprocessed user feedback."""
        return [fb for fb in self.user_feedback if not fb.get("processed", False)]
    
    @workflow.signal
    def mark_feedback_processed(self, feedback_id: str) -> None:
        """Mark a feedback message as processed."""
        for feedback in self.user_feedback:
            if feedback.get("feedback_id") == feedback_id:
                feedback["processed"] = True
                break

    def _get_timestamp(self) -> Dict[str, Any]:
        """Get standardized timestamp information."""
        now = datetime.now(timezone.utc)
        return {
            "timestamp": int(now.timestamp()),
            "iso_timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        }

    @workflow.signal
    def log_decision(self, log_data: Dict[str, Any]) -> None:
        """Log a comprehensive trading decision with all context."""
        log_entry = {
            **self._get_timestamp(),
            "event_type": "decision",
            "entry_id": f"decision_{self.decision_count + 1}",
            **log_data
        }
        
        self.log_entries.append(log_entry)
        self.decision_count += 1

    @workflow.signal
    def log_action(self, log_data: Dict[str, Any]) -> None:
        """Log a specific action taken by the agent."""
        log_entry = {
            **self._get_timestamp(),
            "event_type": "action",
            "entry_id": f"action_{self.action_count + 1}",
            **log_data
        }
        
        self.log_entries.append(log_entry)
        self.action_count += 1

    @workflow.signal
    def log_summary(self, log_data: Dict[str, Any]) -> None:
        """Log summary information (evaluations, performance reports, etc.)."""
        log_entry = {
            **self._get_timestamp(),
            "event_type": "summary",
            "entry_id": f"summary_{self.summary_count + 1}",
            **log_data
        }
        
        self.log_entries.append(log_entry)
        self.summary_count += 1

    @workflow.query
    def get_logs(self, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get log entries with optional filtering."""
        if params is None:
            params = {}
        
        event_type_filter = params.get("event_type")
        since_ts = params.get("since_ts", 0)
        limit = params.get("limit", 1000)
        
        # Filter logs
        filtered_logs = []
        for entry in self.log_entries:
            # Filter by timestamp
            if entry.get("timestamp", 0) < since_ts:
                continue
                
            # Filter by event type
            if event_type_filter and entry.get("event_type") != event_type_filter:
                continue
                
            filtered_logs.append(entry)
        
        # Sort by timestamp (newest first) and limit
        filtered_logs.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return filtered_logs[:limit]

    @workflow.query
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "total_entries": len(self.log_entries),
            "decision_count": self.decision_count,
            "action_count": self.action_count,
            "summary_count": self.summary_count,
            "agent": "execution_agent"
        }

    @workflow.query
    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent decisions for analysis."""
        return self.get_logs({
            "event_type": "decision",
            "limit": limit
        })

    @workflow.query
    def get_recent_actions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent actions for analysis."""
        return self.get_logs({
            "event_type": "action", 
            "limit": limit
        })

    @workflow.run
    async def run(self) -> None:
        await workflow.wait_condition(lambda: False)


def _reset_rate_window(state: SymbolDecisionState, now: datetime) -> None:
    """Reset the hourly call window if necessary."""
    now_ts = now.timestamp()
    if state.current_window_start is None:
        state.current_window_start = now_ts
        state.calls_in_current_window = 0
        return

    elapsed = now_ts - state.current_window_start
    if elapsed >= 3600:
        state.current_window_start = now_ts
        state.calls_in_current_window = 0


def should_call_llm(
    price: float,
    now: datetime,
    state: SymbolDecisionState,
    cfg: ExecutionGatingConfig,
    logger_override: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """Determine whether the execution agent should invoke the LLM."""
    active_logger = logger_override or logger
    _reset_rate_window(state, now)

    if state.calls_in_current_window >= cfg.max_calls_per_hour_per_symbol:
        reason = "RATE_LIMIT"
        active_logger.info(
            "ExecutionAgent gating: SKIP_LLM (%s) price=%.6f calls=%s threshold=%s",
            reason,
            price,
            state.calls_in_current_window,
            cfg.max_calls_per_hour_per_symbol,
        )
        return False, reason

    if state.last_eval_price is None or state.last_eval_time is None:
        reason = "BOOTSTRAP"
        active_logger.info(
            "ExecutionAgent gating: CALL_LLM (%s) price=%.6f", reason, price
        )
        return True, reason

    elapsed = now.timestamp() - state.last_eval_time
    if elapsed >= cfg.max_staleness_seconds:
        reason = "STALE"
        active_logger.info(
            "ExecutionAgent gating: CALL_LLM (%s) price=%.6f elapsed=%.2fs threshold=%s",
            reason,
            price,
            elapsed,
            cfg.max_staleness_seconds,
        )
        return True, reason

    if state.last_eval_price == 0:
        reason = "NO_BASE_PRICE"
        active_logger.info(
            "ExecutionAgent gating: CALL_LLM (%s) price=%.6f last_price=%s",
            reason,
            price,
            state.last_eval_price,
        )
        return True, reason

    price_move = abs(price - state.last_eval_price)
    move_pct = abs(price_move / state.last_eval_price) * 100.0
    if move_pct < cfg.min_price_move_pct:
        reason = f"PRICE_DELTA<{cfg.min_price_move_pct}%"
        active_logger.info(
            "ExecutionAgent gating: SKIP_LLM (%s) price=%.6f last_price=%.6f move_pct=%.4f",
            reason,
            price,
            state.last_eval_price,
            move_pct,
        )
        return False, reason

    reason = "PRICE_MOVE"
    active_logger.info(
        "ExecutionAgent gating: CALL_LLM (%s) price=%.6f last_price=%.6f move_pct=%.4f",
        reason,
        price,
        state.last_eval_price,
        move_pct,
    )
    return True, reason
