"""Tool-loop helper for the LLM strategist.

This module implements a two-pass strategist flow:
1. LLM emits a structured request for read-only tool calls
2. System executes those tools, injects results into context
3. LLM returns a StrategyPlan

This enables adaptation to shifting market conditions while preserving safety.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Maximum tool calls per plan
MAX_TOOL_CALLS_PER_PLAN = 2

# Maximum time for all tool calls (ms)
MAX_TOTAL_TOOL_TIME_MS = 5000

# Timeout per tool call (ms)
TOOL_TIMEOUT_MS = 2000


# ============================================================================
# Tool Registry
# ============================================================================

class ToolCallRequest(BaseModel):
    """A single tool call request from the LLM."""
    tool_name: str = Field(..., description="Name of the tool to call")
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    reason: str = Field(default="", description="Why the tool is being called")


class ToolCallRequestList(BaseModel):
    """List of tool calls requested by the LLM."""
    tool_calls: List[ToolCallRequest] = Field(default_factory=list)


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    tool_name: str
    params: Dict[str, Any]
    duration_ms: int
    status: Literal["success", "error", "timeout"]
    result: Any = None
    error: str | None = None


@dataclass
class ToolRegistry:
    """Registry of available read-only tools."""
    tools: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    tool_descriptions: Dict[str, str] = field(default_factory=dict)

    def register(self, name: str, func: Callable[..., Any], description: str = "") -> None:
        """Register a read-only tool."""
        self.tools[name] = func
        self.tool_descriptions[name] = description

    def is_allowed(self, name: str) -> bool:
        """Check if a tool is in the allowlist."""
        return name in self.tools

    def execute(self, name: str, params: Dict[str, Any], timeout_ms: int = TOOL_TIMEOUT_MS) -> ToolResult:
        """Execute a tool with timeout."""
        start = time.monotonic()

        if not self.is_allowed(name):
            return ToolResult(
                tool_name=name,
                params=params,
                duration_ms=0,
                status="error",
                error=f"Tool '{name}' is not in the allowlist",
            )

        try:
            func = self.tools[name]
            # Simple synchronous execution with timeout approximation
            result = func(**params)
            duration_ms = int((time.monotonic() - start) * 1000)

            return ToolResult(
                tool_name=name,
                params=params,
                duration_ms=duration_ms,
                status="success",
                result=result,
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.warning("Tool execution failed: %s - %s", name, e)
            return ToolResult(
                tool_name=name,
                params=params,
                duration_ms=duration_ms,
                status="error",
                error=str(e),
            )

    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for the prompt."""
        if not self.tool_descriptions:
            return ""

        lines = ["Available read-only tools (optional, max 2 calls):"]
        for name, desc in sorted(self.tool_descriptions.items()):
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)


# ============================================================================
# Default Read-Only Tools
# ============================================================================

def _get_current_time() -> Dict[str, Any]:
    """Get current UTC time."""
    now = datetime.now(timezone.utc)
    return {
        "utc_iso": now.isoformat(),
        "unix_timestamp": now.timestamp(),
        "hour": now.hour,
        "day_of_week": now.strftime("%A"),
    }


def _get_market_hours_info(symbol: str = "BTC-USD") -> Dict[str, Any]:
    """Get market hours information (crypto is 24/7 but useful for session context)."""
    now = datetime.now(timezone.utc)
    hour = now.hour

    # Define trading sessions
    if 0 <= hour < 8:
        session = "asia"
        session_name = "Asian Session"
    elif 8 <= hour < 13:
        session = "europe"
        session_name = "European Session"
    elif 13 <= hour < 21:
        session = "us"
        session_name = "US Session"
    else:
        session = "late_us"
        session_name = "Late US / Pre-Asia"

    return {
        "symbol": symbol,
        "is_open": True,  # Crypto is always open
        "current_session": session,
        "session_name": session_name,
        "utc_hour": hour,
    }


def _get_volatility_regime(
    atr_14: float | None = None,
    realized_vol_short: float | None = None,
    realized_vol_medium: float | None = None,
) -> Dict[str, Any]:
    """Classify current volatility regime based on indicators."""
    if atr_14 is None and realized_vol_short is None:
        return {"regime": "unknown", "confidence": 0.0}

    # Simple classification
    if realized_vol_short is not None and realized_vol_medium is not None:
        vol_ratio = realized_vol_short / realized_vol_medium if realized_vol_medium > 0 else 1.0
        if vol_ratio > 1.5:
            regime = "expanding"
        elif vol_ratio < 0.7:
            regime = "contracting"
        else:
            regime = "stable"
    else:
        regime = "unknown"

    return {
        "regime": regime,
        "atr_14": atr_14,
        "realized_vol_short": realized_vol_short,
        "realized_vol_medium": realized_vol_medium,
        "confidence": 0.7 if regime != "unknown" else 0.3,
    }


# ============================================================================
# Default Tool Registry
# ============================================================================

_default_registry: ToolRegistry | None = None


def get_default_tool_registry() -> ToolRegistry:
    """Get the default tool registry with read-only tools."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()

        # Register read-only tools
        _default_registry.register(
            "get_current_time",
            _get_current_time,
            "Get current UTC time and session info",
        )
        _default_registry.register(
            "get_market_hours_info",
            _get_market_hours_info,
            "Get trading session context (params: symbol)",
        )
        _default_registry.register(
            "get_volatility_regime",
            _get_volatility_regime,
            "Classify volatility regime (params: atr_14, realized_vol_short, realized_vol_medium)",
        )

    return _default_registry


# ============================================================================
# Tool Loop Execution
# ============================================================================

TOOL_REQUEST_PROMPT_SUFFIX = """
Before generating the StrategyPlan, you may optionally request up to 2 read-only tool calls
to gather additional context. If you need additional information, respond with a JSON object:

{"tool_calls": [{"tool_name": "...", "params": {...}, "reason": "..."}]}

If you do not need any tools, proceed directly with the StrategyPlan JSON.
"""


def parse_tool_request(content: str) -> ToolCallRequestList | None:
    """Parse tool call requests from LLM response.

    Returns None if the response is not a tool request.
    """
    content = content.strip()

    # Try to extract JSON
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 3:
            content = parts[1] if parts[1].strip() else parts[2]
    if content.lower().startswith("json"):
        content = content[4:].strip()

    # Look for tool_calls key
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "tool_calls" in data:
            return ToolCallRequestList.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        pass

    return None


def execute_tool_requests(
    requests: ToolCallRequestList,
    registry: ToolRegistry | None = None,
) -> List[ToolResult]:
    """Execute tool requests and return results.

    Enforces:
    - Maximum number of tool calls
    - Maximum total time
    - Tool allowlist
    """
    registry = registry or get_default_tool_registry()
    results: List[ToolResult] = []

    # Enforce max tool calls
    tool_calls = requests.tool_calls[:MAX_TOOL_CALLS_PER_PLAN]

    start_total = time.monotonic()

    for request in tool_calls:
        # Check total time budget
        elapsed_ms = int((time.monotonic() - start_total) * 1000)
        if elapsed_ms >= MAX_TOTAL_TOOL_TIME_MS:
            results.append(ToolResult(
                tool_name=request.tool_name,
                params=request.params,
                duration_ms=0,
                status="timeout",
                error="Total tool time budget exceeded",
            ))
            continue

        remaining_ms = MAX_TOTAL_TOOL_TIME_MS - elapsed_ms
        timeout = min(TOOL_TIMEOUT_MS, remaining_ms)

        result = registry.execute(request.tool_name, request.params, timeout_ms=timeout)
        results.append(result)

    return results


def format_tool_results_for_context(results: List[ToolResult]) -> Dict[str, Any]:
    """Format tool results for injection into global_context."""
    return {
        "tool_results": [
            {
                "tool_name": r.tool_name,
                "params": r.params,
                "duration_ms": r.duration_ms,
                "status": r.status,
                "result": r.result if r.status == "success" else None,
                "error": r.error,
            }
            for r in results
        ]
    }


def inject_tool_results_into_input(
    llm_input_json: str,
    results: List[ToolResult],
) -> str:
    """Inject tool results into the LLM input JSON."""
    try:
        data = json.loads(llm_input_json)
        global_context = data.get("global_context") or {}
        global_context["tool_results"] = format_tool_results_for_context(results)["tool_results"]
        data["global_context"] = global_context
        return json.dumps(data)
    except Exception as e:
        logger.warning("Failed to inject tool results: %s", e)
        return llm_input_json
