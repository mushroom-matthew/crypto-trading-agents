"""Utility helpers shared across agents."""

from __future__ import annotations

from pprint import pformat
from typing import Any, List, Dict
import logging
import json
from mcp.types import CallToolResult, TextContent

logger = logging.getLogger(__name__)


def print_banner(name: str, purpose: str) -> None:
    """Print a simple ASCII banner with ``name`` and ``purpose``."""
    lines = [name, purpose]
    width = max(len(line) for line in lines) + 4
    border = "*" * width
    print(border)
    for line in lines:
        print(f"* {line.ljust(width - 4)} *")
    print(border)


def format_log(data: Any) -> str:
    """Return a pretty string representation of ``data`` for logging."""
    if isinstance(data, str):
        return data
    return pformat(data, width=60)


def tool_result_data(result: Any) -> Any:
    """Return JSON-friendly data from a tool call result.
    
    Parameters
    ----------
    result:
        The result from an MCP tool call
        
    Returns
    -------
    Any
        Parsed JSON data or the original result
    """
    if isinstance(result, CallToolResult):
        if result.content:
            parsed: list[Any] = []
            for item in result.content:
                if isinstance(item, TextContent):
                    try:
                        parsed.append(json.loads(item.text))
                    except Exception:
                        parsed.append(item.text)
                else:
                    parsed.append(item.model_dump() if hasattr(item, "model_dump") else item)
            return parsed if len(parsed) > 1 else parsed[0]
        return []
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result





def stream_chat_completion(client, *, prefix: str = "", color: str = "", reset: str = "", **kwargs) -> dict:
    """Stream Responses API tokens to stdout and return the final message.

    Parameters
    ----------
    client:
        The OpenAI client to use.
    prefix:
        Optional text printed once before the first token.
    color:
        ANSI escape code used for coloring the streamed text.
    reset:
        ANSI escape code used to reset the terminal color when streaming ends.
    kwargs:
        Additional parameters forwarded to ``client.responses.create``.
        Accepts ``messages`` (remapped to ``input``) and Chat Completions
        tool format (automatically unwrapped for the Responses API).
    """
    from agents.llm.model_utils import reasoning_args, output_token_args, temperature_args

    # Remap Chat Completions-style 'messages' to Responses API 'input'
    messages = kwargs.pop("messages", None)
    if messages is not None:
        kwargs["input"] = messages

    model = kwargs.get("model")

    # Transform tools from Chat Completions format to Responses API format
    tools = kwargs.pop("tools", None)
    if tools:
        responses_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                responses_tools.append({
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
            else:
                responses_tools.append(tool)
        kwargs["tools"] = responses_tools

    # Add model-specific parameters if not already specified
    if "reasoning" not in kwargs:
        kwargs.update(reasoning_args(model))
    if "max_output_tokens" not in kwargs:
        kwargs.update(output_token_args(model, 4096))
    if "temperature" not in kwargs:
        kwargs.update(temperature_args(model, 0.7))

    stream = client.responses.create(stream=True, **kwargs)
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    current_tool_idx = -1
    first_token = True

    for event in stream:
        if event.type == "response.output_text.delta":
            token = event.delta
            if token:
                if first_token:
                    if prefix or color:
                        print(f"{color}{prefix}", end="", flush=True)
                    first_token = False
                print(token, end="", flush=True)
                content_parts.append(token)
        elif event.type == "response.output_item.added":
            item = getattr(event, "item", None)
            if item and getattr(item, "type", None) == "function_call":
                current_tool_idx += 1
                tool_calls[current_tool_idx] = {
                    "id": getattr(item, "call_id", f"call_{current_tool_idx}"),
                    "type": "function",
                    "function": {
                        "name": getattr(item, "name", ""),
                        "arguments": "",
                    },
                }
        elif event.type == "response.function_call_arguments.delta":
            if current_tool_idx >= 0 and current_tool_idx in tool_calls:
                tool_calls[current_tool_idx]["function"]["arguments"] += event.delta

    if not first_token and reset:
        print(reset)
    else:
        print()
    message: dict = {"role": "assistant", "content": "".join(content_parts)}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return message


async def check_and_process_feedback(
    client,
    workflow_id: str,
    conversation: List[Dict[str, Any]] = None,
    agent_name: str = "Agent",
    color_start: str = "",
    color_end: str = ""
) -> List[str]:
    """Check for pending user feedback and process it.
    
    Parameters
    ----------
    client:
        Temporal client instance
    workflow_id:
        ID of the workflow to query for feedback
    conversation:
        Optional conversation list to append feedback to (for execution agent)
    agent_name:
        Name of the agent for logging purposes
    color_start:
        ANSI color code for output formatting
    color_end:
        ANSI color code to reset formatting
        
    Returns
    -------
    List[str]
        List of feedback messages that were processed
    """
    feedback_messages = []
    try:
        handle = client.get_workflow_handle(workflow_id)
        pending_feedback = await handle.query("get_pending_feedback")
        
        if pending_feedback:
            print(f"{color_start}[{agent_name}] 📨 Processing {len(pending_feedback)} user feedback message(s){color_end}")
            
            for feedback in pending_feedback:
                feedback_messages.append(feedback['message'])
                
                # Add to conversation if provided (for execution agent)
                if conversation is not None:
                    feedback_message = {
                        "role": "user",
                        "content": f"[USER FEEDBACK]: {feedback['message']}"
                    }
                    conversation.append(feedback_message)
                
                # Mark feedback as processed
                await handle.signal("mark_feedback_processed", feedback["feedback_id"])
                
                print(f"{color_start}[{agent_name}] ✅ Processed feedback: {feedback['message'][:100]}...{color_end}")
                
    except Exception as exc:
        logger.debug(f"Failed to check feedback for {agent_name}: {exc}")
        # Silently continue if there's an error
        pass
    
    return feedback_messages


__all__ = ["print_banner", "format_log", "stream_chat_completion", "check_and_process_feedback", "tool_result_data"]

