"""Model capability helpers for OpenAI calls."""

from __future__ import annotations

from typing import Any, Dict


def is_reasoning_model(model: str | None) -> bool:
    """Return True for models that use reasoning tokens (GPT-5 family)."""
    if not model:
        return False
    return model.lower().startswith("gpt-5")


def supports_temperature(model: str | None) -> bool:
    """Return True when the model accepts a temperature parameter."""
    if not model:
        return True
    # GPT-5 family uses reasoning tokens instead of temperature.
    return not is_reasoning_model(model)


def temperature_args(model: str | None, temperature: float) -> Dict[str, float]:
    """Return a temperature kwarg dict when supported, else empty."""
    if supports_temperature(model):
        return {"temperature": temperature}
    return {}


def reasoning_args(model: str | None, effort: str = "low") -> Dict[str, Any]:
    """Return reasoning configuration for the Responses API, else empty.

    For structured-output tasks (JSON generation) use effort="low" or
    "minimal" to avoid burning the output-token budget on chain-of-thought.
    """
    if not is_reasoning_model(model):
        return {}
    return {"reasoning": {"effort": effort}}


def reasoning_effort_args(model: str | None, effort: str = "low") -> Dict[str, str]:
    """Return reasoning_effort kwarg for Chat Completions API, else empty.

    Chat Completions uses a top-level ``reasoning_effort`` string param,
    unlike the Responses API which nests it under ``reasoning.effort``.
    """
    if not is_reasoning_model(model):
        return {}
    return {"reasoning_effort": effort}


def completion_token_args(
    model: str | None,
    desired_output_tokens: int,
    reasoning_headroom: int = 4096,
) -> Dict[str, int]:
    """Return max_completion_tokens for Chat Completions API.

    For reasoning models, adds headroom for reasoning tokens on top of
    the desired visible output tokens.  For non-reasoning models, returns
    the desired output tokens directly.
    """
    if is_reasoning_model(model):
        return {"max_completion_tokens": desired_output_tokens + reasoning_headroom}
    return {"max_completion_tokens": desired_output_tokens}


__all__ = [
    "is_reasoning_model",
    "supports_temperature",
    "temperature_args",
    "reasoning_args",
    "reasoning_effort_args",
    "completion_token_args",
]
