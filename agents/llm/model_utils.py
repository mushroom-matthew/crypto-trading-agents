"""Model capability helpers for OpenAI calls."""

from __future__ import annotations

from typing import Dict


def supports_temperature(model: str | None) -> bool:
    """Return True when the model accepts a temperature parameter."""
    if not model:
        return True
    model_name = model.lower()
    # GPT-5 family rejects temperature unless explicitly using 5.1 with reasoning_effort=none.
    # We omit temperature for all GPT-5 variants to avoid request failures.
    if model_name.startswith("gpt-5"):
        return False
    return True


def temperature_args(model: str | None, temperature: float) -> Dict[str, float]:
    """Return a temperature kwarg dict when supported, else empty."""
    if supports_temperature(model):
        return {"temperature": temperature}
    return {}


__all__ = ["supports_temperature", "temperature_args"]
