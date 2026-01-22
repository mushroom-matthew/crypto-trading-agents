"""Langfuse-instrumented OpenAI client factory."""

from __future__ import annotations

from agents.langfuse_utils import create_openai_client


def get_llm_client():
    """Return a Langfuse-instrumented OpenAI client."""
    client = create_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY not set; cannot create LLM client")
    return client
