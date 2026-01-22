"""Langfuse observability helpers for OpenAI-powered agents."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from contextlib import contextmanager
from typing import Any, Dict, Optional

from langfuse import Langfuse
from langfuse.openai import openai as langfuse_openai

logger = logging.getLogger(__name__)

# Re-export the instrumented OpenAI namespace for type hints and direct access.
openai = langfuse_openai


@lru_cache(maxsize=1)
def init_langfuse() -> Optional[Langfuse]:
    """Initialise Langfuse tracing if credentials are present."""
    secret = os.getenv("LANGFUSE_SECRET_KEY")
    public = os.getenv("LANGFUSE_PUBLIC_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

    if not secret or not public:
        logger.debug("Langfuse credentials missing; skipping instrumentation.")
        return None

    logger.info("Langfuse instrumentation enabled for OpenAI clients.")
    return Langfuse(secret_key=secret, public_key=public, host=host)


def create_openai_client() -> Optional[langfuse_openai.OpenAI]:
    """Return an instrumented OpenAI client configured for Langfuse."""
    init_langfuse()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; unable to create OpenAI client.")
        return None
    return langfuse_openai.OpenAI(api_key=api_key)


@contextmanager
def langfuse_span(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager that yields a Langfuse span if the client is initialised."""
    client = init_langfuse()
    if not client:
        yield None
        return
    span_factory = getattr(client, "span", None)
    if span_factory is None:
        logger.warning("Langfuse client missing span() API; skipping span creation.")
        yield None
        return
    span = span_factory(name=name, metadata=metadata or {})
    try:
        yield span
    finally:
        span.end()
