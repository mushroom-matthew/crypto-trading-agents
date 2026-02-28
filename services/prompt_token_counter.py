"""Shared tiktoken-based token counter for prompt budget telemetry."""
from __future__ import annotations

import logging
from functools import lru_cache


@lru_cache(maxsize=4)
def _get_encoding(encoding_name: str):
    import tiktoken
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding: str = "o200k_base") -> int:
    """Return token count for `text` using the specified tiktoken encoding.

    Falls back to len(text)//4 if tiktoken is unavailable.
    """
    if not text:
        return 0
    try:
        enc = _get_encoding(encoding)
        return len(enc.encode(text))
    except Exception:
        logging.debug("tiktoken unavailable; using char/4 estimate", exc_info=True)
        return max(1, len(text) // 4)
