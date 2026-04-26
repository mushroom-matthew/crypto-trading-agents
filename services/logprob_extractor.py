"""R97: Token-level log-probability extractor for StrategyPlan fields.

Extracts per-field mean log-probability from the OpenAI Responses API logprob
payload for the three highest-stakes plan fields: regime, stop_loss_pct,
target_pct.

A mean logprob < -1.5 per field (≈ 22% average probability per token) signals
that the model was uncertain about that value, complementing the scratchpad
confidence_map (R88/R93) with a hardware-level uncertainty signal.

Logprob format from OpenAI Responses API:
  output[0].content[0].logprobs  →  list of {token, logprob, bytes, top_logprobs}

Character-offset alignment: we locate each field's JSON value in the serialised
plan JSON string, then identify which tokens overlap that span.  Multi-token
numeric values (e.g. "2.3" → ["2", ".", "3"]) are averaged across sub-tokens.

Failures are always non-fatal — returns empty dict on any error.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Fields to extract logprobs for (highest-stakes plan outputs)
_TARGET_FIELDS = ("regime", "stop_loss_pct", "target_pct")


def _find_field_span(plan_json_str: str, field_name: str) -> tuple[int, int] | None:
    """Return (start, end) character offsets of the JSON value for field_name.

    Handles string and numeric values.  Returns None when field is absent.
    """
    # Match  "field_name": <value>  where value is a string or number
    pattern = re.compile(
        r'"' + re.escape(field_name) + r'"\s*:\s*("(?:[^"\\]|\\.)*"|[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|null|true|false)',
        re.DOTALL,
    )
    m = pattern.search(plan_json_str)
    if not m:
        return None
    # Return span of the value group (group 1)
    return m.start(1), m.end(1)


def _build_char_offset_map(token_logprobs: list[Any]) -> list[tuple[int, int, float]]:
    """Build list of (char_start, char_end, logprob) for each token.

    Tokens are concatenated in order to reconstruct the full string.
    'bytes' field (list of ints) gives the exact UTF-8 byte sequence.
    """
    result: list[tuple[int, int, float]] = []
    pos = 0
    for tok in token_logprobs:
        token_str: str = tok.get("token", "") if isinstance(tok, dict) else getattr(tok, "token", "")
        token_lp: float = tok.get("logprob", 0.0) if isinstance(tok, dict) else float(getattr(tok, "logprob", 0.0))
        token_len = len(token_str)
        result.append((pos, pos + token_len, token_lp))
        pos += token_len
    return result


def extract_field_logprobs(
    response_logprobs: Any,
    plan_json_str: str,
    fields: tuple[str, ...] = _TARGET_FIELDS,
) -> dict[str, float]:
    """Extract mean log-probability per field from Responses API logprob data.

    Args:
        response_logprobs: The logprobs object from the API response.
            Expected shape: list of {token, logprob, bytes, top_logprobs} dicts
            OR an object with a .content attribute containing such a list.
        plan_json_str: The raw JSON string of the generated plan.
        fields: Field names to extract logprobs for.

    Returns:
        Dict mapping field name → mean logprob (negative float).
        Empty dict if extraction fails or logprobs are unavailable.
    """
    if response_logprobs is None:
        return {}

    try:
        # Normalise to a flat list of token dicts
        token_list: list[Any] = []
        if isinstance(response_logprobs, list):
            token_list = response_logprobs
        elif hasattr(response_logprobs, "content"):
            token_list = list(response_logprobs.content or [])
        elif hasattr(response_logprobs, "__iter__"):
            token_list = list(response_logprobs)

        if not token_list:
            return {}

        # Reconstruct the full decoded string from token text to align with plan_json_str
        reconstructed = "".join(
            (t.get("token", "") if isinstance(t, dict) else getattr(t, "token", ""))
            for t in token_list
        )

        # Find the JSON block within the reconstructed string — the response often has
        # preamble text before the JSON.  Locate the first '{' that matches plan_json_str's start.
        json_offset = reconstructed.find("{")
        if json_offset < 0:
            logger.debug("R97: could not find '{' in reconstructed token stream")
            return {}

        # Align: plan_json_str[0] == reconstructed[json_offset]
        # Build offset map relative to the reconstructed string
        char_map = _build_char_offset_map(token_list)  # (abs_start, abs_end, logprob)

        out: dict[str, float] = {}
        for field in fields:
            span = _find_field_span(plan_json_str, field)
            if span is None:
                continue
            val_start_in_json, val_end_in_json = span
            # Convert to absolute offsets in reconstructed string
            abs_start = json_offset + val_start_in_json
            abs_end = json_offset + val_end_in_json

            # Collect tokens that overlap [abs_start, abs_end)
            overlapping_lps: list[float] = []
            for tok_start, tok_end, tok_lp in char_map:
                if tok_end <= abs_start:
                    continue
                if tok_start >= abs_end:
                    break
                overlapping_lps.append(tok_lp)

            if overlapping_lps:
                out[field] = sum(overlapping_lps) / len(overlapping_lps)

        return out

    except Exception as exc:
        logger.debug("R97 extract_field_logprobs failed (non-fatal): %s", exc)
        return {}
