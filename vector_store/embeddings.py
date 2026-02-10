"""Embedding helpers for local vector store retrieval."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Iterable, List

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DIM = 128
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _hash_tokens(tokens: Iterable[str], dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=float)
    for token in tokens:
        token = token.strip().lower()
        if not token:
            continue
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % dim
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def _local_embedding(text: str, dim: int) -> List[float]:
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return [0.0] * dim
    return _hash_tokens(tokens, dim).tolist()


def _openai_embedding(text: str) -> List[float] | None:
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding
    except Exception as exc:
        logger.warning("Vector store OpenAI embedding failed: %s", exc)
        return None


def get_embedding(text: str) -> List[float]:
    """Return embedding vector for text.

    Uses local hashed embeddings by default. Set VECTOR_STORE_EMBEDDINGS to "openai"
    or "auto" to attempt OpenAI embeddings (falls back to local on failure).
    """
    backend = os.environ.get("VECTOR_STORE_EMBEDDINGS", "local").strip().lower()
    dim = int(os.environ.get("VECTOR_STORE_EMBEDDING_DIM", _DEFAULT_DIM))

    if backend in {"openai", "auto"}:
        embedding = _openai_embedding(text)
        if embedding is not None:
            return embedding
        if backend == "openai":
            # Explicit OpenAI mode but failed, fall back to local to keep flow alive.
            logger.warning("Falling back to local embeddings after OpenAI failure.")

    return _local_embedding(text, dim)


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    a = np.array(list(vec_a), dtype=float)
    b = np.array(list(vec_b), dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.shape != b.shape:
        # Normalize to min length if embedding dims differ (e.g., OpenAI vs local).
        size = min(a.shape[0], b.shape[0])
        a = a[:size]
        b = b[:size]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


__all__ = ["get_embedding", "cosine_similarity"]
