"""Vector store for trigger examples to improve LLM trigger generation.

This module provides a simple vector store that:
1. Stores successful trigger examples with their market context
2. Retrieves relevant examples based on similarity to current conditions
3. Injects context into the strategist prompt
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np

logger = logging.getLogger(__name__)

# Default storage location
DEFAULT_STORE_PATH = Path(".cache/trigger_examples")


@dataclass
class TriggerExample:
    """A stored trigger example with context and performance data."""

    trigger_id: str
    symbol: str
    direction: Literal["long", "short", "flat"]
    category: str
    entry_rule: str
    exit_rule: str
    confidence_grade: str

    # Market context when trigger was created
    regime: str  # bull, bear, range, high_vol, mixed
    trend_state: str  # uptrend, downtrend, sideways
    vol_state: str  # low, normal, high, extreme
    rsi_range: str  # oversold, neutral, overbought

    # Performance metrics (optional, filled after backtest)
    win_rate: float | None = None
    avg_return_pct: float | None = None
    times_triggered: int = 0
    times_profitable: int = 0

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = "manual"  # manual, backtest, live

    # Embedding (computed lazily)
    embedding: List[float] | None = None


@dataclass
class TriggerVectorStore:
    """Simple vector store for trigger examples using OpenAI embeddings."""

    store_path: Path = field(default_factory=lambda: DEFAULT_STORE_PATH)
    examples: List[TriggerExample] = field(default_factory=list)
    _embeddings_matrix: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.store_path = Path(self.store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._load_examples()

    def _load_examples(self) -> None:
        """Load examples from disk."""
        examples_file = self.store_path / "examples.json"
        if examples_file.exists():
            try:
                with open(examples_file) as f:
                    data = json.load(f)
                self.examples = [TriggerExample(**ex) for ex in data]
                self._rebuild_embeddings_matrix()
                logger.info("Loaded %d trigger examples from %s", len(self.examples), examples_file)
            except Exception as e:
                logger.warning("Failed to load trigger examples: %s", e)
                self.examples = []

    def _save_examples(self) -> None:
        """Save examples to disk."""
        examples_file = self.store_path / "examples.json"
        try:
            data = [asdict(ex) for ex in self.examples]
            with open(examples_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save trigger examples: %s", e)

    def _rebuild_embeddings_matrix(self) -> None:
        """Rebuild the numpy matrix of embeddings for fast similarity search."""
        embeddings = [ex.embedding for ex in self.examples if ex.embedding]
        if embeddings:
            self._embeddings_matrix = np.array(embeddings)
        else:
            self._embeddings_matrix = None

    def _get_embedding(self, text: str) -> List[float] | None:
        """Get embedding for text using OpenAI."""
        try:
            from openai import OpenAI

            client = OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning("Failed to get embedding: %s", e)
            return None

    def _example_to_text(self, example: TriggerExample) -> str:
        """Convert example to text for embedding."""
        return (
            f"Symbol: {example.symbol}, Direction: {example.direction}, "
            f"Category: {example.category}, Regime: {example.regime}, "
            f"Trend: {example.trend_state}, Volatility: {example.vol_state}, "
            f"RSI: {example.rsi_range}, Entry: {example.entry_rule}"
        )

    def _context_to_text(self, context: Dict[str, Any]) -> str:
        """Convert market context to text for embedding."""
        return (
            f"Symbol: {context.get('symbol', 'unknown')}, "
            f"Regime: {context.get('regime', 'unknown')}, "
            f"Trend: {context.get('trend_state', 'unknown')}, "
            f"Volatility: {context.get('vol_state', 'unknown')}, "
            f"RSI: {context.get('rsi_range', 'neutral')}"
        )

    def add_example(self, example: TriggerExample, compute_embedding: bool = True) -> None:
        """Add a trigger example to the store."""
        if compute_embedding and example.embedding is None:
            text = self._example_to_text(example)
            example.embedding = self._get_embedding(text)

        self.examples.append(example)
        self._rebuild_embeddings_matrix()
        self._save_examples()
        logger.info("Added trigger example: %s", example.trigger_id)

    def add_examples_from_backtest(
        self,
        triggers: List[Dict[str, Any]],
        regime: str,
        performance: Dict[str, Dict[str, Any]] | None = None,
    ) -> int:
        """Add multiple trigger examples from a backtest run.

        Args:
            triggers: List of trigger dicts from StrategyPlan
            regime: Market regime during the backtest
            performance: Optional performance data keyed by trigger_id

        Returns:
            Number of examples added
        """
        added = 0
        for trig in triggers:
            perf = (performance or {}).get(trig.get("id", ""), {})
            example = TriggerExample(
                trigger_id=trig.get("id", f"trigger_{added}"),
                symbol=trig.get("symbol", "unknown"),
                direction=trig.get("direction", "long"),
                category=trig.get("category", "other"),
                entry_rule=trig.get("entry_rule", ""),
                exit_rule=trig.get("exit_rule", ""),
                confidence_grade=trig.get("confidence_grade", "C"),
                regime=regime,
                trend_state=trig.get("trend_state", "unknown"),
                vol_state=trig.get("vol_state", "normal"),
                rsi_range=self._classify_rsi(trig.get("rsi_14")),
                win_rate=perf.get("win_rate"),
                avg_return_pct=perf.get("avg_return_pct"),
                times_triggered=perf.get("times_triggered", 0),
                times_profitable=perf.get("times_profitable", 0),
                source="backtest",
            )
            self.add_example(example, compute_embedding=True)
            added += 1

        return added

    def _classify_rsi(self, rsi: float | None) -> str:
        """Classify RSI into a category."""
        if rsi is None:
            return "neutral"
        if rsi < 30:
            return "oversold"
        if rsi > 70:
            return "overbought"
        return "neutral"

    def search(
        self,
        context: Dict[str, Any],
        top_k: int = 5,
        min_similarity: float = 0.5,
        filter_category: str | None = None,
        filter_direction: str | None = None,
    ) -> List[tuple[TriggerExample, float]]:
        """Search for similar trigger examples.

        Args:
            context: Current market context dict
            top_k: Maximum number of results
            min_similarity: Minimum cosine similarity threshold
            filter_category: Optional category filter
            filter_direction: Optional direction filter

        Returns:
            List of (example, similarity_score) tuples
        """
        if not self.examples or self._embeddings_matrix is None:
            return []

        # Get embedding for query
        query_text = self._context_to_text(context)
        query_embedding = self._get_embedding(query_text)
        if query_embedding is None:
            return []

        # Compute cosine similarities
        query_vec = np.array(query_embedding)
        similarities = np.dot(self._embeddings_matrix, query_vec) / (
            np.linalg.norm(self._embeddings_matrix, axis=1) * np.linalg.norm(query_vec)
        )

        # Filter and sort
        results = []
        for i, (example, sim) in enumerate(zip(self.examples, similarities)):
            if sim < min_similarity:
                continue
            if filter_category and example.category != filter_category:
                continue
            if filter_direction and example.direction != filter_direction:
                continue
            results.append((example, float(sim)))

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_context_injection(
        self,
        context: Dict[str, Any],
        top_k: int = 3,
    ) -> str:
        """Get a formatted string of relevant examples to inject into the prompt.

        Args:
            context: Current market context
            top_k: Number of examples to include

        Returns:
            Formatted string for prompt injection
        """
        results = self.search(context, top_k=top_k)
        if not results:
            return ""

        lines = ["Relevant trigger examples from similar market conditions:"]
        for i, (example, score) in enumerate(results, 1):
            perf_note = ""
            if example.win_rate is not None and example.times_triggered > 0:
                perf_note = f" (win_rate: {example.win_rate:.0%}, n={example.times_triggered})"

            lines.append(
                f"{i}. [{example.category}] {example.symbol} {example.direction}{perf_note}\n"
                f"   entry_rule = \"{example.entry_rule}\"\n"
                f"   exit_rule = \"{example.exit_rule}\""
            )

        return "\n".join(lines)

    def get_category_examples(self, category: str, limit: int = 2) -> List[TriggerExample]:
        """Get examples for a specific category (for diversity)."""
        return [ex for ex in self.examples if ex.category == category][:limit]

    def clear(self) -> None:
        """Clear all examples."""
        self.examples = []
        self._embeddings_matrix = None
        self._save_examples()


# Singleton instance
_store_instance: TriggerVectorStore | None = None


def get_trigger_vector_store(store_path: Path | None = None) -> TriggerVectorStore:
    """Get or create the trigger vector store singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = TriggerVectorStore(store_path=store_path or DEFAULT_STORE_PATH)
    return _store_instance


def seed_default_examples() -> int:
    """Seed the vector store with default trigger examples.

    Returns the number of examples added.
    """
    store = get_trigger_vector_store()

    # Skip if already seeded
    if len(store.examples) > 0:
        return 0

    default_examples = [
        # Trend continuation - BTC uptrend
        TriggerExample(
            trigger_id="seed_btc_trend_cont_1",
            symbol="BTC-USD",
            direction="long",
            category="trend_continuation",
            entry_rule="rsi_14 < 45 and close > sma_medium and macd_hist > 0",
            exit_rule="rsi_14 > 70 or close < sma_medium",
            confidence_grade="B",
            regime="bull",
            trend_state="uptrend",
            vol_state="normal",
            rsi_range="neutral",
            source="seed",
        ),
        # Mean reversion - oversold bounce
        TriggerExample(
            trigger_id="seed_btc_mean_rev_1",
            symbol="BTC-USD",
            direction="long",
            category="mean_reversion",
            entry_rule="rsi_14 < 35 and close < bollinger_lower and close > sma_long",
            exit_rule="close > sma_medium or rsi_14 > 55",
            confidence_grade="B",
            regime="range",
            trend_state="sideways",
            vol_state="normal",
            rsi_range="oversold",
            source="seed",
        ),
        # Volatility breakout
        TriggerExample(
            trigger_id="seed_btc_vol_break_1",
            symbol="BTC-USD",
            direction="long",
            category="volatility_breakout",
            entry_rule="close > bollinger_upper and atr_14 > tf_4h_atr_14 * 0.8 and rsi_14 > 55",
            exit_rule="atr_14 < tf_4h_atr_14 * 0.5 or close < sma_short",
            confidence_grade="B",
            regime="high_vol",
            trend_state="uptrend",
            vol_state="high",
            rsi_range="neutral",
            source="seed",
        ),
        # Reversal - overbought
        TriggerExample(
            trigger_id="seed_eth_reversal_1",
            symbol="ETH-USD",
            direction="short",
            category="reversal",
            entry_rule="rsi_14 > 75 and close > bollinger_upper * 1.01 and macd_hist < 0",
            exit_rule="rsi_14 < 50 or close < sma_short",
            confidence_grade="C",
            regime="bull",
            trend_state="uptrend",
            vol_state="high",
            rsi_range="overbought",
            source="seed",
        ),
        # Emergency exit
        TriggerExample(
            trigger_id="seed_btc_emergency_1",
            symbol="BTC-USD",
            direction="flat",
            category="emergency_exit",
            entry_rule="false",
            exit_rule="vol_state == 'extreme' and atr_14 > tf_4h_atr_14 * 1.5 and position != 'flat'",
            confidence_grade="A",
            regime="high_vol",
            trend_state="unknown",
            vol_state="extreme",
            rsi_range="neutral",
            source="seed",
        ),
        # Simple 2-condition triggers for quiet markets
        TriggerExample(
            trigger_id="seed_btc_simple_1",
            symbol="BTC-USD",
            direction="long",
            category="trend_continuation",
            entry_rule="rsi_14 < 40 and close > sma_medium",
            exit_rule="rsi_14 > 60",
            confidence_grade="C",
            regime="range",
            trend_state="sideways",
            vol_state="low",
            rsi_range="neutral",
            source="seed",
        ),
        TriggerExample(
            trigger_id="seed_eth_simple_1",
            symbol="ETH-USD",
            direction="long",
            category="mean_reversion",
            entry_rule="close < bollinger_lower and rsi_14 < 45",
            exit_rule="close > sma_short",
            confidence_grade="C",
            regime="range",
            trend_state="sideways",
            vol_state="low",
            rsi_range="neutral",
            source="seed",
        ),
    ]

    added = 0
    for example in default_examples:
        store.add_example(example, compute_embedding=True)
        added += 1

    logger.info("Seeded %d default trigger examples", added)
    return added
