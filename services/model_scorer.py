"""ModelScorer interface and default NullModelScorer.

The ModelScorer is the only interface between the model training pipeline
and the live/backtest execution engine. Keep it narrow.

Implementations:
  NullModelScorer  — always returns None scores (default, no model trained)
  LightGBMScorer   — loads a serialized LightGBM model from disk (future)
  XGBoostScorer    — loads a serialized XGBoost model from disk (future)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from schemas.model_score import ModelScorePacket


class ModelScorer(ABC):
    """Abstract base: score a setup given its frozen feature context."""

    @abstractmethod
    def score(self, feature_snapshot: Dict[str, Any]) -> ModelScorePacket:
        """Score a setup from its frozen feature snapshot.

        Args:
            feature_snapshot: SetupEvent.feature_snapshot dict (serialized
                IndicatorSnapshot). Keys are feature names, values are
                float | bool | None.

        Returns:
            ModelScorePacket with model_quality_score and optional sub-scores.
            All fields None if model unavailable or input is insufficient.
        """
        ...

    def is_entry_blocked(self, packet: ModelScorePacket) -> bool:
        """Hard gate: block entry if p_false_breakout > 0.40."""
        if packet.p_false_breakout is None:
            return False
        return packet.p_false_breakout > 0.40

    def size_multiplier(self, packet: ModelScorePacket) -> float:
        """Sizing multiplier from model_quality_score.

        Returns 1.0 (no effect) when model_quality_score is None.
        Clamps to [0.5, 1.25] to prevent extreme sizing.
        """
        if packet.model_quality_score is None:
            return 1.0
        q = packet.model_quality_score
        return max(0.5, min(1.25, 0.5 + 1.0 * (q - 0.5)))


class NullModelScorer(ModelScorer):
    """Default scorer — no trained model.

    Always returns a ModelScorePacket with all None scores.
    size_multiplier returns 1.0, is_entry_blocked returns False.
    This is the correct default: no model means no effect.
    """

    def score(self, feature_snapshot: Dict[str, Any]) -> ModelScorePacket:
        return ModelScorePacket()
