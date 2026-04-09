"""Hypothesis attribution engine (R79).

Classifies completed round trips with attribution tags so the scanner can
build per-symbol/playbook priors and the planner can avoid repeat mistakes.

Tags are deterministic labels derived from episode outcome metrics — no LLM call.
"""
from __future__ import annotations

from typing import List, Optional

from schemas.episode_memory import EpisodeMemoryRecord


# ---------------------------------------------------------------------------
# Attribution tag constants
# ---------------------------------------------------------------------------

ATTRIBUTION_PREMATURE_STOP = "premature_stop"        # stop hit too quickly (< 3 bars)
ATTRIBUTION_TARGET_TOO_CLOSE = "target_too_close"    # target hit but r_multiple < 1.0
ATTRIBUTION_REGIME_MISMATCH = "regime_mismatch"      # invalidation triggered before stop/target
ATTRIBUTION_TIMING_BIAS = "timing_bias"              # held >> expected_bars (early/late entry)
ATTRIBUTION_MODEL_QUALITY_STRONG = "model_quality_strong"  # r_multiple > 2.0 + target hit
ATTRIBUTION_CHOPPY_EXIT = "choppy_exit"              # neither stop nor target — neutral
ATTRIBUTION_SIZE_MISMATCH = "size_mismatch"          # mfe large but r_achieved small

ALL_ATTRIBUTION_TAGS = [
    ATTRIBUTION_PREMATURE_STOP,
    ATTRIBUTION_TARGET_TOO_CLOSE,
    ATTRIBUTION_REGIME_MISMATCH,
    ATTRIBUTION_TIMING_BIAS,
    ATTRIBUTION_MODEL_QUALITY_STRONG,
    ATTRIBUTION_CHOPPY_EXIT,
    ATTRIBUTION_SIZE_MISMATCH,
]

# Thresholds
_PREMATURE_STOP_BARS = 3           # stop hit within this many bars
_TIMING_BIAS_RATIO = 2.5           # bars_held / expected_bars > this = timing bias
_STRONG_R = 2.0                    # r_multiple threshold for model_quality_strong
_POOR_R_DESPITE_TARGET = 1.0       # r_multiple < this even on target hit = target_too_close


def classify_outcome(record: EpisodeMemoryRecord) -> List[str]:
    """Tag a round trip with attribution labels.

    Args:
        record: A resolved EpisodeMemoryRecord (outcome_class set).

    Returns:
        List of attribution tag strings. May be empty for clean trades.
    """
    tags: List[str] = []

    r = record.r_multiple_realized if record.r_multiple_realized is not None else record.r_achieved
    bars = record.bars_held if record.bars_held is not None else record.hold_bars
    expected = record.expected_bars
    stop_hit = record.stop_hit
    target_hit = record.target_hit
    invalidation_hit = record.invalidation_hit

    # --- model_quality_strong: target hit with strong R ---
    if target_hit and r is not None and r > _STRONG_R:
        tags.append(ATTRIBUTION_MODEL_QUALITY_STRONG)
        return tags  # no failure tags for strong winners

    # --- premature_stop: stop hit very quickly ---
    if stop_hit and bars is not None and bars < _PREMATURE_STOP_BARS:
        tags.append(ATTRIBUTION_PREMATURE_STOP)

    # --- target_too_close: target hit but R was poor ---
    if target_hit and r is not None and r < _POOR_R_DESPITE_TARGET:
        tags.append(ATTRIBUTION_TARGET_TOO_CLOSE)

    # --- regime_mismatch: invalidation fired (not stop, not target) ---
    if invalidation_hit and not stop_hit and not target_hit:
        tags.append(ATTRIBUTION_REGIME_MISMATCH)

    # --- timing_bias: held much longer than expected ---
    if bars is not None and expected is not None and expected > 0:
        ratio = bars / expected
        if ratio > _TIMING_BIAS_RATIO:
            tags.append(ATTRIBUTION_TIMING_BIAS)

    # --- choppy_exit: neutral outcome, no clear reason ---
    if record.outcome_class == "neutral" and not stop_hit and not target_hit and not invalidation_hit:
        if not tags:
            tags.append(ATTRIBUTION_CHOPPY_EXIT)

    # --- size_mismatch: MFE was large but R achieved was low (sizing issue) ---
    if record.mfe_pct is not None and record.mfe_pct > 2.0:
        if r is not None and r < 0.5:
            tags.append(ATTRIBUTION_SIZE_MISMATCH)

    return tags


def enrich_episode_with_attribution(
    record: EpisodeMemoryRecord,
    hypothesis_id: Optional[str] = None,
    thesis_text: Optional[str] = None,
    stop_hit: Optional[bool] = None,
    target_hit: Optional[bool] = None,
    invalidation_hit: Optional[bool] = None,
    expected_bars: Optional[int] = None,
) -> EpisodeMemoryRecord:
    """Return a copy of the record with hypothesis attribution fields populated.

    Args:
        record: Base EpisodeMemoryRecord to enrich.
        hypothesis_id: ID of the originating TradeHypothesis.
        thesis_text: LLM's thesis summary (stored for audit).
        stop_hit: Whether the stop price was hit.
        target_hit: Whether the target price was hit.
        invalidation_hit: Whether the invalidation rule fired.
        expected_bars: LLM's estimated_bars_to_resolution.

    Returns:
        Enriched copy with attribution_tags populated.
    """
    r = record.r_achieved
    bars = record.hold_bars

    # Compute timing accuracy
    timing_accuracy: Optional[float] = None
    if expected_bars is not None and expected_bars > 0 and bars is not None and bars > 0:
        timing_accuracy = round(expected_bars / bars, 3)

    enriched = record.model_copy(update={
        "hypothesis_id": hypothesis_id,
        "thesis_text": thesis_text,
        "stop_hit": stop_hit if stop_hit is not None else (record.outcome_class == "loss"),
        "target_hit": target_hit if target_hit is not None else (record.outcome_class == "win"),
        "invalidation_hit": invalidation_hit or False,
        "r_multiple_realized": r,
        "bars_held": bars,
        "expected_bars": expected_bars,
        "timing_accuracy": timing_accuracy,
    })

    # Classify and set attribution_tags
    tags = classify_outcome(enriched)
    enriched = enriched.model_copy(update={"attribution_tags": tags})
    return enriched


# ---------------------------------------------------------------------------
# Scanner prior computation (used by opportunity_scanner rank_universe)
# ---------------------------------------------------------------------------

def compute_playbook_prior(
    episodes: List[EpisodeMemoryRecord],
    playbook_id: Optional[str] = None,
    min_sample: int = 5,
) -> dict:
    """Compute win_rate and r_expectancy from episodes for a playbook.

    Args:
        episodes: Recent EpisodeMemoryRecord list for a symbol.
        playbook_id: Filter to specific playbook; None = all episodes.
        min_sample: Minimum episodes needed to return a meaningful prior.

    Returns:
        dict with keys: win_rate, r_expectancy, sample_size, sufficient_data.
    """
    relevant = [
        e for e in episodes
        if playbook_id is None or e.playbook_id == playbook_id
    ]
    if len(relevant) < min_sample:
        return {
            "win_rate": None,
            "r_expectancy": None,
            "sample_size": len(relevant),
            "sufficient_data": False,
        }

    wins = sum(1 for e in relevant if e.outcome_class == "win")
    win_rate = wins / len(relevant)
    r_values = [e.r_achieved for e in relevant if e.r_achieved is not None]
    r_expectancy = sum(r_values) / len(r_values) if r_values else 0.0

    return {
        "win_rate": round(win_rate, 3),
        "r_expectancy": round(r_expectancy, 3),
        "sample_size": len(relevant),
        "sufficient_data": True,
    }
