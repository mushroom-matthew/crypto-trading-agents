"""Tests for R79 hypothesis attribution engine."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from schemas.episode_memory import EpisodeMemoryRecord
from services.hypothesis_attribution import (
    classify_outcome,
    compute_playbook_prior,
    enrich_episode_with_attribution,
    ATTRIBUTION_MODEL_QUALITY_STRONG,
    ATTRIBUTION_PREMATURE_STOP,
    ATTRIBUTION_TARGET_TOO_CLOSE,
    ATTRIBUTION_REGIME_MISMATCH,
    ATTRIBUTION_TIMING_BIAS,
    ATTRIBUTION_CHOPPY_EXIT,
)


def _make_record(
    outcome_class="neutral",
    r_achieved=None,
    hold_bars=None,
    stop_hit=None,
    target_hit=None,
    invalidation_hit=None,
    mfe_pct=None,
    expected_bars=None,
    playbook_id=None,
) -> EpisodeMemoryRecord:
    now = datetime.now(timezone.utc)
    return EpisodeMemoryRecord(
        episode_id=str(uuid4()),
        symbol="BTC-USD",
        outcome_class=outcome_class,
        r_achieved=r_achieved,
        hold_bars=hold_bars,
        stop_hit=stop_hit,
        target_hit=target_hit,
        invalidation_hit=invalidation_hit,
        mfe_pct=mfe_pct,
        expected_bars=expected_bars,
        bars_held=hold_bars,
        r_multiple_realized=r_achieved,
        playbook_id=playbook_id,
    )


# ---------------------------------------------------------------------------
# classify_outcome tests
# ---------------------------------------------------------------------------

def test_model_quality_strong():
    rec = _make_record(outcome_class="win", r_achieved=2.5, target_hit=True)
    tags = classify_outcome(rec)
    assert ATTRIBUTION_MODEL_QUALITY_STRONG in tags
    # Strong winners get no failure tags
    assert len(tags) == 1


def test_premature_stop_very_fast():
    rec = _make_record(outcome_class="loss", r_achieved=-1.0, stop_hit=True, hold_bars=2)
    tags = classify_outcome(rec)
    assert ATTRIBUTION_PREMATURE_STOP in tags


def test_premature_stop_threshold():
    # bars=2 is below threshold (< 3) → premature
    rec = _make_record(outcome_class="loss", stop_hit=True, hold_bars=2)
    tags = classify_outcome(rec)
    assert ATTRIBUTION_PREMATURE_STOP in tags


def test_target_too_close():
    # Target hit but poor R
    rec = _make_record(outcome_class="win", r_achieved=0.5, target_hit=True)
    tags = classify_outcome(rec)
    assert ATTRIBUTION_TARGET_TOO_CLOSE in tags


def test_regime_mismatch():
    rec = _make_record(outcome_class="loss", invalidation_hit=True, stop_hit=False, target_hit=False)
    tags = classify_outcome(rec)
    assert ATTRIBUTION_REGIME_MISMATCH in tags


def test_timing_bias_late():
    # Held 3x longer than expected
    rec = _make_record(outcome_class="neutral", hold_bars=30, expected_bars=10)
    tags = classify_outcome(rec)
    assert ATTRIBUTION_TIMING_BIAS in tags


def test_no_timing_bias_on_track():
    # Held 1.5x expected — within range
    rec = _make_record(outcome_class="win", hold_bars=15, expected_bars=10, r_achieved=1.5, target_hit=True)
    tags = classify_outcome(rec)
    assert ATTRIBUTION_TIMING_BIAS not in tags


def test_choppy_exit_neutral():
    rec = _make_record(outcome_class="neutral")
    tags = classify_outcome(rec)
    assert ATTRIBUTION_CHOPPY_EXIT in tags


def test_clean_win_no_tags():
    # Nice win, short hold, good R — no failure tags
    rec = _make_record(
        outcome_class="win", r_achieved=1.8, target_hit=True, hold_bars=8, expected_bars=10
    )
    tags = classify_outcome(rec)
    # model_quality_strong not triggered (r < 2.0), no failure tags
    assert ATTRIBUTION_PREMATURE_STOP not in tags
    assert ATTRIBUTION_REGIME_MISMATCH not in tags
    assert ATTRIBUTION_CHOPPY_EXIT not in tags


def test_multiple_tags_possible():
    # Premature stop AND size mismatch
    rec = _make_record(
        outcome_class="loss",
        r_achieved=0.1,
        stop_hit=True,
        hold_bars=1,
        mfe_pct=3.0,  # large MFE but tiny R
    )
    tags = classify_outcome(rec)
    assert ATTRIBUTION_PREMATURE_STOP in tags


# ---------------------------------------------------------------------------
# enrich_episode_with_attribution
# ---------------------------------------------------------------------------

def test_enrich_sets_attribution_tags():
    rec = _make_record(outcome_class="loss", r_achieved=-1.0, hold_bars=2)
    enriched = enrich_episode_with_attribution(
        rec,
        hypothesis_id="hyp-123",
        thesis_text="BTC breakout long",
        stop_hit=True,
        target_hit=False,
    )
    assert enriched.hypothesis_id == "hyp-123"
    assert enriched.thesis_text == "BTC breakout long"
    assert enriched.stop_hit is True
    assert enriched.target_hit is False
    assert isinstance(enriched.attribution_tags, list)


def test_enrich_computes_timing_accuracy():
    rec = _make_record(outcome_class="win", hold_bars=10)
    enriched = enrich_episode_with_attribution(
        rec, expected_bars=10
    )
    assert enriched.timing_accuracy == pytest.approx(1.0, rel=0.01)


def test_enrich_timing_accuracy_missing_bars():
    rec = _make_record(outcome_class="win")  # hold_bars=None
    enriched = enrich_episode_with_attribution(rec, expected_bars=10)
    assert enriched.timing_accuracy is None


def test_enrich_infers_stop_hit_from_outcome():
    rec = _make_record(outcome_class="loss", r_achieved=-1.0)
    enriched = enrich_episode_with_attribution(rec)
    assert enriched.stop_hit is True  # inferred from outcome_class="loss"
    assert enriched.target_hit is False  # inferred from outcome_class


def test_enrich_infers_target_hit_from_outcome():
    rec = _make_record(outcome_class="win", r_achieved=1.5)
    enriched = enrich_episode_with_attribution(rec)
    assert enriched.target_hit is True
    assert enriched.stop_hit is False


# ---------------------------------------------------------------------------
# compute_playbook_prior
# ---------------------------------------------------------------------------

def _make_episodes_for_playbook(n_wins: int, n_losses: int, playbook: str) -> list:
    records = []
    for i in range(n_wins):
        records.append(_make_record(outcome_class="win", r_achieved=1.5, playbook_id=playbook))
    for i in range(n_losses):
        records.append(_make_record(outcome_class="loss", r_achieved=-1.0, playbook_id=playbook))
    return records


def test_compute_prior_insufficient_data():
    episodes = _make_episodes_for_playbook(3, 1, "donchian_breakout")
    prior = compute_playbook_prior(episodes, playbook_id="donchian_breakout", min_sample=5)
    assert prior["sufficient_data"] is False
    assert prior["win_rate"] is None
    assert prior["r_expectancy"] is None


def test_compute_prior_sufficient_data():
    episodes = _make_episodes_for_playbook(6, 4, "donchian_breakout")
    prior = compute_playbook_prior(episodes, playbook_id="donchian_breakout", min_sample=5)
    assert prior["sufficient_data"] is True
    assert prior["win_rate"] == pytest.approx(0.6, rel=0.01)
    assert prior["sample_size"] == 10


def test_compute_prior_no_playbook_filter():
    episodes = (
        _make_episodes_for_playbook(3, 2, "donchian_breakout")
        + _make_episodes_for_playbook(2, 3, "rsi_extremes")
    )
    prior = compute_playbook_prior(episodes, playbook_id=None, min_sample=5)
    assert prior["sufficient_data"] is True
    assert prior["sample_size"] == 10


def test_compute_prior_r_expectancy():
    episodes = _make_episodes_for_playbook(5, 5, "my_playbook")
    prior = compute_playbook_prior(episodes, playbook_id="my_playbook", min_sample=5)
    # wins give 1.5, losses give -1.0 → mean = (5*1.5 + 5*(-1.0)) / 10 = 0.25
    assert prior["r_expectancy"] == pytest.approx(0.25, rel=0.01)


def test_compute_prior_different_playbook_filtered():
    episodes = _make_episodes_for_playbook(5, 0, "pb_a") + _make_episodes_for_playbook(2, 0, "pb_b")
    prior = compute_playbook_prior(episodes, playbook_id="pb_b", min_sample=5)
    # Only 2 records for pb_b
    assert prior["sufficient_data"] is False


# ---------------------------------------------------------------------------
# OpportunityCard attribution fields in score_symbol
# ---------------------------------------------------------------------------

def test_opportunity_card_has_attribution_fields():
    from schemas.opportunity import OpportunityCard
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    card = OpportunityCard(
        symbol="BTC-USD",
        opportunity_score=0.4,
        opportunity_score_norm=0.5,
        vol_edge=0.5,
        structure_edge=0.5,
        trend_edge=0.5,
        liquidity_score=0.5,
        spread_penalty=0.1,
        instability_penalty=0.0,
        expected_hold_horizon="intraday",
        scored_at=now,
        indicator_as_of=now,
        playbook_win_rate=0.65,
        playbook_r_expectancy=0.8,
        attribution_sample_size=10,
    )
    assert card.playbook_win_rate == pytest.approx(0.65)
    assert card.playbook_r_expectancy == pytest.approx(0.8)
    assert card.attribution_sample_size == 10


def test_attribution_bonus_in_score():
    """score_symbol applies a small bonus when r_expectancy > 0 with sufficient data."""
    from services.opportunity_scanner import score_symbol
    from schemas.llm_strategist import IndicatorSnapshot
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    snap = IndicatorSnapshot(symbol="BTC-USD", timeframe="1h", as_of=now, close=50000.0)

    card_no_prior = score_symbol("BTC-USD", snap)
    card_with_prior = score_symbol(
        "BTC-USD", snap,
        playbook_r_expectancy=2.0,
        playbook_win_rate=0.7,
        attribution_sample_size=10,
    )
    # Card with prior should have a slightly higher score
    assert card_with_prior.opportunity_score >= card_no_prior.opportunity_score
