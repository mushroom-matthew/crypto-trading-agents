"""Tests for services/playbook_stats_service.py (Runbook 52).

Validates:
- attach_stats with 0 episodes returns stats with n=0
- win_rate, expectancy, hold bars, z-score, mae/mfe computed correctly
- Only episodes matching playbook_id are used
- regime filtering logic
- evidence_source set to "episode_memory"
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import List
from uuid import uuid4

import pytest

from schemas.episode_memory import EpisodeMemoryRecord
from schemas.playbook_definition import HorizonExpectations, PlaybookDefinition
from services.playbook_stats_service import attach_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    playbook_id: str = "test_pb",
    outcome_class: str = "win",
    r_achieved: float | None = None,
    hold_bars: int | None = None,
    mae: float | None = None,
    mfe: float | None = None,
    regime_fingerprint: dict | None = None,
) -> EpisodeMemoryRecord:
    return EpisodeMemoryRecord(
        episode_id=str(uuid4()),
        symbol="BTC-USD",
        playbook_id=playbook_id,
        outcome_class=outcome_class,  # type: ignore[arg-type]
        r_achieved=r_achieved,
        hold_bars=hold_bars,
        mae=mae,
        mfe=mfe,
        regime_fingerprint=regime_fingerprint,
    )


def _make_playbook(
    playbook_id: str = "test_pb",
    expected_p50: int | None = None,
    expected_p90: int | None = None,
) -> PlaybookDefinition:
    horizon = HorizonExpectations(
        expected_hold_bars_p50=expected_p50,
        expected_hold_bars_p90=expected_p90,
    )
    return PlaybookDefinition(playbook_id=playbook_id, horizon_expectations=horizon)


# ---------------------------------------------------------------------------
# attach_stats — core stats computation
# ---------------------------------------------------------------------------


class TestAttachStatsEmpty:
    def test_no_episodes_returns_n_zero(self):
        pb = _make_playbook()
        result = attach_stats(pb, [])
        assert len(result.performance_stats) == 1
        stats = result.performance_stats[0]
        assert stats.n == 0

    def test_no_episodes_win_rate_is_none(self):
        pb = _make_playbook()
        result = attach_stats(pb, [])
        assert result.performance_stats[0].win_rate is None

    def test_no_episodes_evidence_source_episode_memory(self):
        pb = _make_playbook()
        result = attach_stats(pb, [])
        assert result.performance_stats[0].evidence_source == "episode_memory"

    def test_no_episodes_performance_stats_length_one(self):
        """attach_stats always produces exactly one stats entry."""
        pb = _make_playbook()
        result = attach_stats(pb, [])
        assert len(result.performance_stats) == 1


class TestAttachStatsWins:
    def test_all_wins_sets_win_rate_one(self):
        pb = _make_playbook()
        episodes = [
            _make_episode(r_achieved=1.5),
            _make_episode(r_achieved=2.0),
            _make_episode(r_achieved=0.8),
        ]
        result = attach_stats(pb, episodes)
        stats = result.performance_stats[0]
        assert stats.win_rate == pytest.approx(1.0)

    def test_mixed_win_rate(self):
        pb = _make_playbook()
        episodes = [
            _make_episode(outcome_class="win", r_achieved=2.0),
            _make_episode(outcome_class="win", r_achieved=1.5),
            _make_episode(outcome_class="loss", r_achieved=-1.0),
            _make_episode(outcome_class="loss", r_achieved=-0.5),
        ]
        result = attach_stats(pb, episodes)
        stats = result.performance_stats[0]
        assert stats.win_rate == pytest.approx(0.5)
        assert stats.n == 4

    def test_win_rate_greater_than_zero_when_wins_present(self):
        pb = _make_playbook()
        episodes = [
            _make_episode(outcome_class="win", r_achieved=1.0),
            _make_episode(outcome_class="loss", r_achieved=-1.0),
        ]
        result = attach_stats(pb, episodes)
        assert result.performance_stats[0].win_rate > 0


class TestAttachStatsExpectancy:
    def test_expectancy_computed_from_wins_and_losses(self):
        """expectancy = win_rate * avg_win_r - (1-win_rate) * avg_loss_r"""
        pb = _make_playbook()
        episodes = [
            _make_episode(outcome_class="win", r_achieved=2.0),
            _make_episode(outcome_class="loss", r_achieved=-1.0),
        ]
        result = attach_stats(pb, episodes)
        stats = result.performance_stats[0]
        # win_rate = 0.5, avg_win_r = 2.0, avg_loss_r = 1.0
        expected = 0.5 * 2.0 - 0.5 * 1.0
        assert stats.expectancy == pytest.approx(expected)

    def test_expectancy_none_when_no_losses(self):
        """If there are no loss episodes, expectancy cannot be computed."""
        pb = _make_playbook()
        episodes = [
            _make_episode(outcome_class="win", r_achieved=1.5),
            _make_episode(outcome_class="win", r_achieved=2.0),
        ]
        result = attach_stats(pb, episodes)
        # expectancy requires both win and loss r values — only wins: avg_loss_r is None
        assert result.performance_stats[0].expectancy is None


class TestAttachStatsHoldBars:
    def test_p50_hold_bars_computed(self):
        pb = _make_playbook()
        # hold_bars sorted: [2, 4, 6, 8, 10]
        episodes = [
            _make_episode(hold_bars=10),
            _make_episode(hold_bars=2),
            _make_episode(hold_bars=6),
            _make_episode(hold_bars=4),
            _make_episode(hold_bars=8),
        ]
        result = attach_stats(pb, episodes)
        stats = result.performance_stats[0]
        # median of [2,4,6,8,10] at index 2 = 6
        assert stats.p50_hold_bars == 6

    def test_p90_hold_bars_computed(self):
        pb = _make_playbook()
        # sorted hold bars: [1,2,3,4,5,6,7,8,9,10]
        episodes = [_make_episode(hold_bars=i) for i in range(1, 11)]
        result = attach_stats(pb, episodes)
        stats = result.performance_stats[0]
        # p90 index = int(10 * 0.9) = 9 → [1..10][9] = 10
        assert stats.p90_hold_bars == 10

    def test_hold_bars_mean_computed(self):
        pb = _make_playbook()
        episodes = [
            _make_episode(hold_bars=4),
            _make_episode(hold_bars=8),
        ]
        result = attach_stats(pb, episodes)
        stats = result.performance_stats[0]
        assert stats.hold_bars_mean == pytest.approx(6.0)


class TestAttachStatsZScore:
    def test_hold_time_z_score_none_when_no_horizon_set(self):
        pb = _make_playbook()  # no p50/p90 set
        episodes = [_make_episode(hold_bars=10), _make_episode(hold_bars=20)]
        result = attach_stats(pb, episodes)
        assert result.performance_stats[0].hold_time_z_score is None

    def test_hold_time_z_score_computed_when_horizon_set(self):
        """z-score should be computed when p50 and p90 are set and data is available."""
        pb = _make_playbook(expected_p50=10, expected_p90=20)
        # expected_mean = 10, expected_std = (20-10)/1.28 ≈ 7.8125
        episodes = [_make_episode(hold_bars=h) for h in [8, 10, 12, 14]]
        result = attach_stats(pb, episodes)
        stats = result.performance_stats[0]
        assert stats.hold_time_z_score is not None
        # hold_mean ≈ 11.0, z ≈ (11.0 - 10.0) / 7.8125 ≈ 0.128
        assert isinstance(stats.hold_time_z_score, float)


class TestAttachStatsMaeMfe:
    def test_mae_p50_computed(self):
        pb = _make_playbook()
        episodes = [
            _make_episode(mae=-0.02),
            _make_episode(mae=-0.04),
            _make_episode(mae=-0.06),
        ]
        result = attach_stats(pb, episodes)
        # sorted mae: [-0.06, -0.04, -0.02]; p50 index = 1 → -0.04
        assert result.performance_stats[0].mae_p50 == pytest.approx(-0.04)

    def test_mfe_p50_computed(self):
        pb = _make_playbook()
        episodes = [
            _make_episode(mfe=0.05),
            _make_episode(mfe=0.10),
            _make_episode(mfe=0.15),
        ]
        result = attach_stats(pb, episodes)
        # sorted mfe: [0.05, 0.10, 0.15]; p50 index = 1 → 0.10
        assert result.performance_stats[0].mfe_p50 == pytest.approx(0.10)

    def test_mae_mfe_none_when_no_data(self):
        pb = _make_playbook()
        episodes = [_make_episode()]  # no mae/mfe set
        result = attach_stats(pb, episodes)
        assert result.performance_stats[0].mae_p50 is None
        assert result.performance_stats[0].mfe_p50 is None


class TestAttachStatsFiltering:
    def test_only_matching_playbook_id_used(self):
        """attach_stats must only use episodes whose playbook_id matches."""
        pb = _make_playbook("my_pb")
        episodes = [
            _make_episode(playbook_id="my_pb", outcome_class="win", r_achieved=1.0),
            _make_episode(playbook_id="other_pb", outcome_class="win", r_achieved=5.0),
            _make_episode(playbook_id="my_pb", outcome_class="loss", r_achieved=-1.0),
        ]
        result = attach_stats(pb, episodes)
        stats = result.performance_stats[0]
        # Only 2 episodes match "my_pb"
        assert stats.n == 2

    def test_regime_none_returns_stats_for_all_episodes(self):
        """When regime=None, all matching playbook episodes are used."""
        pb = _make_playbook()
        episodes = [
            _make_episode(outcome_class="win"),
            _make_episode(outcome_class="loss"),
            _make_episode(outcome_class="win"),
        ]
        result = attach_stats(pb, episodes, regime=None)
        assert result.performance_stats[0].n == 3

    def test_regime_label_propagated_to_stats(self):
        """The regime parameter is propagated to the stats regime field."""
        pb = _make_playbook()
        episodes = [_make_episode()]
        result = attach_stats(pb, episodes, regime="range")
        assert result.performance_stats[0].regime == "range"

    def test_regime_none_propagated_to_stats(self):
        pb = _make_playbook()
        episodes = [_make_episode()]
        result = attach_stats(pb, episodes, regime=None)
        assert result.performance_stats[0].regime is None

    def test_evidence_source_is_episode_memory(self):
        pb = _make_playbook()
        result = attach_stats(pb, [_make_episode()])
        assert result.performance_stats[0].evidence_source == "episode_memory"

    def test_last_updated_is_set(self):
        """last_updated should be populated when episodes are present."""
        pb = _make_playbook()
        result = attach_stats(pb, [_make_episode()])
        assert result.performance_stats[0].last_updated is not None

    def test_original_playbook_not_mutated(self):
        """attach_stats must not mutate the input PlaybookDefinition."""
        pb = _make_playbook()
        assert pb.performance_stats == []
        _ = attach_stats(pb, [_make_episode()])
        assert pb.performance_stats == []
