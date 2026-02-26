"""PlaybookStatsService — attaches empirical performance stats to PlaybookDefinition objects.

Sources consulted (in order of priority):
  1. EpisodeMemoryStore (Runbook 51) — most recent outcome data
  2. Manual validation evidence in .md files (fallback, not implemented here)

Stats computed:
  - n, win_rate, avg_r, p50/p90 hold bars, hold_bars_mean/std
  - hold_time_z_score (calibration check vs HorizonExpectations)
  - mae_p50, mfe_p50
  - expectancy = win_rate * avg_win_r - (1 - win_rate) * avg_loss_r

All functions are pure / side-effect-free. Callers are responsible for
persisting or caching the returned PlaybookDefinition copies.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import List, Optional

from schemas.episode_memory import EpisodeMemoryRecord
from schemas.playbook_definition import PlaybookDefinition, PlaybookPerformanceStats


def _compute_stats_from_episodes(
    episodes: List[EpisodeMemoryRecord],
    regime: Optional[str],
    expected_hold_mean: Optional[float],
    expected_hold_std: Optional[float],
) -> PlaybookPerformanceStats:
    """Compute PlaybookPerformanceStats from a list of resolved episode records.

    Args:
        episodes: Resolved EpisodeMemoryRecord objects for one playbook (possibly
                  filtered by regime).
        regime: Optional regime label attached to the stats slice.
        expected_hold_mean: Expected average hold duration in bars (from
                            HorizonExpectations.expected_hold_bars_p50).
        expected_hold_std: Expected std-dev of hold duration (approximated from
                           p50→p90 span).

    Returns:
        A fully populated PlaybookPerformanceStats instance.
    """
    if not episodes:
        return PlaybookPerformanceStats(
            n=0,
            regime=regime,
            evidence_source="episode_memory",
        )

    n = len(episodes)
    wins = [e for e in episodes if e.outcome_class == "win"]
    win_rate = len(wins) / n

    r_values = [e.r_achieved for e in episodes if e.r_achieved is not None]
    avg_r = statistics.mean(r_values) if r_values else None

    win_r = [e.r_achieved for e in wins if e.r_achieved is not None]
    loss_episodes = [e for e in episodes if e.outcome_class == "loss"]
    loss_r = [abs(e.r_achieved) for e in loss_episodes if e.r_achieved is not None]
    avg_win_r = statistics.mean(win_r) if win_r else None
    avg_loss_r = statistics.mean(loss_r) if loss_r else None

    expectancy: Optional[float] = None
    if avg_win_r is not None and avg_loss_r is not None:
        expectancy = win_rate * avg_win_r - (1 - win_rate) * avg_loss_r

    hold_bars_list = sorted([e.hold_bars for e in episodes if e.hold_bars is not None])
    p50_hold: Optional[int] = None
    p90_hold: Optional[int] = None
    hold_mean: Optional[float] = None
    hold_std: Optional[float] = None

    if hold_bars_list:
        p50_hold = hold_bars_list[len(hold_bars_list) // 2]
        p90_hold = hold_bars_list[int(len(hold_bars_list) * 0.9)]
        hold_mean = statistics.mean(hold_bars_list)
        hold_std = statistics.stdev(hold_bars_list) if len(hold_bars_list) >= 2 else None

    # Hold-time calibration: z-score of mean realized vs expected
    z_score: Optional[float] = None
    if (
        hold_mean is not None
        and expected_hold_mean is not None
        and expected_hold_std is not None
        and expected_hold_std > 0
    ):
        z_score = (hold_mean - expected_hold_mean) / expected_hold_std

    mae_list = sorted([e.mae for e in episodes if e.mae is not None])
    mfe_list = sorted([e.mfe for e in episodes if e.mfe is not None])
    mae_p50 = mae_list[len(mae_list) // 2] if mae_list else None
    mfe_p50 = mfe_list[len(mfe_list) // 2] if mfe_list else None

    return PlaybookPerformanceStats(
        n=n,
        win_rate=win_rate,
        avg_r=avg_r,
        expectancy=expectancy,
        p50_hold_bars=p50_hold,
        p90_hold_bars=p90_hold,
        hold_bars_mean=hold_mean,
        hold_bars_std=hold_std,
        hold_time_z_score=z_score,
        mae_p50=mae_p50,
        mfe_p50=mfe_p50,
        last_updated=datetime.now(timezone.utc),
        evidence_source="episode_memory",
        regime=regime,
    )


def attach_stats(
    playbook: PlaybookDefinition,
    episodes: List[EpisodeMemoryRecord],
    regime: Optional[str] = None,
) -> PlaybookDefinition:
    """Return a copy of `playbook` with performance_stats populated from episode records.

    Only episodes whose `playbook_id` matches `playbook.playbook_id` are used.
    When `regime` is provided, episodes that carry a regime_fingerprint are preferred;
    if none have a fingerprint, all matching episodes are used as a fallback.

    Args:
        playbook: The PlaybookDefinition to annotate. Not mutated.
        episodes: All available EpisodeMemoryRecord objects (may include other playbooks).
        regime: Optional regime label to attach to the stats slice and use for filtering.

    Returns:
        A new PlaybookDefinition instance with `performance_stats` set to a single-element
        list containing the computed stats.
    """
    relevant = [e for e in episodes if e.playbook_id == playbook.playbook_id]

    if regime:
        # Prefer episodes that have a regime_fingerprint (best proxy for regime-tagged data).
        # Fall back to all episodes for this playbook if none have fingerprints.
        regime_eps = [e for e in relevant if e.regime_fingerprint is not None]
        if not regime_eps:
            regime_eps = relevant
    else:
        regime_eps = relevant

    # Derive expected hold-time parameters from horizon expectations for calibration check.
    expected_mean: Optional[float] = None
    expected_std: Optional[float] = None
    if (
        playbook.horizon_expectations.expected_hold_bars_p50 is not None
        and playbook.horizon_expectations.expected_hold_bars_p90 is not None
    ):
        # Rough normal approximation: mean ≈ p50, std from p50→p90 span / 1.28
        expected_mean = float(playbook.horizon_expectations.expected_hold_bars_p50)
        span = (
            playbook.horizon_expectations.expected_hold_bars_p90
            - playbook.horizon_expectations.expected_hold_bars_p50
        )
        expected_std = span / 1.28

    stats = _compute_stats_from_episodes(regime_eps, regime, expected_mean, expected_std)
    return playbook.model_copy(update={"performance_stats": [stats]})
