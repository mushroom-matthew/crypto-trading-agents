"""High-level (slow) reflection service (Runbook 50).

Runs on a scheduled batch cadence (daily/weekly or on-demand after a minimum
batch of resolved episodes).  NEVER invoked inside the tick loop.

Gating rules
------------
- Minimum elapsed interval (daily by default, weekly for structural changes)
- Minimum resolved episodes (N >= 20 for structural recommendations)
- Minimum regime-cluster sample size for regime eligibility updates
- If gates fail: emit monitor-only findings, set insufficient_sample=True

Output contract
---------------
- HighLevelReflectionReport is produced for every invocation (even skipped
  ones include the skip meta).
- Structural recommendations are gated; monitor findings are always emitted.
- The judge consumes the report as evidence, not as a direct command.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from schemas.episode_memory import EpisodeMemoryRecord
from schemas.reflection import (
    HighLevelReflectionReport,
    HighLevelReflectionRequest,
    PlaybookFinding,
    ReflectionInvocationMeta,
    RegimeClusterSummary,
)

logger = logging.getLogger(__name__)

_MIN_EPISODES_STRUCTURAL = 20
_MIN_REGIME_CLUSTER_SAMPLES = 10


# ---------------------------------------------------------------------------
# Cluster / grouping helpers
# ---------------------------------------------------------------------------


def _cluster_key(record: EpisodeMemoryRecord) -> str:
    playbook = record.playbook_id or "unknown"
    regime = _regime_label(record)
    return f"playbook={playbook}|regime={regime}"


def _regime_label(record: EpisodeMemoryRecord) -> str:
    """Best-effort regime label from fingerprint or template/trigger info."""
    if record.template_id:
        return record.template_id
    if record.trigger_category:
        return record.trigger_category
    return "unknown"


# ---------------------------------------------------------------------------
# Outcome statistics
# ---------------------------------------------------------------------------


def _win_rate(records: List[EpisodeMemoryRecord]) -> float:
    if not records:
        return 0.0
    wins = sum(1 for r in records if r.outcome_class == "win")
    return wins / len(records)


def _avg(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _dominant_failure_modes(records: List[EpisodeMemoryRecord], top_n: int = 3) -> List[str]:
    counts: Dict[str, int] = defaultdict(int)
    for r in records:
        for mode in r.failure_modes:
            counts[mode] += 1
    return sorted(counts, key=lambda m: -counts[m])[:top_n]


# ---------------------------------------------------------------------------
# Drift detection helpers
# ---------------------------------------------------------------------------


def _hold_time_deviation_pct(
    records: List[EpisodeMemoryRecord],
    expected_p50: Optional[float],
) -> Optional[float]:
    """Fractional deviation of actual avg hold_bars from expected P50."""
    if expected_p50 is None or expected_p50 <= 0:
        return None
    actual_avg = _avg([r.hold_bars for r in records if r.hold_bars is not None])
    if actual_avg is None:
        return None
    return (actual_avg - expected_p50) / expected_p50


def _drift_findings_for_cluster(
    cluster_key: str,
    records: List[EpisodeMemoryRecord],
) -> List[str]:
    findings: List[str] = []
    modes = _dominant_failure_modes(records)
    win_r = _win_rate(records)
    if win_r < 0.35 and len(records) >= 5:
        findings.append(
            f"DRIFT: cluster '{cluster_key}' win rate is {win_r:.0%} — significantly below "
            "expected breakeven; potential regime or playbook mis-alignment"
        )
    if "false_breakout_reversion" in modes and "low_volume_breakout_failure" in modes:
        findings.append(
            f"DRIFT: cluster '{cluster_key}' shows co-occurring breakout failure modes — "
            "consider tightening breakout confirmation criteria"
        )
    return findings


# ---------------------------------------------------------------------------
# Public service
# ---------------------------------------------------------------------------


class HighLevelReflectionService:
    """Scheduled batch reflection service.

    Usage::

        svc = HighLevelReflectionService()
        report = svc.reflect(request, episodes)
    """

    def reflect(
        self,
        request: HighLevelReflectionRequest,
        episodes: List[EpisodeMemoryRecord],
    ) -> HighLevelReflectionReport:
        """Run batch reflection and return a typed report.

        ``episodes`` must be pre-fetched (e.g. from EpisodeMemoryStore) for
        the requested window and filters.  The service is pure computation —
        no I/O.
        """
        n = len(episodes)

        # Sample-size gate for structural recommendations
        structural_eligible = n >= request.min_episodes_for_structural_recommendation
        insufficient_reason: Optional[str] = None
        if not structural_eligible:
            insufficient_reason = (
                f"Only {n} resolved episodes in window; need >= "
                f"{request.min_episodes_for_structural_recommendation} for "
                "structural recommendations"
            )

        # Cluster episodes
        clusters: Dict[str, List[EpisodeMemoryRecord]] = defaultdict(list)
        for ep in episodes:
            clusters[_cluster_key(ep)].append(ep)

        # Build cluster summaries
        cluster_summaries: List[RegimeClusterSummary] = []
        all_drift_findings: List[str] = []
        for key, recs in clusters.items():
            win_r = _win_rate(recs)
            avg_r = _avg([r.r_achieved for r in recs])
            avg_hold = _avg([r.hold_bars for r in recs if r.hold_bars is not None])
            modes = _dominant_failure_modes(recs)
            cluster_summaries.append(
                RegimeClusterSummary(
                    cluster_key=key,
                    n_episodes=len(recs),
                    win_rate=win_r,
                    avg_r_achieved=avg_r,
                    avg_hold_bars=avg_hold,
                    dominant_failure_modes=modes,
                )
            )
            all_drift_findings.extend(_drift_findings_for_cluster(key, recs))

        # Build playbook-level findings
        by_playbook: Dict[str, List[EpisodeMemoryRecord]] = defaultdict(list)
        for ep in episodes:
            pb = ep.playbook_id or "unknown"
            by_playbook[pb].append(ep)

        playbook_findings: List[PlaybookFinding] = []
        for pb_id, recs in by_playbook.items():
            n_pb = len(recs)
            win_r = _win_rate(recs)
            avg_r = _avg([r.r_achieved for r in recs])
            mae_avg = _avg([r.mae_pct for r in recs if r.mae_pct is not None])
            mfe_avg = _avg([r.mfe_pct for r in recs if r.mfe_pct is not None])
            modes = _dominant_failure_modes(recs)

            # Regime-cluster gate for structural eligibility
            pb_structural_eligible = (
                structural_eligible and n_pb >= request.min_regime_cluster_samples
            )
            pb_insufficient_reason: Optional[str] = None
            if not pb_structural_eligible:
                if not structural_eligible:
                    pb_insufficient_reason = insufficient_reason
                else:
                    pb_insufficient_reason = (
                        f"Playbook '{pb_id}' has only {n_pb} episodes; need >= "
                        f"{request.min_regime_cluster_samples} for structural change"
                    )

            # Choose recommended action
            action: str = "hold"
            if pb_structural_eligible and win_r < 0.35 and n_pb >= 10:
                action = "research_experiment"
            elif win_r < 0.40 and n_pb >= 5:
                action = "policy_adjust"

            playbook_findings.append(
                PlaybookFinding(
                    playbook_id=pb_id,
                    n_episodes=n_pb,
                    win_rate=win_r,
                    avg_r_achieved=avg_r,
                    mae_drift=mae_avg,
                    mfe_drift=mfe_avg,
                    dominant_failure_modes=modes,
                    recommended_action=action,  # type: ignore[arg-type]
                    structural_change_eligible=pb_structural_eligible,
                    insufficient_sample_reason=pb_insufficient_reason,
                )
            )

        # Top-level recommendations (only structural ones are gated)
        recommendations: List[Dict] = []
        for pf in playbook_findings:
            if pf.recommended_action in ("research_experiment", "policy_adjust"):
                rec: Dict = {
                    "playbook_id": pf.playbook_id,
                    "action": pf.recommended_action,
                    "win_rate": pf.win_rate,
                    "n_episodes": pf.n_episodes,
                    "structural_change_eligible": pf.structural_change_eligible,
                }
                if not pf.structural_change_eligible and pf.insufficient_sample_reason:
                    rec["monitor_only"] = True
                    rec["reason"] = pf.insufficient_sample_reason
                recommendations.append(rec)

        evidence_refs = [ep.episode_id for ep in episodes[:50]]

        return HighLevelReflectionReport(
            window_start=request.window_start,
            window_end=request.window_end,
            n_episodes=n,
            regime_cluster_summary=cluster_summaries,
            playbook_findings=playbook_findings,
            drift_findings=all_drift_findings,
            recommendations=recommendations,
            evidence_refs=evidence_refs,
            insufficient_sample=not structural_eligible,
            insufficient_sample_reason=insufficient_reason,
            structural_recommendations_suppressed=not structural_eligible,
            meta=request.meta,
        )


# ---------------------------------------------------------------------------
# Cadence gate helper
# ---------------------------------------------------------------------------


def should_run_high_level_reflection(
    last_run_at: Optional[datetime],
    cadence: str,
    now: Optional[datetime] = None,
    force_run: bool = False,
) -> tuple[bool, Optional[str]]:
    """Check whether the time gate allows a high-level reflection run.

    Returns (should_run, skip_reason).  skip_reason is non-None when skipping.
    """
    if force_run:
        return True, None

    _now = now or datetime.now(tz=timezone.utc)

    if last_run_at is None:
        return True, None

    if last_run_at.tzinfo is None:
        last_run_at = last_run_at.replace(tzinfo=timezone.utc)

    elapsed = _now - last_run_at

    min_interval: timedelta
    if cadence == "weekly":
        min_interval = timedelta(days=7)
    else:  # daily (default)
        min_interval = timedelta(hours=23)

    if elapsed < min_interval:
        remaining = min_interval - elapsed
        return False, (
            f"High-level reflection cadence gate: last run {elapsed.total_seconds() / 3600:.1f}h ago; "
            f"need {min_interval.total_seconds() / 3600:.1f}h interval. "
            f"Next run in ~{remaining.total_seconds() / 3600:.1f}h."
        )

    return True, None
