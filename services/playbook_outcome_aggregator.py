"""PlaybookOutcomeAggregator — reads research trade outcomes and writes validation
evidence to vector_store/playbooks/*.md files (Runbook 48).
"""

from __future__ import annotations

import logging
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from schemas.research_budget import PlaybookValidationResult, ResearchTrade

logger = logging.getLogger(__name__)

PLAYBOOK_DIR = Path("vector_store/playbooks")
MIN_EVIDENCE_WRITE = 5  # minimum trades before writing to .md


class PlaybookOutcomeAggregator:
    """Reads ResearchTrade outcomes tagged with a playbook_id and writes
    validation statistics to the ``## Validation Evidence`` section in
    the corresponding ``vector_store/playbooks/<playbook_id>.md`` file.

    Usage::

        agg = PlaybookOutcomeAggregator()
        result = agg.aggregate(playbook_id, trades)
        agg.write_evidence_to_playbook(result)
    """

    def __init__(self, playbook_dir: Path | None = None) -> None:
        self.playbook_dir = playbook_dir or PLAYBOOK_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        playbook_id: str,
        trades: List[ResearchTrade],
    ) -> PlaybookValidationResult:
        """Compute validation statistics for *playbook_id* from *trades*.

        Only closed trades (outcome != "open") are considered.  Returns
        a ``PlaybookValidationResult`` regardless of sample size — callers
        check ``status == "insufficient_data"`` to decide whether to write.
        """
        closed = [t for t in trades if t.playbook_id == playbook_id and t.outcome != "open"]

        if not closed:
            return PlaybookValidationResult(
                playbook_id=playbook_id,
                status="insufficient_data",
                n_trades=0,
                last_updated=datetime.now(timezone.utc),
            )

        n = len(closed)
        hits = sum(1 for t in closed if t.outcome == "hit_1r")
        win_rate = hits / n

        r_values = [t.r_achieved for t in closed if t.r_achieved is not None]
        avg_r = statistics.mean(r_values) if r_values else None

        min_sample = self._get_min_sample_size(playbook_id)
        status = self._evaluate_status(n, min_sample, win_rate, avg_r)

        return PlaybookValidationResult(
            playbook_id=playbook_id,
            status=status,
            n_trades=n,
            win_rate=round(win_rate, 3),
            avg_r=round(avg_r, 3) if avg_r is not None else None,
            last_updated=datetime.now(timezone.utc),
        )

    def write_evidence_to_playbook(
        self,
        result: PlaybookValidationResult,
        judge_notes: Optional[str] = None,
        min_trades: int | None = None,
    ) -> None:
        """Update the ``## Validation Evidence`` section in the playbook .md.

        Skips writing when ``result.n_trades`` is below *min_trades*
        (default: ``PLAYBOOK_MIN_EVIDENCE_WRITE`` env var or 5).
        """
        import os

        threshold = min_trades if min_trades is not None else int(
            os.environ.get("PLAYBOOK_MIN_EVIDENCE_WRITE", str(MIN_EVIDENCE_WRITE))
        )

        if result.n_trades < threshold:
            logger.debug(
                "Skipping evidence write for %s: n_trades=%d < threshold=%d",
                result.playbook_id, result.n_trades, threshold,
            )
            return

        path = self.playbook_dir / f"{result.playbook_id}.md"
        if not path.exists():
            logger.warning("Playbook not found, cannot write evidence: %s", path)
            return

        content = path.read_text(encoding="utf-8")
        evidence_block = self._render_evidence_block(result, judge_notes)

        if "## Validation Evidence" in content:
            content = re.sub(
                r"## Validation Evidence\n.*?(?=\n##|\Z)",
                evidence_block,
                content,
                flags=re.DOTALL,
            )
        else:
            content = content.rstrip() + "\n\n" + evidence_block + "\n"

        path.write_text(content, encoding="utf-8")
        logger.info(
            "Updated playbook evidence: %s (status=%s, n=%d)",
            result.playbook_id, result.status, result.n_trades,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_evidence_block(
        self,
        result: PlaybookValidationResult,
        judge_notes: Optional[str],
    ) -> str:
        ts = result.last_updated.isoformat() if result.last_updated else "null"
        win_rate_str = str(result.win_rate) if result.win_rate is not None else "null"
        avg_r_str = str(result.avg_r) if result.avg_r is not None else "null"
        median_bars_str = (
            str(result.median_bars_to_outcome)
            if result.median_bars_to_outcome is not None
            else "null"
        )
        notes_str = judge_notes or (
            result.judge_notes if result.judge_notes else "null"
        )
        return (
            "## Validation Evidence\n"
            "<!-- Auto-updated by PlaybookOutcomeAggregator. Do not edit manually. -->\n"
            f"status: {result.status}\n"
            f"n_trades: {result.n_trades}\n"
            f"win_rate: {win_rate_str}\n"
            f"avg_r: {avg_r_str}\n"
            f"median_bars_to_outcome: {median_bars_str}\n"
            f"last_updated: {ts}\n"
            f"judge_notes: {notes_str}\n"
        )

    def _get_min_sample_size(self, playbook_id: str) -> int:
        """Read min_sample_size from the playbook frontmatter, default 20."""
        path = self.playbook_dir / f"{playbook_id}.md"
        if not path.exists():
            return 20
        try:
            text = path.read_text(encoding="utf-8")
            match = re.search(r"^min_sample_size:\s*(\d+)", text, re.MULTILINE)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return 20

    def _evaluate_status(
        self,
        n: int,
        min_sample: int,
        win_rate: float,
        avg_r: Optional[float],
    ) -> str:
        if n < min_sample:
            return "insufficient_data"
        # Validated: win_rate >= 50% and avg_r >= 0.6 (if known)
        if win_rate >= 0.50 and (avg_r is None or avg_r >= 0.6):
            return "validated"
        # Refuted: clearly losing edge
        if win_rate < 0.40:
            return "refuted"
        return "mixed"
