"""Unit tests for PlaybookOutcomeAggregator (Runbook 48)."""

from __future__ import annotations

import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from schemas.research_budget import ResearchTrade
from services.playbook_outcome_aggregator import PlaybookOutcomeAggregator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PLAYBOOK_MD = """\
---
title: Bollinger Squeeze
type: playbook
min_sample_size: 5
playbook_id: bollinger_squeeze
---
# Bollinger Squeeze

Patterns
- Compression followed by expansion.

Notes
- Avoid forcing direction.

## Validation Evidence
<!-- Auto-updated by PlaybookOutcomeAggregator. Do not edit manually. -->
status: insufficient_data
n_trades: 0
win_rate: null
avg_r: null
median_bars_to_outcome: null
last_updated: null
judge_notes: null
"""


def _make_trade(
    outcome: str,
    r_achieved: float | None = None,
    playbook_id: str = "bollinger_squeeze",
) -> ResearchTrade:
    return ResearchTrade(
        experiment_id="exp-001",
        playbook_id=playbook_id,
        symbol="BTC-USD",
        direction="long",
        entry_price=50_000.0,
        qty=0.01,
        entry_ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        outcome=outcome,
        r_achieved=r_achieved,
    )


@pytest.fixture
def tmp_playbook_dir(tmp_path: Path) -> Path:
    """Create a temp dir with a sample bollinger_squeeze.md playbook."""
    playbook = tmp_path / "bollinger_squeeze.md"
    playbook.write_text(SAMPLE_PLAYBOOK_MD, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# aggregate() — statistics computation
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_empty_trades(self) -> None:
        agg = PlaybookOutcomeAggregator()
        result = agg.aggregate("bollinger_squeeze", [])
        assert result.status == "insufficient_data"
        assert result.n_trades == 0
        assert result.win_rate is None

    def test_only_open_trades_ignored(self) -> None:
        trades = [_make_trade("open") for _ in range(5)]
        agg = PlaybookOutcomeAggregator()
        result = agg.aggregate("bollinger_squeeze", trades)
        assert result.status == "insufficient_data"
        assert result.n_trades == 0

    def test_win_rate_computation(self) -> None:
        trades = (
            [_make_trade("hit_1r", r_achieved=1.0)] * 3
            + [_make_trade("hit_stop", r_achieved=-1.0)] * 2
        )
        agg = PlaybookOutcomeAggregator()
        result = agg.aggregate("bollinger_squeeze", trades)
        assert result.n_trades == 5
        assert result.win_rate == pytest.approx(0.6, abs=0.001)

    def test_avg_r_computation(self) -> None:
        trades = (
            [_make_trade("hit_1r", r_achieved=1.0)] * 2
            + [_make_trade("hit_stop", r_achieved=-1.0)] * 2
        )
        agg = PlaybookOutcomeAggregator()
        result = agg.aggregate("bollinger_squeeze", trades)
        assert result.avg_r == pytest.approx(0.0, abs=0.001)

    def test_filters_by_playbook_id(self) -> None:
        trades = [
            _make_trade("hit_1r", r_achieved=1.0, playbook_id="bollinger_squeeze"),
            _make_trade("hit_stop", r_achieved=-1.0, playbook_id="other_playbook"),
        ]
        agg = PlaybookOutcomeAggregator()
        result = agg.aggregate("bollinger_squeeze", trades)
        assert result.n_trades == 1
        assert result.win_rate == 1.0

    def test_status_validated(self, tmp_playbook_dir: Path) -> None:
        # 5 trades, 4 wins → win_rate=0.8, avg_r=0.9 → validated
        trades = (
            [_make_trade("hit_1r", r_achieved=1.0)] * 4
            + [_make_trade("hit_stop", r_achieved=-0.5)]
        )
        agg = PlaybookOutcomeAggregator(playbook_dir=tmp_playbook_dir)
        result = agg.aggregate("bollinger_squeeze", trades)
        assert result.status == "validated"

    def test_status_refuted(self, tmp_playbook_dir: Path) -> None:
        # 5 trades, 1 win → win_rate=0.2 → refuted
        trades = (
            [_make_trade("hit_1r", r_achieved=1.0)]
            + [_make_trade("hit_stop", r_achieved=-1.0)] * 4
        )
        agg = PlaybookOutcomeAggregator(playbook_dir=tmp_playbook_dir)
        result = agg.aggregate("bollinger_squeeze", trades)
        assert result.status == "refuted"

    def test_status_mixed(self, tmp_playbook_dir: Path) -> None:
        # 5 trades, 2 wins → win_rate=0.4 → mixed
        trades = (
            [_make_trade("hit_1r", r_achieved=1.0)] * 2
            + [_make_trade("hit_stop", r_achieved=-1.0)] * 3
        )
        agg = PlaybookOutcomeAggregator(playbook_dir=tmp_playbook_dir)
        result = agg.aggregate("bollinger_squeeze", trades)
        assert result.status == "mixed"

    def test_insufficient_data_below_min_sample(self, tmp_playbook_dir: Path) -> None:
        # min_sample_size=5; 4 trades total is insufficient
        trades = [_make_trade("hit_1r", r_achieved=1.0)] * 4
        agg = PlaybookOutcomeAggregator(playbook_dir=tmp_playbook_dir)
        result = agg.aggregate("bollinger_squeeze", trades)
        assert result.status == "insufficient_data"


# ---------------------------------------------------------------------------
# write_evidence_to_playbook()
# ---------------------------------------------------------------------------

class TestWriteEvidence:
    def test_updates_existing_section(self, tmp_playbook_dir: Path) -> None:
        trades = [_make_trade("hit_1r", r_achieved=1.0)] * 5
        agg = PlaybookOutcomeAggregator(playbook_dir=tmp_playbook_dir)
        result = agg.aggregate("bollinger_squeeze", trades)
        agg.write_evidence_to_playbook(result, min_trades=5)

        content = (tmp_playbook_dir / "bollinger_squeeze.md").read_text()
        assert "status: validated" in content
        assert "n_trades: 5" in content
        assert "win_rate: 1.0" in content

    def test_skips_below_min_trades(self, tmp_playbook_dir: Path) -> None:
        trades = [_make_trade("hit_1r", r_achieved=1.0)] * 3
        agg = PlaybookOutcomeAggregator(playbook_dir=tmp_playbook_dir)
        result = agg.aggregate("bollinger_squeeze", trades)
        agg.write_evidence_to_playbook(result, min_trades=5)

        content = (tmp_playbook_dir / "bollinger_squeeze.md").read_text()
        # Should NOT have been updated — still shows initial "insufficient_data"
        assert "n_trades: 0" in content

    def test_does_not_corrupt_surrounding_sections(self, tmp_playbook_dir: Path) -> None:
        trades = [_make_trade("hit_1r", r_achieved=1.0)] * 5
        agg = PlaybookOutcomeAggregator(playbook_dir=tmp_playbook_dir)
        result = agg.aggregate("bollinger_squeeze", trades)
        agg.write_evidence_to_playbook(result, min_trades=5)

        content = (tmp_playbook_dir / "bollinger_squeeze.md").read_text()
        # Frontmatter, title, and Notes section must still be intact
        assert "## Bollinger Squeeze" in content or "# Bollinger Squeeze" in content
        assert "Notes" in content
        assert "Patterns" in content

    def test_missing_playbook_file_does_not_raise(self, tmp_playbook_dir: Path) -> None:
        from schemas.research_budget import PlaybookValidationResult
        result = PlaybookValidationResult(
            playbook_id="nonexistent_playbook",
            status="validated",
            n_trades=10,
            win_rate=0.6,
        )
        agg = PlaybookOutcomeAggregator(playbook_dir=tmp_playbook_dir)
        # Should log a warning and return without raising
        agg.write_evidence_to_playbook(result, min_trades=1)

    def test_judge_notes_written(self, tmp_playbook_dir: Path) -> None:
        trades = [_make_trade("hit_1r", r_achieved=1.0)] * 5
        agg = PlaybookOutcomeAggregator(playbook_dir=tmp_playbook_dir)
        result = agg.aggregate("bollinger_squeeze", trades)
        agg.write_evidence_to_playbook(result, judge_notes="Strong edge confirmed.", min_trades=5)

        content = (tmp_playbook_dir / "bollinger_squeeze.md").read_text()
        assert "Strong edge confirmed." in content

    def test_appends_section_if_missing(self, tmp_path: Path) -> None:
        """Playbook without ## Validation Evidence gets the section appended."""
        playbook = tmp_path / "no_evidence.md"
        playbook.write_text("# No Evidence Playbook\n\nSome text.\n", encoding="utf-8")

        from schemas.research_budget import PlaybookValidationResult
        result = PlaybookValidationResult(
            playbook_id="no_evidence",
            status="validated",
            n_trades=10,
            win_rate=0.6,
        )
        agg = PlaybookOutcomeAggregator(playbook_dir=tmp_path)
        agg.write_evidence_to_playbook(result, min_trades=1)

        content = playbook.read_text()
        assert "## Validation Evidence" in content
        assert "status: validated" in content
