"""Unit tests for ResearchBudgetState and ResearchTrade schemas (Runbook 48)."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from schemas.research_budget import (
    ExperimentAttribution,
    PlaybookValidationResult,
    ResearchBudgetState,
    ResearchTrade,
)
from schemas.judge_feedback import (
    ExperimentSuggestion,
    JudgeAction,
    JudgeActionType,
    PlaybookEditSuggestion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(
    outcome: str = "open",
    r_achieved: float | None = None,
    playbook_id: str = "bollinger_squeeze",
    experiment_id: str = "exp-001",
) -> ResearchTrade:
    return ResearchTrade(
        experiment_id=experiment_id,
        playbook_id=playbook_id,
        symbol="BTC-USD",
        direction="long",
        entry_price=50_000.0,
        qty=0.01,
        entry_ts=datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        outcome=outcome,
        r_achieved=r_achieved,
    )


def _make_budget(cash: float = 1000.0, max_loss_usd: float = 500.0) -> ResearchBudgetState:
    return ResearchBudgetState(
        initial_capital=1000.0,
        cash=cash,
        max_loss_usd=max_loss_usd,
    )


# ---------------------------------------------------------------------------
# ResearchTrade
# ---------------------------------------------------------------------------

class TestResearchTrade:
    def test_defaults(self) -> None:
        t = _make_trade()
        assert t.outcome == "open"
        assert t.r_achieved is None
        assert t.exit_price is None
        assert t.pnl is None
        assert len(t.trade_id) > 0  # uuid generated

    def test_closed_hit(self) -> None:
        t = _make_trade(outcome="hit_1r", r_achieved=1.0)
        assert t.outcome == "hit_1r"
        assert t.r_achieved == 1.0

    def test_closed_stop(self) -> None:
        t = _make_trade(outcome="hit_stop", r_achieved=-1.0)
        assert t.outcome == "hit_stop"
        assert t.r_achieved == -1.0

    def test_invalid_outcome_rejected(self) -> None:
        with pytest.raises(Exception):
            ResearchTrade(
                experiment_id="exp-001",
                playbook_id="bollinger_squeeze",
                symbol="BTC-USD",
                direction="long",
                entry_price=50_000.0,
                qty=0.01,
                entry_ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
                outcome="not_a_valid_outcome",
            )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(Exception):
            ResearchTrade(
                experiment_id="exp-001",
                playbook_id="bollinger_squeeze",
                symbol="BTC-USD",
                direction="long",
                entry_price=50_000.0,
                qty=0.01,
                entry_ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
                unexpected_field="oops",
            )


# ---------------------------------------------------------------------------
# ResearchBudgetState
# ---------------------------------------------------------------------------

class TestResearchBudgetState:
    def test_basic_state(self) -> None:
        budget = _make_budget()
        assert budget.cash == 1000.0
        assert budget.initial_capital == 1000.0
        assert budget.paused is False
        assert budget.total_pnl == 0.0
        assert budget.active_experiment_id is None

    def test_paused_state(self) -> None:
        budget = ResearchBudgetState(
            initial_capital=1000.0,
            cash=400.0,
            max_loss_usd=500.0,
            paused=True,
            pause_reason="max_loss_usd exceeded",
        )
        assert budget.paused is True
        assert budget.pause_reason == "max_loss_usd exceeded"

    def test_with_trades(self) -> None:
        t1 = _make_trade(outcome="hit_1r", r_achieved=1.0)
        t2 = _make_trade(outcome="hit_stop", r_achieved=-1.0)
        budget = ResearchBudgetState(
            initial_capital=1000.0,
            cash=1000.0,
            max_loss_usd=500.0,
            trades=[t1, t2],
        )
        assert len(budget.trades) == 2

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(Exception):
            ResearchBudgetState(
                initial_capital=1000.0,
                cash=1000.0,
                max_loss_usd=500.0,
                unknown_field="oops",
            )

    def test_serialization_roundtrip(self) -> None:
        budget = ResearchBudgetState(
            initial_capital=1000.0,
            cash=900.0,
            max_loss_usd=500.0,
            active_experiment_id="exp-001",
            total_pnl=-100.0,
        )
        dumped = budget.model_dump()
        restored = ResearchBudgetState.model_validate(dumped)
        assert restored.cash == 900.0
        assert restored.total_pnl == -100.0
        assert restored.active_experiment_id == "exp-001"


# ---------------------------------------------------------------------------
# ExperimentAttribution
# ---------------------------------------------------------------------------

class TestExperimentAttribution:
    def test_basic(self) -> None:
        attr = ExperimentAttribution(
            signal_event_id="sig-001",
            experiment_id="exp-001",
            playbook_id="bollinger_squeeze",
            hypothesis="BB squeeze leads to breakout within 48h",
        )
        assert attr.experiment_id == "exp-001"
        assert attr.setup_event_id is None

    def test_no_signal_or_setup(self) -> None:
        # Both IDs are optional
        attr = ExperimentAttribution(
            experiment_id="exp-002",
            playbook_id=None,
            hypothesis="Testing",
        )
        assert attr.signal_event_id is None
        assert attr.playbook_id is None


# ---------------------------------------------------------------------------
# PlaybookValidationResult
# ---------------------------------------------------------------------------

class TestPlaybookValidationResult:
    def test_insufficient_data(self) -> None:
        r = PlaybookValidationResult(
            playbook_id="bollinger_squeeze",
            status="insufficient_data",
            n_trades=0,
        )
        assert r.win_rate is None
        assert r.avg_r is None

    def test_validated(self) -> None:
        r = PlaybookValidationResult(
            playbook_id="bollinger_squeeze",
            status="validated",
            n_trades=25,
            win_rate=0.6,
            avg_r=0.9,
        )
        assert r.status == "validated"

    def test_refuted(self) -> None:
        r = PlaybookValidationResult(
            playbook_id="rsi_extremes",
            status="refuted",
            n_trades=22,
            win_rate=0.36,
        )
        assert r.status == "refuted"


# ---------------------------------------------------------------------------
# JudgeActionType literal and new action payloads (Runbook 48)
# ---------------------------------------------------------------------------

class TestJudgeActionTypes:
    def test_experiment_suggestion_model(self) -> None:
        s = ExperimentSuggestion(
            playbook_id="bollinger_squeeze",
            hypothesis="Band squeeze predicts breakout",
            target_symbols=["BTC-USD"],
            trigger_categories=["volatility_breakout"],
            rationale="Win rate below 45% on breakout triggers for 15 trades",
        )
        assert s.min_sample_size == 20
        assert s.max_loss_usd == 50.0

    def test_playbook_edit_suggestion_model(self) -> None:
        e = PlaybookEditSuggestion(
            playbook_id="bollinger_squeeze",
            section="Notes",
            suggested_text="Updated guidance based on evidence",
            evidence_summary="win_rate=0.52 over 21 trades",
        )
        assert e.requires_human_review is True
        assert e.section == "Notes"

    def test_judge_action_with_experiment_suggestion(self) -> None:
        action = JudgeAction(
            action_id="judge-action-001",
            recommended_action="hold",
            experiment_suggestion=ExperimentSuggestion(
                playbook_id="bollinger_squeeze",
                hypothesis="Test",
                target_symbols=["ETH-USD"],
                trigger_categories=["volatility_breakout"],
                rationale="Low win rate observed",
            ),
        )
        assert action.experiment_suggestion is not None
        assert action.playbook_edit_suggestion is None

    def test_judge_action_with_playbook_edit(self) -> None:
        action = JudgeAction(
            action_id="judge-action-002",
            recommended_action="hold",
            playbook_edit_suggestion=PlaybookEditSuggestion(
                playbook_id="rsi_extremes",
                section="Notes",
                suggested_text="Remove volume filter — evidence shows no edge.",
                evidence_summary="win_rate_high_vol=0.48, win_rate_low_vol=0.47 over 30 trades",
            ),
        )
        assert action.playbook_edit_suggestion is not None
        assert action.experiment_suggestion is None


# ---------------------------------------------------------------------------
# Research budget auto-pause logic
# ---------------------------------------------------------------------------

class TestResearchBudgetAutoPause:
    """Verify that callers can detect when max_loss_usd is exceeded."""

    def test_should_pause_when_loss_exceeds_limit(self) -> None:
        budget = ResearchBudgetState(
            initial_capital=1000.0,
            cash=400.0,  # Lost 600 from initial 1000
            max_loss_usd=500.0,
            total_pnl=-600.0,
        )
        # Loss = initial - cash = 600; exceeds max_loss_usd = 500
        loss = budget.initial_capital - budget.cash
        should_pause = loss >= budget.max_loss_usd
        assert should_pause is True

    def test_should_not_pause_when_within_limit(self) -> None:
        budget = ResearchBudgetState(
            initial_capital=1000.0,
            cash=600.0,
            max_loss_usd=500.0,
            total_pnl=-400.0,
        )
        loss = budget.initial_capital - budget.cash
        should_pause = loss >= budget.max_loss_usd
        assert should_pause is False


# ---------------------------------------------------------------------------
# SignalEvent research attribution fields
# ---------------------------------------------------------------------------

class TestSignalEventResearchFields:
    """Verify that SignalEvent now accepts experiment_id and playbook_id."""

    def test_signal_event_with_research_tags(self) -> None:
        from datetime import datetime, timezone
        from schemas.signal_event import SignalEvent

        sig = SignalEvent(
            engine_version="1.0.0",
            ts=datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
            valid_until=datetime(2026, 1, 3, 12, tzinfo=timezone.utc),
            timeframe="1h",
            symbol="BTC-USD",
            direction="long",
            trigger_id="trig-001",
            strategy_type="compression_breakout",
            regime_snapshot_hash="abc123",
            entry_price=50000.0,
            stop_price_abs=49000.0,
            target_price_abs=52000.0,
            risk_r_multiple=2.0,
            expected_hold_bars=24,
            thesis="Breakout above compression band",
            experiment_id="exp-001",
            playbook_id="bollinger_squeeze",
        )
        assert sig.experiment_id == "exp-001"
        assert sig.playbook_id == "bollinger_squeeze"

    def test_signal_event_research_fields_optional(self) -> None:
        from datetime import datetime, timezone
        from schemas.signal_event import SignalEvent

        sig = SignalEvent(
            engine_version="1.0.0",
            ts=datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
            valid_until=datetime(2026, 1, 3, 12, tzinfo=timezone.utc),
            timeframe="1h",
            symbol="BTC-USD",
            direction="long",
            trigger_id="trig-002",
            strategy_type="trend_continuation",
            regime_snapshot_hash="def456",
            entry_price=50000.0,
            stop_price_abs=49000.0,
            target_price_abs=52000.0,
            risk_r_multiple=2.0,
            expected_hold_bars=24,
            thesis="Normal trade, no research tag",
        )
        assert sig.experiment_id is None
        assert sig.playbook_id is None
