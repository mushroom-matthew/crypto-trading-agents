"""Schemas for research budget in paper trading (Runbook 48)."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ResearchTrade(BaseModel):
    """Record of a single research trade charged against the research budget."""

    model_config = {"extra": "forbid"}

    trade_id: str = Field(default_factory=lambda: str(uuid4()))
    experiment_id: str
    playbook_id: Optional[str]
    symbol: str
    direction: str  # "long" | "short"
    entry_price: float
    exit_price: Optional[float] = None
    qty: float
    entry_ts: datetime
    exit_ts: Optional[datetime] = None
    pnl: Optional[float] = None
    outcome: Literal["hit_1r", "hit_stop", "ttl_expired", "open"] = "open"
    r_achieved: Optional[float] = None
    entry_indicators: Dict = Field(default_factory=dict)


class ResearchBudgetState(BaseModel):
    """Isolated capital pool and position state for research trades."""

    model_config = {"extra": "forbid"}

    initial_capital: float
    cash: float
    positions: Dict = Field(default_factory=dict)
    active_experiment_id: Optional[str] = None
    active_playbook_id: Optional[str] = None
    trades: List[ResearchTrade] = Field(default_factory=list)
    total_pnl: float = 0.0
    max_loss_usd: float  # from MetricSpec.max_loss_usd; pauses on breach
    paused: bool = False
    pause_reason: Optional[str] = None


class ExperimentAttribution(BaseModel):
    """Maps a SignalEvent or SetupEvent to an experiment and playbook."""

    model_config = {"extra": "forbid"}

    signal_event_id: Optional[str] = None
    setup_event_id: Optional[str] = None
    experiment_id: str
    playbook_id: Optional[str]
    hypothesis: str


class PlaybookValidationResult(BaseModel):
    """Computed validation stats for one playbook from closed research trades."""

    model_config = {"extra": "forbid"}

    playbook_id: str
    status: Literal["insufficient_data", "validated", "refuted", "mixed"]
    n_trades: int
    win_rate: Optional[float] = None
    avg_r: Optional[float] = None
    median_bars_to_outcome: Optional[float] = None
    last_updated: Optional[datetime] = None
    judge_notes: Optional[str] = None
