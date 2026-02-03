"""Learning gate evaluator: checks market conditions and kill switches
to determine whether learning-book trades should be allowed."""

from __future__ import annotations

from typing import Any, Dict

from schemas.learning_gate import LearningGateStatus, LearningGateThresholds, LearningKillSwitchConfig


class LearningGateEvaluator:
    """Evaluates market conditions and cumulative performance to decide
    whether the learning gate is open or closed.

    Call ``evaluate()`` before each bar to get the current gate status.
    Call ``record_learning_trade()`` after each learning fill to update
    kill-switch counters.
    """

    def __init__(
        self,
        thresholds: LearningGateThresholds | None = None,
        kill_switches: LearningKillSwitchConfig | None = None,
    ) -> None:
        self.thresholds = thresholds or LearningGateThresholds()
        self.kill_switches = kill_switches or LearningKillSwitchConfig()
        # Kill-switch state
        self._cumulative_learning_loss: float = 0.0
        self._consecutive_losses: int = 0
        self._killed: bool = False

    def record_learning_trade(self, pnl: float) -> None:
        """Update kill-switch counters after a learning trade closes."""
        if pnl < 0:
            self._cumulative_learning_loss += abs(pnl)
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def reset_daily(self) -> None:
        """Reset daily counters (call at day boundaries)."""
        self._cumulative_learning_loss = 0.0
        self._consecutive_losses = 0
        self._killed = False

    def evaluate(
        self,
        equity: float,
        realized_vol: float | None = None,
        median_vol: float | None = None,
        volume: float | None = None,
        median_volume: float | None = None,
        spread_pct: float | None = None,
    ) -> LearningGateStatus:
        """Check all gate conditions and return status.

        Args:
            equity: Current portfolio equity (for kill-switch % calculation).
            realized_vol: Current realized volatility of the asset.
            median_vol: Median realized volatility (rolling baseline).
            volume: Current bar/period volume.
            median_volume: Median volume (rolling baseline).
            spread_pct: Current bid-ask spread as % of mid price.
        """
        reasons: list[str] = []

        # If previously killed this day, stay closed
        if self._killed:
            return LearningGateStatus(open=False, reasons=["kill_switch_active"])

        # --- Market condition gates ---
        if realized_vol is not None and median_vol is not None and median_vol > 0:
            vol_multiple = realized_vol / median_vol
            if vol_multiple >= self.thresholds.volatility_spike_multiple:
                reasons.append("volatility_spike")

        if volume is not None and median_volume is not None and median_volume > 0:
            vol_fraction = volume / median_volume
            if vol_fraction <= self.thresholds.liquidity_thin_volume_multiple:
                reasons.append("liquidity_thin")

        if spread_pct is not None:
            if spread_pct >= self.thresholds.spread_wide_pct:
                reasons.append("spread_wide")

        # --- Kill switches ---
        if equity > 0:
            loss_pct = (self._cumulative_learning_loss / equity) * 100.0
            if loss_pct >= self.kill_switches.daily_loss_limit_pct:
                reasons.append("daily_loss_limit")
                self._killed = True

        if self._consecutive_losses >= self.kill_switches.consecutive_loss_limit:
            reasons.append("consecutive_loss_limit")
            self._killed = True

        return LearningGateStatus(open=len(reasons) == 0, reasons=reasons)
