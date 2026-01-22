"""Deterministic trade quality evaluation metrics.

This module provides reusable trade quality metrics that work across:
- Backtesting
- Paper trading
- Live trading

All metrics are computed deterministically from trade data - no LLM involved.
The results are fed to the judge for context-aware evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence
import math


@dataclass
class TradeMetrics:
    """Deterministic metrics computed from a set of trades."""

    # Core counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # P&L metrics
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Ratios
    win_rate: float = 0.0
    loss_rate: float = 0.0
    profit_factor: float = 0.0
    risk_reward_ratio: float = 0.0

    # Frequency metrics
    trades_per_hour: float = 0.0
    avg_hold_minutes: float = 0.0

    # Category breakdown
    category_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Warning signals
    emergency_exit_count: int = 0
    emergency_exit_pct: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0

    # Quality score (0-100, deterministic)
    quality_score: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "total_pnl": self.total_pnl,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "profit_factor": self.profit_factor,
            "risk_reward_ratio": self.risk_reward_ratio,
            "trades_per_hour": self.trades_per_hour,
            "avg_hold_minutes": self.avg_hold_minutes,
            "category_stats": self.category_stats,
            "emergency_exit_count": self.emergency_exit_count,
            "emergency_exit_pct": self.emergency_exit_pct,
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_losses": self.max_consecutive_losses,
            "quality_score": self.quality_score,
        }


def compute_trade_metrics(
    fills: Sequence[Mapping[str, Any]],
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    trigger_catalog: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> TradeMetrics:
    """Compute deterministic trade quality metrics from fills.

    Args:
        fills: List of fill records with timestamp, pnl, reason, etc.
        start_time: Start of evaluation window (for frequency calculation)
        end_time: End of evaluation window
        trigger_catalog: Optional mapping of trigger_id -> {category, ...}

    Returns:
        TradeMetrics with all computed values
    """
    metrics = TradeMetrics()

    if not fills:
        return metrics

    # Filter to window if specified
    filtered_fills = list(fills)
    if start_time or end_time:
        filtered_fills = []
        for fill in fills:
            ts = fill.get("timestamp")
            if ts is None:
                continue
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if start_time and ts < start_time:
                continue
            if end_time and ts > end_time:
                continue
            filtered_fills.append(fill)

    if not filtered_fills:
        return metrics

    metrics.total_trades = len(filtered_fills)

    # Compute P&L metrics
    pnl_values = []
    consecutive_losses = 0
    max_consecutive = 0
    category_data: Dict[str, Dict[str, Any]] = {}

    for fill in filtered_fills:
        pnl = float(fill.get("pnl", 0.0) or 0.0)
        pnl_values.append(pnl)

        # Win/loss classification
        if pnl > 0.01:  # Small threshold for "winning"
            metrics.winning_trades += 1
            metrics.gross_profit += pnl
            metrics.largest_win = max(metrics.largest_win, pnl)
            consecutive_losses = 0
        elif pnl < -0.01:
            metrics.losing_trades += 1
            metrics.gross_loss += abs(pnl)
            metrics.largest_loss = max(metrics.largest_loss, abs(pnl))
            consecutive_losses += 1
            max_consecutive = max(max_consecutive, consecutive_losses)
        else:
            metrics.breakeven_trades += 1
            consecutive_losses = 0

        # Track category stats
        reason = fill.get("reason", "")
        category = _get_category(reason, trigger_catalog)
        if category not in category_data:
            category_data[category] = {
                "count": 0,
                "wins": 0,
                "losses": 0,
                "pnl": 0.0,
            }
        category_data[category]["count"] += 1
        category_data[category]["pnl"] += pnl
        if pnl > 0.01:
            category_data[category]["wins"] += 1
        elif pnl < -0.01:
            category_data[category]["losses"] += 1

        # Track emergency exits
        if "emergency" in reason.lower() or reason.endswith("_exit"):
            metrics.emergency_exit_count += 1

    metrics.total_pnl = sum(pnl_values)
    metrics.consecutive_losses = consecutive_losses
    metrics.max_consecutive_losses = max_consecutive

    # Compute ratios
    if metrics.total_trades > 0:
        metrics.win_rate = metrics.winning_trades / metrics.total_trades
        metrics.loss_rate = metrics.losing_trades / metrics.total_trades
        metrics.emergency_exit_pct = metrics.emergency_exit_count / metrics.total_trades

    if metrics.winning_trades > 0:
        metrics.avg_win = metrics.gross_profit / metrics.winning_trades

    if metrics.losing_trades > 0:
        metrics.avg_loss = metrics.gross_loss / metrics.losing_trades

    if metrics.gross_loss > 0:
        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
    elif metrics.gross_profit > 0:
        metrics.profit_factor = float("inf")

    if metrics.avg_loss > 0:
        metrics.risk_reward_ratio = metrics.avg_win / metrics.avg_loss

    # Compute frequency metrics
    if start_time and end_time:
        hours = max(0.01, (end_time - start_time).total_seconds() / 3600)
        metrics.trades_per_hour = metrics.total_trades / hours

    # Compute category stats with win rates
    for cat, data in category_data.items():
        count = data["count"]
        data["win_rate"] = data["wins"] / count if count > 0 else 0.0
        data["avg_pnl"] = data["pnl"] / count if count > 0 else 0.0
    metrics.category_stats = category_data

    # Compute deterministic quality score
    metrics.quality_score = _compute_quality_score(metrics)

    return metrics


def _get_category(reason: str, catalog: Optional[Mapping[str, Mapping[str, Any]]]) -> str:
    """Extract category from trigger reason."""
    if not reason:
        return "unknown"

    # Strip suffixes
    base_id = reason
    for suffix in ("_exit", "_flat", "_emergency"):
        if base_id.endswith(suffix):
            base_id = base_id[: -len(suffix)]
            break

    # Check catalog
    if catalog and base_id in catalog:
        entry = catalog[base_id]
        if isinstance(entry, dict) and "category" in entry:
            return entry["category"]

    # Infer from naming conventions
    reason_lower = reason.lower()
    if "emergency" in reason_lower:
        return "emergency_exit"
    if "exit" in reason_lower or "flat" in reason_lower:
        return "exit"
    if "trend" in reason_lower:
        return "trend_continuation"
    if "breakout" in reason_lower:
        return "volatility_breakout"
    if "reversion" in reason_lower or "mean" in reason_lower:
        return "mean_reversion"
    if "reversal" in reason_lower:
        return "reversal"

    return "other"


def _compute_quality_score(metrics: TradeMetrics) -> float:
    """Compute a deterministic quality score (0-100) from metrics.

    Scoring factors:
    - Win rate contribution (0-30 points): 50%+ win rate is baseline
    - Profit factor contribution (0-25 points): 1.5+ is good
    - Risk/reward contribution (0-20 points): 1.0+ is neutral
    - Consistency penalty (-15 points): High consecutive losses
    - Emergency exit penalty (-10 points): >20% emergency exits
    """
    score = 50.0  # Baseline

    # Win rate contribution (0-30)
    # 50% = 15 points, 60% = 21 points, 70% = 27 points
    win_rate_contribution = min(30, metrics.win_rate * 50)
    score += win_rate_contribution - 15  # Neutral at 50%

    # Profit factor contribution (0-25)
    # 1.0 = 0 points, 1.5 = 12.5 points, 2.0 = 25 points
    if metrics.profit_factor == float("inf"):
        pf_contribution = 25
    else:
        pf_contribution = min(25, max(0, (metrics.profit_factor - 1.0) * 25))
    score += pf_contribution

    # Risk/reward contribution (0-20)
    # 0.5 = -10, 1.0 = 0, 2.0 = 20
    rr_contribution = min(20, max(-10, (metrics.risk_reward_ratio - 1.0) * 20))
    score += rr_contribution

    # Consistency penalty (0 to -15)
    if metrics.max_consecutive_losses >= 5:
        score -= 15
    elif metrics.max_consecutive_losses >= 3:
        score -= 8
    elif metrics.max_consecutive_losses >= 2:
        score -= 3

    # Emergency exit penalty (0 to -10)
    if metrics.emergency_exit_pct > 0.3:
        score -= 10
    elif metrics.emergency_exit_pct > 0.2:
        score -= 6
    elif metrics.emergency_exit_pct > 0.1:
        score -= 3

    # No trades penalty
    if metrics.total_trades == 0:
        score = 40.0  # Neutral-low for no data

    return max(0.0, min(100.0, round(score, 1)))


def format_metrics_for_judge(metrics: TradeMetrics) -> str:
    """Format metrics as text for the judge prompt."""
    lines = [
        "DETERMINISTIC TRADE QUALITY METRICS:",
        f"- Total trades: {metrics.total_trades}",
        f"- Win rate: {metrics.win_rate * 100:.1f}% ({metrics.winning_trades}W / {metrics.losing_trades}L / {metrics.breakeven_trades}BE)",
        f"- Profit factor: {metrics.profit_factor:.2f}" if metrics.profit_factor != float("inf") else "- Profit factor: ∞ (no losses)",
        f"- Risk/reward ratio: {metrics.risk_reward_ratio:.2f}",
        f"- Total P&L: ${metrics.total_pnl:.2f}",
        f"- Avg win: ${metrics.avg_win:.2f}, Avg loss: ${metrics.avg_loss:.2f}",
        f"- Largest win: ${metrics.largest_win:.2f}, Largest loss: ${metrics.largest_loss:.2f}",
        f"- Emergency exits: {metrics.emergency_exit_count} ({metrics.emergency_exit_pct * 100:.1f}%)",
        f"- Max consecutive losses: {metrics.max_consecutive_losses}",
        f"- Quality score (deterministic): {metrics.quality_score}/100",
    ]

    if metrics.category_stats:
        lines.append("\nCATEGORY BREAKDOWN:")
        for cat, data in sorted(metrics.category_stats.items(), key=lambda x: -x[1]["count"]):
            wr = data.get("win_rate", 0) * 100
            lines.append(f"  - {cat}: {data['count']} trades, {wr:.0f}% win rate, ${data['pnl']:.2f} P&L")

    return "\n".join(lines)


@dataclass
class PositionQuality:
    """Quality assessment for current positions."""

    symbol: str
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    hold_duration_hours: float
    is_underwater: bool = False
    is_extended: bool = False  # Held too long without profit
    risk_score: float = 50.0  # 0-100, lower is riskier


def assess_position_quality(
    positions: Mapping[str, float],
    entry_prices: Mapping[str, float],
    current_prices: Mapping[str, float],
    position_opened_times: Mapping[str, datetime],
    current_time: datetime,
) -> List[PositionQuality]:
    """Assess quality of current open positions."""
    assessments = []

    for symbol, qty in positions.items():
        if abs(qty) < 1e-9:
            continue

        entry = entry_prices.get(symbol, 0.0)
        current = current_prices.get(symbol, entry)
        opened_at = position_opened_times.get(symbol)

        if entry <= 0:
            continue

        unrealized = (current - entry) * qty
        unrealized_pct = ((current / entry) - 1) * 100 if entry > 0 else 0.0

        hold_hours = 0.0
        if opened_at:
            hold_hours = (current_time - opened_at).total_seconds() / 3600

        is_underwater = unrealized < 0
        is_extended = hold_hours > 24 and unrealized_pct < 1.0  # Held >24h without 1% gain

        # Risk score: lower for underwater/extended positions
        risk_score = 50.0
        if is_underwater:
            risk_score -= min(30, abs(unrealized_pct) * 3)  # -3 points per % underwater
        if is_extended:
            risk_score -= 10
        if hold_hours > 48:
            risk_score -= 10
        risk_score = max(0.0, min(100.0, risk_score))

        assessments.append(
            PositionQuality(
                symbol=symbol,
                entry_price=entry,
                current_price=current,
                unrealized_pnl=unrealized,
                unrealized_pnl_pct=unrealized_pct,
                hold_duration_hours=hold_hours,
                is_underwater=is_underwater,
                is_extended=is_extended,
                risk_score=risk_score,
            )
        )

    return assessments


def format_position_quality_for_judge(assessments: List[PositionQuality]) -> str:
    """Format position assessments for judge prompt."""
    if not assessments:
        return "POSITION QUALITY: No open positions"

    lines = ["POSITION QUALITY:"]
    for pq in assessments:
        status = []
        if pq.is_underwater:
            status.append("UNDERWATER")
        if pq.is_extended:
            status.append("EXTENDED")
        status_str = f" [{', '.join(status)}]" if status else ""

        lines.append(
            f"  - {pq.symbol}: {pq.unrealized_pnl_pct:+.2f}% (${pq.unrealized_pnl:+.2f}), "
            f"held {pq.hold_duration_hours:.1f}h, risk={pq.risk_score:.0f}{status_str}"
        )

    return "\n".join(lines)
