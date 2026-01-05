"""Live daily report generation - matches backtest daily report format."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import BlockEvent, Order, RiskAllocation

logger = logging.getLogger(__name__)


async def generate_live_daily_report(
    db: AsyncSession,
    date: str,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate backtest-style daily report for live trading.

    Args:
        db: Database session
        date: Date in YYYY-MM-DD format
        run_id: Optional run ID to filter by (defaults to latest run)

    Returns:
        Dict matching backtest daily report format with:
        - trade_count: Number of filled orders
        - blocks: Total blocks
        - block_breakdown: Blocks by reason
        - risk_budget: Budget used/available
        - pnl: Profit/loss metrics
        - win_rate: Win percentage
    """
    # Parse date range
    start_date = datetime.fromisoformat(date)
    end_date = start_date + timedelta(days=1)

    logger.info(f"Generating live daily report for {date} (run_id={run_id})")

    # Query orders for this date range
    orders_query = select(Order).where(
        Order.created_at >= start_date,
        Order.created_at < end_date
    )
    result = await db.execute(orders_query)
    orders = result.scalars().all()

    # Query block events
    blocks_query = select(BlockEvent).where(
        BlockEvent.timestamp >= start_date,
        BlockEvent.timestamp < end_date
    )
    if run_id:
        blocks_query = blocks_query.where(BlockEvent.run_id == run_id)

    result = await db.execute(blocks_query)
    blocks = result.scalars().all()

    # Query risk allocations
    risk_query = select(RiskAllocation).where(
        RiskAllocation.claim_timestamp >= start_date,
        RiskAllocation.claim_timestamp < end_date
    )
    if run_id:
        risk_query = risk_query.where(RiskAllocation.run_id == run_id)

    result = await db.execute(risk_query)
    risk_allocations = result.scalars().all()

    # Compute trade metrics
    filled_orders = [o for o in orders if hasattr(o, 'status') and o.status == 'filled']
    trade_count = len(filled_orders)

    # Calculate P&L from filled orders (simplified)
    total_pnl = Decimal(0)
    wins = 0
    losses = 0

    for order in filled_orders:
        if hasattr(order, 'realized_pnl') and order.realized_pnl:
            pnl = Decimal(str(order.realized_pnl))
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1

    win_rate = (wins / trade_count * 100) if trade_count > 0 else 0.0

    # Block breakdown by reason
    block_breakdown: Dict[str, int] = {}
    for block in blocks:
        reason = block.reason
        block_breakdown[reason] = block_breakdown.get(reason, 0) + 1

    # Risk budget metrics
    total_claimed = sum(Decimal(str(r.claim_amount)) for r in risk_allocations)

    # TODO: Get actual budget from config (hardcoded for now)
    risk_budget_abs = Decimal("1000.0")
    risk_used_abs = float(total_claimed)
    risk_used_pct = float(total_claimed / risk_budget_abs * 100) if risk_budget_abs > 0 else 0.0
    risk_available_abs = float(risk_budget_abs - total_claimed)

    # Build report matching backtest format
    report = {
        "date": date,
        "run_id": run_id,

        # Trade metrics
        "trade_count": trade_count,
        "executed_trades": trade_count,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,

        # P&L metrics
        "total_pnl": float(total_pnl),
        "equity_return_pct": 0.0,  # TODO: Calculate from starting equity

        # Block metrics
        "blocks": len(blocks),
        "block_breakdown": block_breakdown,

        # Risk budget
        "risk_budget": {
            "budget_abs": float(risk_budget_abs),
            "used_abs": risk_used_abs,
            "available_abs": risk_available_abs,
            "used_pct": risk_used_pct,
            "utilization_pct": risk_used_pct,
        },

        # Trade details (for drill-down)
        "orders": [
            {
                "order_id": str(o.order_id),
                "symbol": o.product_id,
                "side": o.side.value if hasattr(o.side, 'value') else str(o.side),
                "qty": float(o.base_size) if hasattr(o, 'base_size') else 0.0,
                "price": float(o.price) if o.price else 0.0,
                "status": getattr(o, 'status', 'unknown'),
                "timestamp": o.created_at.isoformat(),
            }
            for o in orders[:50]  # Limit to 50 most recent
        ],

        # Block details (for drill-down)
        "block_events": [
            {
                "timestamp": b.timestamp.isoformat(),
                "symbol": b.symbol,
                "side": b.side,
                "qty": float(b.qty),
                "reason": b.reason,
                "detail": b.detail,
                "trigger_id": b.trigger_id,
            }
            for b in blocks[:50]  # Limit to 50 most recent
        ],

        # Risk allocation details
        "risk_allocations": [
            {
                "trigger_id": r.trigger_id,
                "claim_timestamp": r.claim_timestamp.isoformat(),
                "claim_amount": float(r.claim_amount),
                "status": r.status.value if hasattr(r, 'status') and hasattr(r.status, 'value') else None,
            }
            for r in risk_allocations[:50]  # Limit to 50 most recent
        ],

        # Metadata
        "generated_at": datetime.utcnow().isoformat(),
    }

    logger.info(
        f"Generated live daily report: {trade_count} trades, "
        f"{len(blocks)} blocks, {risk_used_pct:.1f}% risk used"
    )

    return report
