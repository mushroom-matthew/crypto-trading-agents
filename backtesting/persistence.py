"""Persistence helpers for backtest results and block events."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable

from sqlalchemy import delete, select

from app.db.models import BacktestRun, BacktestStatus, BlockEvent
from app.db.repo import Database


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _status_from_results(status: Any) -> BacktestStatus:
    if isinstance(status, BacktestStatus):
        return status
    if isinstance(status, str):
        try:
            return BacktestStatus(status)
        except ValueError:
            pass
    return BacktestStatus.completed


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def _build_results_payload(results: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(results or {})
    llm_data = payload.get("llm_data") or {}
    results_summary = payload.get("results_summary")
    if not results_summary:
        results_summary = {
            "final_equity": payload.get("final_equity"),
            "equity_return_pct": payload.get("equity_return_pct"),
            "sharpe_ratio": payload.get("sharpe_ratio"),
            "max_drawdown_pct": payload.get("max_drawdown_pct"),
            "win_rate": payload.get("win_rate"),
            "total_trades": payload.get("total_trades"),
            "avg_win": payload.get("avg_win"),
            "avg_loss": payload.get("avg_loss"),
            "profit_factor": payload.get("profit_factor"),
        }
    # Fallback: if still all-null, try the "summary" sub-dict
    summary_sub = payload.get("summary") or {}
    if summary_sub and not any(v is not None for v in results_summary.values()):
        results_summary = {
            "final_equity": summary_sub.get("final_equity"),
            "equity_return_pct": summary_sub.get("equity_return_pct"),
            "sharpe_ratio": summary_sub.get("sharpe_ratio"),
            "max_drawdown_pct": summary_sub.get("max_drawdown_pct"),
            "win_rate": summary_sub.get("win_rate"),
            "total_trades": summary_sub.get("total_trades"),
            "avg_win": summary_sub.get("avg_win"),
            "avg_loss": summary_sub.get("avg_loss"),
            "profit_factor": summary_sub.get("profit_factor"),
        }
    payload["results_summary"] = results_summary
    payload.setdefault("summary", results_summary)
    if "plan_log" not in payload:
        payload["plan_log"] = llm_data.get("plan_log") or []
    if "bar_decisions" not in payload:
        payload["bar_decisions"] = llm_data.get("bar_decisions") or {}
    if "limit_enforcement" not in payload:
        daily_reports = llm_data.get("daily_reports") or []
        payload["limit_enforcement"] = [
            report.get("limit_stats") for report in daily_reports if report.get("limit_stats") is not None
        ]
    return payload


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        coerced = float(value)  # handles numpy floats/decimals
        if not math.isfinite(coerced):
            return None
    except (TypeError, ValueError):
        return value
    return value


def _iter_block_events(results_payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    llm_data = results_payload.get("llm_data") or {}
    daily_reports = llm_data.get("daily_reports") or []
    for report in daily_reports:
        limit_stats = report.get("limit_stats") or {}
        for block in limit_stats.get("blocked_details") or []:
            yield block


async def persist_backtest_results(run_id: str, results: Dict[str, Any]) -> None:
    """Upsert backtest results and emit block_events."""

    db = Database()
    results_payload = _sanitize_json(_build_results_payload(results))
    config = results_payload.get("config") or {}
    status = _status_from_results(results_payload.get("status"))
    started_at = _parse_ts(results_payload.get("started_at"))
    completed_at = _parse_ts(results_payload.get("completed_at"))
    candles_total = results_payload.get("candles_total")
    candles_processed = results_payload.get("candles_processed")
    results_json = json.dumps(results_payload, default=_json_default, allow_nan=False)

    async with db.session() as session:
        existing = await session.execute(select(BacktestRun).where(BacktestRun.run_id == run_id))
        record = existing.scalar_one_or_none()
        if record:
            if started_at is None:
                started_at = record.started_at
            if completed_at is None:
                completed_at = record.completed_at
            if not config:
                try:
                    config = json.loads(record.config) if record.config else {}
                except json.JSONDecodeError:
                    config = {}
            config_json = json.dumps(config, default=_json_default)
            record.config = config_json
            record.status = status
            record.started_at = started_at
            record.completed_at = completed_at
            record.candles_total = candles_total
            record.candles_processed = candles_processed
            record.results = results_json
        else:
            if started_at is None:
                started_at = datetime.now(timezone.utc)
            config_json = json.dumps(config, default=_json_default)
            record = BacktestRun(
                run_id=run_id,
                config=config_json,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                candles_total=candles_total,
                candles_processed=candles_processed,
                results=results_json,
            )
            session.add(record)

        await session.execute(delete(BlockEvent).where(BlockEvent.run_id == run_id))
        block_rows = []
        for block in _iter_block_events(results_payload):
            timestamp = _parse_ts(block.get("timestamp")) or datetime.now(timezone.utc)
            reason = (block.get("reason") or "blocked")[:50]
            qty = block.get("quantity") or block.get("qty") or 0.0
            try:
                qty_value = Decimal(str(qty))
            except Exception:
                qty_value = Decimal("0")
            block_rows.append(
                BlockEvent(
                    timestamp=timestamp,
                    run_id=run_id,
                    correlation_id=block.get("correlation_id"),
                    trigger_id=str(block.get("trigger_id") or "unknown"),
                    symbol=str(block.get("symbol") or "unknown"),
                    side=str(block.get("side") or "unknown"),
                    qty=qty_value,
                    reason=reason,
                    detail=block.get("detail"),
                )
            )
        if block_rows:
            session.add_all(block_rows)
